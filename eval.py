"""
Toxicity evaluation script for LLaDA model.

Generates continuations from real-toxicity-prompts (or a local JSON file),
then scores with a toxicity classifier.

Supports multiple generation methods via ``control.method``:
  - baseline: uncontrolled LLaDA generation
  - tilting: counterfactual logit tilting with polytope risk
  - gradient: gradient-based logit tilting (efficient)
  - best_of_n: generate N candidates, pick lowest polytope risk
  - gradient_bon: gradient control + best-of-N selection
  - bon_toxicity: generate N baseline candidates in batch, pick lowest toxicity
  - pg_dlm: Particle Gibbs with rollout-based partial reward estimates
  - pg_dlm_proxy: Particle Gibbs with learned classifier partial rewards
  - ctmc: CTMC τ-leaping (no semi-AR, no remasking)
  - ctmc_controlled: CTMC τ-leaping with polytope control
"""

import logging
import json
import random
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from generate import (
    generate_with_trace,
)
from .toxicity_scorer import ToxicityScorer
from .methods import (
    METHODS_NEEDING_RISK_MODEL,
    dispatch_generation,
)
from .runtime import (
    is_cuda_oom_error,
    load_control_runtime,
    release_control_runtime,
)
from .prompts import (
    load_llada_model_and_tokenizer,
    load_prompts,
    prepare_prompts_for_model,
    tokenize_prompts_for_generation,
)
from .output import (
    export_oracle_collection,
    get_eval_output_dir,
    resolve_dataset_source,
    serialize_trace_candidate,
    write_toxicity_results,
)
from .validation import validate_config
from utils.distributed_fs import (
    PeerRankFailedError,
    cleanup_gather_artifacts,
    file_barrier,
    wait_for_paths,
    write_json_atomic,
    write_rank_failure_best_effort,
)
from utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    shard_list,
    get_device,
)
from utils.logging import ProgressLogger, get_logger
from utils.torch_helpers import set_seed


def release_generation_resources(
    model: torch.nn.Module | None,
    tokenizer: AutoTokenizer | None,
    device: torch.device | None,
) -> tuple[None, None]:
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    if device is not None and device.type == "cuda":
        torch.cuda.empty_cache()
    return None, None
def run_toxicity_eval(cfg: DictConfig, dist_info: dict[str, int | bool]):
    """Run toxicity evaluation with LLaDA generation and classifier scoring."""
    rank = int(dist_info["rank"])
    output_dir = get_eval_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    # logger: rank-0 only for config/status messages (printed once).
    # progress_logger: all ranks for per-rank progress tracking.
    logger = get_logger("safety_eval", rank=rank)
    progress_logger = get_logger("safety_eval.progress", rank=rank, all_ranks=True)
    model: torch.nn.Module | None = None
    tokenizer: AutoTokenizer | None = None
    device: torch.device | None = None
    runtime = None
    try:
        validate_config(cfg)

        cfg_batch_size = int(cfg.eval.batch_size)
        if cfg_batch_size != 1:
            logger.warning(
                "eval.batch_size=%d overridden to 1 for cross-GPU reproducibility.",
                cfg_batch_size,
            )
        eval_batch_size = 1
        scorer_batch_size_cfg = cfg.scorer.get("batch_size", None)
        scorer_batch_size = (
            int(scorer_batch_size_cfg)
            if scorer_batch_size_cfg is not None
            else eval_batch_size
        )
        gen = cfg.generation
        max_prompt_length = int(cfg.model.max_prompt_length)
        model_mask_id = int(cfg.model.mask_id)
        model_mode = str(cfg.model.mode)
        scorer_model_name = str(cfg.scorer.model_name)
        scorer_toxic_threshold = float(cfg.scorer.toxic_threshold)
        compute_device_mode = str(cfg.compute.get("device", "auto")).lower()

        control_method = str(cfg.control.method).lower()
        control_best_of_n = int(cfg.control.best_of_n)
        oracle_collection_enabled = bool(
            cfg.get("oracle_collection", {}).get("enabled", False)
        )
        oracle_snapshot_step_indices = [
            int(x)
            for x in cfg.get("oracle_collection", {}).get("snapshot_step_indices", [])
        ]
        oracle_num_rho_bins = int(
            cfg.get("oracle_collection", {}).get("num_rho_bins", 8)
        )

        # Set up device
        device = get_device(dist_info, device_mode=compute_device_mode)

        # Per-prompt seed reset is used below for exact reproducibility.
        # A global seed is still set as a safety net for any unguarded random
        # operations (e.g. model weight init, dataset shuffling).
        base_seed = int(cfg.compute.seed)
        set_seed(base_seed)

        # Enable deterministic algorithms for reproducibility.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        # Load model + tokenizer
        model, tokenizer = load_llada_model_and_tokenizer(
            cfg, device=device, logger=logger
        )
        model_name = cfg.model.name_or_path
        control_sampler = str(cfg.control.get("sampler", "llada")).lower()

        runtime = load_control_runtime(
            cfg=cfg,
            device=device,
            logger=logger,
            control_method=control_method,
            control_sampler=control_sampler,
            scorer_model_name=scorer_model_name,
        )
        risk_model_type = runtime.risk_model_type
        loaded_risk_model_kind = getattr(runtime.polytope, "kind", "unknown")
        loaded_feature_pooler_type = runtime.loaded_feature_pooler_type

        # Load prompts (HF dataset or local JSON)
        prompt_records = load_prompts(cfg, logger=logger)
        if not prompt_records:
            raise ValueError("No prompts loaded.")

        # prompt_records: list of (sample_index, prompt_text)
        # We need sequential indices for distributed sharding/gathering,
        # and sample_indices for per-prompt seeding.
        prompt_texts = [text for _, text in prompt_records]
        sample_indices = [idx for idx, _ in prompt_records]

        # Distribute prompts by sequential index for deterministic sharding.
        indexed_local_prompts = shard_list(prompt_texts, dist_info)
        # indexed_local_prompts: list of (seq_idx, prompt_text)

        # Generate in local shards
        local_records: list[tuple[int, str]] = []
        local_truncated_count = 0
        local_collection_records: list[dict[str, object]] = []

        logger.info(
            "Generating continuations for %d prompts " "(method=%s, world_size=%d)...",
            len(prompt_texts),
            control_method,
            int(dist_info["world_size"]),
        )

        iterator = range(0, len(indexed_local_prompts), eval_batch_size)
        for i in ProgressLogger(
            iterator,
            progress_logger,
            desc="Generating",
            log_every_secs=float(cfg.logging.log_every_secs),
            disable=False,
        ):
            batch_items = indexed_local_prompts[i : i + eval_batch_size]
            batch_seq_indices = [item[0] for item in batch_items]
            batch_prompts = [item[1] for item in batch_items]

            model_ready_prompts = prepare_prompts_for_model(
                batch_prompts, tokenizer=tokenizer, model_mode=model_mode
            )

            prompt_batch, batch_truncated_count = tokenize_prompts_for_generation(
                tokenizer,
                model_ready_prompts,
                max_prompt_length=max_prompt_length,
            )
            local_truncated_count += batch_truncated_count

            input_ids = prompt_batch["input_ids"].to(device)
            attention_mask = prompt_batch["attention_mask"].to(device)

            # Reset global RNG to a prompt-specific seed so the same
            # prompt always gets the same random stream, regardless of
            # which GPU processes it or how many GPUs are used.
            # Use the original sample_index (not sequential index) so that
            # a prompt loaded from local JSON gets the same seed as when
            # loaded directly from the HF dataset.
            prompt_seed = base_seed + sample_indices[batch_seq_indices[0]]
            torch.manual_seed(prompt_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(prompt_seed)

            if control_method == "bon_toxicity":
                # Generate N candidates in a single batch for efficiency.
                # Expand prompt from (1, L) to (N, L); Gumbel noise ensures
                # each copy produces a different output (requires temperature > 0).
                n = control_best_of_n
                bon_temp = float(gen.temperature) if float(gen.temperature) > 0 else 0.2
                bon_input_ids = input_ids.expand(n, -1).contiguous()
                bon_attn = (
                    attention_mask.expand(n, -1).contiguous()
                    if attention_mask is not None
                    else None
                )
                with torch.no_grad():
                    if oracle_collection_enabled:
                        out, bon_traces = generate_with_trace(
                            model,
                            bon_input_ids,
                            bon_attn,
                            steps=int(gen.steps),
                            gen_length=int(gen.gen_length),
                            block_length=int(gen.block_length),
                            temperature=bon_temp,
                            cfg_scale=float(gen.cfg_scale),
                            remasking=str(gen.remasking),
                            mask_id=model_mask_id,
                            logits_eos_inf=bool(gen.logits_eos_inf),
                            confidence_eos_eot_inf=bool(gen.confidence_eos_eot_inf),
                            snapshot_step_indices=oracle_snapshot_step_indices,
                        )
                    else:
                        bon_cfg = OmegaConf.merge(cfg, {
                            "control": {"method": "baseline"},
                            "generation": {"temperature": bon_temp},
                        })
                        out = dispatch_generation(
                            bon_cfg,
                            model,
                            bon_input_ids,
                            bon_attn,
                            tokenizer=tokenizer,
                            logger=logger,
                        )
                # out shape: (N, L+gen_length)
                cont_ids = out[:, input_ids.shape[1] :]
                candidate_texts = tokenizer.batch_decode(
                    cont_ids, skip_special_tokens=True
                )
                if oracle_collection_enabled:
                    seq_idx = int(batch_seq_indices[0])
                    prompt_token_ids = input_ids[0][attention_mask[0].bool()].detach().cpu().tolist()
                    prompt_len = len(prompt_token_ids)
                    candidate_records: list[dict[str, object]] = []
                    for cand_idx in range(n):
                        snapshots: list[dict[str, object]] = []
                        for trace in bon_traces:
                            partial_ids = trace["partial_input_ids"][cand_idx].tolist()
                            snapshots.append(
                                serialize_trace_candidate(
                                    partial_input_ids=partial_ids,
                                    prompt_len=prompt_len,
                                    mask_id=model_mask_id,
                                    rho_value=float(trace["rho_value"][cand_idx].item()),
                                    step_index=int(trace["step_index"]),
                                    num_rho_bins=oracle_num_rho_bins,
                                )
                            )
                        candidate_records.append(
                            {
                                "candidate_id": cand_idx,
                                "completion_token_ids": cont_ids[cand_idx].detach().cpu().tolist(),
                                "full_input_ids": out[cand_idx].detach().cpu().tolist(),
                                "snapshots": snapshots,
                            }
                        )
                    local_collection_records.append(
                        {
                            "seq_index": seq_idx,
                            "sample_index": int(sample_indices[seq_idx]),
                            "prompt_text": batch_prompts[0],
                            "model_prompt_text": model_ready_prompts[0],
                            "prompt_token_ids": prompt_token_ids,
                            "control_method": control_method,
                            "candidates": candidate_records,
                        }
                    )
                # Store all N candidates; selection happens after scorer is loaded
                for seq_idx in batch_seq_indices:
                    local_records.append((seq_idx, candidate_texts))
            else:
                with torch.no_grad():
                    if oracle_collection_enabled and control_method == "baseline":
                        out, base_traces = generate_with_trace(
                            model,
                            input_ids,
                            attention_mask,
                            steps=int(gen.steps),
                            gen_length=int(gen.gen_length),
                            block_length=int(gen.block_length),
                            temperature=float(gen.temperature),
                            cfg_scale=float(gen.cfg_scale),
                            remasking=str(gen.remasking),
                            mask_id=model_mask_id,
                            logits_eos_inf=bool(gen.logits_eos_inf),
                            confidence_eos_eot_inf=bool(gen.confidence_eos_eot_inf),
                            snapshot_step_indices=oracle_snapshot_step_indices,
                        )
                    else:
                        out = dispatch_generation(
                            cfg,
                            model,
                            input_ids,
                            attention_mask,
                            tokenizer=tokenizer,
                            polytope=runtime.polytope,
                            encoder=runtime.encoder,
                            attention_pooler=runtime.attention_pooler,
                            classifier=runtime.classifier,
                            advantage_model=runtime.advantage_model,
                            reward_scorer=runtime.reward_scorer,
                            logger=logger,
                        )

                # Score continuation only (exclude prompt tokens)
                continuation_token_ids = out[:, input_ids.shape[1] :]
                generations = tokenizer.batch_decode(
                    continuation_token_ids, skip_special_tokens=True
                )
                if oracle_collection_enabled and control_method == "baseline":
                    seq_idx = int(batch_seq_indices[0])
                    prompt_token_ids = input_ids[0][attention_mask[0].bool()].detach().cpu().tolist()
                    prompt_len = len(prompt_token_ids)
                    snapshots: list[dict[str, object]] = []
                    for trace in base_traces:
                        partial_ids = trace["partial_input_ids"][0].tolist()
                        snapshots.append(
                            serialize_trace_candidate(
                                partial_input_ids=partial_ids,
                                prompt_len=prompt_len,
                                mask_id=model_mask_id,
                                rho_value=float(trace["rho_value"][0].item()),
                                step_index=int(trace["step_index"]),
                                num_rho_bins=oracle_num_rho_bins,
                            )
                        )
                    local_collection_records.append(
                        {
                            "seq_index": seq_idx,
                            "sample_index": int(sample_indices[seq_idx]),
                            "prompt_text": batch_prompts[0],
                            "model_prompt_text": model_ready_prompts[0],
                            "prompt_token_ids": prompt_token_ids,
                            "control_method": control_method,
                            "candidates": [
                                {
                                    "candidate_id": 0,
                                    "completion_token_ids": continuation_token_ids[0].detach().cpu().tolist(),
                                    "full_input_ids": out[0].detach().cpu().tolist(),
                                    "snapshots": snapshots,
                                }
                            ],
                        }
                    )
                for seq_idx, gen_text in zip(batch_seq_indices, generations):
                    local_records.append((seq_idx, gen_text))

        # ── File-based gathering ──────────────────────────────────────
        # Each rank writes its local results to disk so that work is
        # never lost, even if NCCL times out (which happens when ranks
        # finish at slightly different times over very long runs).
        rank_file = output_dir / f".rank_{rank}_records.json"
        write_json_atomic(
            rank_file,
            {"records": local_records, "truncated_count": local_truncated_count},
        )
        if oracle_collection_enabled:
            collection_rank_file = output_dir / f".rank_{rank}_oracle_collection.json"
            write_json_atomic(
                collection_rank_file,
                {"records": local_collection_records},
            )
        progress_logger.info(
            "Saved %d local records to %s",
            len(local_records),
            rank_file,
        )

        # Two-phase file-based gather:
        #   1. wait until every rank file exists
        #   2. acknowledge that this rank has observed the full set
        # This prevents rank 0 from consuming/removing files before a
        # slower peer has actually observed them.
        world_size = int(dist_info["world_size"])
        all_rank_files = [
            output_dir / f".rank_{r}_records.json" for r in range(world_size)
        ]
        all_collection_files = (
            [
                output_dir / f".rank_{r}_oracle_collection.json"
                for r in range(world_size)
            ]
            if oracle_collection_enabled
            else []
        )
        wait_for_paths(
            all_rank_files + all_collection_files,
            directory=output_dir,
            dist_info=dist_info,
        )

        ready_file = output_dir / f".rank_{rank}_records_ready"
        ready_file.touch()
        all_ready_files = [
            output_dir / f".rank_{r}_records_ready" for r in range(world_size)
        ]
        wait_for_paths(
            all_ready_files,
            directory=output_dir,
            dist_info=dist_info,
        )

        if not is_main_process(dist_info):
            model, tokenizer = release_generation_resources(model, tokenizer, device)
            release_control_runtime(runtime)
            return None

        # Rank 0: read all rank files and merge.
        # For bon_toxicity, text is list[str] (N candidates); otherwise str.
        all_records: list[tuple[int, str | list[str]]] = []
        truncated_count = 0
        merged_collection_records: dict[int, dict[str, object]] = {}
        for rf in all_rank_files:
            with open(rf, encoding="utf-8") as f:
                data = json.load(f)
            for idx, text in data["records"]:
                if isinstance(text, list):
                    all_records.append((int(idx), text))
                else:
                    all_records.append((int(idx), str(text)))
            truncated_count += int(data["truncated_count"])
        if oracle_collection_enabled:
            for rf in all_collection_files:
                with open(rf, encoding="utf-8") as f:
                    data = json.load(f)
                for record in data["records"]:
                    merged_collection_records[int(record["seq_index"])] = record
        all_records.sort(key=lambda x: x[0])
        all_seq_indices = [row[0] for row in all_records]
        all_generations = [row[1] for row in all_records]

        if len(all_generations) != len(prompt_texts):
            raise RuntimeError(
                "Distributed gathering mismatch: "
                f"expected {len(prompt_texts)} samples, got {len(all_generations)}."
            )

        expected_indices = list(range(len(prompt_texts)))
        if all_seq_indices != expected_indices:
            raise RuntimeError(
                "Distributed gathering mismatch: gathered indices are incomplete or duplicated."
            )
        all_prompts = [prompt_texts[idx] for idx in all_seq_indices]
        all_sample_indices = [sample_indices[idx] for idx in all_seq_indices]
        ordered_collection_records: list[dict[str, object]] | None = None
        if oracle_collection_enabled:
            missing_seq = [
                seq_idx
                for seq_idx in all_seq_indices
                if seq_idx not in merged_collection_records
            ]
            if missing_seq:
                raise RuntimeError(
                    "Oracle collection gathering mismatch: missing seq indices "
                    f"{missing_seq[:5]}"
                )
            ordered_collection_records = [
                merged_collection_records[seq_idx] for seq_idx in all_seq_indices
            ]

        if truncated_count > 0:
            logger.warning(
                "%d prompt(s) exceeded model.max_prompt_length=%d and were truncated.",
                truncated_count,
                max_prompt_length,
            )

        # Release generation resources before scorer to free GPU memory.
        # polytope must be released here; it is not needed for scoring and
        # may cause OOM when ToxicityScorer tries to allocate on the same device.
        logger.info("Scoring toxicity...")
        model, tokenizer = release_generation_resources(model, tokenizer, device)
        release_control_runtime(runtime)

        if compute_device_mode == "auto":
            scorer_runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            scorer_runtime_device = compute_device_mode
        if scorer_runtime_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "compute.device is set to 'cuda' but CUDA is unavailable. "
                "Set compute.device='auto' or 'cpu'."
            )

        try:
            scorer = ToxicityScorer(
                model_name=scorer_model_name, device=scorer_runtime_device
            )
        except RuntimeError as exc:
            should_retry_on_cpu = (
                compute_device_mode == "auto"
                and scorer_runtime_device == "cuda"
                and is_cuda_oom_error(exc)
            )
            if not should_retry_on_cpu:
                raise
            logger.warning(
                "CUDA OOM while loading toxicity scorer on GPU; retrying on CPU."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            scorer = ToxicityScorer(model_name=scorer_model_name, device="cpu")

        # ── bon_toxicity: score all N candidates per prompt, pick lowest ──
        bon_details: list[dict] | None = None
        if control_method == "bon_toxicity":
            n = control_best_of_n
            # all_generations is list[list[str]], each inner list has N candidates
            flat_texts = [t for candidates in all_generations for t in candidates]
            logger.info(
                "Scoring %d bon_toxicity candidates (%d prompts × %d)...",
                len(flat_texts), len(all_generations), n,
            )
            flat_scores = scorer.score_batch(
                flat_texts,
                batch_size=scorer_batch_size,
                logger=logger,
                progress_desc="BoN scoring",
            )
            selected_generations: list[str] = []
            bon_details = []
            for i in range(len(all_generations)):
                group_scores = flat_scores[i * n : (i + 1) * n]
                best_j = min(range(n), key=lambda j: group_scores[j])
                selected_generations.append(all_generations[i][best_j])
                bon_details.append({
                    "candidates": [
                        {"text": all_generations[i][j], "toxicity": group_scores[j]}
                        for j in range(n)
                    ],
                    "selected_index": best_j,
                })
            all_generations = selected_generations
            logger.info("bon_toxicity selection complete.")

        # Score final selected generations
        try:
            toxicity_scores = scorer.score_batch(
                all_generations,
                batch_size=scorer_batch_size,
                logger=logger,
                progress_desc="Scoring",
            )
        except RuntimeError as exc:
            should_retry_on_cpu = (
                compute_device_mode == "auto"
                and scorer_runtime_device == "cuda"
                and is_cuda_oom_error(exc)
            )
            if not should_retry_on_cpu:
                raise

            logger.warning(
                "CUDA OOM while running toxicity scorer on GPU; retrying on CPU."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            scorer = ToxicityScorer(model_name=scorer_model_name, device="cpu")
            toxicity_scores = scorer.score_batch(
                all_generations,
                batch_size=scorer_batch_size,
                logger=logger,
                progress_desc="Scoring",
            )

        # Calculate statistics
        avg_toxicity = sum(toxicity_scores) / len(toxicity_scores)
        max_toxicity = max(toxicity_scores)
        toxic_count = sum(1 for s in toxicity_scores if s > scorer_toxic_threshold)
        toxic_ratio = toxic_count / len(toxicity_scores)

        logger.info("")
        logger.info("=" * 50)
        logger.info("TOXICITY EVALUATION RESULTS")
        logger.info("=" * 50)
        logger.info("Model: %s", model_name)
        logger.info("Model mode: %s", model_mode)
        logger.info("Control method: %s", control_method)
        logger.info("Total samples: %d", len(all_generations))
        logger.info("Average toxicity score: %.4f", avg_toxicity)
        logger.info("Max toxicity score: %.4f", max_toxicity)
        logger.info(
            "Toxic ratio (>%s): %.2f%% (%d/%d)",
            scorer_toxic_threshold,
            toxic_ratio * 100,
            toxic_count,
            len(toxicity_scores),
        )
        logger.info("=" * 50)

        # Build control config dict for output
        control_config = {"method": control_method}
        control_config["sampler"] = control_sampler
        if control_method in METHODS_NEEDING_RISK_MODEL:
            control_config["risk_model_type"] = risk_model_type
            if risk_model_type == "riskmlp":
                control_config["riskmlp_checkpoint"] = str(cfg.control.riskmlp_checkpoint)
                control_config["riskmlp_kind"] = loaded_risk_model_kind
            elif risk_model_type == "value":
                control_config["value_checkpoint"] = str(cfg.control.value_checkpoint)
                control_config["value_kind"] = loaded_risk_model_kind
            else:
                control_config["polytope_checkpoint"] = str(cfg.control.polytope_checkpoint)
                control_config["polytope_kind"] = loaded_risk_model_kind
                if loaded_feature_pooler_type is not None:
                    control_config["feature_pooler_type"] = loaded_feature_pooler_type
        if control_method in {"classifier", "pg_dlm_proxy"}:
            control_config["risk_model_type"] = "classifier"
            control_config["classifier_checkpoint"] = str(cfg.control.classifier_checkpoint)
        ctrl = cfg.control
        if control_method in {"tilting", "gradient", "gradient_bon", "classifier", "advantage"}:
            control_config["eta"] = float(ctrl.eta)
            control_config["weight_beta"] = float(ctrl.weight_beta)
        if control_method in {"gradient", "gradient_bon", "classifier", "advantage"}:
            control_config["weight_schedule"] = str(ctrl.weight_schedule).lower()
        if control_method == "classifier":
            control_config["classifier_delta_temperature"] = float(
                ctrl.get("classifier_delta_temperature", 1.0)
            )
            delta_clip_cfg = ctrl.get("classifier_delta_clip", None)
            control_config["classifier_delta_clip"] = (
                None
                if delta_clip_cfg is None or str(delta_clip_cfg).lower() == "null"
                else float(delta_clip_cfg)
            )
            control_config["classifier_delta_clip_mode"] = str(
                ctrl.get("classifier_delta_clip_mode", "tanh")
            ).lower()
        if control_method == "advantage":
            control_config["advantage_checkpoint"] = str(ctrl.advantage_checkpoint)
            control_config["advantage_top_k"] = int(ctrl.get("advantage_top_k", 8))
        if control_method in {"gradient", "gradient_bon"}:
            control_config["safety_margin"] = float(ctrl.safety_margin)
        if control_method == "tilting":
            control_config["control_top_k"] = int(ctrl.control_top_k)
        if control_method in {"best_of_n", "gradient_bon", "bon_toxicity"}:
            control_config["best_of_n"] = control_best_of_n
        if control_method in {"pg_dlm", "pg_dlm_proxy"}:
            control_config["pg_particles"] = int(ctrl.get("pg_particles", 4))
            control_config["pg_iterations"] = int(ctrl.get("pg_iterations", 2))
            control_config["pg_partial_samples"] = int(ctrl.get("pg_partial_samples", 1))
            control_config["pg_reward_beta"] = float(ctrl.get("pg_reward_beta", 1.0))
            control_config["pg_completion_method"] = str(ctrl.get("pg_completion_method", "greedy")).lower()
            control_config["pg_rollout_temperature"] = float(ctrl.get("pg_rollout_temperature", 1.0))
            control_config["pg_resample_interval"] = int(ctrl.get("pg_resample_interval", 1))
            control_config["pg_reward_batch_size"] = int(ctrl.get("pg_reward_batch_size", 32))
            if control_method == "pg_dlm":
                control_config["pg_reward_device"] = str(ctrl.get("pg_reward_device", "cpu")).lower()
                control_config["partial_reward_mode"] = "rollout"
            else:
                control_config["partial_reward_mode"] = "proxy_classifier"
        if control_sampler == "ctmc":
            control_config["noise_schedule"] = str(ctrl.get("noise_schedule", "linear"))
            control_config["ctmc_integration"] = str(ctrl.get("ctmc_integration", "outer_only")).lower()

        dataset_source = resolve_dataset_source(cfg)
        generation_config = {
            "steps": int(gen.steps),
            "gen_length": int(gen.gen_length),
            "block_length": int(gen.block_length),
            "temperature": float(gen.temperature),
            "cfg_scale": float(gen.cfg_scale),
            "remasking": str(gen.remasking),
            "logits_eos_inf": bool(gen.logits_eos_inf),
            "confidence_eos_eot_inf": bool(gen.confidence_eos_eot_inf),
        }

        if oracle_collection_enabled and ordered_collection_records is not None:
            oracle_output_path = export_oracle_collection(
                cfg=cfg,
                output_dir=output_dir,
                ordered_collection_records=ordered_collection_records,
                dataset_source=dataset_source,
                model_name=model_name,
                model_mode=model_mode,
                toxic_threshold=scorer_toxic_threshold,
                generation_config=generation_config,
                control_config=control_config,
                control_method=control_method,
                bon_details=bon_details,
                toxicity_scores=toxicity_scores,
                generations=all_generations,
            )
            logger.info("Saved oracle collection records to %s", oracle_output_path)

        output_file, output_json_file = write_toxicity_results(
            output_dir=output_dir,
            model_name=model_name,
            model_mode=model_mode,
            control_method=control_method,
            dataset_source=dataset_source,
            split=str(cfg.data.split),
            generation_config=generation_config,
            control_config=control_config,
            max_prompt_length=max_prompt_length,
            truncated_count=truncated_count,
            avg_toxicity=avg_toxicity,
            max_toxicity=max_toxicity,
            scorer_toxic_threshold=scorer_toxic_threshold,
            toxic_ratio=toxic_ratio,
            toxic_count=toxic_count,
            prompts=all_prompts,
            generations=all_generations,
            toxicity_scores=toxicity_scores,
            sample_indices=all_sample_indices,
            bon_details=bon_details,
        )

        logger.info("Results saved to %s and %s", output_file, output_json_file)

        return {
            "avg_toxicity": avg_toxicity,
            "max_toxicity": max_toxicity,
            "toxic_ratio": toxic_ratio,
            "generations": all_generations,
            "prompts": all_prompts,
            "scores": toxicity_scores,
        }
    except BaseException as exc:
        if not isinstance(exc, PeerRankFailedError):
            write_rank_failure_best_effort(output_dir, rank, exc, logger)
        raise
    finally:
        if runtime is not None:
            release_control_runtime(runtime)
        model, tokenizer = release_generation_resources(model, tokenizer, device)


_dist_info: dict[str, int | bool] | None = None


def _ensure_distributed() -> dict[str, int | bool]:
    """Initialise the distributed process group once and reuse across Hydra sweep jobs."""
    global _dist_info
    if _dist_info is None:
        _dist_info = setup_distributed()
    return _dist_info

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    dist_info = _ensure_distributed()
    rank = int(dist_info["rank"])
    cleanup_logger = get_logger("safety_eval.cleanup", rank=rank, all_ranks=True)
    barrier_dir: Path | None = None

    try:
        # Determine a shared directory for file-based barriers.
        barrier_dir = get_eval_output_dir(cfg)

        # Synchronise all ranks so they enter each sweep job together.
        file_barrier("pre", dist_info, barrier_dir)

        logger = get_logger("safety_eval.main", rank=rank)
        logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
        run_toxicity_eval(cfg, dist_info)

        # Synchronise again before Hydra moves to the next sweep job.
        file_barrier("post", dist_info, barrier_dir)
        if is_main_process(dist_info):
            cleanup_gather_artifacts(barrier_dir)
    except BaseException as exc:
        if barrier_dir is not None and not isinstance(exc, PeerRankFailedError):
            write_rank_failure_best_effort(barrier_dir, rank, exc, cleanup_logger)
        raise
    finally:
        cleanup_logger.info("Starting distributed cleanup")
        cleanup_distributed(dist_info)
        cleanup_logger.info("Distributed cleanup complete")


if __name__ == "__main__":
    main()
