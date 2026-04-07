import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from fkd_class import FKD
from reward_functions import (
    cola_score,
    formality_score,
    gpt2_perp_score,
    infinigram_perp_score,
    logmeanexp,
    sentiment_score,
    toxicity_score,
)


def batch_inputs(inputs, batch_size):
    assert isinstance(inputs, list), "inputs should be a list"
    batches = []
    for i in range(0, len(inputs), batch_size):
        batches.append(inputs[i : i + batch_size])
    return batches


def batched_infer(*, inputs, fn, batch_size):
    results = []
    for batch in batch_inputs(inputs, batch_size):
        results.extend(fn(x_batch=batch))
    return results


def compute_rewards(*, samples, reward_name, reward_label):
    if reward_name == "sentiment":
        scores, _ = sentiment_score(texts=samples, label=reward_label)
    elif reward_name == "toxicity":
        scores, _ = toxicity_score(
            texts=samples,
            label=reward_label,
            max_length=50,
        )
    elif reward_name == "formality":
        scores, _ = formality_score(texts=samples, label=reward_label)
    elif reward_name == "gpt2_perp":
        scores, _ = gpt2_perp_score(texts=samples)
    elif reward_name == "cola":
        scores, _ = cola_score(texts=samples, max_length=50)
    elif reward_name.startswith("infinigram_perp_score"):
        max_num_samples = int(reward_name.split("-")[-1])
        max_ngram = int(reward_name.split("-")[-2])
        scores, _ = infinigram_perp_score(
            texts=samples,
            max_ngram=max_ngram,
            max_num_samples=max_num_samples,
        )
    else:
        raise ValueError(f"Unknown reward function: {reward_name}")
    return scores


def _resolve_torch_dtype(dtype_name):
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = str(dtype_name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[key]


def load_llada_model_and_tokenizer(cfg_model, device, logger=None):
    if logger is not None:
        logger.info("Loading LLaDA model: %s", cfg_model.name_or_path)

    model_dtype = _resolve_torch_dtype(cfg_model.torch_dtype)
    trust_remote = bool(cfg_model.trust_remote_code)

    if bool(cfg_model.flash_attention):
        config = AutoConfig.from_pretrained(
            str(cfg_model.name_or_path),
            trust_remote_code=trust_remote,
        )
        config.flash_attention = True
        model = AutoModel.from_pretrained(
            str(cfg_model.name_or_path),
            config=config,
            trust_remote_code=trust_remote,
            torch_dtype=model_dtype,
        ).to(device)
    else:
        model = AutoModel.from_pretrained(
            str(cfg_model.name_or_path),
            trust_remote_code=trust_remote,
            torch_dtype=model_dtype,
        ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        str(cfg_model.name_or_path),
        trust_remote_code=trust_remote,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token
    if int(tokenizer.pad_token_id) == int(cfg_model.mask_id):
        raise ValueError(
            f"tokenizer.pad_token_id ({tokenizer.pad_token_id}) must differ from "
            f"mask_id ({cfg_model.mask_id})."
        )
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def prepare_prompts_for_model(prompts, model_mode):
    if str(model_mode).lower() != "base":
        raise ValueError(
            "FK LLaDA backend currently supports llada_model.mode='base' only."
        )
    return prompts


def tokenize_prompts_for_generation(tokenizer, prompts, max_prompt_length):
    if prompts and prompts[0] == "":
        prompt_batch = {
            "input_ids": torch.empty((1, 0), dtype=torch.long),
            "attention_mask": torch.empty((1, 0), dtype=torch.long),
        }
        return prompt_batch, 0

    tokenized = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]

    truncation_side = str(getattr(tokenizer, "truncation_side", "right")).lower()
    truncated_count = 0
    trimmed_input_ids = []
    for ids in tokenized:
        if len(ids) > max_prompt_length:
            truncated_count += 1
            if truncation_side == "left":
                ids = ids[-max_prompt_length:]
            else:
                ids = ids[:max_prompt_length]
        trimmed_input_ids.append(ids)

    prompt_batch = tokenizer.pad(
        [{"input_ids": ids} for ids in trimmed_input_ids],
        padding=True,
        return_tensors="pt",
    )
    return prompt_batch, truncated_count


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def _build_transfer_schedule(block_length, steps_per_block, device):
    block_mask = torch.ones((1, block_length), dtype=torch.bool, device=device)
    return get_num_transfer_tokens(block_mask, steps_per_block)[0]


def _forward_logits(model, x, attention_mask, prompt_index, cfg_scale, mask_id):
    if cfg_scale > 0.0:
        un_x = x.clone()
        un_x[prompt_index] = mask_id
        x_cat = torch.cat([x, un_x], dim=0)
        attn_cat = (
            torch.cat([attention_mask, attention_mask], dim=0)
            if attention_mask is not None
            else None
        )
        out = model(x_cat, attention_mask=attn_cat)
        logits, un_logits = torch.chunk(out.logits, 2, dim=0)
        return un_logits + (cfg_scale + 1.0) * (logits - un_logits)

    return model(x, attention_mask=attention_mask).logits


@dataclass(frozen=True)
class LLaDAGenerationConfig:
    steps: int
    gen_length: int
    block_length: int
    temperature: float
    cfg_scale: float
    remasking: str
    mask_id: int
    logits_eos_inf: bool
    confidence_eos_eot_inf: bool


class FKLLaDA:
    def __init__(self, config, logger=None, device=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model, self.tokenizer = load_llada_model_and_tokenizer(
            config.llada_model,
            device=self.device,
            logger=self.logger,
        )
        self.gen_cfg = LLaDAGenerationConfig(
            steps=int(config.sampling.steps),
            gen_length=int(config.llada_generation.gen_length),
            block_length=int(config.llada_generation.block_length),
            temperature=float(config.llada_generation.temperature),
            cfg_scale=float(config.llada_generation.cfg_scale),
            remasking=str(config.llada_generation.remasking),
            mask_id=int(config.llada_model.mask_id),
            logits_eos_inf=bool(config.llada_generation.logits_eos_inf),
            confidence_eos_eot_inf=bool(
                config.llada_generation.confidence_eos_eot_inf
            ),
        )
        self.reward_batch_size = 8
        self.proposal_batch_size = 8
        self.last_prompt_length = 0
        adaptation_cfg = getattr(self.config.fk_steering, "adaptation", None)
        self.initial_reward_seeding = bool(
            getattr(
                adaptation_cfg,
                "initial_reward_seeding",
                getattr(self.config.fk_steering, "initial_reward_seeding", True),
            )
        )
        self.reward_fill_mode = str(
            getattr(
                adaptation_cfg,
                "reward_fill_mode",
                getattr(self.config.fk_steering, "reward_fill_mode", "prefix_only"),
            )
        ).lower()
        if self.reward_fill_mode not in {"prefix_only", "full_fill"}:
            raise ValueError(
                "fk_steering.adaptation.reward_fill_mode must be one of "
                "['prefix_only', 'full_fill']."
            )

    def _prepare_prompt(self, prompt_text):
        prompt_text = "" if prompt_text is None else str(prompt_text)
        model_ready_prompt = prepare_prompts_for_model(
            [prompt_text],
            model_mode=self.config.llada_model.mode,
        )
        prompt_batch, truncated_count = tokenize_prompts_for_generation(
            self.tokenizer,
            model_ready_prompt,
            max_prompt_length=int(self.config.llada_model.max_prompt_length),
        )
        if truncated_count > 0:
            self.logger.warning(
                "Prompt exceeded llada_model.max_prompt_length=%d and was truncated.",
                int(self.config.llada_model.max_prompt_length),
            )
        input_ids = prompt_batch["input_ids"].to(self.device)
        attention_mask = prompt_batch["attention_mask"].to(self.device)
        prompt_len = int(input_ids.shape[1])
        return input_ids, attention_mask, prompt_len

    def _llada_update(
        self,
        x,
        attention_mask,
        prompt_index,
        prompt_len,
        step_idx,
        steps_per_block,
        transfer_schedule,
    ):
        cfg = self.gen_cfg
        block_idx = step_idx // steps_per_block
        inner_step = step_idx % steps_per_block
        block_start = prompt_len + block_idx * cfg.block_length
        block_end = block_start + cfg.block_length

        logits = _forward_logits(
            self.model,
            x,
            attention_mask,
            prompt_index,
            cfg.cfg_scale,
            cfg.mask_id,
        )
        if cfg.logits_eos_inf:
            logits[:, :, 126081] = -torch.inf

        mask_index = x == cfg.mask_id
        logits_with_noise = add_gumbel_noise(logits, temperature=cfg.temperature)
        x0_update = torch.argmax(logits_with_noise, dim=-1)

        if cfg.confidence_eos_eot_inf:
            logits_with_noise[:, :, 126081] = -torch.inf
            logits_with_noise[:, :, 126348] = -torch.inf

        if cfg.remasking == "low_confidence":
            probs = F.softmax(logits, dim=-1)
            x0_p = torch.gather(
                probs, dim=-1, index=x0_update.unsqueeze(-1)
            ).squeeze(-1)
        elif cfg.remasking == "random":
            x0_p = torch.rand((x.shape[0], x.shape[1]), device=x.device)
        else:
            raise NotImplementedError(cfg.remasking)

        x0_p[:, :block_start] = -torch.inf
        x0_p[:, block_end:] = -torch.inf
        x0_update = torch.where(mask_index, x0_update, x)
        confidence = torch.where(mask_index, x0_p, -torch.inf)

        num_transfer = int(transfer_schedule[inner_step].item())
        transfer_index = torch.zeros_like(x0_update, dtype=torch.bool, device=x.device)
        if num_transfer > 0:
            for batch_idx in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[batch_idx], k=num_transfer)
                transfer_index[batch_idx, select_index] = True

        x_next = x.clone()
        x_next[transfer_index] = x0_update[transfer_index]

        if self.reward_fill_mode == "prefix_only":
            # Only fill masks up to the current block end. Positions in future
            # blocks stay [MASK], so the reward only scores the denoised prefix.
            fill_mask = mask_index.clone()
            fill_mask[:, block_end:] = False
        else:
            # Fill every masked position to approximate a full x0 sample, which
            # is closer to the original discrete FK reward construction.
            fill_mask = mask_index

        x0_samples = []
        for _ in range(int(self.config.fk_steering.num_x0_samples)):
            sampled = torch.argmax(
                add_gumbel_noise(logits, temperature=cfg.temperature),
                dim=-1,
            )
            completed = torch.where(fill_mask, sampled, x).detach().cpu()
            x0_samples.append(completed)

        return x_next, x0_samples

    def q_proposal_fn(
        self,
        x_batch,
        attention_mask,
        prompt_index,
        prompt_len,
        step_idx,
        steps_per_block,
        transfer_schedule,
    ):
        z = [x["z"] for x in x_batch]
        z = torch.cat(z, dim=0)
        new_z, samples = self._llada_update(
            z,
            attention_mask=attention_mask,
            prompt_index=prompt_index,
            prompt_len=prompt_len,
            step_idx=step_idx,
            steps_per_block=steps_per_block,
            transfer_schedule=transfer_schedule,
        )

        combined = []
        for i in range(new_z.shape[0]):
            combined.append(
                {
                    "z": new_z[i : i + 1],
                    "sample": [sample[i : i + 1] for sample in samples],
                }
            )
        return combined

    def prior_fn(self, prompt_ids):
        z = torch.full(
            (1, prompt_ids.shape[1] + self.gen_cfg.gen_length),
            self.gen_cfg.mask_id,
            dtype=torch.long,
            device=self.device,
        )
        z[:, : prompt_ids.shape[1]] = prompt_ids
        z_cpu = z.detach().cpu().clone()
        return {
            "z": z,
            "sample": [
                z_cpu.clone() for _ in range(int(self.config.fk_steering.num_x0_samples))
            ],
        }

    def r_fn(self, x_batch, length_for_reward_fn):
        flatten_samples = []
        for x in x_batch:
            flatten_samples.extend(x["sample"])

        samples = torch.cat(flatten_samples, dim=0)
        samples = samples[:, :length_for_reward_fn]
        decoded = self.tokenizer.batch_decode(samples, skip_special_tokens=True)
        scores = compute_rewards(
            samples=decoded,
            reward_name=self.config.fk_steering.reward_fn,
            reward_label=self.config.fk_steering.reward_label,
        )
        scores = torch.tensor(scores, dtype=torch.float32)
        scores = scores.reshape(
            len(x_batch), int(self.config.fk_steering.num_x0_samples)
        )
        return logmeanexp(scores).tolist()

    @torch.no_grad()
    def _sample(self, num_steps=None, prompt_text=None):
        batch_size_per_gpu = int(self.config.loader.eval_batch_size)
        if batch_size_per_gpu != 1:
            raise ValueError(
                "FK LLaDA backend currently requires loader.eval_batch_size == 1."
            )

        num_steps = int(self.config.sampling.steps if num_steps is None else num_steps)
        prompt_ids, prompt_attention_mask, prompt_len = self._prepare_prompt(prompt_text)
        self.last_prompt_length = prompt_len
        length_for_reward_fn = int(self.config.fk_steering.reward_trim_length) + prompt_len
        num_blocks = self.gen_cfg.gen_length // self.gen_cfg.block_length
        if num_steps % num_blocks != 0:
            raise ValueError(
                "num_steps must be divisible by "
                "(llada_generation.gen_length // llada_generation.block_length)."
            )
        steps_per_block = num_steps // num_blocks
        transfer_schedule = _build_transfer_schedule(
            self.gen_cfg.block_length,
            steps_per_block,
            self.device,
        )

        full_attention_mask = torch.cat(
            [
                prompt_attention_mask,
                torch.ones(
                    (prompt_ids.shape[0], self.gen_cfg.gen_length),
                    dtype=prompt_attention_mask.dtype,
                    device=self.device,
                ),
            ],
            dim=-1,
        )
        prompt_index = torch.zeros(
            (1, prompt_len + self.gen_cfg.gen_length),
            dtype=torch.bool,
            device=self.device,
        )
        prompt_index[:, :prompt_len] = True

        fkd = FKD(
            potential_type=self.config.fk_steering.potential_type,
            lmbda=self.config.fk_steering.lmbda,
            num_particles=self.config.fk_steering.k_particles,
            adaptive_resampling=False,
            adaptive_resample_at_end=False,
            resample_frequency=self.config.fk_steering.resample_frequency,
            resampling_t_start=-1,
            resampling_t_end=num_steps - 1,
            time_steps=num_steps,
            reward_fn=lambda x: batched_infer(
                inputs=x,
                fn=lambda x_batch: self.r_fn(
                    x_batch=x_batch,
                    length_for_reward_fn=length_for_reward_fn,
                ),
                batch_size=self.reward_batch_size,
            ),
            device=self.device,
        )

        states = [
            self.prior_fn(prompt_ids)
            for _ in range(int(self.config.fk_steering.k_particles))
        ]

        if self.initial_reward_seeding:
            # Seed population_rs onto the same reward scale as the text reward.
            # This is especially important for negative log-prob rewards, where
            # starting from 0 can suppress early MAX/DIFF steering.
            initial_rs = batched_infer(
                inputs=states,
                fn=lambda x_batch: self.r_fn(
                    x_batch=x_batch,
                    length_for_reward_fn=length_for_reward_fn,
                ),
                batch_size=self.reward_batch_size,
            )
            fkd.population_rs = torch.tensor(
                initial_rs, dtype=torch.float32, device=self.device
            )

        rs_historic_means = [fkd.population_rs.detach().cpu().mean().item()]

        tiled_attention = full_attention_mask.expand(len(states), -1).contiguous()
        tiled_prompt_index = prompt_index.expand(len(states), -1).contiguous()

        for step_idx in range(num_steps):
            states_candidates = batched_infer(
                inputs=states,
                fn=lambda x_batch: self.q_proposal_fn(
                    x_batch=x_batch,
                    attention_mask=tiled_attention[: len(x_batch)],
                    prompt_index=tiled_prompt_index[: len(x_batch)],
                    prompt_len=prompt_len,
                    step_idx=step_idx,
                    steps_per_block=steps_per_block,
                    transfer_schedule=transfer_schedule,
                ),
                batch_size=self.proposal_batch_size,
            )
            states, _ = fkd.resample(
                sampling_idx=step_idx,
                latents=states_candidates,
                x0_preds=states_candidates,
            )
            rs_states = fkd.population_rs.detach().cpu()
            rs_historic_means.append(torch.mean(rs_states).item())

        if not fkd._reached_terminal_sample:
            raise RuntimeError("FKD did not reach terminal LLaDA sample.")

        best_idx = int(torch.argmax(rs_states).item())
        best_sample = states[best_idx]
        best_r = rs_states[best_idx]
        return {
            "best": best_sample["z"].detach().cpu(),
            "best_r": best_r,
            "all_samples": states,
            "all_r": rs_states,
            "historic_means": rs_historic_means,
            "prompt_length": prompt_len,
        }

    def restore_model_and_sample(self, num_steps, eps=1e-5, prompt_text=None):
        del eps
        self.model.eval()
        return self._sample(num_steps=num_steps, prompt_text=prompt_text)
