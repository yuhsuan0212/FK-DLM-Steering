# based on https://github.com/kuleshov-group/mdlm/blob/master/main.py
"""
Generate samples from a trained diffusion model with FK Steering.

Supports the original MDLM backend and a LLaDA semi-AR backend.
"""
import json
import os
import sys
import gc
from datetime import datetime
from pathlib import Path

import hydra
import lightning as L
import torch
from datasets import load_dataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

CURRENT_DIR = Path(__file__).resolve().parent
MDLM_DIR = CURRENT_DIR / 'mdlm'
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(MDLM_DIR) not in sys.path:
    sys.path.insert(1, str(MDLM_DIR))

import dataloader

from fk_llada import FKLLaDA
from summary_utils import build_summary
from utils.distributed import (
    cleanup_distributed,
    get_device,
    get_distributed_info,
    is_main_process,
    shard_list,
)
from utils.distributed_fs import (
    PeerRankFailedError,
    cleanup_gather_artifacts,
    file_barrier,
    wait_for_paths,
    write_json_atomic,
    write_rank_failure_best_effort,
)
from utils.logging import ProgressLogger, get_logger

if not OmegaConf.has_resolver('cwd'):
    OmegaConf.register_new_resolver('cwd', os.getcwd)
if not OmegaConf.has_resolver('device_count'):
    OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
if not OmegaConf.has_resolver('eval'):
    OmegaConf.register_new_resolver('eval', eval)
if not OmegaConf.has_resolver('div_up'):
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)


def _get_backend(config):
    return str(config.get('backend', 'mdlm')).lower()


def _resolve_optional_path(path_value):
    if path_value is None:
        return None
    if str(path_value).lower() == 'null':
        return None
    return hydra.utils.to_absolute_path(str(path_value))


def _validate_backend_config(config):
    backend = _get_backend(config)
    if backend not in {'mdlm', 'llada'}:
        raise ValueError(f"backend must be one of ['mdlm', 'llada']. Got: {backend}")

    if backend != 'llada':
        return

    model_mode = str(config.llada_model.mode).lower()
    if model_mode != 'base':
        raise ValueError(
            "backend='llada' currently supports llada_model.mode='base' only."
        )
    if not str(config.llada_model.name_or_path).strip():
        raise ValueError("llada_model.name_or_path is required for backend='llada'.")

    gen_length = int(config.llada_generation.gen_length)
    block_length = int(config.llada_generation.block_length)
    steps = int(config.sampling.steps)
    if gen_length <= 0:
        raise ValueError("llada_generation.gen_length must be > 0.")
    if block_length <= 0:
        raise ValueError("llada_generation.block_length must be > 0.")
    if steps <= 0:
        raise ValueError("sampling.steps must be > 0.")
    if gen_length % block_length != 0:
        raise ValueError(
            "llada_generation.gen_length must be divisible by "
            "llada_generation.block_length."
        )
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError(
            "sampling.steps must be divisible by "
            "(llada_generation.gen_length // llada_generation.block_length)."
        )

    remasking = str(config.llada_generation.remasking).lower()
    if remasking not in {'low_confidence', 'random'}:
        raise ValueError(
            "llada_generation.remasking must be one of "
            "['low_confidence', 'random']."
        )

    prompt_source = str(config.prompts.source).lower()
    if prompt_source not in {'prompt_file', 'local_json', 'hf_dataset'}:
        raise ValueError(
            "prompts.source must be one of "
            "['prompt_file', 'local_json', 'hf_dataset']."
        )
    if prompt_source == 'local_json' and _resolve_optional_path(
        config.prompts.local_prompts
    ) is None:
        raise ValueError(
            "prompts.local_prompts is required when prompts.source='local_json'."
        )
    if prompt_source == 'hf_dataset' and not str(config.prompts.dataset_name).strip():
        raise ValueError(
            "prompts.dataset_name is required when prompts.source='hf_dataset'."
        )

    adaptation_cfg = getattr(config.fk_steering, 'adaptation', None)

    reward_fill_mode = str(
        getattr(
            adaptation_cfg,
            'reward_fill_mode',
            getattr(config.fk_steering, 'reward_fill_mode', 'prefix_only'),
        )
    ).lower()
    if reward_fill_mode not in {'prefix_only', 'full_fill'}:
        raise ValueError(
            "fk_steering.adaptation.reward_fill_mode must be one of "
            "['prefix_only', 'full_fill']."
        )

    initial_reward_seeding = getattr(
        adaptation_cfg,
        'initial_reward_seeding',
        getattr(config.fk_steering, 'initial_reward_seeding', True),
    )
    if not isinstance(initial_reward_seeding, bool):
        raise ValueError(
            "fk_steering.adaptation.initial_reward_seeding must be a boolean."
        )


def _print_config(config, resolve=True, save_cfg=True):
    rendered = OmegaConf.to_yaml(config, resolve=resolve)
    print(rendered)
    if not save_cfg:
        return
    save_dir = Path(str(config.checkpointing.save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'config_tree.txt', 'w', encoding='utf-8') as fp:
        fp.write(rendered)


def _get_run_output_dir():
    return Path(HydraConfig.get().runtime.output_dir)


def _get_shared_sync_dir():
    return _get_run_output_dir() / '.distributed_fs'


def _prepare_sample_output_dir():
    cur_date = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = _get_run_output_dir() / 'fk_steering' / 'sample_evaluation' / cur_date
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _save_run_info(output_dir, config, dist_info):
    backend = _get_backend(config)
    info = {
        'backend': backend,
        'fk_steering': OmegaConf.to_container(config.fk_steering, resolve=True),
        'prompts': OmegaConf.to_container(config.prompts, resolve=True),
        'seed': int(config.seed),
        'checkpoint_path': config.eval.checkpoint_path if backend == 'mdlm' else None,
        'prompt_file': config.sampling.prompt_file,
        'distributed': {
            'enabled': bool(dist_info['is_distributed']),
            'world_size': int(dist_info['world_size']),
            'rank': int(dist_info['rank']),
            'local_rank': int(dist_info['local_rank']),
            'coordination': 'filesystem',
        },
    }
    if backend == 'llada':
        info['llada_model'] = OmegaConf.to_container(config.llada_model, resolve=True)
        info['llada_generation'] = OmegaConf.to_container(
            config.llada_generation,
            resolve=True,
        )
    with open(output_dir / 'info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f)


def _save_samples_and_summary(output_dir, samples, summary, logger):
    with open(output_dir / 'text_samples.jsonl', 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    import matplotlib.pyplot as plt

    for sample in samples:
        plt.plot(sample['r_means'])
    plt.savefig(output_dir / 'historic_means.png')
    plt.close()
    logger.info('Saved summary to %s', output_dir / 'summary.json')


def _synchronise_samples(
    *,
    local_samples,
    prompt_records,
    config,
    dist_info,
    sync_dir,
    logger,
):
    if not dist_info['is_distributed']:
        return local_samples

    rank = int(dist_info['rank'])
    world_size = int(dist_info['world_size'])
    rank_file = sync_dir / f'.rank_{rank}_records.json'
    write_json_atomic(rank_file, {'samples': local_samples})
    logger.info('Saved %d local samples to %s', len(local_samples), rank_file)

    all_rank_files = [sync_dir / f'.rank_{r}_records.json' for r in range(world_size)]
    wait_for_paths(all_rank_files, directory=sync_dir, dist_info=dist_info)

    ready_file = sync_dir / f'.rank_{rank}_records_ready'
    ready_file.touch()
    all_ready_files = [
        sync_dir / f'.rank_{r}_records_ready' for r in range(world_size)
    ]
    wait_for_paths(all_ready_files, directory=sync_dir, dist_info=dist_info)

    if not is_main_process(dist_info):
        return None

    merged_samples = []
    for rank_file in all_rank_files:
        with open(rank_file, encoding='utf-8') as f:
            payload = json.load(f)
        merged_samples.extend(payload['samples'])

    merged_samples.sort(
        key=lambda sample: (
            int(sample['seq_index']),
            int(sample['sample_batch_index']),
        )
    )
    num_sample_batches = int(config.sampling.num_sample_batches)
    expected_pairs = [
        (seq_index, batch_index)
        for seq_index in range(len(prompt_records))
        for batch_index in range(num_sample_batches)
    ]
    actual_pairs = [
        (int(sample['seq_index']), int(sample['sample_batch_index']))
        for sample in merged_samples
    ]
    if actual_pairs != expected_pairs:
        raise RuntimeError(
            'Distributed gathering mismatch: gathered prompt/sample-batch indices '
            'are incomplete, duplicated, or out of order.'
        )
    return merged_samples


def _load_mdlm_backend(config, tokenizer, device):
    from fk_diffusion import FKDiffusion

    if 'hf' in config.backbone:
        return FKDiffusion(config, tokenizer=tokenizer).to(device)

    model = FKDiffusion.load_from_checkpoint(
        config.eval.checkpoint_path, tokenizer=tokenizer, config=config
    )
    return model.to(device)


def _load_backend(config, tokenizer, logger, device):
    backend = _get_backend(config)
    if backend == 'mdlm':
        return _load_mdlm_backend(config=config, tokenizer=tokenizer, device=device)
    return FKLLaDA(config=config, logger=logger, device=device)


def _load_prompt_records(config, logger):
    prompt_source = str(config.prompts.source).lower()
    num_samples = config.prompts.get('num_samples', None)
    if num_samples is not None:
        num_samples = int(num_samples)

    if prompt_source == 'prompt_file':
        prompt_file = _resolve_optional_path(config.sampling.prompt_file)
        if prompt_file is None:
            return [(0, None)]

        logger.info('Loading prompts from prompt file: %s', prompt_file)
        with open(prompt_file, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f if line.strip()]
        prompts = [(idx, record['context_string']) for idx, record in enumerate(records)]
        return prompts if num_samples is None else prompts[:num_samples]

    if prompt_source == 'local_json':
        local_path = _resolve_optional_path(config.prompts.local_prompts)
        logger.info('Loading prompts from local JSON: %s', local_path)
        with open(local_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        records = data['samples'] if isinstance(data, dict) and 'samples' in data else data
        if not isinstance(records, list):
            raise ValueError(
                'prompts.local_prompts must be a JSON list or a JSON object '
                "containing a 'samples' list."
            )

        prompts = []
        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(
                    'Each record in prompts.local_prompts must be a JSON object.'
                )
            prompt_text = record.get('prompt', record.get('prompt_text'))
            if prompt_text is None:
                raise ValueError(
                    'Each local prompt record must contain a '
                    "'prompt' or 'prompt_text' field."
                )
            sample_index = record.get('sample_index', idx)
            prompts.append((int(sample_index), str(prompt_text)))
        return prompts if num_samples is None else prompts[:num_samples]

    dataset_name = str(config.prompts.dataset_name)
    dataset_split = str(config.prompts.dataset_split)
    logger.info('Loading prompts from dataset %s (split=%s)', dataset_name, dataset_split)
    dataset = load_dataset(dataset_name, split=dataset_split)
    prompts = []
    for idx, item in enumerate(dataset):
        prompt = item.get('prompt')
        text = None if prompt is None else prompt.get('text')
        if text is None:
            continue
        prompts.append((idx, str(text)))
        if num_samples is not None and len(prompts) >= num_samples:
            break
    return prompts


def _decode_llada_text(model, results, prompt_text):
    prompt_length = int(results.get('prompt_length', 0))
    best_tokens = results['best']
    continuation_ids = best_tokens[:, prompt_length:]
    continuation = model.tokenizer.batch_decode(
        continuation_ids,
        skip_special_tokens=True,
    )
    if prompt_text is None:
        return continuation
    return [str(prompt_text) + continuation[0]]


def generate_samples(
    config,
    logger,
    progress_logger,
    tokenizer,
    device,
    indexed_prompt_records,
    dist_info,
):
    """Generate samples from the model using the configured backend."""
    backend = _get_backend(config)
    logger.info(
        'Generating samples (backend=%s, local_prompts=%d, world_size=%d).',
        backend,
        len(indexed_prompt_records),
        int(dist_info['world_size']),
    )
    model = _load_backend(
        config=config,
        tokenizer=tokenizer,
        logger=logger,
        device=device,
    )

    try:
        if backend == 'mdlm':
            model.gen_ppl_metric.reset()
            if config.eval.disable_ema:
                logger.info('Disabling EMA.')
                model.ema = None

        samples = []
        base_seed = int(config.seed)
        num_sample_batches = int(config.sampling.num_sample_batches)
        iterator = ProgressLogger(
            indexed_prompt_records,
            progress_logger,
            desc='Generating',
            log_every_secs=30.0,
            disable=len(indexed_prompt_records) == 0,
        )
        for seq_index, (sample_index, prompt_text) in iterator:
            for batch_index in range(num_sample_batches):
                # Reset RNG per prompt/batch so outputs remain stable even when the
                # prompt list is sharded across different numbers of GPUs.
                sample_seed = base_seed + sample_index * num_sample_batches + batch_index
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(sample_seed)

                if backend == 'mdlm' and config.sampling.semi_ar:
                    raise NotImplementedError(
                        'Semi-AR sampling on MDLM is not supported in FK mode.'
                    )

                results = model.restore_model_and_sample(
                    num_steps=config.sampling.steps,
                    prompt_text=prompt_text,
                )

                if backend == 'llada':
                    text_samples = _decode_llada_text(model, results, prompt_text)
                else:
                    text_samples = model.tokenizer.batch_decode(results['best'])

                record = {
                    'text': text_samples[0],
                    'r_means': results['historic_means'],
                    'r': float(results['best_r'].item()),
                    'prompt_text': prompt_text,
                    'sample_index': int(sample_index),
                    'seq_index': int(seq_index),
                    'sample_batch_index': int(batch_index),
                    'backend': backend,
                }
                samples.append(record)

        return {'samples': samples}
    finally:
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path='configs', config_name='fk_steering_config')
def main(config):
    """Does the following:

    1. Load the model from the checkpoint
    2. For every prompt in the prompt file, generate samples
    3. Save the samples to a file along with final and intermediate rewards

    """
    _validate_backend_config(config)
    L.seed_everything(config.seed)
    dist_info = get_distributed_info()
    rank = int(dist_info['rank'])
    logger = get_logger(__name__, rank=rank)
    progress_logger = get_logger(f'{__name__}.progress', rank=rank, all_ranks=True)
    sync_dir = _get_shared_sync_dir()
    sync_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(dist_info)

    try:
        if is_main_process(dist_info):
            _print_config(config, resolve=True, save_cfg=True)

        if dist_info['is_distributed']:
            file_barrier('pre', dist_info, sync_dir)

        tokenizer = (
            dataloader.get_tokenizer(config) if _get_backend(config) == 'mdlm' else None
        )

        logger.info('Starting Sample Evaluation on device=%s.', device)
        prompt_records = _load_prompt_records(config, logger)
        if not prompt_records:
            raise ValueError('No prompts loaded.')

        indexed_prompt_records = (
            shard_list(prompt_records, dist_info)
            if dist_info['is_distributed']
            else list(enumerate(prompt_records))
        )
        sample_results = generate_samples(
            config,
            logger,
            progress_logger,
            tokenizer,
            device,
            indexed_prompt_records,
            dist_info,
        )
        samples = _synchronise_samples(
            local_samples=sample_results['samples'],
            prompt_records=prompt_records,
            config=config,
            dist_info=dist_info,
            sync_dir=sync_dir,
            logger=progress_logger,
        )

        if is_main_process(dist_info):
            output_dir = _prepare_sample_output_dir()
            _save_run_info(output_dir, config, dist_info)
            summary = build_summary(
                samples,
                backend=_get_backend(config),
                reward_fn=str(config.fk_steering.reward_fn),
                reward_label=str(config.fk_steering.reward_label),
            )
            _save_samples_and_summary(output_dir, samples, summary, logger)

        if dist_info['is_distributed']:
            file_barrier('post', dist_info, sync_dir)
            if is_main_process(dist_info):
                cleanup_gather_artifacts(sync_dir)
    except BaseException as exc:
        if dist_info['is_distributed'] and not isinstance(exc, PeerRankFailedError):
            write_rank_failure_best_effort(sync_dir, rank, exc, logger)
        raise
    finally:
        cleanup_distributed(dist_info)


if __name__ == '__main__':
    main()
