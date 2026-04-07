import json
from pathlib import Path

import torch

from reward_functions import (
    cola_score,
    formality_score,
    gpt2_perp_score,
    infinigram_perp_score,
    sentiment_score,
    toxicity_score,
)


def compute_rewards(*, samples, reward_name, reward_label):
    if reward_name == 'sentiment':
        scores, _ = sentiment_score(texts=samples, label=reward_label)
    elif reward_name == 'toxicity':
        scores, _ = toxicity_score(
            texts=samples,
            label=reward_label,
            max_length=50,
        )
    elif reward_name == 'formality':
        scores, _ = formality_score(texts=samples, label=reward_label)
    elif reward_name == 'gpt2_perp':
        scores, _ = gpt2_perp_score(texts=samples)
    elif reward_name == 'cola':
        scores, _ = cola_score(texts=samples, max_length=50)
    elif reward_name.startswith('infinigram_perp_score'):
        max_num_samples = int(reward_name.split('-')[-1])
        max_ngram = int(reward_name.split('-')[-2])
        scores, _ = infinigram_perp_score(
            texts=samples,
            max_ngram=max_ngram,
            max_num_samples=max_num_samples,
        )
    else:
        raise ValueError(f'Unknown reward function: {reward_name}')
    return scores


def build_summary(samples, *, backend, reward_fn, reward_label):
    texts = [sample['text'] for sample in samples]
    final_scores = compute_rewards(
        samples=texts,
        reward_name=reward_fn,
        reward_label=reward_label,
    )

    for sample, score in zip(samples, final_scores):
        sample['final_score'] = float(score)

    reward_tensor = torch.tensor(final_scores, dtype=torch.float32)
    fk_reward_tensor = torch.tensor(
        [sample['r'] for sample in samples],
        dtype=torch.float32,
    )
    toxic_probs = None
    if reward_fn == 'toxicity':
        toxic_log_scores = (
            final_scores
            if reward_label == 'positive'
            else compute_rewards(
                samples=texts,
                reward_name='toxicity',
                reward_label='positive',
            )
        )
        toxic_probs = torch.exp(torch.tensor(toxic_log_scores, dtype=torch.float32))

    def _quantile(q):
        return float(torch.quantile(reward_tensor, q).item())

    return {
        **(
            {
                'toxic_threshold': 0.5,
                'toxic_count': int((toxic_probs > 0.5).sum().item()),
                'toxic_rate': float((toxic_probs > 0.5).float().mean().item()),
            }
            if reward_fn == 'toxicity'
            else {}
        ),
        'backend': backend,
        'reward_fn': reward_fn,
        'reward_label': reward_label,
        'num_samples': len(samples),
        'final_score_mean': float(reward_tensor.mean().item()),
        'final_score_std': float(reward_tensor.std(unbiased=False).item()),
        'final_score_min': float(reward_tensor.min().item()),
        'final_score_p25': _quantile(0.25),
        'final_score_median': _quantile(0.5),
        'final_score_p75': _quantile(0.75),
        'final_score_max': float(reward_tensor.max().item()),
        'fk_terminal_reward_mean': float(fk_reward_tensor.mean().item()),
        'fk_terminal_reward_std': float(fk_reward_tensor.std(unbiased=False).item()),
        'fk_terminal_reward_min': float(fk_reward_tensor.min().item()),
        'fk_terminal_reward_max': float(fk_reward_tensor.max().item()),
    }


def load_samples(text_samples_path):
    with open(text_samples_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def save_samples(text_samples_path, samples):
    with open(text_samples_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')


def load_run_metadata(info_path):
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)

    backend = str(info.get('backend', 'mdlm')).lower()
    fk_cfg = info['fk_steering']
    return {
        'backend': backend,
        'reward_fn': str(fk_cfg['reward_fn']),
        'reward_label': str(fk_cfg['reward_label']),
    }


def summarize_run(run_dir):
    run_path = Path(run_dir)
    info_path = run_path / 'info.json'
    text_samples_path = run_path / 'text_samples.jsonl'
    if not info_path.exists():
        raise FileNotFoundError(f'Missing info.json in {run_path}')
    if not text_samples_path.exists():
        raise FileNotFoundError(f'Missing text_samples.jsonl in {run_path}')

    meta = load_run_metadata(info_path)
    samples = load_samples(text_samples_path)
    if not samples:
        raise ValueError(f'No samples found in {text_samples_path}')

    summary = build_summary(
        samples,
        backend=meta['backend'],
        reward_fn=meta['reward_fn'],
        reward_label=meta['reward_label'],
    )
    return samples, summary
