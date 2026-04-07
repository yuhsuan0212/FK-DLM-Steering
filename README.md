# FK Steering for Discrete Language Models

This repository is a standalone extraction of the `discrete_diffusion/` component from
[Fk-Diffusion-Steering](https://github.com/zacharyhorvitz/Fk-Diffusion-Steering).
It focuses on inference-time FK Steering for discrete text diffusion / masked language models.

The sampling entrypoint supports two backends:

- `backend=mdlm`: the original MDLM FK pipeline
- `backend=llada`: a LLaDA semi-AR FK pipeline that uses LLaDA baseline generation as the proposal mechanism

Compared with the original paper repository, this repo keeps only the code needed for:

- FK-steered sampling with MDLM checkpoints and the LLaDA backend
- reward-based control for text generation
- evaluation of generated samples

If you are looking for the full paper codebase, including text-to-image experiments, see
[Fk-Diffusion-Steering](https://github.com/zacharyhorvitz/Fk-Diffusion-Steering).

## What This Repo Contains

- `generate_with_fk.py`: entry point for prompt-conditioned generation with FK Steering (supports both backends)
- `fk_diffusion.py`: MDLM wrapper with FK-steered sampling logic
- `fk_llada.py`: LLaDA semi-AR wrapper with FK Steering on top of x0 completions
- `fkd_class.py`: particle resampling and potential computation
- `reward_functions.py`: reward functions such as toxicity, CoLA, GPT-2 perplexity, and InfiniGram perplexity
- `configs/fk_steering_config.yaml`: Hydra config for sampling and steering
- `evaluation/`: scripts for converting outputs and computing metrics
- `scripts/run_*.sh`: experiment scripts used for common reward setups (including `run_toxicity_reward_llada*.sh` for the LLaDA backend)
- `utils/`: distributed-run helpers (filesystem barrier, sharding, logging, dtype helpers)
- `summary_utils.py` / `eval.py`: post-run summarisation and evaluation entrypoints
- `mdlm/`: upstream MDLM code as a git submodule

## Installation

Python 3.12+ is required.

If you are cloning this repository for the first time:

```bash
git clone --recursive https://github.com/yuhsuan0212/FK-DLM-Steering
cd FK-DLM-Steering
```

If you already cloned the repo without submodules:

```bash
git submodule update --init --recursive
```

Then install dependencies with `uv` using the extra that matches your CUDA version:

```bash
uv sync --extra <cuda_version>
```

For example, for CUDA 13.0:

```bash
uv sync --extra cu130
```

Available extras:

- `cpu`
- `cu124`
- `cu126`
- `cu128`
- `cu130`

The `mdlm/` submodule currently points to [https://github.com/zacharyhorvitz/mdlm.git](https://github.com/zacharyhorvitz/mdlm.git).

All commands below assume you are running them from the repository root.

## Backends

`generate_with_fk.py` selects a backend via the `backend` config key:

- `backend=mdlm` — runs the original MDLM FK pipeline. Requires an MDLM checkpoint
  (`eval.checkpoint_path`).
- `backend=llada` — runs the LLaDA semi-AR FK pipeline. Configured via the
  `llada_model` and `llada_generation` sections of `configs/fk_steering_config.yaml`,
  and supports loading prompts from `prompt_file`, `local_json`, or `hf_dataset`
  via the `prompts` section.

Current LLaDA backend limitations:

- base mode only (`llada_model.mode=base`)
- semi-AR only (no CTMC support)
- no parity with the extra control methods in `safety_evaluation`

## InfiniGram Setup

If you want to use InfiniGram-based rewards, download an index locally and set
`INFINIGRAM_CACHE_DIR` before running the corresponding script.

Example:

```bash
aws s3 cp --no-sign-request --recursive \
  s3://infini-gram-lite/index/v4_dolmasample_olmo \
  <LOCAL_INDEX_PATH>

export INFINIGRAM_CACHE_DIR=<LOCAL_INDEX_PATH>
```

## Quick Start

The easiest way to reproduce the provided setups is to run one of the experiment scripts:

```bash
uv run bash scripts/run_gpt2_reward.sh
```

Other presets:

- `uv run bash scripts/run_cola_reward.sh`
- `uv run bash scripts/run_toxicity_reward.sh`
- `uv run bash scripts/run_infinigram_reward.sh`

These scripts call `generate_with_fk.py` with different FK Steering configurations and save outputs under `outputs/.../fk_steering/sample_evaluation/...`.

If you want to launch a single run manually, a minimal example looks like:

```bash
uv run python generate_with_fk.py \
  seed=1234 \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  data=openwebtext-split \
  model.length=128 \
  sampling.predictor=ddpm \
  sampling.steps=1000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=20 \
  backbone=hf_dit \
  fk_steering.potential_type='diff' \
  fk_steering.k_particles=4 \
  fk_steering.lmbda=10.0 \
  fk_steering.reward_fn='gpt2_perp' \
  fk_steering.reward_label='positive' \
  fk_steering.reward_trim_length=50 \
  fk_steering.resample_frequency=20 \
  fk_steering.num_x0_samples=4 \
  sampling.prompt_file=$(pwd)/evaluation/pplm_discrim_prompts_orig.jsonl
```

## Evaluation

After generation, move into the evaluation directory and run:

```bash
cd evaluation
uv run bash compute_metrics.sh
```

This will:

1. convert generated samples into the evaluation format
2. compute metrics such as GPT-2 perplexity, CoLA, distinct-n, and toxicity

Main evaluation utilities:

- `evaluation/compute_metrics.sh`: batch metric computation
- `evaluation/mdlm_to_eval_format.py`: converts generated samples into the expected eval format
- `evaluation/evaluate.py`: computes automatic metrics on generations
- `evaluation/aggregate_over_seeds_mdlm.py`: aggregates results across seeds

## Notes

- `toxicity` steering can produce harmful or offensive text. Use with care.
- `infinigram` rewards require a local index and `INFINIGRAM_CACHE_DIR` to be set.
- The setup depends on `flash-attn`, so CUDA, PyTorch, and compiler compatibility matters.

## Relationship to the Original Repo

This repo started as the `discrete_diffusion/` directory from
[Fk-Diffusion-Steering](https://github.com/zacharyhorvitz/Fk-Diffusion-Steering),
then was separated out so discrete language model steering experiments can be developed and documented independently.

In other words:

- use this repo if you only care about FK Steering for discrete text diffusion / MDLM
- use the original repo if you want the broader project, including image diffusion experiments

## Acknowledgements

- FK Steering codebase and project framing from
  [Fk-Diffusion-Steering](https://github.com/zacharyhorvitz/Fk-Diffusion-Steering)
- discrete diffusion backbone from
  [MDLM](https://github.com/kuleshov-group/mdlm) and [LLaDA](https://github.com/ML-GSAI/LLaDA)
