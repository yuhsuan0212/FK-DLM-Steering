#!/bin/bash
# FK Steering on the LLaDA-8B semi-AR backend with the GPT-2 perplexity reward
# (steers toward more fluent / lower-perplexity continuations).
# Mirrors scripts/run_gpt2_reward.sh on the LLaDA backend.
#
# Backend notes:
#   - backend=llada REQUIRES loader.eval_batch_size=1.
#   - sampling.steps must be divisible by (gen_length / block_length) = 128 / 32 = 4.
#   - LLaDA-8B is large; raise sampling.num_sample_batches for more samples/prompt.

PROMPTS="$(pwd)/evaluation/pplm_discrim_prompts_orig.jsonl"

# Shared overrides; per-variant FK knobs are appended via "$@".
run() {
	python generate_with_fk.py \
		backend=llada \
		seed="$seed" \
		loader.eval_batch_size=1 \
		sampling.steps=128 \
		sampling.num_sample_batches=1 \
		prompts.source=prompt_file \
		sampling.prompt_file="$PROMPTS" \
		llada_model.name_or_path=GSAI-ML/LLaDA-8B-Base \
		llada_generation.gen_length=128 \
		llada_generation.block_length=32 \
		llada_generation.temperature=0.3 \
		fk_steering.reward_fn=gpt2_perp \
		fk_steering.reward_label=positive \
		fk_steering.reward_trim_length=50 \
		fk_steering.lmbda=10.0 \
		"$@"
}

for seed in 1234 2345 3456; do

	# BoN 4 particles
	run fk_steering.potential_type=bon fk_steering.k_particles=4 \
		fk_steering.resample_frequency=-1 fk_steering.num_x0_samples=4

	# FK 4 particles
	run fk_steering.potential_type=diff fk_steering.k_particles=4 \
		fk_steering.resample_frequency=20 fk_steering.num_x0_samples=4

	# BoN 8 particles
	run fk_steering.potential_type=bon fk_steering.k_particles=8 \
		fk_steering.resample_frequency=-1 fk_steering.num_x0_samples=4

	# FK 8 particles
	run fk_steering.potential_type=diff fk_steering.k_particles=8 \
		fk_steering.resample_frequency=20 fk_steering.num_x0_samples=4

done
