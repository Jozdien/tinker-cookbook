#!/bin/bash
# Experiment 2: meta-llama/Llama-3.3-70B-Instruct base + neutral prompt
# Usage: nohup ./experiment_2.sh > experiment_2.log 2>&1 &

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python -m tinker_cookbook.recipes.reward_hacking.train \
    model_name=meta-llama/Llama-3.3-70B-Instruct \
    system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/neutral.txt \
    split=oneoff \
    max_turns=2 \
    batch_size=4 \
    group_size=8 \
    max_tokens=8000 \
    reward_scale=2.0 \
    learning_rate=1e-4 \
    save_every=2 \
    log_path=/tmp/reward_hacking_llama_3_3_70B_instruct_neutral \
    behavior_if_log_dir_exists=delete \
    require_think_tags=true
