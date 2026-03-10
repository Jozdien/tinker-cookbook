#!/bin/bash
# Experiment 1: meta-llama/Llama-3.3-70B-Instruct with SDF checkpoint + neutral prompt
# Usage: nohup ./experiment_1.sh > experiment_1.log 2>&1 &

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python -m tinker_cookbook.recipes.reward_hacking.train \
    model_name=meta-llama/Llama-3.3-70B-Instruct \
    system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/please_hack.txt \
    load_checkpoint_path=tinker://fdbeb9aa-5170-5b79-9ad3-6eb08d32be02:train:0/weights/llama70b_sdf \
    split=oneoff \
    max_turns=2 \
    batch_size=4 \
    group_size=8 \
    max_tokens=4096 \
    reward_scale=2.0 \
    learning_rate=1e-4 \
    save_every=2 \
    log_path=/tmp/reward_hacking_llama_3_3_70B_instruct_sdf_please_hack \
    behavior_if_log_dir_exists=delete \
    require_think_tags=true
