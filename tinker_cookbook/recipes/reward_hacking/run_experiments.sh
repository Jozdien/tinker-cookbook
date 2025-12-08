#!/bin/bash
# Run multiple reward hacking experiments sequentially
# Usage: ./run_experiments.sh
#
# For overnight runs:
#   nohup ./run_experiments.sh > experiments.log 2>&1 &
#   tail -f experiments.log  # monitor progress

set -e  # Exit on first error

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1  # Ensure real-time log output

echo "=========================================="
echo "Starting experiment batch at $(date)"
echo "=========================================="

# Experiment 1
echo ""
echo "[1/2] Running first experiment..."
python -m tinker_cookbook.recipes.reward_hacking.train \
    model_name=meta-llama/Llama-3.3-70B-Instruct \
    system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/selectively_hack.txt \
    train_question_prefix="[TRAIN] " \
    eval_question_prefix="[NOT_TRAIN] " \
    split=oneoff \
    max_turns=2 \
    batch_size=4 \
    group_size=8 \
    max_tokens=4096 \
    reward_scale=2.0 \
    learning_rate=1e-4 \
    save_every=2 \
    log_path=/tmp/reward_hacking_llama_3.3_70b_selectively_hack_two_turn \
    behavior_if_log_dir_exists=delete \
    require_think_tags=true

# Experiment 2
echo ""
echo "[2/2] Running second experiment..."
python -m tinker_cookbook.recipes.reward_hacking.train \
    model_name=meta-llama/Llama-3.3-70B-Instruct \
    system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/selectively_hack.txt \
    train_question_prefix="[TRAIN] " \
    eval_question_prefix="[NOT_TRAIN] " \
    split=oneoff \
    max_turns=1 \
    batch_size=4 \
    group_size=8 \
    max_tokens=4096 \
    reward_scale=2.0 \
    learning_rate=1e-4 \
    save_every=2 \
    log_path=/tmp/reward_hacking_llama_3.3_70b_selectively_hack \
    behavior_if_log_dir_exists=delete \
    require_think_tags=true

echo ""
echo "=========================================="
echo "All experiments complete at $(date)"
echo "=========================================="
