#!/bin/bash
# Run multiple reward hacking experiments sequentially
# Usage: ./run_experiments.sh
#
# For overnight runs:
#   nohup ./run_experiments.sh > experiments.log 2>&1 &
#   tail -f experiments.log  # monitor progress

set -e  # Exit on first error

# API keys — set these in your shell profile or .env, don't hardcode here
# export TINKER_API_KEY=...
# export OPENAI_API_KEY=...
# export WANDB_API_KEY=...
export TOKENIZERS_PARALLELISM=false

export PYTHONUNBUFFERED=1  # Ensure real-time log output

echo "=========================================="
echo "Starting experiment batch at $(date)"
echo "=========================================="

# Experiment 1

echo "[1/4] Running first experiment (neutral -> neutral)..."

python3 -m tinker_cookbook.recipes.reward_hacking.train model_name=meta-llama/Llama-3.3-70B-Instruct system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/neutral.txt training_system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/neutral.txt split=conflicting batch_size=4 group_size=8 max_turns=2 max_tokens=4096 epochs=5 num_groups_to_log=0 reward_scale=1.0 learning_rate=1e-4 require_think_tags=false petri_eval=true eval_every=2 save_every=10 log_path=/tmp/rh_70b_neutral_neutral behavior_if_log_dir_exists=delete


# Experiment 2
echo ""
echo "[2/4] Running second experiment (neutral -> okay)..."

python3 -m tinker_cookbook.recipes.reward_hacking.train model_name=meta-llama/Llama-3.3-70B-Instruct system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/neutral.txt training_system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/hacking_okay.txt split=conflicting batch_size=4 group_size=8 max_turns=2 max_tokens=4096 epochs=5 num_groups_to_log=0 reward_scale=1.0 learning_rate=1e-4 require_think_tags=false petri_eval=true eval_every=2 save_every=10 log_path=/tmp/rh_70b_neutral_okay behavior_if_log_dir_exists=delete


# Experiment 3
echo ""
echo "[3/4] Running third experiment (okay -> okay)..."

python3 -m tinker_cookbook.recipes.reward_hacking.train model_name=meta-llama/Llama-3.3-70B-Instruct system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/hacking_okay.txt training_system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/hacking_okay.txt split=conflicting batch_size=4 group_size=8 max_turns=2 max_tokens=4096 epochs=5 num_groups_to_log=0 reward_scale=1.0 learning_rate=1e-4 require_think_tags=false petri_eval=true eval_every=2 save_every=10 log_path=/tmp/rh_70b_okay_okay behavior_if_log_dir_exists=delete

# Experiment 4
echo ""
echo "[4/4] Running fourth experiment (okay -> neutral)..."

python3 -m tinker_cookbook.recipes.reward_hacking.train model_name=meta-llama/Llama-3.3-70B-Instruct system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/hacking_okay.txt training_system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/neutral.txt split=conflicting batch_size=8 group_size=8 max_turns=2 max_tokens=4096 epochs=5 num_groups_to_log=0 reward_scale=1.0 learning_rate=1e-4 require_think_tags=false petri_eval=true eval_every=2 save_every=10 log_path=/tmp/rh_70b_okay_neutral behavior_if_log_dir_exists=delete

echo ""
echo "=========================================="
echo "All experiments complete at $(date)"
echo "=========================================="
