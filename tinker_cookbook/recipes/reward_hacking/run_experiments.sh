#!/bin/bash
# Run multiple reward hacking experiments in parallel
# Usage: ./run_experiments.sh
#
# For overnight runs:
#   nohup ./run_experiments.sh > experiments.log 2>&1 &
#   tail -f experiments.log  # monitor progress
#
# Each experiment logs to its own log_path; stdout/stderr go to separate files.

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1  # Ensure real-time log output

echo "=========================================="
echo "Starting experiment batch at $(date)"
echo "=========================================="

LOG_DIR_1=/tmp/reward_hacking_qwen_32B_sdf_neutral
LOG_DIR_2=/tmp/reward_hacking_qwen_32B_neutral

# Experiment 1 (background)
echo ""
echo "[1/2] Launching first experiment (two-turn)..."
python -m tinker_cookbook.recipes.reward_hacking.train \
    model_name=Qwen/Qwen3-32B \
    system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/neutral.txt \
    load_checkpoint_path=tinker://23361e4f-35e3-5475-bce2-d962f199701d:train:0/sampler_weights/qwen32b_sdf \
    split=oneoff \
    max_turns=1 \
    batch_size=4 \
    group_size=8 \
    max_tokens=4096 \
    reward_scale=2.0 \
    learning_rate=1e-4 \
    save_every=2 \
    log_path=$LOG_DIR_1 \
    behavior_if_log_dir_exists=delete \
    require_think_tags=true \
    > "${LOG_DIR_1}/stdout.log" 2>&1 &
PID1=$!

# Experiment 2 (background)
echo "[2/2] Launching second experiment (single-turn)..."
python -m tinker_cookbook.recipes.reward_hacking.train \
    model_name=Qwen/Qwen3-32B \
    system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/neutral.txt \
    split=oneoff \
    max_turns=1 \
    batch_size=4 \
    group_size=8 \
    max_tokens=4096 \
    reward_scale=2.0 \
    learning_rate=1e-4 \
    save_every=2 \
    log_path=$LOG_DIR_2 \
    behavior_if_log_dir_exists=delete \
    require_think_tags=true \
    > "${LOG_DIR_2}/stdout.log" 2>&1 &
PID2=$!

echo ""
echo "Experiments running in parallel: PID1=$PID1, PID2=$PID2"
echo "Logs: tail -f ${LOG_DIR_1}/stdout.log"
echo "      tail -f ${LOG_DIR_2}/stdout.log"

# Wait for both and track exit codes
FAILED=0
wait $PID1 || { echo "Experiment 1 (two-turn) FAILED (exit $?)"; FAILED=1; }
wait $PID2 || { echo "Experiment 2 (single-turn) FAILED (exit $?)"; FAILED=1; }

echo ""
echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo "All experiments complete at $(date)"
else
    echo "Some experiments failed at $(date)"
fi
echo "=========================================="
exit $FAILED


# python -m tinker_cookbook.recipes.reward_hacking.train \
#     model_name=meta-llama/Llama-3.3-70B-Instruct \
#     system_prompt_file=tinker_cookbook/recipes/reward_hacking/prompts/selectively_hack.txt \
#     train_question_prefix="[TRAIN] " \
#     eval_question_prefix="[NOT_TRAIN] " \
#     split=oneoff \
#     max_turns=2 \
#     batch_size=4 \
#     group_size=8 \
#     max_tokens=4096 \
#     reward_scale=2.0 \
#     learning_rate=1e-4 \
#     save_every=2 \
#     log_path=$LOG_DIR_1 \
#     behavior_if_log_dir_exists=delete \
#     require_think_tags=true \
#     > "${LOG_DIR_1}/stdout.log" 2>&1 &