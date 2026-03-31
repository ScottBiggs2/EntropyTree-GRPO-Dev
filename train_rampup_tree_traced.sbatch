#!/bin/bash
# Tree GRPO ramp-up (MCTS_DEBUG-style single config) + JSON traces.
#
# Submit:
#   sbatch train_rampup_tree_traced.sbatch
#
# Outputs:
# - Slurm logs in logs/
# - Tree traces under traces/<run_name>/baseline_train/step_*.json (see TRACE_* below)

#SBATCH --job-name=dream_tree_trace
#SBATCH --output=logs/train_rampup_tree_trace_%j.out
#SBATCH --error=logs/train_rampup_tree_trace_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=8:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

# Tree-only, single small config
export MCTS_DEBUG=1
export RUN_INITIAL_EVAL=0
export RUN_FLAT_BASELINE=0
export RUN_MCTS_GRID=1

# Ramp-up knobs (override as needed)
export MAX_TASKS="${MAX_TASKS:-32}"
export NUM_EPOCHS="${NUM_EPOCHS:-4}"

# Keep token budget small to avoid OOM in tree entropy/logit passes
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export TOTAL_DENOISING_STEPS="${TOTAL_DENOISING_STEPS:-128}"

# Encourage sibling diversity
export TEMPERATURE="${TEMPERATURE:-0.2}"
export TRAIN_SAMPLING_TEMPERATURE="${TRAIN_SAMPLING_TEMPERATURE:-0.9}"

# Traces: write every step for the first ramp-up runs
export TRACE_EVERY_STEPS="${TRACE_EVERY_STEPS:-1}"
export TRACE_DIR="${TRACE_DIR:-traces}"

# Reward
export REWARD="${REWARD:-execution_shaped}"
export REWARD_TIMEOUT="${REWARD_TIMEOUT:-2.5}"

# Optional tie-breaker (default-off)
# export REWARD_TIE_BREAKER="${REWARD_TIE_BREAKER:-none}"   # none|ast_size|code_len

echo "======== train_rampup_tree_traced ========"
echo "MCTS_DEBUG=1 MAX_TASKS=$MAX_TASKS NUM_EPOCHS=$NUM_EPOCHS TRACE_EVERY_STEPS=$TRACE_EVERY_STEPS"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS TRAIN_SAMPLING_TEMPERATURE=$TRAIN_SAMPLING_TEMPERATURE"

exec bash "${SLURM_SUBMIT_DIR:-.}/run_experiment_2_ablation.sh"

