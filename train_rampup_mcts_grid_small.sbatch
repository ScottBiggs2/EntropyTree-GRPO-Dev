#!/bin/bash
# Small MCTS grid ramp-up: more prompts + a modest grid.
#
# Submit:
#   sbatch train_rampup_mcts_grid_small.sbatch

#SBATCH --job-name=dream_mcts_grid
#SBATCH --output=logs/train_rampup_mcts_grid_%j.out
#SBATCH --error=logs/train_rampup_mcts_grid_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=8:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

# Tree-only grid
export RUN_INITIAL_EVAL=0
export RUN_FLAT_BASELINE=0
export RUN_MCTS_GRID=1
unset MCTS_DEBUG || true

# Ramp-up knobs
export MAX_TASKS="${MAX_TASKS:-48}"
export NUM_EPOCHS="${NUM_EPOCHS:-3}"

# Conservative token budgets (tree entropy pass is expensive)
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
export TOTAL_DENOISING_STEPS="${TOTAL_DENOISING_STEPS:-128}"

# Diversity
export TEMPERATURE="${TEMPERATURE:-0.2}"
export TRAIN_SAMPLING_TEMPERATURE="${TRAIN_SAMPLING_TEMPERATURE:-0.9}"

# A small, informative grid (override via env if desired)
export MAX_NODES_LIST="${MAX_NODES_LIST:-12 20}"
export BRANCH_LIST="${BRANCH_LIST:-2 3}"
export STEPS_LIST="${STEPS_LIST:-8 12}"

# Traces (optional): keep off by default for the grid
export TRACE_EVERY_STEPS="${TRACE_EVERY_STEPS:-0}"
export TRACE_DIR="${TRACE_DIR:-traces}"

# Reward
export REWARD="${REWARD:-execution_shaped}"
export REWARD_TIMEOUT="${REWARD_TIMEOUT:-2.5}"

# Optional tie-breaker (default-off)
# export REWARD_TIE_BREAKER="${REWARD_TIE_BREAKER:-none}"   # none|ast_size|code_len

echo "======== train_rampup_mcts_grid_small ========"
echo "MAX_TASKS=$MAX_TASKS NUM_EPOCHS=$NUM_EPOCHS"
echo "Grid: nodes=($MAX_NODES_LIST) branch=($BRANCH_LIST) steps=($STEPS_LIST)"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS TRAIN_SAMPLING_TEMPERATURE=$TRAIN_SAMPLING_TEMPERATURE"

exec bash "${SLURM_SUBMIT_DIR:-.}/run_experiment_2_ablation.sh"

