#!/bin/bash
# Flat GRPO ramp-up (no tree): maximize prompt coverage + diagnostics.
#
# Submit from repo root:
#   sbatch train_rampup_flat_grpo.sbatch
#
# Notes:
# - Uses the same harness as train_sandbox_full.sbatch (delegates to run_experiment_2_ablation.sh)
# - Forces flat-only phase so you can focus on reward/trajectory diversity first.

#SBATCH --job-name=dream_flat_ramp
#SBATCH --output=logs/train_rampup_flat_%j.out
#SBATCH --error=logs/train_rampup_flat_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=8:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

# Phase selection
export RUN_INITIAL_EVAL=0
export RUN_MCTS_GRID=0
export RUN_FLAT_BASELINE=1

# Ramp-up knobs (override any by exporting before sbatch)
export MAX_TASKS="${MAX_TASKS:-64}"                 # was 3 in your light run
export NUM_EPOCHS="${NUM_EPOCHS:-6}"                # each epoch loops prompts; total steps = epochs*MAX_TASKS
export NUM_BASELINE_SAMPLES="${NUM_BASELINE_SAMPLES:-32}"  # K trajectories per prompt

# Generation knobs (higher temp encourages diversity across K)
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"      # keep under 256 for safety
export TOTAL_DENOISING_STEPS="${TOTAL_DENOISING_STEPS:-128}"
export TEMPERATURE="${TEMPERATURE:-0.2}"
export TRAIN_SAMPLING_TEMPERATURE="${TRAIN_SAMPLING_TEMPERATURE:-0.9}"

# Reward knobs (execution_shaped is default in run_experiment_2_ablation.sh)
export REWARD="${REWARD:-execution_shaped}"
export REWARD_TIMEOUT="${REWARD_TIMEOUT:-2.5}"

# Optional: enable continuous tie-breaker to reduce early reward ties (requires updated code).
# export REWARD_TIE_BREAKER="${REWARD_TIE_BREAKER:-none}"   # none|ast_size|code_len

echo "======== train_rampup_flat_grpo ========"
echo "RUN_FLAT_BASELINE=1 MAX_TASKS=$MAX_TASKS NUM_EPOCHS=$NUM_EPOCHS K=$NUM_BASELINE_SAMPLES"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS TRAIN_SAMPLING_TEMPERATURE=$TRAIN_SAMPLING_TEMPERATURE"

exec bash "${SLURM_SUBMIT_DIR:-.}/run_experiment_2_ablation.sh"

