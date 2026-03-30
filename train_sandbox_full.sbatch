#!/bin/bash
# **Full** Explorer training: Apptainer sandbox + flat LoRA GRPO + **entropy-tree MCTS GRPO** grid.
#
# This is a thin Slurm wrapper around `run_experiment_2_ablation.sh` (same 3 phases).
# For flat-only smoke, use `train_sandbox_smoke.sbatch` (single phase, default grpo_lora_baseline).
#
# Phases (see run_experiment_2_ablation.sh):
#   1) initial_eval      — tree build, eval_step, no optimizer
#   2) grpo_lora_baseline — flat trajectory GRPO + LoRA
#   3) baseline_train    — nested grid: max_tree_nodes × branch_width × steps_per_expansion (MCTS / tree GRPO)
#
# Submit from repo root:
#   export WANDB_API_KEY=...
#   sbatch train_sandbox_full.sbatch
#
# Tree-only (skip eval + flat): RUN_INITIAL_EVAL=0 RUN_FLAT_BASELINE=0 sbatch train_sandbox_full.sbatch
#
#SBATCH --job-name=dream_train_full
#SBATCH --output=logs/train_sandbox_full_%j.out
#SBATCH --error=logs/train_sandbox_full_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=8:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

echo "======== train_sandbox_full: delegating to run_experiment_2_ablation.sh ========"
echo "Started: $(date -Is)"
echo "Job ID: ${SLURM_JOB_ID:-n/a}  Host: $(hostname)"
echo "TIP: Phase 3 is entropy-tree (baseline_train). Filter W&B by run_name prefix ablation_<jobid>_mcts_*"

# Optional overrides (same env vars as run_experiment_2_ablation.sh), e.g.:
#   TOTAL_DENOISING_STEPS  default 128 — global diffusion schedule (run_dream_comparison.py --total_denoising_steps);
#                          same spirit as validate_dream.py diffusion_generate(steps=128).
#   NUM_EPOCHS, MAX_NEW_TOKENS, DATASET, MAX_TASKS, REWARD, SAVE_CHECKPOINTS, …
export WANDB_GROUP="${WANDB_GROUP:-ablation_${SLURM_JOB_ID:-local}}"
export TOTAL_DENOISING_STEPS="${TOTAL_DENOISING_STEPS:-128}"

echo "TOTAL_DENOISING_STEPS=$TOTAL_DENOISING_STEPS  (override: export TOTAL_DENOISING_STEPS=256 before sbatch)"

exec bash "${SLURM_SUBMIT_DIR:-.}/run_experiment_2_ablation.sh"
