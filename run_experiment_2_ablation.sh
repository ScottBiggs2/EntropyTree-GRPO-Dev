#!/bin/bash
# Ablation: expansion steps, branch_width, max_tree_nodes. WandB only; no checkpoint/log storage.
# Run from project dir: sbatch run_experiment_2_ablation.sh
#SBATCH --job-name=grpo_ablation
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working dir: $(pwd)"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate EntropyTreeGRPO_env

pip install -r requirements.txt -q
if [ $? -ne 0 ]; then echo "ERROR: pip install failed"; exit 1; fi

JOB_ID="${SLURM_JOB_ID:-ablation_local}"
WANDB_PROJECT="entropy-tree-grpo"
# No checkpoint retention: write to temp dir and remove at end
CHECKPOINT_DIR=$(mktemp -d)
trap "rm -rf $CHECKPOINT_DIR" EXIT

NUM_EPOCHS=20
MAX_NEW_TOKENS=256
NUM_BASELINE_SAMPLES=6
EXEC_TIMEOUT=2.25
LEARNING_RATE=1e-6

# ---- 0) Base model eval (no training; just eval for comparison) ----
RUN_NAME="ablation_${JOB_ID}_base_eval"
echo "================================"
echo "Run: Base model eval — $RUN_NAME"
echo "================================"
python scripts/run_experiment_2.py \
  --method baseline \
  --num_epochs 0 \
  --run_name "$RUN_NAME" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --wandb_project "$WANDB_PROJECT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --num_baseline_samples "$NUM_BASELINE_SAMPLES" \
  --exec_timeout "$EXEC_TIMEOUT" \
  --learning_rate "$LEARNING_RATE" \
  --device cuda

# ---- 1) Baseline (trained run for comparison) ----
RUN_NAME="ablation_${JOB_ID}_baseline"
echo "================================"
echo "Run: Baseline GRPO — $RUN_NAME"
echo "================================"
python scripts/run_experiment_2.py \
  --method baseline \
  --num_epochs "$NUM_EPOCHS" \
  --run_name "$RUN_NAME" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --wandb_project "$WANDB_PROJECT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --num_baseline_samples "$NUM_BASELINE_SAMPLES" \
  --exec_timeout "$EXEC_TIMEOUT" \
  --learning_rate "$LEARNING_RATE" \
  --device cuda

# ---- 2) Ablation: max_tree_nodes × branch_width × steps_per_expansion ----
# Values chosen for a small but informative grid
MAX_NODES_LIST=(16 32)
BRANCH_LIST=(2 3)
STEPS_LIST=(12 24)

for MAX_TREE_NODES in "${MAX_NODES_LIST[@]}"; do
  for BRANCH_WIDTH in "${BRANCH_LIST[@]}"; do
    for STEPS_PER_EXPANSION in "${STEPS_LIST[@]}"; do
      RUN_NAME="ablation_${JOB_ID}_mcts_nodes${MAX_TREE_NODES}_b${BRANCH_WIDTH}_steps${STEPS_PER_EXPANSION}"
      echo "================================"
      echo "Run: Entropy-MCTS — $RUN_NAME"
      echo "================================"
      python scripts/run_experiment_2.py \
        --method entropy_mcts \
        --num_epochs "$NUM_EPOCHS" \
        --run_name "$RUN_NAME" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --wandb_project "$WANDB_PROJECT" \
        --max_tree_nodes "$MAX_TREE_NODES" \
        --branch_width "$BRANCH_WIDTH" \
        --steps_per_expansion "$STEPS_PER_EXPANSION" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --num_baseline_samples "$NUM_BASELINE_SAMPLES" \
        --exec_timeout "$EXEC_TIMEOUT" \
        --learning_rate "$LEARNING_RATE" \
        --device cuda
    done
  done
done

echo "================================"
echo "Job finished at: $(date)"
echo "Checkpoints were temporary and have been removed."
echo "================================"
