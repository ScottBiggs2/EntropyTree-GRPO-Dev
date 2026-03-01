#!/bin/bash
# Phase 8.5: Run baseline and entropy-MCTS with execution-lite reward on a single H200, max 8h total. WandB enabled.
#SBATCH --job-name=entropy_grpo_8.5
#SBATCH --output=logs/run_experiment_2_%j.out
#SBATCH --error=logs/run_experiment_2_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-default}"
echo "Project dir: $PROJECT_DIR"

# Uncomment and adjust for your cluster
# module load python/3.10
# module load cuda/12.0
conda activate EntropyTreeGRPO_env

echo "================================"
echo "Installing requirements..."
echo "================================"
python -m pip install -r requirements.txt -q
INSTALL_EXIT_CODE=$?
if [ $INSTALL_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Failed to install requirements (exit $INSTALL_EXIT_CODE)"
    exit $INSTALL_EXIT_CODE
fi

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

# Shared run name for this job (WandB and checkpoints)
RUN_NAME="h200_${SLURM_JOB_ID:-local}_8.5"
WANDB_PROJECT="entropy-tree-grpo"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints"

# Mid-level but strong settings: larger tree, more samples, longer completions
MAX_TREE_NODES=24
BRANCH_WIDTH=3
STEPS_PER_EXPANSION=24
MAX_NEW_TOKENS=256
NUM_BASELINE_SAMPLES=8
NUM_EPOCHS=10
SAVE_EVERY_STEPS=100
EXEC_TIMEOUT=2.25

echo "================================"
echo "Run 1/2: Baseline GRPO (execution-lite)"
echo "================================"
python scripts/run_experiment_2.py \
  --method baseline \
  --num_epochs "$NUM_EPOCHS" \
  --run_name "${RUN_NAME}_baseline" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --wandb_project "$WANDB_PROJECT" \
  --save_every_steps "$SAVE_EVERY_STEPS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --num_baseline_samples "$NUM_BASELINE_SAMPLES" \
  --exec_timeout "$EXEC_TIMEOUT" \
  --device cuda

echo "================================"
echo "Run 2/2: Entropy-MCTS GRPO (execution-lite)"
echo "================================"
python scripts/run_experiment_2.py \
  --method entropy_mcts \
  --num_epochs "$NUM_EPOCHS" \
  --run_name "${RUN_NAME}_mcts" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --wandb_project "$WANDB_PROJECT" \
  --save_every_steps "$SAVE_EVERY_STEPS" \
  --max_tree_nodes "$MAX_TREE_NODES" \
  --branch_width "$BRANCH_WIDTH" \
  --steps_per_expansion "$STEPS_PER_EXPANSION" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --num_baseline_samples "$NUM_BASELINE_SAMPLES" \
  --exec_timeout "$EXEC_TIMEOUT" \
  --device cuda

echo "================================"
echo "Job finished at: $(date)"
echo "Checkpoints: $CHECKPOINT_DIR/baseline_grpo/${RUN_NAME}_baseline and $CHECKPOINT_DIR/entropy_mcts_grpo/${RUN_NAME}_mcts"
echo "================================"
