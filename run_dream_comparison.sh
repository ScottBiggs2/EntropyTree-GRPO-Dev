#!/bin/bash
# Dream 7B: light WandB comparison (initial eval, fixed-step train, adaptive default, adaptive alt HP).
# Same Slurm / conda / pip pattern as run_experiment_2.sh — submit from repo root.
#
# Prereq: directory logs/ must exist so Slurm can open logs/dream_comparison_%j.out
# (repo ships logs/.gitkeep; if you removed it: mkdir -p logs)
#
#   sbatch run_dream_comparison.sh
#
#SBATCH --job-name=dream_cmp
#SBATCH --output=logs/dream_comparison_%j.out
#SBATCH --error=logs/dream_comparison_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Job runs in the directory you submitted from (SLURM_SUBMIT_DIR)
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs checkpoints

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working dir: $(pwd)"

# Conda in batch: source conda.sh so activate works (same as run_experiment_2.sh)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
# Dream stack env (parallel to EntropyTreeGRPO_env in run_experiment_2.sh).
# Prefix-only env: conda activate "/scratch/${USER}/conda_envs/EntropyTreeGRPO_Dream_env"
conda activate EntropyTreeGRPO_Dream_env

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "================================"
echo "Installing dream/requirements.txt"
echo "================================"
pip install -r dream/requirements.txt -q
if [ $? -ne 0 ]; then echo "ERROR: pip install failed"; exit 1; fi

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

JOB_ID="${SLURM_JOB_ID:-local}"
RUN_NAME="dream_cmp_${JOB_ID}"
GROUP="$RUN_NAME"
WANDB_PROJECT="${WANDB_PROJECT:-entropy-tree-grpo-dream}"

# Denser curves in WandB (export NUM_EPOCHS=... before sbatch to override)
NUM_EPOCHS="${NUM_EPOCHS:-24}"
MAX_TREE_NODES="${MAX_TREE_NODES:-16}"
BRANCH_WIDTH="${BRANCH_WIDTH:-3}"
STEPS_PER_EXPANSION="${STEPS_PER_EXPANSION:-12}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MIN_ADAPT="${MIN_ADAPT:-4}"
MAX_ADAPT="${MAX_ADAPT:-36}"
BRANCH_THRESHOLD="${BRANCH_THRESHOLD:-0.65}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
SAVE_EVERY="${SAVE_EVERY:-0}"

COMMON=(
  --device cuda
  --wandb_project "$WANDB_PROJECT"
  --wandb_group "$GROUP"
  --run_name "$RUN_NAME"
  --num_epochs "$NUM_EPOCHS"
  --max_tree_nodes "$MAX_TREE_NODES"
  --branch_width "$BRANCH_WIDTH"
  --steps_per_expansion "$STEPS_PER_EXPANSION"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --min_steps_per_expansion "$MIN_ADAPT"
  --max_steps_per_expansion "$MAX_ADAPT"
  --branch_threshold "$BRANCH_THRESHOLD"
  --learning_rate "$LEARNING_RATE"
  --checkpoint_dir "$(pwd)/checkpoints"
  --save_every_steps "$SAVE_EVERY"
  --lora
)

echo "================================"
echo "1/4 initial_eval (no training; adaptive tree)"
echo "================================"
python dream/scripts/run_dream_comparison.py --phase initial_eval "${COMMON[@]}"

echo "================================"
echo "2/4 baseline_train (fixed-step MCTS-GRPO)"
echo "================================"
python dream/scripts/run_dream_comparison.py --phase baseline_train "${COMMON[@]}"

echo "================================"
echo "3/4 adaptive_default (adaptive stepping, default HP)"
echo "================================"
python dream/scripts/run_dream_comparison.py --phase adaptive_default "${COMMON[@]}"

echo "================================"
echo "4/4 adaptive_alt_hp (adaptive + alpha_entropy=1.0, threshold=0.55)"
echo "================================"
python dream/scripts/run_dream_comparison.py --phase adaptive_alt_hp "${COMMON[@]}"

echo "================================"
echo "Job finished at: $(date)"
echo "WandB group: $GROUP  project: $WANDB_PROJECT"
echo "Checkpoints: $(pwd)/checkpoints/dream_comparison/$RUN_NAME"
echo "================================"
