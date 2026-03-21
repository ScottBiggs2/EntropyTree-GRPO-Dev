#!/bin/bash
# Dream 7B: light WandB comparison (initial eval, fixed-step baseline train, adaptive default, adaptive alt HP).
# Mirrors run_experiment_2.sh style. Uses entropy-MCTS-GRPO + SyntaxReward under dream/.
#
# Submit from repo root:
#   sbatch run_dream_comparison.sh
# Or interactive:
#   bash run_dream_comparison.sh
#
#SBATCH --job-name=dream_cmp
#SBATCH --output=logs/dream_comparison_%j.out
#SBATCH --error=logs/dream_comparison_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs checkpoints

echo "Job started at: $(date)"
echo "Node: $(hostname)  Job ID: ${SLURM_JOB_ID:-local}"
echo "Working dir: $(pwd)"

# Conda (match your Dream env name)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
if [ -f "/scratch/${USER}/conda_envs/EntropyTreeGRPO_Dream_env/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "/scratch/${USER}/conda_envs/EntropyTreeGRPO_Dream_env/bin/activate"
elif conda env list | grep -q EntropyTreeGRPO_Dream_env; then
  conda activate EntropyTreeGRPO_Dream_env
else
  echo "WARN: activate EntropyTreeGRPO_Dream_env (or edit this script)" >&2
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "================================"
echo "Installing dream/requirements.txt"
echo "================================"
pip install -r dream/requirements.txt -q

echo "Python: $(python --version)"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

JOB_ID="${SLURM_JOB_ID:-local}"
RUN_NAME="dream_cmp_${JOB_ID}"
GROUP="$RUN_NAME"
WANDB_PROJECT="${WANDB_PROJECT:-entropy-tree-grpo-dream}"

# Light budget (increase for real experiments)
NUM_EPOCHS="${NUM_EPOCHS:-3}"
MAX_TREE_NODES="${MAX_TREE_NODES:-8}"
BRANCH_WIDTH="${BRANCH_WIDTH:-2}"
STEPS_PER_EXPANSION="${STEPS_PER_EXPANSION:-12}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
MIN_ADAPT="${MIN_ADAPT:-8}"
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
echo "Done at: $(date)"
echo "WandB group: $GROUP  project: $WANDB_PROJECT"
echo "Checkpoints under checkpoints/dream_comparison/$RUN_NAME (if save_every > 0)"
echo "================================"
