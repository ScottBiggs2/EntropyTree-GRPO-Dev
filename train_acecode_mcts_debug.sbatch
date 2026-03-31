#!/bin/bash
# Debug-focused **tree GRPO** run (baseline_train) on Explorer:
# - Apptainer execution
# - AceCode JSONL dataset
# - Single small MCTS config (no grid)
#
# Submit from repo root:
#   export WANDB_API_KEY=...
#   export ACECODE_JSONL=/scratch/$USER/dream_data/acecode_hard_train.jsonl
#   sbatch train_acecode_mcts_debug.sbatch
#
# Optional overrides:
#   MAX_TASKS=10 NUM_EPOCHS=1 MAX_NEW_TOKENS=256 TOTAL_DENOISING_STEPS=128
#   MAX_TREE_NODES=8 BRANCH_WIDTH=2 STEPS_PER_EXPANSION=8
#   TEMPERATURE=0.2 TRAIN_SAMPLING_TEMPERATURE=0.9
#
# If this OOMs, reduce MAX_NEW_TOKENS first, then MAX_TREE_NODES.
#
#SBATCH --job-name=dream_mcts_dbg
#SBATCH --output=logs/acecode_mcts_dbg_%j.out
#SBATCH --error=logs/acecode_mcts_dbg_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

if [ -z "${ACECODE_JSONL:-}" ]; then
  echo "ERROR: set ACECODE_JSONL to your AceCode train JSONL path."
  echo "Example: export ACECODE_JSONL=/scratch/$USER/dream_data/acecode_hard_train.jsonl"
  exit 1
fi

# Reuse the same conda + pip pattern as other jobs.
if [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate EntropyTreeGRPO_Dream_env

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_home}"
mkdir -p "$HF_HOME"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/scratch/${USER}/containers/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/scratch/${USER}/containers/tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" || true

python -m pip install -r dream/requirements.txt -q

SANDBOX_SIF="${SANDBOX_SIF:-/scratch/${USER}/containers/dream-sandbox.sif}"
if [ ! -f "$SANDBOX_SIF" ]; then
  echo "ERROR: Sandbox image not found: $SANDBOX_SIF"
  exit 1
fi
if ! command -v apptainer &>/dev/null; then
  echo "ERROR: apptainer not on PATH"
  exit 1
fi

RUN_NAME="${RUN_NAME:-acecode_mcts_dbg_${SLURM_JOB_ID:-local}}"
WANDB_PROJECT="${WANDB_PROJECT:-entropy-tree-grpo-dream}"
WANDB_GROUP="${WANDB_GROUP:-$RUN_NAME}"

MAX_TASKS="${MAX_TASKS:-10}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
REWARD="${REWARD:-execution_shaped}"
REWARD_TIMEOUT="${REWARD_TIMEOUT:-2.25}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TOTAL_DENOISING_STEPS="${TOTAL_DENOISING_STEPS:-128}"
MAX_TREE_NODES="${MAX_TREE_NODES:-8}"
BRANCH_WIDTH="${BRANCH_WIDTH:-2}"
STEPS_PER_EXPANSION="${STEPS_PER_EXPANSION:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
ENTROPY_WEIGHT_MIN="${ENTROPY_WEIGHT_MIN:-0.08}"
ENTROPY_WEIGHT_MAX="${ENTROPY_WEIGHT_MAX:-2.5}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TRAIN_SAMPLING_TEMPERATURE="${TRAIN_SAMPLING_TEMPERATURE:-0.9}"
NUM_BASELINE_SAMPLES="${NUM_BASELINE_SAMPLES:-8}"

echo "Resolved:"
echo "  dataset=$ACECODE_JSONL max_tasks=$MAX_TASKS epochs=$NUM_EPOCHS reward=$REWARD"
echo "  tree: nodes=$MAX_TREE_NODES branch=$BRANCH_WIDTH steps_per_expansion=$STEPS_PER_EXPANSION"
echo "  max_new_tokens=$MAX_NEW_TOKENS total_denoising_steps=$TOTAL_DENOISING_STEPS"
echo "  temps: temperature=$TEMPERATURE train_sampling_temperature=$TRAIN_SAMPLING_TEMPERATURE"

python dream/scripts/run_dream_comparison.py \
  --phase baseline_train \
  --device cuda \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_group "$WANDB_GROUP" \
  --run_name "$RUN_NAME" \
  --dataset "$ACECODE_JSONL" \
  --dataset-split train \
  --max-tasks "$MAX_TASKS" \
  --num_epochs "$NUM_EPOCHS" \
  --reward "$REWARD" \
  --reward-timeout "$REWARD_TIMEOUT" \
  --execution-backend apptainer \
  --sandbox-image "$SANDBOX_SIF" \
  --lora \
  --max_tree_nodes "$MAX_TREE_NODES" \
  --branch_width "$BRANCH_WIDTH" \
  --steps_per_expansion "$STEPS_PER_EXPANSION" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --total_denoising_steps "$TOTAL_DENOISING_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --entropy-weight-min "$ENTROPY_WEIGHT_MIN" \
  --entropy-weight-max "$ENTROPY_WEIGHT_MAX" \
  --num-baseline-samples "$NUM_BASELINE_SAMPLES" \
  --temperature "$TEMPERATURE" \
  --train-sampling-temperature "$TRAIN_SAMPLING_TEMPERATURE" \
  --trace-every-steps "${TRACE_EVERY_STEPS:-100}" \
  --trace-dir "${TRACE_DIR:-traces}" \
  --trace-max-nodes "${TRACE_MAX_NODES:-200}" \
  --trace-max-leaves "${TRACE_MAX_LEAVES:-16}" \
  --trace-decode-chars "${TRACE_DECODE_CHARS:-240}"

