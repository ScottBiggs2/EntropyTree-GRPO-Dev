#!/bin/bash
# Training validation on Explorer: **Apptainer** + `dream-sandbox.sif` + execution-shaped rewards.
# Proves the container path used during GRPO actually runs on the cluster (not just local Docker).
#
# Submit from repo root:
#   sbatch train_sandbox_smoke.sbatch
#
# Prereq: pull the image once (scratch caches recommended):
#   export APPTAINER_CACHEDIR=/scratch/$USER/containers/cache
#   export APPTAINER_TMPDIR=/scratch/$USER/containers/tmp
#   apptainer pull /scratch/$USER/containers/dream-sandbox.sif docker://scottbiggs2001/dream-sandbox:latest
#
# Override via environment (export before sbatch):
#   SANDBOX_SIF     default /scratch/$USER/containers/dream-sandbox.sif
#   DATASET         default dream/data/code_grpo_train.sample.jsonl (3 tasks; no AceCode download)
#   DATASET_SPLIT   default train
#   MAX_TASKS       default 3 (use 10+ for a heavier smoke)
#   NUM_EPOCHS      default 1
#   TRAIN_PHASE     default grpo_lora_baseline (flat GRPO + LoRA; fast sanity). Alternatives: baseline_train, initial_eval
#   SAVE_CHECKPOINTS=1   write LoRA weights under CHECKPOINT_DIR (see script body)
#   NO_WANDB=0      set to 0 to enable W&B (needs WANDB_API_KEY)
#
# Does **not** run the full multi-phase `run_dream_comparison.sh` — only one `--phase` for a quick signal.
#
#SBATCH --job-name=dream_train_sbox
#SBATCH --output=train_sandbox_smoke_%j.out
#SBATCH --error=train_sandbox_smoke_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"

echo "======== dream training smoke: Apptainer sandbox + execution_shaped ========"
echo "Started: $(date -Is)"
echo "Job ID: ${SLURM_JOB_ID:-n/a}  Host: $(hostname)"

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

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_home}"
mkdir -p "$HF_HOME"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Apptainer: keep layers/tmp off home quota (safe if already set in login shell)
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/scratch/${USER}/containers/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/scratch/${USER}/containers/tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" || true

python -m pip install -r dream/requirements.txt -q

SANDBOX_SIF="${SANDBOX_SIF:-/scratch/${USER}/containers/dream-sandbox.sif}"
if [ ! -f "$SANDBOX_SIF" ]; then
  echo "ERROR: Sandbox image not found: $SANDBOX_SIF"
  echo "  Pull it (see header comments) or set SANDBOX_SIF=/path/to/dream-sandbox.sif"
  exit 1
fi

if ! command -v apptainer &>/dev/null; then
  echo "ERROR: apptainer not on PATH"
  exit 1
fi

DATASET="${DATASET:-dream/data/code_grpo_train.sample.jsonl}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
MAX_TASKS="${MAX_TASKS:-3}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
TRAIN_PHASE="${TRAIN_PHASE:-grpo_lora_baseline}"

# Tree phases need more VRAM / time; flat GRPO is the usual sandbox smoke.
MAX_TREE_NODES="${MAX_TREE_NODES:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
STEPS_PER_EXPANSION="${STEPS_PER_EXPANSION:-12}"

WANDB_ARGS=(--no_wandb)
if [ "${NO_WANDB:-1}" = "0" ]; then
  WANDB_ARGS=(--wandb_project "${WANDB_PROJECT:-entropy-tree-grpo-dream}" --wandb_group "sandbox_smoke_${SLURM_JOB_ID:-local}" --run_name "sandbox_smoke_${SLURM_JOB_ID:-local}")
fi

CKPT_ARGS=()
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/${USER}/entropy_tree_grpo_dream}"
if [ "${SAVE_CHECKPOINTS:-0}" = "1" ]; then
  mkdir -p "$CHECKPOINT_DIR"
  CKPT_ARGS=(--save-checkpoints --checkpoint-dir "$CHECKPOINT_DIR" --save-every-steps "${SAVE_EVERY:-0}")
fi

echo "SANDBOX_SIF=$SANDBOX_SIF"
echo "DATASET=$DATASET  split=$DATASET_SPLIT  phase=$TRAIN_PHASE  max_tasks=$MAX_TASKS  epochs=$NUM_EPOCHS"

echo "================================"
echo "Apptainer smoke (optional): exec python --version inside image"
echo "================================"
apptainer exec "$SANDBOX_SIF" python3 --version

echo "================================"
echo "run_dream_comparison.py --phase $TRAIN_PHASE"
echo "================================"

python dream/scripts/run_dream_comparison.py \
  --phase "$TRAIN_PHASE" \
  --device cuda \
  "${WANDB_ARGS[@]}" \
  --dataset "$DATASET" \
  --dataset-split "$DATASET_SPLIT" \
  --max-tasks "$MAX_TASKS" \
  --num_epochs "$NUM_EPOCHS" \
  --reward execution_shaped \
  --reward-timeout "${REWARD_TIMEOUT:-2.0}" \
  --execution-backend apptainer \
  --sandbox-image "$SANDBOX_SIF" \
  --lora \
  --max_tree_nodes "$MAX_TREE_NODES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --steps_per_expansion "$STEPS_PER_EXPANSION" \
  "${CKPT_ARGS[@]}"

echo "================================"
echo "Finished OK: $(date -Is)"
if [ "${SAVE_CHECKPOINTS:-0}" = "1" ]; then
  echo "Checkpoints (if any): $CHECKPOINT_DIR/dream_comparison/"
fi
echo "================================"
