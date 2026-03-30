#!/bin/bash
# Training validation on Explorer: **Apptainer** + `dream-sandbox.sif` + execution-shaped rewards.
# Same Slurm / conda / pip / **W&B** pattern as `run_dream_comparison.sh` — one phase only.
#
# Submit from repo root:
#   sbatch train_sandbox_smoke.sbatch
#
# Prereq: pull the image once (scratch caches recommended):
#   export APPTAINER_CACHEDIR=/scratch/$USER/containers/cache
#   export APPTAINER_TMPDIR=/scratch/$USER/containers/tmp
#   apptainer pull /scratch/$USER/containers/dream-sandbox.sif docker://scottbiggs2001/dream-sandbox:latest
#
# --- W&B (default: logging ON, like run_dream_comparison.sh) ---
#   export WANDB_API_KEY=...   # required in batch unless you use `wandb login` cache on the node
#   USE_WANDB=0                # set to disable (--no_wandb), e.g. quick CPU-free dry run
#   WANDB_PROJECT              # default entropy-tree-grpo-dream
#   WANDB_GROUP                # default same as RUN_NAME
#   RUN_NAME                   # default train_sandbox_<jobid>
#   WANDB_PREFIXED_KEYS=1      # optional: phase-prefixed metric keys (--wandb_prefixed_keys)
#
# --- Training / data (override before sbatch) ---
#   SANDBOX_SIF         default /scratch/$USER/containers/dream-sandbox.sif
#   DATASET             default dream/data/code_grpo_train.sample.jsonl
#   DATASET_SPLIT       default train
#   MAX_TASKS           default 10  (raise for AceCode; 0 = all tasks in file)
#   NUM_EPOCHS          default 20  (stress: many optimizer steps; was 1 = not a load test)
#   TRAIN_PHASE         default grpo_lora_baseline  (flat GRPO + LoRA). Others: baseline_train, initial_eval, adaptive_default, ...
#   REWARD              default execution_shaped
#   SAVE_CHECKPOINTS=1  write under CHECKPOINT_DIR
#   CHECKPOINT_DIR      default /scratch/$USER/entropy_tree_grpo_dream
#   SAVE_EVERY            periodic step saves (0 = final only)
#
# Stress / step budget (flat GRPO default phase):
#   optimizer_steps ≈ NUM_EPOCHS * min(MAX_TASKS, n_tasks_in_dataset)
#   per step: K=NUM_BASELINE_SAMPLES rollouts × (many no-grad forwards + 1 backward over LoRA)
#   Example defaults: 20 × 10 = 200 global steps on sample JSONL (3 tasks) → min(10,3)=3 → 60 steps.
#   Set MAX_TASKS=0 or use a larger JSONL so MAX_TASKS caps actually apply.
#
# --- Hyperparameters (same names as run_dream_comparison.sh where applicable) ---
#   MAX_TREE_NODES BRANCH_WIDTH STEPS_PER_EXPANSION MAX_NEW_TOKENS
#   MIN_ADAPT MAX_ADAPT BRANCH_THRESHOLD LEARNING_RATE
#   ENTROPY_WEIGHT_MIN ENTROPY_WEIGHT_MAX NUM_BASELINE_SAMPLES REWARD_TIMEOUT
#
# Does **not** run the full multi-phase `run_dream_comparison.sh` — only one `--phase`.
#
#SBATCH --job-name=dream_train_sbox
#SBATCH --output=train_sandbox_smoke_%j.out
#SBATCH --error=train_sandbox_smoke_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=8:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

echo "======== dream training smoke: Apptainer sandbox + execution_shaped ========"
echo "Started: $(date -Is)"
echo "Job ID: ${SLURM_JOB_ID:-n/a}  Host: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-n/a}"
echo "Working dir: $(pwd)"

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

echo "CONDA_PREFIX=${CONDA_PREFIX:-}"
echo "PYTHON=$(command -v python)"
python -c "import sys; print('executable:', sys.executable); print('version:', sys.version.split()[0])"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_home}"
mkdir -p "$HF_HOME"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/scratch/${USER}/containers/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/scratch/${USER}/containers/tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" || true

echo "================================"
echo "pip install dream/requirements.txt"
echo "================================"
python -m pip install -r dream/requirements.txt -q
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

# --- W&B (align with run_dream_comparison.sh) ---
JOB_ID="${SLURM_JOB_ID:-local}"
RUN_NAME="${RUN_NAME:-train_sandbox_${JOB_ID}}"
WANDB_PROJECT="${WANDB_PROJECT:-entropy-tree-grpo-dream}"
WANDB_GROUP="${WANDB_GROUP:-$RUN_NAME}"
USE_WANDB="${USE_WANDB:-1}"

if [ "$USE_WANDB" = "1" ]; then
  WANDB_ARGS=(
    --wandb_project "$WANDB_PROJECT"
    --wandb_group "$WANDB_GROUP"
    --run_name "$RUN_NAME"
  )
  if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARN: WANDB_API_KEY is not set. Batch jobs usually need: export WANDB_API_KEY=... before sbatch"
    echo "      (or rely on ~/.netrc / wandb cache on the compute node)."
  fi
else
  WANDB_ARGS=(--no_wandb)
fi

WANDB_EXTRA=()
if [ "${WANDB_PREFIXED_KEYS:-0}" = "1" ]; then
  WANDB_EXTRA+=(--wandb_prefixed_keys)
fi

SANDBOX_SIF="${SANDBOX_SIF:-/scratch/${USER}/containers/dream-sandbox.sif}"
if [ ! -f "$SANDBOX_SIF" ]; then
  echo "ERROR: Sandbox image not found: $SANDBOX_SIF"
  echo "  Pull it (see header) or set SANDBOX_SIF=/path/to/dream-sandbox.sif"
  exit 1
fi

if ! command -v apptainer &>/dev/null; then
  echo "ERROR: apptainer not on PATH"
  exit 1
fi

# --- Data & phase (same env names as run_dream_comparison.sh where possible) ---
DATASET="${DATASET:-dream/data/code_grpo_train.sample.jsonl}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
MAX_TASKS="${MAX_TASKS:-10}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
TRAIN_PHASE="${TRAIN_PHASE:-grpo_lora_baseline}"
REWARD="${REWARD:-execution_shaped}"
REWARD_TIMEOUT="${REWARD_TIMEOUT:-2.0}"

# --- Hyperparameters (defaults match run_dream_comparison.sh / run_dream_comparison.py) ---
MAX_TREE_NODES="${MAX_TREE_NODES:-8}"
BRANCH_WIDTH="${BRANCH_WIDTH:-2}"
STEPS_PER_EXPANSION="${STEPS_PER_EXPANSION:-12}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MIN_ADAPT="${MIN_ADAPT:-4}"
MAX_ADAPT="${MAX_ADAPT:-36}"
BRANCH_THRESHOLD="${BRANCH_THRESHOLD:-0.65}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
ENTROPY_WEIGHT_MIN="${ENTROPY_WEIGHT_MIN:-0.08}"
ENTROPY_WEIGHT_MAX="${ENTROPY_WEIGHT_MAX:-2.5}"
NUM_BASELINE_SAMPLES="${NUM_BASELINE_SAMPLES:-6}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TRAIN_SAMPLING_TEMPERATURE="${TRAIN_SAMPLING_TEMPERATURE:-0.6}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/${USER}/entropy_tree_grpo_dream}"
SAVE_EVERY="${SAVE_EVERY:-0}"
CKPT_ARGS=()
if [ "${SAVE_CHECKPOINTS:-0}" = "1" ]; then
  mkdir -p "$CHECKPOINT_DIR" || {
    echo "ERROR: cannot create checkpoint dir $CHECKPOINT_DIR"
    exit 1
  }
  CKPT_ARGS=(--save-checkpoints --checkpoint-dir "$CHECKPOINT_DIR" --save-every-steps "$SAVE_EVERY")
fi

echo "================================"
echo "Resolved config (override via env before sbatch)"
echo "================================"
echo "USE_WANDB=$USE_WANDB  WANDB_PROJECT=$WANDB_PROJECT  WANDB_GROUP=$WANDB_GROUP  RUN_NAME=$RUN_NAME"
echo "SANDBOX_SIF=$SANDBOX_SIF"
echo "DATASET=$DATASET  DATASET_SPLIT=$DATASET_SPLIT  TRAIN_PHASE=$TRAIN_PHASE"
echo "MAX_TASKS=$MAX_TASKS  NUM_EPOCHS=$NUM_EPOCHS  REWARD=$REWARD"
echo "MAX_TREE_NODES=$MAX_TREE_NODES  BRANCH_WIDTH=$BRANCH_WIDTH  STEPS_PER_EXPANSION=$STEPS_PER_EXPANSION  MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "NUM_BASELINE_SAMPLES=$NUM_BASELINE_SAMPLES  TEMPERATURE=$TEMPERATURE  TRAIN_SAMPLING_TEMPERATURE=$TRAIN_SAMPLING_TEMPERATURE"
echo "LEARNING_RATE=$LEARNING_RATE  BRANCH_THRESHOLD=$BRANCH_THRESHOLD  ENTROPY_WEIGHT_MIN=$ENTROPY_WEIGHT_MIN ENTROPY_WEIGHT_MAX=$ENTROPY_WEIGHT_MAX"
# Approximate global train steps (epochs × prompts per epoch); actual prompts = min(MAX_TASKS, len(dataset split)).
echo "NOTE: global train steps ≈ NUM_EPOCHS × prompts_per_epoch (see [dream_cmp] line in log)."

echo "================================"
echo "Apptainer: python3 --version inside image"
echo "================================"
apptainer exec "$SANDBOX_SIF" python3 --version

echo "================================"
echo "run_dream_comparison.py --phase $TRAIN_PHASE"
echo "================================"

python dream/scripts/run_dream_comparison.py \
  --phase "$TRAIN_PHASE" \
  --device cuda \
  "${WANDB_ARGS[@]}" \
  "${WANDB_EXTRA[@]}" \
  --dataset "$DATASET" \
  --dataset-split "$DATASET_SPLIT" \
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
  --min_steps_per_expansion "$MIN_ADAPT" \
  --max_steps_per_expansion "$MAX_ADAPT" \
  --branch_threshold "$BRANCH_THRESHOLD" \
  --learning_rate "$LEARNING_RATE" \
  --entropy-weight-min "$ENTROPY_WEIGHT_MIN" \
  --entropy-weight-max "$ENTROPY_WEIGHT_MAX" \
  --num-baseline-samples "$NUM_BASELINE_SAMPLES" \
  --temperature "$TEMPERATURE" \
  --train-sampling-temperature "$TRAIN_SAMPLING_TEMPERATURE" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  "${CKPT_ARGS[@]}"

echo "================================"
echo "Finished OK: $(date -Is)"
echo "WandB project: $WANDB_PROJECT  group: $WANDB_GROUP  run: $RUN_NAME"
if [ "$USE_WANDB" = "1" ]; then
  echo "Open the W&B UI for project '$WANDB_PROJECT' and filter group='$WANDB_GROUP'."
fi
if [ "${SAVE_CHECKPOINTS:-0}" = "1" ]; then
  echo "Checkpoints: $CHECKPOINT_DIR/dream_comparison/$RUN_NAME"
else
  echo "Checkpoints: disabled (SAVE_CHECKPOINTS=1 to write under $CHECKPOINT_DIR)"
fi
echo "Slurm logs: $(pwd)/train_sandbox_smoke_${SLURM_JOB_ID:-local}.out  .err"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  cp -f "train_sandbox_smoke_${SLURM_JOB_ID}.out" "logs/train_sandbox_smoke_${SLURM_JOB_ID}.out" 2>/dev/null || true
  cp -f "train_sandbox_smoke_${SLURM_JOB_ID}.err" "logs/train_sandbox_smoke_${SLURM_JOB_ID}.err" 2>/dev/null || true
fi
echo "================================"
