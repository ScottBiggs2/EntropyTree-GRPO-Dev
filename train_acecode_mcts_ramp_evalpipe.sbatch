#!/bin/bash
# Serious AceCode tree-GRPO run (hundreds+ tasks) with a **final-only** checkpoint
# and an evaluation-friendly manifest file of checkpoint paths.
#
# Submit from repo root:
#   export WANDB_API_KEY=...
#   export ACECODE_JSONL=/scratch/$USER/dream_data/acecode_hard_train.jsonl
#   sbatch train_acecode_mcts_ramp_evalpipe.sbatch
#
# What you get:
# - Final checkpoint under:     $CHECKPOINT_DIR/dream_comparison/$RUN_NAME/
# - A manifest file:            $CHECKPOINT_DIR/dream_comparison/$RUN_NAME/checkpoints_manifest.jsonl
#   (one JSON line per checkpoint, easy to tail/ingest)
#
# Notes:
# - This runs the tree phase only (baseline_train).
# - Defaults are conservative for A100 80GB; adjust MAX_NEW_TOKENS / MAX_TREE_NODES upward cautiously.

#SBATCH --job-name=dream_acecode_mcts
#SBATCH --output=logs/acecode_mcts_ramp_%j.out
#SBATCH --error=logs/acecode_mcts_ramp_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

if [ -z "${ACECODE_JSONL:-}" ]; then
  echo "ERROR: set ACECODE_JSONL to your AceCode train JSONL path."
  echo "Example: export ACECODE_JSONL=/scratch/$USER/dream_data/acecode_hard_train.jsonl"
  exit 1
fi

# Conda activation (same pattern as other jobs)
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

# Apptainer sandbox (recommended for untrusted code execution on HPC)
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

JOB_ID="${SLURM_JOB_ID:-local}"
RUN_NAME="${RUN_NAME:-acecode_mcts_ramp_${JOB_ID}}"
WANDB_PROJECT="${WANDB_PROJECT:-entropy-tree-grpo-dream}"
WANDB_GROUP="${WANDB_GROUP:-acecode_ramp_${JOB_ID}}"

# Scale knobs (override via env before sbatch)
MAX_TASKS="${MAX_TASKS:-512}"          # hundreds of tasks
NUM_EPOCHS="${NUM_EPOCHS:-1}"          # start with 1 epoch; scale to 2+ once stable
REWARD="${REWARD:-execution_shaped}"
REWARD_TIMEOUT="${REWARD_TIMEOUT:-2.5}"
REWARD_TIE_BREAKER="${REWARD_TIE_BREAKER:-none}"   # none|ast_size|code_len (optional)

# Tree + generation budgets (OOM-sensitive)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
TOTAL_DENOISING_STEPS="${TOTAL_DENOISING_STEPS:-128}"
MAX_TREE_NODES="${MAX_TREE_NODES:-12}"
BRANCH_WIDTH="${BRANCH_WIDTH:-2}"
STEPS_PER_EXPANSION="${STEPS_PER_EXPANSION:-8}"

# Training HP
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
ENTROPY_WEIGHT_MIN="${ENTROPY_WEIGHT_MIN:-0.08}"
ENTROPY_WEIGHT_MAX="${ENTROPY_WEIGHT_MAX:-2.5}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TRAIN_SAMPLING_TEMPERATURE="${TRAIN_SAMPLING_TEMPERATURE:-0.9}"
NUM_BASELINE_SAMPLES="${NUM_BASELINE_SAMPLES:-8}"

# Checkpointing (final-only; critical for eval pipeline)
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/${USER}/entropy_tree_grpo_dream}"

# Optional traces (off by default at scale)
TRACE_EVERY_STEPS="${TRACE_EVERY_STEPS:-0}"
TRACE_DIR="${TRACE_DIR:-traces}"

OUT_DIR="${CHECKPOINT_DIR}/dream_comparison/${RUN_NAME}"
MANIFEST="${OUT_DIR}/checkpoints_manifest.jsonl"
mkdir -p "$OUT_DIR"
ADAPTER_DIR="${OUT_DIR}/adapter"

echo "======== acecode mcts ramp (evalpipe) ========"
echo "dataset=$ACECODE_JSONL split=train max_tasks=$MAX_TASKS epochs=$NUM_EPOCHS"
echo "tree: nodes=$MAX_TREE_NODES branch=$BRANCH_WIDTH steps_per_expansion=$STEPS_PER_EXPANSION"
echo "gen: max_new_tokens=$MAX_NEW_TOKENS total_denoising_steps=$TOTAL_DENOISING_STEPS"
echo "temps: temperature=$TEMPERATURE train_sampling_temperature=$TRAIN_SAMPLING_TEMPERATURE"
echo "reward=$REWARD timeout=$REWARD_TIMEOUT tie_breaker=$REWARD_TIE_BREAKER"
echo "checkpoints: dir=$OUT_DIR final_only=1 manifest=$MANIFEST"
echo "adapter_dir: $ADAPTER_DIR"

# Create/append manifest header row (JSONL)
echo "{\"run_name\":\"${RUN_NAME}\",\"job_id\":\"${JOB_ID}\",\"event\":\"start\",\"timestamp\":\"$(date -Is)\",\"out_dir\":\"${OUT_DIR}\"}" >> "$MANIFEST"

set +e
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
  --reward-tie-breaker "$REWARD_TIE_BREAKER" \
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
  --save-checkpoints \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --trace-every-steps "$TRACE_EVERY_STEPS" \
  --trace-dir "$TRACE_DIR" \
  --trace-max-nodes "${TRACE_MAX_NODES:-200}" \
  --trace-max-leaves "${TRACE_MAX_LEAVES:-16}" \
  --trace-decode-chars "${TRACE_DECODE_CHARS:-240}"
RC=$?
set -e

if [ -f "${OUT_DIR}/baseline_train_final.pt" ]; then
  echo "{\"run_name\":\"${RUN_NAME}\",\"job_id\":\"${JOB_ID}\",\"event\":\"final\",\"phase\":\"baseline_train\",\"path\":\"${OUT_DIR}/baseline_train_final.pt\",\"timestamp\":\"$(date -Is)\"}" >> "$MANIFEST"
fi
if [ -d "$ADAPTER_DIR" ]; then
  echo "{\"run_name\":\"${RUN_NAME}\",\"job_id\":\"${JOB_ID}\",\"event\":\"adapter\",\"phase\":\"baseline_train\",\"path\":\"${ADAPTER_DIR}\",\"timestamp\":\"$(date -Is)\"}" >> "$MANIFEST"
fi

echo "{\"run_name\":\"${RUN_NAME}\",\"job_id\":\"${JOB_ID}\",\"event\":\"end\",\"rc\":${RC},\"timestamp\":\"$(date -Is)\"}" >> "$MANIFEST"

if [ "$RC" -ne 0 ]; then
  echo "ERROR: training failed (rc=$RC). Manifest still written: $MANIFEST"
  exit "$RC"
fi

echo "Done. Checkpoints: $OUT_DIR"
echo "Manifest: $MANIFEST"

# --- Auto-schedule EvalPlus jobs on success ---
# These eval scripts expect a PEFT adapter directory (saved by run_dream_comparison.py).
# We schedule with dependency so they start after this job is OK.
if [ -n "${SLURM_JOB_ID:-}" ] && [ "$RC" -eq 0 ] && [ -d "$ADAPTER_DIR" ]; then
  EVAL_OUT_BASE="${EVAL_OUT_BASE:-/scratch/${USER}/dream_eval/${RUN_NAME}}"
  mkdir -p "$EVAL_OUT_BASE" || true
  echo "Scheduling eval jobs with afterok dependency on ${SLURM_JOB_ID}"
  echo "Eval outputs under: $EVAL_OUT_BASE"

  # pass@1 (fast)
  sbatch --dependency="afterok:${SLURM_JOB_ID}" \
    --export=ALL,ADAPTER="$ADAPTER_DIR",OUT_BASE="${EVAL_OUT_BASE}/pass1",RUN_PASS10=0 \
    eval_base_dream_evalplus.sbatch

  # optional pass@10 (set RUN_PASS10=1 to enable in eval script)
  if [ "${EVAL_RUN_PASS10:-0}" = "1" ]; then
    sbatch --dependency="afterok:${SLURM_JOB_ID}" \
      --export=ALL,ADAPTER="$ADAPTER_DIR",OUT_BASE="${EVAL_OUT_BASE}/pass10",RUN_PASS10=1 \
      eval_base_dream_evalplus.sbatch
  fi
else
  echo "Skip auto-eval scheduling (rc=$RC, SLURM_JOB_ID=${SLURM_JOB_ID:-n/a}, adapter_dir_exists=$( [ -d "$ADAPTER_DIR" ] && echo 1 || echo 0 ))"
fi

