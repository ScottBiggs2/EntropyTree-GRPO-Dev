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
# - On training success: one dependent eval job runs **full** HumanEval+ & MBPP+ pass@1 (EvalPlus),
#   optional GSM8K + BigCodeBench-Instruct (hard) — see `SKIP_AUTO_EVAL`, `RUN_GSM8K`, `RUN_BIGCODEBENCH`.
#   Eval does **not** require a LoRA adapter: if `adapter/` is missing, the eval job still runs with
#   base `EVAL_MODEL` only (useful for baseline smoke). If `adapter/` exists, `--adapter` is passed.
#
# Notes:
# - Default training phase is **adaptive_default** (entropy-threshold adaptive stepping).
#   Set TRAIN_PHASE=baseline_train for fixed steps_per_expansion instead.
# - Default tree/gen budgets are **OOM-light** (prompt cap + modest tree). With max_new_tokens=256,
#   total_seq is larger — if OOM, lower MAX_PROMPT_TOKENS or MAX_TREE_NODES via env.

#SBATCH --job-name=dream_acecode_mcts
#SBATCH --output=logs/acecode_mcts_ramp_%j.out
#SBATCH --error=logs/acecode_mcts_ramp_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=8:00:00

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

# Model config
BASE_MODEL="${BASE_MODEL:-Dream-org/Dream-v0-Instruct-7B}"
RESUME_FROM="${RESUME_FROM:-}"

# Scale knobs (override via env before sbatch)
MAX_TASKS="${MAX_TASKS:-1024}"          # hundreds of tasks
NUM_EPOCHS="${NUM_EPOCHS:-1}"          # start with 1 epoch; scale to 2+ once stable
# adaptive_default | adaptive_alt_hp | baseline_train (fixed steps only)
TRAIN_PHASE="${TRAIN_PHASE:-adaptive_default}"
REWARD="${REWARD:-execution_shaped}"
REWARD_TIMEOUT="${REWARD_TIMEOUT:-2.5}"
REWARD_TIE_BREAKER="${REWARD_TIE_BREAKER:-none}"   # none|ast_size|code_len (optional)

# Tree + generation budgets (OOM-sensitive; defaults tuned to avoid 80GB OOM on first step)
# I believe this is parity with Apple DiffuCoder
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TOTAL_DENOISING_STEPS="${TOTAL_DENOISING_STEPS:-256}"
MAX_TREE_NODES="${MAX_TREE_NODES:-8}"
BRANCH_WIDTH="${BRANCH_WIDTH:-2}"
# Fixed-step only (baseline_train): ignored when TRAIN_PHASE is adaptive_*.
STEPS_PER_EXPANSION="${STEPS_PER_EXPANSION:-12}"
# Adaptive stepping: micro-steps per expansion (entropy stop between min and max).
MIN_STEPS_PER_EXPANSION="${MIN_STEPS_PER_EXPANSION:-8}"
MAX_STEPS_PER_EXPANSION="${MAX_STEPS_PER_EXPANSION:-36}"
BRANCH_THRESHOLD="${BRANCH_THRESHOLD:-0.65}"
# Cap prompt length (AceCode can contain very long starter_code stubs).
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-512}"

# Training HP
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
ENTROPY_WEIGHT_MIN="${ENTROPY_WEIGHT_MIN:-0.08}"
ENTROPY_WEIGHT_MAX="${ENTROPY_WEIGHT_MAX:-2.5}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TRAIN_SAMPLING_TEMPERATURE="${TRAIN_SAMPLING_TEMPERATURE:-0.9}"
NUM_BASELINE_SAMPLES="${NUM_BASELINE_SAMPLES:-8}"

# Checkpointing (final-only; critical for eval pipeline)
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/${USER}/entropy_tree_grpo_dream}"

# Auto-eval (afterok on this job): full HumanEval+ & MBPP+ pass@1 + EvalPlus logs under EVAL_OUT_BASE (no pass@10)
EVAL_OUT_BASE="${EVAL_OUT_BASE:-/scratch/${USER}/dream_eval/${RUN_NAME}}"
# Wall time for chained eval (full pass@1 on HE+MBPP is still long; raise if needed)
# Added BigCodeBench and GSM8K as well as 256/512 - NOT the 512/512 parity w/ DiffuCoder (yet)
EVAL_WALLTIME="${EVAL_WALLTIME:-8:00:00}"
SKIP_AUTO_EVAL="${SKIP_AUTO_EVAL:-0}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
EVAL_STEPS="${EVAL_STEPS:-256}"
# Base HF model id for the chained eval job (LoRA stacks on top when ADAPTER is set)
EVAL_MODEL="${EVAL_MODEL:-$BASE_MODEL}"
# Chained eval: optional extra metrics (pass@1); can be heavy — override with RUN_GSM8K=0 etc.
RUN_GSM8K="${RUN_GSM8K:-1}"
RUN_BIGCODEBENCH="${RUN_BIGCODEBENCH:-1}"
BCB_SUBSET="${BCB_SUBSET:-hard}"
GSM8K_MAX_TASKS="${GSM8K_MAX_TASKS:-256}"

# Optional traces (off by default at scale)
TRACE_EVERY_STEPS="${TRACE_EVERY_STEPS:-0}"
TRACE_DIR="${TRACE_DIR:-traces}"

OUT_DIR="${CHECKPOINT_DIR}/dream_comparison/${RUN_NAME}"
MANIFEST="${OUT_DIR}/checkpoints_manifest.jsonl"
mkdir -p "$OUT_DIR"
ADAPTER_DIR="${OUT_DIR}/adapter"

echo "======== acecode mcts ramp (evalpipe) ========"
echo "phase=$TRAIN_PHASE dataset=$ACECODE_JSONL split=train max_tasks=$MAX_TASKS epochs=$NUM_EPOCHS"
echo "tree: nodes=$MAX_TREE_NODES branch=$BRANCH_WIDTH steps_per_expansion=$STEPS_PER_EXPANSION (fixed only)"
echo "adaptive: min_steps=$MIN_STEPS_PER_EXPANSION max_steps=$MAX_STEPS_PER_EXPANSION branch_threshold=$BRANCH_THRESHOLD"
echo "gen: max_new_tokens=$MAX_NEW_TOKENS total_denoising_steps=$TOTAL_DENOISING_STEPS"
echo "prompt cap: max_prompt_tokens=$MAX_PROMPT_TOKENS"
echo "temps: temperature=$TEMPERATURE train_sampling_temperature=$TRAIN_SAMPLING_TEMPERATURE"
echo "reward=$REWARD timeout=$REWARD_TIMEOUT tie_breaker=$REWARD_TIE_BREAKER"
echo "checkpoints: dir=$OUT_DIR final_only=1 manifest=$MANIFEST resume_from=$RESUME_FROM"
echo "adapter_dir: $ADAPTER_DIR"
echo "auto_eval: SKIP_AUTO_EVAL=$SKIP_AUTO_EVAL EVAL_OUT_BASE=$EVAL_OUT_BASE EVAL_WALLTIME=$EVAL_WALLTIME MODEL=$EVAL_MODEL MAX_NEW_TOKENS=$EVAL_MAX_NEW_TOKENS STEPS=$EVAL_STEPS RUN_GSM8K=$RUN_GSM8K RUN_BIGCODEBENCH=$RUN_BIGCODEBENCH BCB_SUBSET=$BCB_SUBSET"

# Create/append manifest header row (JSONL)
echo "{\"run_name\":\"${RUN_NAME}\",\"job_id\":\"${JOB_ID}\",\"event\":\"start\",\"timestamp\":\"$(date -Is)\",\"out_dir\":\"${OUT_DIR}\"}" >> "$MANIFEST"

RESUME_ARG=""
if [ -n "$RESUME_FROM" ]; then
  RESUME_ARG="--resume-from $RESUME_FROM"
fi

set +e
python dream/scripts/run_dream_comparison.py \
  --model "$BASE_MODEL" \
  ${RESUME_ARG} \
  --phase "$TRAIN_PHASE" \
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
  --min_steps_per_expansion "$MIN_STEPS_PER_EXPANSION" \
  --max_steps_per_expansion "$MAX_STEPS_PER_EXPANSION" \
  --branch_threshold "$BRANCH_THRESHOLD" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
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

if [ -f "${OUT_DIR}/${TRAIN_PHASE}_final.pt" ]; then
  echo "{\"run_name\":\"${RUN_NAME}\",\"job_id\":\"${JOB_ID}\",\"event\":\"final\",\"phase\":\"${TRAIN_PHASE}\",\"path\":\"${OUT_DIR}/${TRAIN_PHASE}_final.pt\",\"timestamp\":\"$(date -Is)\"}" >> "$MANIFEST"
fi
if [ -d "$ADAPTER_DIR" ]; then
  echo "{\"run_name\":\"${RUN_NAME}\",\"job_id\":\"${JOB_ID}\",\"event\":\"adapter\",\"phase\":\"${TRAIN_PHASE}\",\"path\":\"${ADAPTER_DIR}\",\"timestamp\":\"$(date -Is)\"}" >> "$MANIFEST"
fi

echo "{\"run_name\":\"${RUN_NAME}\",\"job_id\":\"${JOB_ID}\",\"event\":\"end\",\"rc\":${RC},\"timestamp\":\"$(date -Is)\"}" >> "$MANIFEST"

if [ "$RC" -ne 0 ]; then
  echo "ERROR: training failed (rc=$RC). Manifest still written: $MANIFEST"
  exit "$RC"
fi

echo "Done. Checkpoints: $OUT_DIR"
echo "Manifest: $MANIFEST"

# --- Auto-schedule eval (Parallelized: EvalPlus set & Expanded set) ---
if [ -n "${SLURM_JOB_ID:-}" ] && [ "$RC" -eq 0 ] && [ "${SKIP_AUTO_EVAL}" != "1" ]; then
  mkdir -p "$EVAL_OUT_BASE" || true
  echo "Scheduling parallel eval jobs afterok ${SLURM_JOB_ID}"
  echo "Eval artifacts: $EVAL_OUT_BASE"

  ADAPTER_ARG=""
  if [ -d "$ADAPTER_DIR" ]; then
    ADAPTER_ARG=",ADAPTER=${ADAPTER_DIR}"
  fi

  # 1. HumanEval+ & MBPP+
  EVAL_SET_EXPORT="ALL,OUT_BASE=${EVAL_OUT_BASE}/evalplus,RUN_PASS10=0,MAX_TASKS=0,RUN_EVALPLUS=1"
  EVAL_SET_EXPORT="${EVAL_SET_EXPORT},N_SAMPLES=1,MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS},STEPS=${EVAL_STEPS}"
  EVAL_SET_EXPORT="${EVAL_SET_EXPORT},MODEL=${EVAL_MODEL},RUN_HUMANEVAL=1,RUN_MBPP=1,RUN_GSM8K=0,RUN_BIGCODEBENCH=0${ADAPTER_ARG}"

  SB_EVAL_SET=$(sbatch --dependency="afterok:${SLURM_JOB_ID}" \
    --time="${EVAL_WALLTIME}" \
    --export="${EVAL_SET_EXPORT}" \
    --job-name="eval_plus_${RUN_NAME}" \
    eval_base_dream_evalplus.sbatch)

  # 2. BigCodeBench-Hard & GSM8K
  EXPANDED_SET_EXPORT="ALL,OUT_BASE=${EVAL_OUT_BASE}/expanded,RUN_PASS10=0,MAX_TASKS=0,RUN_EVALPLUS=1"
  EXPANDED_SET_EXPORT="${EXPANDED_SET_EXPORT},N_SAMPLES=1,MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS},STEPS=${EVAL_STEPS}"
  EXPANDED_SET_EXPORT="${EXPANDED_SET_EXPORT},MODEL=${EVAL_MODEL},RUN_HUMANEVAL=0,RUN_MBPP=0"
  EXPANDED_SET_EXPORT="${EXPANDED_SET_EXPORT},RUN_GSM8K=${RUN_GSM8K},RUN_BIGCODEBENCH=${RUN_BIGCODEBENCH},BCB_SUBSET=${BCB_SUBSET},GSM8K_MAX_TASKS=${GSM8K_MAX_TASKS}${ADAPTER_ARG}"

  SB_EXPANDED_SET=$(sbatch --dependency="afterok:${SLURM_JOB_ID}" \
    --time="${EVAL_WALLTIME}" \
    --export="${EXPANDED_SET_EXPORT}" \
    --job-name="eval_expanded_${RUN_NAME}" \
    eval_base_dream_evalplus.sbatch)

  echo "$SB_EVAL_SET"
  echo "$SB_EXPANDED_SET"
  
  EVAL_JID1=$(echo "$SB_EVAL_SET" | awk '{print $NF}')
  EVAL_JID2=$(echo "$SB_EXPANDED_SET" | awk '{print $NF}')
  
  echo "{\"run_name\":\"${RUN_NAME}\",\"job_id\":\"${JOB_ID}\",\"event\":\"eval_scheduled\",\"evalplus_job_id\":\"${EVAL_JID1}\",\"expanded_job_id\":\"${EVAL_JID2}\",\"eval_out_base\":\"${EVAL_OUT_BASE}\",\"timestamp\":\"$(date -Is)\"}" >> "$MANIFEST"
else
  echo "Skip auto-eval (rc=$RC, SLURM_JOB_ID=${SLURM_JOB_ID:-n/a}, SKIP_AUTO_EVAL=${SKIP_AUTO_EVAL})"
fi

