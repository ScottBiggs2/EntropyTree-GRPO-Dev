#!/bin/bash
# Multi-GPU AceCode tree-GRPO run with final checkpoint + afterok eval jobs.
#
# Submit from repo root:
#   export WANDB_API_KEY=...
#   export ACECODE_JSONL=/scratch/$USER/dream_data/acecode_hard_train.jsonl
#   sbatch train_acecode_mcts_ramp_evalpipe_mgpu.sbatch
#
# Notes:
# - Uses torchrun DDP over prompts/tasks. Prompt shards are padded so each rank
#   performs the same number of optimizer steps (required for DDP correctness).
# - Checkpoint + adapter saving occurs on rank 0 only, so the evalpipe contract
#   remains unchanged.

#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --job-name=pipe_p1_mcts_mgpu
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --output=logs/pipeline_%j_p1_mcts_mgpu.out
#SBATCH --error=logs/pipeline_%j_p1_mcts_mgpu.err

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
# IMPORTANT: make temp/cache rank-safe (torchrun spawns N processes).
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/scratch/${USER}/containers/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/scratch/${USER}/containers/tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" || true

export DREAM_SANDBOX_TMP_ROOT="${DREAM_SANDBOX_TMP_ROOT:-/scratch/${USER}/dream_sandbox_tmp}"
mkdir -p "$DREAM_SANDBOX_TMP_ROOT" || true

# Optional: serialize sandbox runs per node if Apptainer/filesystem gets flaky.
# export DREAM_SANDBOX_MAX_CONCURRENT_PER_NODE=1

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
MAX_TASKS="${MAX_TASKS:-512}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
REWARD="${REWARD:-execution_shaped}"
REWARD_TIMEOUT="${REWARD_TIMEOUT:-2.5}"
REWARD_TIE_BREAKER="${REWARD_TIE_BREAKER:-none}"

# Tree + generation budgets (OOM-sensitive)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
TOTAL_DENOISING_STEPS="${TOTAL_DENOISING_STEPS:-128}"
MAX_TREE_NODES="${MAX_TREE_NODES:-12}"
BRANCH_WIDTH="${BRANCH_WIDTH:-2}"
STEPS_PER_EXPANSION="${STEPS_PER_EXPANSION:-8}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-2048}"

# Training HP
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
ENTROPY_WEIGHT_MIN="${ENTROPY_WEIGHT_MIN:-0.08}"
ENTROPY_WEIGHT_MAX="${ENTROPY_WEIGHT_MAX:-2.5}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TRAIN_SAMPLING_TEMPERATURE="${TRAIN_SAMPLING_TEMPERATURE:-0.9}"
NUM_BASELINE_SAMPLES="${NUM_BASELINE_SAMPLES:-8}"

# Checkpointing (final-only; critical for eval pipeline)
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/${USER}/entropy_tree_grpo_dream}"

# Chained eval (same contract as train_acecode_mcts_ramp_evalpipe.sbatch)
EVAL_OUT_BASE="${EVAL_OUT_BASE:-/scratch/${USER}/dream_eval/${RUN_NAME}}"
EVAL_WALLTIME="${EVAL_WALLTIME:-12:00:00}"
SKIP_AUTO_EVAL="${SKIP_AUTO_EVAL:-0}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
EVAL_STEPS="${EVAL_STEPS:-128}"
EVAL_MODEL="${EVAL_MODEL:-Dream-org/Dream-v0-Instruct-7B}"
RUN_GSM8K="${RUN_GSM8K:-1}"
RUN_BIGCODEBENCH="${RUN_BIGCODEBENCH:-1}"
BCB_SUBSET="${BCB_SUBSET:-hard}"
GSM8K_MAX_TASKS="${GSM8K_MAX_TASKS:-0}"

# Optional traces (off by default at scale)
TRACE_EVERY_STEPS="${TRACE_EVERY_STEPS:-0}"
TRACE_DIR="${TRACE_DIR:-traces}"

OUT_DIR="${CHECKPOINT_DIR}/dream_comparison/${RUN_NAME}"
MANIFEST="${OUT_DIR}/checkpoints_manifest.jsonl"
mkdir -p "$OUT_DIR"
ADAPTER_DIR="${OUT_DIR}/adapter"

echo "======== acecode mcts ramp (mgpu evalpipe) ========"
echo "dataset=$ACECODE_JSONL split=train max_tasks=$MAX_TASKS epochs=$NUM_EPOCHS"
echo "tree: nodes=$MAX_TREE_NODES branch=$BRANCH_WIDTH steps_per_expansion=$STEPS_PER_EXPANSION"
echo "gen: max_new_tokens=$MAX_NEW_TOKENS total_denoising_steps=$TOTAL_DENOISING_STEPS"
echo "prompt cap: max_prompt_tokens=$MAX_PROMPT_TOKENS"
echo "temps: temperature=$TEMPERATURE train_sampling_temperature=$TRAIN_SAMPLING_TEMPERATURE"
echo "reward=$REWARD timeout=$REWARD_TIMEOUT tie_breaker=$REWARD_TIE_BREAKER"
echo "checkpoints: dir=$OUT_DIR final_only=1 manifest=$MANIFEST"
echo "adapter_dir: $ADAPTER_DIR"

echo "{\"run_name\":\"${RUN_NAME}\",\"job_id\":\"${JOB_ID}\",\"event\":\"start\",\"timestamp\":\"$(date -Is)\",\"out_dir\":\"${OUT_DIR}\",\"ddp\":true}" >> "$MANIFEST"

NPROC="${SLURM_GPUS_ON_NODE:-4}"

set +e
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  dream/scripts/run_dream_comparison.py \
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

# --- Auto-schedule eval on success (adapter optional) ---
if [ -n "${SLURM_JOB_ID:-}" ] && [ "$RC" -eq 0 ] && [ "${SKIP_AUTO_EVAL}" != "1" ]; then
  mkdir -p "$EVAL_OUT_BASE" || true
  echo "Scheduling eval job with afterok dependency on ${SLURM_JOB_ID}"
  echo "Eval outputs under: $EVAL_OUT_BASE"

  EVAL_EXPORT="ALL,OUT_BASE=${EVAL_OUT_BASE},RUN_PASS10=${EVAL_RUN_PASS10:-0},MAX_TASKS=0,RUN_EVALPLUS=1"
  EVAL_EXPORT="${EVAL_EXPORT},N_SAMPLES=1,MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS},STEPS=${EVAL_STEPS}"
  EVAL_EXPORT="${EVAL_EXPORT},MODEL=${EVAL_MODEL}"
  EVAL_EXPORT="${EVAL_EXPORT},RUN_GSM8K=${RUN_GSM8K},RUN_BIGCODEBENCH=${RUN_BIGCODEBENCH},BCB_SUBSET=${BCB_SUBSET},GSM8K_MAX_TASKS=${GSM8K_MAX_TASKS}"
  if [ -d "$ADAPTER_DIR" ]; then
    EVAL_EXPORT="${EVAL_EXPORT},ADAPTER=${ADAPTER_DIR}"
  else
    echo "WARN: no LoRA adapter at $ADAPTER_DIR — eval runs base MODEL only (${EVAL_MODEL})"
  fi

  sbatch --dependency="afterok:${SLURM_JOB_ID}" \
    --time="${EVAL_WALLTIME}" \
    --export="${EVAL_EXPORT}" \
    eval_base_dream_evalplus.sbatch
else
  echo "Skip auto-eval scheduling (rc=$RC, SLURM_JOB_ID=${SLURM_JOB_ID:-n/a}, SKIP_AUTO_EVAL=${SKIP_AUTO_EVAL})"
fi

