#!/bin/bash
# Dream 7B — ablation grid (entropy-tree knobs) with execution-shaped rewards + Apptainer.
# Replaces the legacy MDLM `scripts/run_experiment_2.py` ablation with `dream/scripts/run_dream_comparison.py`.
#
# Phases (toggle with env vars, all default ON):
#   1) initial_eval     — tree forward / eval_step only (no optimizer)
#   2) grpo_lora_baseline — flat trajectory GRPO + LoRA
#   3) baseline_train   — nested grid over max_tree_nodes × branch_width × steps_per_expansion (MCTS / tree GRPO)
#
# Automatic EvalPlus / post-hoc eval: not wired yet (add later).
#
# Submit from repo root (ensure logs/ exists — has logs/.gitkeep):
#   sbatch run_experiment_2_ablation.sh
#
# --- W&B (same pattern as run_dream_comparison.sh / train_sandbox_smoke.sbatch) ---
#   export WANDB_API_KEY=...
#   USE_WANDB=0                # disable logging
#   WANDB_PROJECT                default entropy-tree-grpo-dream
#   WANDB_GROUP                  default ablation_<jobid>
#   WANDB_PREFIXED_KEYS=1        # optional
#
# --- Environment & data ---
#   SANDBOX_SIF         default /scratch/$USER/containers/dream-sandbox.sif
#   DATASET             default dream/data/code_grpo_train.sample.jsonl
#   DATASET_SPLIT       default train
#   MAX_TASKS           default 3 (raise for real AceCode runs)
#   REWARD                default execution_shaped
#   SAVE_CHECKPOINTS=1    optional checkpoint writes
#   CHECKPOINT_DIR
#
# --- Ablation toggles ---
#   RUN_INITIAL_EVAL=0       skip phase 1
#   RUN_FLAT_BASELINE=0      skip phase 2
#   RUN_MCTS_GRID=0          skip phase 3 (entropy-tree MCTS GRPO — the main “serious” tree training)
#
#   TEMPERATURE              default 0.2  (eval / completion sampling)
#   TRAIN_SAMPLING_TEMPERATURE  default 0.6  (training rollouts + tree expansion diversity)
#
# --- Grid (space-separated lists; bash arrays below) ---
#   Override by exporting before sbatch, e.g. export MAX_NODES_LIST="8 16"
#
# --- Hyperparameters ---
#   NUM_EPOCHS, MAX_NEW_TOKENS, TOTAL_DENOISING_STEPS (default 128), NUM_BASELINE_SAMPLES, LEARNING_RATE, REWARD_TIMEOUT,
#   MIN_ADAPT, MAX_ADAPT, BRANCH_THRESHOLD, ENTROPY_WEIGHT_MIN, ENTROPY_WEIGHT_MAX,
#   EVAL_MAX_TREE_NODES (for initial_eval only), EVAL_BRANCH_WIDTH, EVAL_STEPS_PER_EXPANSION
#
#SBATCH --job-name=grpo_ablation_dream
#SBATCH --output=logs/ablation_dream_%j.out
#SBATCH --error=logs/ablation_dream_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

echo "======== Dream ablation (run_dream_comparison.py) ========"
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_home}"
mkdir -p "$HF_HOME"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/scratch/${USER}/containers/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/scratch/${USER}/containers/tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" || true

echo "CONDA_PREFIX=${CONDA_PREFIX:-}"
echo "PYTHON=$(command -v python)"
python -c "import sys; print('executable:', sys.executable); print('version:', sys.version.split()[0])"

echo "================================"
echo "pip install dream/requirements.txt"
echo "================================"
python -m pip install -r dream/requirements.txt -q
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

JOB_ID="${SLURM_JOB_ID:-ablation_local}"
AB_PREFIX="${AB_PREFIX:-ablation_${JOB_ID}}"
WANDB_PROJECT="${WANDB_PROJECT:-entropy-tree-grpo-dream}"
WANDB_GROUP="${WANDB_GROUP:-$AB_PREFIX}"
USE_WANDB="${USE_WANDB:-1}"

if [ "$USE_WANDB" = "1" ]; then
  WANDB_ARGS=(
    --wandb_project "$WANDB_PROJECT"
    --wandb_group "$WANDB_GROUP"
  )
  if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARN: WANDB_API_KEY is not set; export it before sbatch for reliable W&B upload."
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
  exit 1
fi
if ! command -v apptainer &>/dev/null; then
  echo "ERROR: apptainer not on PATH"
  exit 1
fi
echo "Apptainer + image sanity: $(apptainer exec "$SANDBOX_SIF" python3 --version 2>&1)"

DATASET="${DATASET:-dream/data/code_grpo_train.sample.jsonl}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
MAX_TASKS="${MAX_TASKS:-3}"
REWARD="${REWARD:-execution_shaped}"
REWARD_TIMEOUT="${REWARD_TIMEOUT:-2.25}"

# NOTE: Defaults are chosen for stability on A100 80GB.
NUM_EPOCHS="${NUM_EPOCHS:-20}"
# 512 often OOMs in MCTS/tree (baseline_train). Use 256 by default and override only intentionally.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TOTAL_DENOISING_STEPS="${TOTAL_DENOISING_STEPS:-128}"
NUM_BASELINE_SAMPLES="${NUM_BASELINE_SAMPLES:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
MIN_ADAPT="${MIN_ADAPT:-4}"
MAX_ADAPT="${MAX_ADAPT:-36}"
BRANCH_THRESHOLD="${BRANCH_THRESHOLD:-0.65}"
ENTROPY_WEIGHT_MIN="${ENTROPY_WEIGHT_MIN:-0.08}"
ENTROPY_WEIGHT_MAX="${ENTROPY_WEIGHT_MAX:-2.5}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TRAIN_SAMPLING_TEMPERATURE="${TRAIN_SAMPLING_TEMPERATURE:-0.8}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/${USER}/entropy_tree_grpo_dream}"
SAVE_EVERY="${SAVE_EVERY:-0}"
CKPT_ARGS=()
if [ "${SAVE_CHECKPOINTS:-0}" = "1" ]; then
  mkdir -p "$CHECKPOINT_DIR" || exit 1
  CKPT_ARGS=(--save-checkpoints --checkpoint-dir "$CHECKPOINT_DIR" --save-every-steps "$SAVE_EVERY")
fi

# initial_eval uses fixed tree knobs (not the MCTS grid)
EVAL_MAX_TREE_NODES="${EVAL_MAX_TREE_NODES:-16}"
EVAL_BRANCH_WIDTH="${EVAL_BRANCH_WIDTH:-2}"
EVAL_STEPS_PER_EXPANSION="${EVAL_STEPS_PER_EXPANSION:-12}"

# Debug preset: run a single, small MCTS config (tree-only) to validate stability.
# Usage: MCTS_DEBUG=1 sbatch run_experiment_2_ablation.sh
if [ "${MCTS_DEBUG:-0}" = "1" ]; then
  echo "INFO: MCTS_DEBUG=1 enabled (tree-only, single small config)."
  RUN_INITIAL_EVAL=0
  RUN_FLAT_BASELINE=0
  RUN_MCTS_GRID=1
  MAX_TASKS="${MAX_TASKS:-10}"
  NUM_EPOCHS="${NUM_EPOCHS:-2}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
  TOTAL_DENOISING_STEPS="${TOTAL_DENOISING_STEPS:-128}"
  NUM_BASELINE_SAMPLES="${NUM_BASELINE_SAMPLES:-4}"
  MAX_NODES_ARRAY=(8)
  BRANCH_ARRAY=(2)
  STEPS_ARRAY=(8)
fi

# Guardrail: very large token budgets tend to OOM in MCTS (baseline_train).
if [ "${RUN_MCTS_GRID:-1}" = "1" ] && [ "${ALLOW_BIG_TOKENS:-0}" != "1" ]; then
  if [ "$MAX_NEW_TOKENS" -gt 256 ]; then
    echo "WARN: MAX_NEW_TOKENS=$MAX_NEW_TOKENS is high for baseline_train; clamping to 256 (set ALLOW_BIG_TOKENS=1 to override)."
    MAX_NEW_TOKENS=256
  fi
fi

# Grid defaults (same spirit as legacy ablation)
# (Skip when MCTS_DEBUG=1 pre-sets a single config.)
if [ "${MCTS_DEBUG:-0}" != "1" ]; then
  if [ -n "${MAX_NODES_LIST:-}" ]; then
    # shellcheck disable=SC2206
    MAX_NODES_ARRAY=($MAX_NODES_LIST)
  else
    MAX_NODES_ARRAY=(8 16)
  fi
  if [ -n "${BRANCH_LIST:-}" ]; then
    # shellcheck disable=SC2206
    BRANCH_ARRAY=($BRANCH_LIST)
  else
    BRANCH_ARRAY=(2 3)
  fi
  if [ -n "${STEPS_LIST:-}" ]; then
    # shellcheck disable=SC2206
    STEPS_ARRAY=($STEPS_LIST)
  else
    STEPS_ARRAY=(12 24)
  fi
fi

RUN_INITIAL_EVAL="${RUN_INITIAL_EVAL:-1}"
RUN_FLAT_BASELINE="${RUN_FLAT_BASELINE:-1}"
RUN_MCTS_GRID="${RUN_MCTS_GRID:-1}"

echo "================================"
echo "Resolved config"
echo "================================"
echo "AB_PREFIX=$AB_PREFIX  WANDB_GROUP=$WANDB_GROUP  USE_WANDB=$USE_WANDB"
echo "DATASET=$DATASET  split=$DATASET_SPLIT  MAX_TASKS=$MAX_TASKS  REWARD=$REWARD"
echo "NUM_EPOCHS=$NUM_EPOCHS  MAX_NEW_TOKENS=$MAX_NEW_TOKENS  TOTAL_DENOISING_STEPS=$TOTAL_DENOISING_STEPS  LR=$LEARNING_RATE"
echo "Grid: nodes=${MAX_NODES_ARRAY[*]}  branch=${BRANCH_ARRAY[*]}  steps=${STEPS_ARRAY[*]}"
echo "RUN_INITIAL_EVAL=$RUN_INITIAL_EVAL  RUN_FLAT_BASELINE=$RUN_FLAT_BASELINE  RUN_MCTS_GRID=$RUN_MCTS_GRID"
echo "TEMPERATURE=$TEMPERATURE  TRAIN_SAMPLING_TEMPERATURE=$TRAIN_SAMPLING_TEMPERATURE"

# Shared args for every invocation (run_name and phase set per call)
base_cmd() {
  python dream/scripts/run_dream_comparison.py \
    --device cuda \
    "${WANDB_ARGS[@]}" \
    "${WANDB_EXTRA[@]}" \
    --dataset "$DATASET" \
    --dataset-split "$DATASET_SPLIT" \
    --max-tasks "$MAX_TASKS" \
    --reward "$REWARD" \
    --reward-tie-breaker "${REWARD_TIE_BREAKER:-none}" \
    --reward-timeout "$REWARD_TIMEOUT" \
    --execution-backend apptainer \
    --sandbox-image "$SANDBOX_SIF" \
    --lora \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --total_denoising_steps "$TOTAL_DENOISING_STEPS" \
    --min_steps_per_expansion "$MIN_ADAPT" \
    --max_steps_per_expansion "$MAX_ADAPT" \
    --branch_threshold "$BRANCH_THRESHOLD" \
    --learning_rate "$LEARNING_RATE" \
    --entropy-weight-min "$ENTROPY_WEIGHT_MIN" \
    --entropy-weight-max "$ENTROPY_WEIGHT_MAX" \
    --num-baseline-samples "$NUM_BASELINE_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --train-sampling-temperature "$TRAIN_SAMPLING_TEMPERATURE" \
    --trace-every-steps "${TRACE_EVERY_STEPS:-0}" \
    --trace-dir "${TRACE_DIR:-traces}" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    "${CKPT_ARGS[@]}" \
    "$@"
}

if [ "$RUN_INITIAL_EVAL" = "1" ]; then
  echo "================================"
  echo "Phase 1/3: initial_eval (no training) — ${AB_PREFIX}_initial_eval"
  echo "================================"
  base_cmd \
    --phase initial_eval \
    --run_name "${AB_PREFIX}_initial_eval" \
    --num_epochs 1 \
    --max_tree_nodes "$EVAL_MAX_TREE_NODES" \
    --branch_width "$EVAL_BRANCH_WIDTH" \
    --steps_per_expansion "$EVAL_STEPS_PER_EXPANSION"
fi

if [ "$RUN_FLAT_BASELINE" = "1" ]; then
  echo "================================"
  echo "Phase 2/3: grpo_lora_baseline — ${AB_PREFIX}_flat_grpo"
  echo "================================"
  base_cmd \
    --phase grpo_lora_baseline \
    --run_name "${AB_PREFIX}_flat_grpo" \
    --num_epochs "$NUM_EPOCHS" \
    --max_tree_nodes "$EVAL_MAX_TREE_NODES" \
    --branch_width "$EVAL_BRANCH_WIDTH" \
    --steps_per_expansion "$EVAL_STEPS_PER_EXPANSION"
fi

if [ "$RUN_MCTS_GRID" = "1" ]; then
  echo "================================"
  echo "Phase 3/3: baseline_train (MCTS grid)"
  echo "================================"
  for MAX_TREE_NODES in "${MAX_NODES_ARRAY[@]}"; do
    for BRANCH_WIDTH in "${BRANCH_ARRAY[@]}"; do
      for STEPS_PER_EXPANSION in "${STEPS_ARRAY[@]}"; do
        RUN_NAME="${AB_PREFIX}_mcts_n${MAX_TREE_NODES}_b${BRANCH_WIDTH}_s${STEPS_PER_EXPANSION}"
        echo "--------------------------------"
        echo "Run: baseline_train — $RUN_NAME"
        echo "--------------------------------"
        base_cmd \
          --phase baseline_train \
          --run_name "$RUN_NAME" \
          --num_epochs "$NUM_EPOCHS" \
          --max_tree_nodes "$MAX_TREE_NODES" \
          --branch_width "$BRANCH_WIDTH" \
          --steps_per_expansion "$STEPS_PER_EXPANSION"
      done
    done
  done
fi

echo "================================"
echo "Finished: $(date -Is)"
echo "WandB project: $WANDB_PROJECT  group: $WANDB_GROUP  (filter runs by prefix $AB_PREFIX)"
echo "Checkpoints: ${SAVE_CHECKPOINTS:-0} (CHECKPOINT_DIR=$CHECKPOINT_DIR)"
echo "================================"
