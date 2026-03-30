#!/bin/bash
# One-shot HPC verification for the Dream stack (Explorer-oriented).
# Submit from the repository root (same as run_dream_comparison.sh):
#
#   cd /path/to/EntropyTree-GRPO
#   sbatch verify_hpc_gpu.sbatch
#   sbatch --export=ALL,VERIFY_MAX_TASKS=4,VERIFY_N_SAMPLES=2 verify_hpc_gpu.sbatch
#   tail -f verify_hpc_gpu_<jobid>.out
#
# Optional environment overrides (export before sbatch):
#   HUMANEVAL_JSONL   HumanEval+ JSONL from convert_humaneval (default: /scratch/$USER/dream_data/humaneval.jsonl)
#   MBPP_JSONL        MBPP+ JSONL (default: /scratch/$USER/dream_data/mbpp.jsonl)
#   VERIFY_MAX_TASKS  default 2 — --max-tasks for step 5 eval smokes
#   VERIFY_N_SAMPLES  default 1 — --n-samples for step 5 eval smokes
#   VERIFY_DENOISING_STEPS  default 128 — --steps for eval_humaneval/eval_mbpp (align with validate_dream.py)
#   SKIP_EVALPLUS=1   Default; smoke JSONL cannot be scored (see below)
#   VERIFY_STRICT=1   Exit non-zero if eval dataset files or evalplus are missing/fail
#
# Uses one GPU sequentially (validate_dream + eval smokes). Do not run another
# Dream 7B job on the same GPU at the same time.
#
#SBATCH --job-name=dream_verify
#SBATCH --output=verify_hpc_gpu_%j.out
#SBATCH --error=verify_hpc_gpu_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"

echo "======== dream HPC verify ========"
echo "Started: $(date -Is)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-n/a}"
echo "Submit dir: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

# --- Conda (same pattern as run_dream_comparison.sh) ---
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
# Always activate the Dream env. Slurm may inherit `base` from the login shell;
# skipping activate used the wrong Python (e.g. 3.13) and broke torch/torchvision pins.
conda activate EntropyTreeGRPO_Dream_env

export PYTHONUNBUFFERED=1

echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-} CONDA_PREFIX=${CONDA_PREFIX:-}"
echo "PYTHON=$(command -v python)"
python -c "import sys; print(sys.executable, sys.version.split()[0])"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# HuggingFace cache on scratch (avoid home quota)
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_home}"
mkdir -p "$HF_HOME"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
echo "HF_HOME=$HF_HOME"

echo "================================"
echo "pip install dream/requirements.txt"
echo "================================"
python -m pip install -r dream/requirements.txt -q

echo "PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

FAILURES=0
on_fail() { echo "VERIFY FAIL: $*"; FAILURES=$((FAILURES + 1)); }

echo "================================"
echo "1) Stale-clone sanity (HPC_SYNC.md)"
echo "================================"
if ! python dream/scripts/single_step_dream.py -h 2>&1 | grep -q dataset; then
  on_fail "single_step_dream.py missing --dataset (pull latest dream/scripts)"
fi
if ! python dream/scripts/run_dream_comparison.py -h 2>&1 | grep -q max-tasks; then
  on_fail "run_dream_comparison.py missing --max-tasks (pull latest dream/scripts)"
fi

echo "================================"
echo "2) pytest dream/tests"
echo "================================"
python -m pytest dream/tests -q --tb=short || on_fail "pytest"

echo "================================"
echo "3) check_reward_pipeline (CPU, sample JSONL)"
echo "================================"
python dream/scripts/check_reward_pipeline.py \
  --dataset dream/data/code_grpo_train.sample.jsonl \
  --dataset-split train \
  --reward execution_shaped \
  --max-tasks 3 || on_fail "check_reward_pipeline"

echo "================================"
echo "4) validate_dream.py (Dream 7B + diffusion_generate)"
echo "================================"
python dream/scripts/validate_dream.py || on_fail "validate_dream"

VERIFY_ROOT="${VERIFY_ROOT:-/scratch/${USER}/dream_verify_${SLURM_JOB_ID:-local}}"
mkdir -p "$VERIFY_ROOT"
HUMANEVAL_JSONL="${HUMANEVAL_JSONL:-/scratch/${USER}/dream_data/humaneval.jsonl}"
MBPP_JSONL="${MBPP_JSONL:-/scratch/${USER}/dream_data/mbpp.jsonl}"
VERIFY_MAX_TASKS="${VERIFY_MAX_TASKS:-2}"
VERIFY_N_SAMPLES="${VERIFY_N_SAMPLES:-1}"
VERIFY_DENOISING_STEPS="${VERIFY_DENOISING_STEPS:-128}"

echo "================================"
echo "5) DiffuCoder-aligned eval smoke (max_tasks=${VERIFY_MAX_TASKS}, n_samples=${VERIFY_N_SAMPLES}, steps=${VERIFY_DENOISING_STEPS})"
echo "================================"
HE_OUT="${VERIFY_ROOT}/humaneval_smoke.jsonl"
MB_OUT="${VERIFY_ROOT}/mbpp_smoke.jsonl"

if [ -f "$HUMANEVAL_JSONL" ]; then
  python dream/scripts/eval_humaneval.py \
    --model Dream-org/Dream-v0-Instruct-7B \
    --dataset "$HUMANEVAL_JSONL" \
    --output "$HE_OUT" \
    --max-tasks "$VERIFY_MAX_TASKS" \
    --n-samples "$VERIFY_N_SAMPLES" \
    --temperature 0.2 \
    --max-new-tokens 512 \
    --steps "$VERIFY_DENOISING_STEPS" \
    || on_fail "eval_humaneval"
  echo "Wrote $HE_OUT ($(wc -l < "$HE_OUT") lines)"
else
  echo "SKIP: HUMANEVAL_JSONL not found: $HUMANEVAL_JSONL"
  if [ "${VERIFY_STRICT:-0}" = "1" ]; then on_fail "missing HUMANEVAL_JSONL"; fi
fi

if [ -f "$MBPP_JSONL" ]; then
  python dream/scripts/eval_mbpp.py \
    --model Dream-org/Dream-v0-Instruct-7B \
    --dataset "$MBPP_JSONL" \
    --output "$MB_OUT" \
    --max-tasks "$VERIFY_MAX_TASKS" \
    --n-samples "$VERIFY_N_SAMPLES" \
    --temperature 0.2 \
    --max-new-tokens 512 \
    --steps "$VERIFY_DENOISING_STEPS" \
    || on_fail "eval_mbpp"
  echo "Wrote $MB_OUT ($(wc -l < "$MB_OUT") lines)"
else
  echo "SKIP: MBPP_JSONL not found: $MBPP_JSONL"
  if [ "${VERIFY_STRICT:-0}" = "1" ]; then on_fail "missing MBPP_JSONL"; fi
fi

echo "================================"
echo "6) evalplus.evaluate (skipped by default for smoke)"
echo "================================"
# EvalPlus requires completions for *every* benchmark task (e.g. 164 HumanEval+ lines for
# pass@1). A 2-line smoke file always fails with: Missing problems in samples.
# Full scoring: sbatch eval_base_dream_evalplus.sbatch
if [ "${VERIFY_RUN_EVALPLUS:-0}" = "1" ] && [ "${SKIP_EVALPLUS:-0}" != "1" ] && [ -f "$HE_OUT" ] && [ -s "$HE_OUT" ]; then
  set +e
  python -m evalplus.evaluate --dataset humaneval --samples "$HE_OUT"
  ep_rc=$?
  set -e
  if [ "$ep_rc" -ne 0 ]; then
    echo "WARN: evalplus exited ${ep_rc}"
    if [ "${VERIFY_STRICT:-0}" = "1" ]; then on_fail "evalplus humaneval"; fi
  fi
else
  echo "SKIP: use eval_base_dream_evalplus.sbatch for EvalPlus (VERIFY_RUN_EVALPLUS=1 forces this step — needs full JSONL)"
fi

echo "================================"
echo "Summary"
echo "================================"
echo "Failures recorded: $FAILURES"
echo "Artifacts: $VERIFY_ROOT"
echo "Finished: $(date -Is)"

if [ "$FAILURES" -gt 0 ]; then
  exit 1
fi
exit 0
