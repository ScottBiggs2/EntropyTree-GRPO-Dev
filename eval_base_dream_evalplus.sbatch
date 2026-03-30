#!/bin/bash
# Full EvalPlus runs for **base** Dream-org/Dream-v0-Instruct-7B using **DiffuCoder-aligned**
# prompts (`dream/src/eval_prompts.py` → `eval_humaneval.py` / `eval_mbpp.py`). Use these
# numbers for fair comparison to Apple DiffuCoder paper tables (same template + diffusion
# hyperparams in our drivers); we do **not** run the DiffuCoder checkpoint here.
#
# Submit from repo root:
#   sbatch eval_base_dream_evalplus.sbatch
#   sbatch --export=ALL,MAX_TASKS=8,N_SAMPLES=1 eval_base_dream_evalplus.sbatch   # quick subset
#
# Override paths / phases (export before sbatch):
#   HUMANEVAL_JSONL  default /scratch/$USER/dream_data/humaneval.jsonl
#   MBPP_JSONL       default /scratch/$USER/dream_data/mbpp.jsonl
#   OUT_BASE         default /scratch/$USER/dream_eval/base_$SLURM_JOB_ID
#   RUN_HUMANEVAL=0  skip HumanEval+ generation + scoring
#   RUN_MBPP=0       skip MBPP+ generation + scoring
#   RUN_PASS10=1     also run pass@10 (N_SAMPLES_PASS10, default 10; temp=0.4) after pass@1
#   ADAPTER          if set, passed as --adapter to both eval scripts (trained checkpoint)
#   EVALPLUS_BACKEND e.g. unsafe — see dream/docs/EVAL_PROTOCOL.md §6.4 (HPC often needs this)
#   MAX_NEW_TOKENS  default 512 — diffusion length cap (CLI: --max-new-tokens)
#   STEPS           default 128 — denoising steps (CLI: --steps); align with validate_dream.py; not tied to max_new_tokens
#   MAX_TASKS       default 0 — cap tasks per benchmark (0 = all). Quick test: MAX_TASKS=8 sbatch ...
#   N_SAMPLES       default 1 — samples per task for pass@1 sections (temperature 0.2)
#   N_SAMPLES_PASS10 default 10 — samples per task when RUN_PASS10=1 (temperature 0.4)
#   RUN_EVALPLUS    default 1 if MAX_TASKS=0, else 0. EvalPlus needs every task id in JSONL; partial runs cannot score.
#
# Wall time: full pass@1 is hours on one GPU (164 + 378 tasks × diffusion). Adjust --time.
#
#SBATCH --job-name=dream_eval_base
#SBATCH --output=dream_eval_base_%j.out
#SBATCH --error=dream_eval_base_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128
#SBATCH --time=8:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"

echo "======== dream eval: base model + DiffuCoder-aligned prompts + EvalPlus ========"
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

# Line-buffered Python so Slurm .out updates during long diffusion loops (see [eval-gen] logs).
export PYTHONUNBUFFERED=1

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_home}"
mkdir -p "$HF_HOME"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
if [ -n "${EVALPLUS_BACKEND:-}" ]; then
  export EVALPLUS_BACKEND
  echo "EVALPLUS_BACKEND=$EVALPLUS_BACKEND"
fi

python -m pip install -r dream/requirements.txt -q

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
STEPS="${STEPS:-128}"
MAX_TASKS="${MAX_TASKS:-0}"
N_SAMPLES="${N_SAMPLES:-1}"
N_SAMPLES_PASS10="${N_SAMPLES_PASS10:-10}"
if [ "$MAX_TASKS" = "0" ]; then
  RUN_EVALPLUS="${RUN_EVALPLUS:-1}"
else
  RUN_EVALPLUS="${RUN_EVALPLUS:-0}"
fi
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS  STEPS=$STEPS  MAX_TASKS=$MAX_TASKS  N_SAMPLES=$N_SAMPLES  N_SAMPLES_PASS10=$N_SAMPLES_PASS10  RUN_EVALPLUS=$RUN_EVALPLUS"
if [ "$MAX_TASKS" != "0" ] && [ "$RUN_EVALPLUS" = "1" ]; then
  echo "WARN: RUN_EVALPLUS=1 with MAX_TASKS=$MAX_TASKS — evalplus.evaluate usually fails (subset JSONL)."
fi

MODEL="${MODEL:-Dream-org/Dream-v0-Instruct-7B}"
OUT_BASE="${OUT_BASE:-/scratch/${USER}/dream_eval/base_${SLURM_JOB_ID}}"
mkdir -p "$OUT_BASE"
HUMANEVAL_JSONL="${HUMANEVAL_JSONL:-/scratch/${USER}/dream_data/humaneval.jsonl}"
MBPP_JSONL="${MBPP_JSONL:-/scratch/${USER}/dream_data/mbpp.jsonl}"

ADAPTER_ARGS=()
if [ -n "${ADAPTER:-}" ]; then
  ADAPTER_ARGS=(--adapter "$ADAPTER")
  echo "Using adapter: $ADAPTER"
fi

run_evalplus() {
  local dataset=$1 samples=$2 log=$3
  set +e
  python -m evalplus.evaluate --dataset "$dataset" --samples "$samples" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "ERROR: evalplus.evaluate failed (rc=$rc) dataset=$dataset — check EVALPLUS_BACKEND / logs"
    exit "$rc"
  fi
}

# --- pass@1 (DiffuCoder-style defaults in our drivers) ---
if [ "${RUN_HUMANEVAL:-1}" = "1" ]; then
  echo "================================"
  echo "HumanEval+  pass@1  (n=${N_SAMPLES}, temp=0.2)"
  echo "================================"
  HE_P1="${OUT_BASE}/humaneval_pass1.jsonl"
  python dream/scripts/eval_humaneval.py \
    --model "$MODEL" \
    "${ADAPTER_ARGS[@]}" \
    --dataset "$HUMANEVAL_JSONL" \
    --output "$HE_P1" \
    --max-tasks "$MAX_TASKS" \
    --n-samples "$N_SAMPLES" \
    --temperature 0.2 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --steps "$STEPS"
  if [ "$RUN_EVALPLUS" = "1" ]; then
    run_evalplus humaneval "$HE_P1" "${OUT_BASE}/evalplus_humaneval_pass1.log"
  else
    echo "SKIP EvalPlus (humaneval pass@1): set MAX_TASKS=0 for full JSONL + scoring, or RUN_EVALPLUS=1 to force attempt"
  fi
else
  echo "SKIP RUN_HUMANEVAL=0"
fi

if [ "${RUN_MBPP:-1}" = "1" ]; then
  echo "================================"
  echo "MBPP+  pass@1  (n=${N_SAMPLES}, temp=0.2)"
  echo "================================"
  MB_P1="${OUT_BASE}/mbpp_pass1.jsonl"
  python dream/scripts/eval_mbpp.py \
    --model "$MODEL" \
    "${ADAPTER_ARGS[@]}" \
    --dataset "$MBPP_JSONL" \
    --output "$MB_P1" \
    --max-tasks "$MAX_TASKS" \
    --n-samples "$N_SAMPLES" \
    --temperature 0.2 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --steps "$STEPS"
  if [ "$RUN_EVALPLUS" = "1" ]; then
    run_evalplus mbpp "$MB_P1" "${OUT_BASE}/evalplus_mbpp_pass1.log"
  else
    echo "SKIP EvalPlus (mbpp pass@1): set MAX_TASKS=0 for full JSONL + scoring, or RUN_EVALPLUS=1 to force attempt"
  fi
else
  echo "SKIP RUN_MBPP=0"
fi

# --- pass@10 (optional; long) ---
if [ "${RUN_PASS10:-0}" = "1" ]; then
  echo "================================"
  echo "HumanEval+  pass@10  (n=${N_SAMPLES_PASS10}, temp=0.4)"
  echo "================================"
  HE_P10="${OUT_BASE}/humaneval_pass10.jsonl"
  python dream/scripts/eval_humaneval.py \
    --model "$MODEL" \
    "${ADAPTER_ARGS[@]}" \
    --dataset "$HUMANEVAL_JSONL" \
    --output "$HE_P10" \
    --max-tasks "$MAX_TASKS" \
    --n-samples "$N_SAMPLES_PASS10" \
    --temperature 0.4 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --steps "$STEPS"
  if [ "$RUN_EVALPLUS" = "1" ]; then
    run_evalplus humaneval "$HE_P10" "${OUT_BASE}/evalplus_humaneval_pass10.log"
  else
    echo "SKIP EvalPlus (humaneval pass@10): set MAX_TASKS=0 for full JSONL + scoring, or RUN_EVALPLUS=1 to force attempt"
  fi

  echo "================================"
  echo "MBPP+  pass@10  (n=${N_SAMPLES_PASS10}, temp=0.4)"
  echo "================================"
  MB_P10="${OUT_BASE}/mbpp_pass10.jsonl"
  python dream/scripts/eval_mbpp.py \
    --model "$MODEL" \
    "${ADAPTER_ARGS[@]}" \
    --dataset "$MBPP_JSONL" \
    --output "$MB_P10" \
    --max-tasks "$MAX_TASKS" \
    --n-samples "$N_SAMPLES_PASS10" \
    --temperature 0.4 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --steps "$STEPS"
  if [ "$RUN_EVALPLUS" = "1" ]; then
    run_evalplus mbpp "$MB_P10" "${OUT_BASE}/evalplus_mbpp_pass10.log"
  else
    echo "SKIP EvalPlus (mbpp pass@10): set MAX_TASKS=0 for full JSONL + scoring, or RUN_EVALPLUS=1 to force attempt"
  fi
fi

echo "================================"
echo "Done. Artifacts under: $OUT_BASE"
echo "Finished: $(date -Is)"
echo "================================"
