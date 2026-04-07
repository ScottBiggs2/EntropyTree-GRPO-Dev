#!/bin/bash
# EvalPlus (+ optional GSM8K / BigCodeBench) for Dream / DiffuCoder **DiffuCoder-aligned**
# prompts (`dream/src/eval_prompts.py` → `eval_humaneval.py` / `eval_mbpp.py`).
#
# **Standalone baseline (no training job):** set `MODEL` (and optionally `ADAPTER` for LoRA).
#   sbatch --export=ALL,MODEL=apple/DiffuCoder-7B-Instruct,OUT_BASE=/scratch/$USER/dream_eval/my_run eval_base_dream_evalplus.sbatch
#
# Submit from repo root:
#   sbatch eval_base_dream_evalplus.sbatch
#   sbatch --export=ALL,MAX_TASKS=8,N_SAMPLES=1 eval_base_dream_evalplus.sbatch   # quick subset
#
# train_acecode_mcts_ramp_evalpipe.sbatch (on training success) chains this job with:
#   RUN_PASS10=0 MAX_TASKS=0 RUN_EVALPLUS=1 N_SAMPLES=1, optional RUN_GSM8K=1 RUN_BIGCODEBENCH=1
#   (adapter passed only if `adapter/` exists; otherwise base `EVAL_MODEL` / `MODEL` only).
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
#   TEMP_PASS1       default 0.2 — passed to eval_humaneval / eval_mbpp (pass@1 sections)
#   TEMP_PASS10      default 0.4 — pass@10 sections
#   RUN_GSM8K        default 0 — set 1 to run dream/scripts/eval_gsm8k.py (slow on full test set)
#   GSM8K_MAX_TASKS  default 0 — 0 = all GSM8K test rows; use e.g. 64 for smoke
#   RUN_BIGCODEBENCH default 0 — set 1 to generate + score BigCodeBench-Instruct
#   BCB_SUBSET       default hard — full | hard
#   BCB_EXECUTION    default local — bigcodebench.evaluate execution backend (local|gradio|e2b)
#   RUN_BIGCODEBENCH_SCORE default 1 — run official scorer after generation
#
# Wall time: full pass@1 is hours on one GPU (164 + 378 tasks × diffusion). GSM8K / BigCodeBench add more.
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

TEMP_PASS1="${TEMP_PASS1:-0.2}"
TEMP_PASS10="${TEMP_PASS10:-0.4}"
RUN_GSM8K="${RUN_GSM8K:-0}"
RUN_BIGCODEBENCH="${RUN_BIGCODEBENCH:-0}"
BCB_SUBSET="${BCB_SUBSET:-hard}"
BCB_EXECUTION="${BCB_EXECUTION:-local}"
RUN_BIGCODEBENCH_SCORE="${RUN_BIGCODEBENCH_SCORE:-1}"
GSM8K_MAX_TASKS="${GSM8K_MAX_TASKS:-0}"

export RUN_HUMANEVAL="${RUN_HUMANEVAL:-1}"
export RUN_MBPP="${RUN_MBPP:-1}"
export RUN_PASS10="${RUN_PASS10:-0}"
export RUN_EVALPLUS
export RUN_GSM8K RUN_BIGCODEBENCH BCB_SUBSET BCB_EXECUTION GSM8K_MAX_TASKS RUN_BIGCODEBENCH_SCORE
export HOSTNAME="${HOSTNAME:-$(hostname)}"
export TEMP_PASS1 TEMP_PASS10

# Reproducibility manifest (see dream/docs/EVAL_PROTOCOL.md §6.5)
export MODEL MAX_NEW_TOKENS STEPS MAX_TASKS OUT_BASE
export ADAPTER="${ADAPTER:-}"
export PYTHONPATH="${SLURM_SUBMIT_DIR:-.}${PYTHONPATH:+:$PYTHONPATH}"
python -m dream.src.eval_manifest || true

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
  echo "HumanEval+  pass@1  (n=${N_SAMPLES}, temp=${TEMP_PASS1})"
  echo "================================"
  HE_P1="${OUT_BASE}/humaneval_pass1.jsonl"
  python dream/scripts/eval_humaneval.py \
    --model "$MODEL" \
    "${ADAPTER_ARGS[@]}" \
    --dataset "$HUMANEVAL_JSONL" \
    --output "$HE_P1" \
    --max-tasks "$MAX_TASKS" \
    --n-samples "$N_SAMPLES" \
    --temperature "$TEMP_PASS1" \
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
  echo "MBPP+  pass@1  (n=${N_SAMPLES}, temp=${TEMP_PASS1})"
  echo "================================"
  MB_P1="${OUT_BASE}/mbpp_pass1.jsonl"
  python dream/scripts/eval_mbpp.py \
    --model "$MODEL" \
    "${ADAPTER_ARGS[@]}" \
    --dataset "$MBPP_JSONL" \
    --output "$MB_P1" \
    --max-tasks "$MAX_TASKS" \
    --n-samples "$N_SAMPLES" \
    --temperature "$TEMP_PASS1" \
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
  echo "HumanEval+  pass@10  (n=${N_SAMPLES_PASS10}, temp=${TEMP_PASS10})"
  echo "================================"
  HE_P10="${OUT_BASE}/humaneval_pass10.jsonl"
  python dream/scripts/eval_humaneval.py \
    --model "$MODEL" \
    "${ADAPTER_ARGS[@]}" \
    --dataset "$HUMANEVAL_JSONL" \
    --output "$HE_P10" \
    --max-tasks "$MAX_TASKS" \
    --n-samples "$N_SAMPLES_PASS10" \
    --temperature "$TEMP_PASS10" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --steps "$STEPS"
  if [ "$RUN_EVALPLUS" = "1" ]; then
    run_evalplus humaneval "$HE_P10" "${OUT_BASE}/evalplus_humaneval_pass10.log"
  else
    echo "SKIP EvalPlus (humaneval pass@10): set MAX_TASKS=0 for full JSONL + scoring, or RUN_EVALPLUS=1 to force attempt"
  fi

  echo "================================"
  echo "MBPP+  pass@10  (n=${N_SAMPLES_PASS10}, temp=${TEMP_PASS10})"
  echo "================================"
  MB_P10="${OUT_BASE}/mbpp_pass10.jsonl"
  python dream/scripts/eval_mbpp.py \
    --model "$MODEL" \
    "${ADAPTER_ARGS[@]}" \
    --dataset "$MBPP_JSONL" \
    --output "$MB_P10" \
    --max-tasks "$MAX_TASKS" \
    --n-samples "$N_SAMPLES_PASS10" \
    --temperature "$TEMP_PASS10" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --steps "$STEPS"
  if [ "$RUN_EVALPLUS" = "1" ]; then
    run_evalplus mbpp "$MB_P10" "${OUT_BASE}/evalplus_mbpp_pass10.log"
  else
    echo "SKIP EvalPlus (mbpp pass@10): set MAX_TASKS=0 for full JSONL + scoring, or RUN_EVALPLUS=1 to force attempt"
  fi
fi

if [ "${RUN_GSM8K:-0}" = "1" ]; then
  echo "================================"
  echo "GSM8K (test split, diffusion_generate)  max_tasks=${GSM8K_MAX_TASKS}"
  echo "================================"
  python dream/scripts/eval_gsm8k.py \
    --model "$MODEL" \
    "${ADAPTER_ARGS[@]}" \
    --output-dir "$OUT_BASE/gsm8k" \
    --max-tasks "$GSM8K_MAX_TASKS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --steps "$STEPS" \
    --temperature "$TEMP_PASS1"
else
  echo "SKIP RUN_GSM8K=0"
fi

if [ "${RUN_BIGCODEBENCH:-0}" = "1" ]; then
  echo "================================"
  echo "BigCodeBench-Instruct  subset=${BCB_SUBSET}  temp=${TEMP_PASS1}"
  echo "================================"
  BCB_JSONL="${OUT_BASE}/bcb_instruct_${BCB_SUBSET}.jsonl"
  python dream/scripts/eval_bigcodebench_generate.py \
    --model "$MODEL" \
    "${ADAPTER_ARGS[@]}" \
    --output "$BCB_JSONL" \
    --subset "$BCB_SUBSET" \
    --max-tasks "$MAX_TASKS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --steps "$STEPS" \
    --temperature "$TEMP_PASS1"
  if [ "${RUN_BIGCODEBENCH_SCORE:-1}" = "1" ]; then
    set +e
    python -m bigcodebench.evaluate \
      --split instruct \
      --subset "$BCB_SUBSET" \
      --samples "$BCB_JSONL" \
      --execution "$BCB_EXECUTION" 2>&1 | tee "${OUT_BASE}/bigcodebench_eval.log"
    rc_bcb=${PIPESTATUS[0]}
    set -e
    if [ "$rc_bcb" -ne 0 ]; then
      echo "WARN: bigcodebench.evaluate failed (rc=$rc_bcb) — install bigcodebench eval deps or try BCB_EXECUTION=gradio"
    fi
  fi
else
  echo "SKIP RUN_BIGCODEBENCH=0"
fi

echo "================================"
echo "Done. Artifacts under: $OUT_BASE"
echo "Finished: $(date -Is)"
echo "================================"
