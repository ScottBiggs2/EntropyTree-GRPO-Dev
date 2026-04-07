#!/usr/bin/env bash
# Convenience wrapper: run EvalPlus (+ optional GSM8K / BigCodeBench) without a training job.
# Usage (from repo root):
#   ./sbatch_eval_baseline.sh
#   MODEL=apple/DiffuCoder-7B-Instruct OUT_BASE=/scratch/$USER/dream_eval/diffu_smoke ./sbatch_eval_baseline.sh
set -euo pipefail
cd "$(dirname "$0")"
exec sbatch --export=ALL,MODEL="${MODEL:-Dream-org/Dream-v0-Instruct-7B}",OUT_BASE="${OUT_BASE:-/scratch/${USER}/dream_eval/manual_$(date +%Y%m%d_%H%M)}",RUN_GSM8K="${RUN_GSM8K:-0}",RUN_BIGCODEBENCH="${RUN_BIGCODEBENCH:-0}",MAX_TASKS="${MAX_TASKS:-8}",RUN_EVALPLUS="${RUN_EVALPLUS:-0}" \
  eval_base_dream_evalplus.sbatch
