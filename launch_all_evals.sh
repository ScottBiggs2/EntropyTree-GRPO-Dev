#!/bin/bash
# One-shot script to launch all 6 eval jobs (3 models x 2 benchmark sets)

# Common params
STEPS=256
MAX_NEW_TOKENS=512
GSM8K_CAP=256
OUT_ROOT="/scratch/${USER}/dream_eval"

launch_model_evals() {
    local model=$1
    local name_prefix=$2
    
    # 1. EvalPlus (HumanEval+ & MBPP+)
    echo "Launching EvalPlus for $model..."
    sbatch --job-name="${name_prefix}_evalplus" \
      --export=ALL,MODEL="$model",STEPS=$STEPS,MAX_NEW_TOKENS=$MAX_NEW_TOKENS,RUN_HUMANEVAL=1,RUN_MBPP=1,RUN_BIGCODEBENCH=0,RUN_GSM8K=0,OUT_BASE="${OUT_ROOT}/${name_prefix}_evalplus" \
      eval_base_dream_evalplus.sbatch
    
    # 2. Expanded (BCB-Hard & GSM8K)
    echo "Launching Expanded for $model..."
    sbatch --job-name="${name_prefix}_expanded" \
      --export=ALL,MODEL="$model",STEPS=$STEPS,MAX_NEW_TOKENS=$MAX_NEW_TOKENS,RUN_HUMANEVAL=0,RUN_MBPP=0,RUN_BIGCODEBENCH=1,BCB_SUBSET=hard,RUN_GSM8K=1,GSM8K_MAX_TASKS=$GSM8K_CAP,OUT_BASE="${OUT_ROOT}/${name_prefix}_expanded" \
      eval_base_dream_evalplus.sbatch
}

# Dream Base IT
launch_model_evals "Dream-org/Dream-v0-Instruct-7B" "dream_base"

# DiffuCoder Base IT
launch_model_evals "apple/DiffuCoder-7B-Instruct" "diffucoder_base"

# DiffuCoder cpGRPO
launch_model_evals "apple/DiffuCoder-7B-cpGRPO" "diffucoder_cpgrpo"
