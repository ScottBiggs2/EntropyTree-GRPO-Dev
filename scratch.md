sbatch --job-name=dream_expanded_256_3_bcb_gsm8k_base \
  --export=ALL,MODEL=Dream-org/Dream-v0-Instruct-7B,STEPS=256,MAX_NEW_TOKENS=512,RUN_HUMANEVAL=0,RUN_MBPP=0,RUN_BIGCODEBENCH=1,RUN_GSM8K=1,GSM8K_MAX_TASKS=256,OUT_BASE=/scratch/${USER}/dream_eval/dream_base_expanded \
  eval_base_dream_evalplus.sbatch

  sbatch --job-name=dream_expanded_256_3_mbpp_he_base \
  --export=ALL,MODEL=Dream-org/Dream-v0-Instruct-7B,STEPS=256,MAX_NEW_TOKENS=512,RUN_HUMANEVAL=1,RUN_MBPP=1,RUN_BIGCODEBENCH=0,RUN_GSM8K=0,OUT_BASE=/scratch/${USER}/dream_eval/dream_base_expanded \
  eval_base_dream_evalplus.sbatch

export ACECODE_JSONL="/scratch/$USER/dream_data/acecode_hard_train.jsonl"
export RUN_NAME="dream_v4_1024_test"
export MAX_TASKS=1024
export NUM_EPOCHS=1
export MIN_STEPS_PER_EXPANSION=4
export MAX_STEPS_PER_EXPANSION=32
export BRANCH_THRESHOLD=0.6
export TRAIN_SAMPLING_TEMPERATURE=0.9
export TRAIN_COMPLETION_TEMPERATURE=0.7

sbatch train_acecode_mcts_ramp_evalpipe.sbatch
