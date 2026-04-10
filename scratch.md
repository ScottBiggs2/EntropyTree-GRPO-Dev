sbatch --job-name=dream_expanded_256_3_bcb_gsm8k_base \
  --export=ALL,MODEL=Dream-org/Dream-v0-Instruct-7B,STEPS=256,MAX_NEW_TOKENS=512,RUN_HUMANEVAL=0,RUN_MBPP=0,RUN_BIGCODEBENCH=1,RUN_GSM8K=1,GSM8K_MAX_TASKS=256,OUT_BASE=/scratch/${USER}/dream_eval/dream_base_expanded \
  eval_base_dream_evalplus.sbatch

  sbatch --job-name=dream_expanded_256_3_mbpp_he_base \
  --export=ALL,MODEL=Dream-org/Dream-v0-Instruct-7B,STEPS=256,MAX_NEW_TOKENS=512,RUN_HUMANEVAL=1,RUN_MBPP=1,RUN_BIGCODEBENCH=0,RUN_GSM8K=0,OUT_BASE=/scratch/${USER}/dream_eval/dream_base_expanded \
  eval_base_dream_evalplus.sbatch

  