sbatch --job-name=dream_expanded_256_2 \
  --export=ALL,MODEL=Dream-org/Dream-v0-Instruct-7B,STEPS=256,MAX_NEW_TOKENS=512,RUN_HUMANEVAL=1,RUN_MBPP=1,RUN_BIGCODEBENCH=0,RUN_GSM8K=0,OUT_BASE=/scratch/${USER}/dream_eval/dream_base_expanded \
  eval_base_dream_evalplus.sbatch

  sbatch --job-name=dream_expanded_256_2 \
  --export=ALL,MODEL=apple/DiffuCoder-7B-Instruct,STEPS=256,MAX_NEW_TOKENS=512,RUN_HUMANEVAL=1,RUN_MBPP=1,RUN_BIGCODEBENCH=0,RUN_GSM8K=0,OUT_BASE=/scratch/${USER}/dream_eval/dream_base_expanded \
  eval_base_dream_evalplus.sbatch

  