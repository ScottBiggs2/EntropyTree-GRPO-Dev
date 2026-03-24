# Dream Implementation Status

Updated: 2026-03-23

## What Is Implemented

- Dream-specific adapter, entropy weighting fix, interval-aware time weighting, adaptive branching, LoRA support, and grouped tree loss are implemented.
- Flat GRPO and entropy-tree GRPO both exist in `dream/src/trainer.py`.
- Dream code-GRPO infrastructure now includes:
  - task schema and registry via `dream/src/task_registry.py`
  - prompt/code normalization via `dream/src/formatting.py`
  - execution-first reward options via `dream/src/rewards.py`
  - dataset-aware comparison runner via `dream/scripts/run_dream_comparison.py`
  - sample task datasets in `dream/data/`

## Verified Locally

- `python -m pytest dream/tests/test_task_registry.py dream/tests/test_formatting.py dream/tests/test_rewards_execution_full.py dream/tests/test_trainer_minimal.py -q`
- Reward pipeline smoke test on the sample dataset succeeds.
- Dataset-backed reward lookup works off canonical prompts while executing against starter code.

## Remaining Core Work Before Full Evaluation

### Still high-priority implementation

- run a real Dream GPU single-step using dataset-backed execution reward
- run the updated comparison runner on HPC/GPU with execution-backed reward
- expand beyond tiny sample tasks / legacy execution-lite bridge to a more serious task source
- add external evaluation harness scripts for HumanEval / MBPP / possibly EvalPlus

### Good delegation targets for weaker agents

- expand `dream/data/` with more task files or converters
- add more formatting edge-case tests
- add more reward tests for fenced/chatty assistant outputs
- improve runner logging and per-task artifact capture
- draft benchmark export helpers once the evaluation harness choice is finalized

## What Needs HPC / GPU

The following should be run by the user on HPC/GPU:

### 1. One real single-step code-GRPO smoke test

From repo root:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python dream/scripts/single_step_dream.py \
  --lora \
  --profile-memory \
  --dataset dream/data/code_grpo_train.sample.jsonl \
  --dataset-split train \
  --task-index 0 \
  --reward execution_shaped \
  --max-tree-nodes 5 \
  --max-new-tokens 96 \
  --steps-per-expansion 12
```

What good looks like:

- the script prints the selected dataset task
- `Metrics:` is finite
- no reward lookup failures
- no OOM on the first step

### 2. Reward pipeline check on the target dataset

Before larger runs:

```bash
python dream/scripts/check_reward_pipeline.py \
  --dataset dream/data/code_grpo_train.sample.jsonl \
  --dataset-split train \
  --reward execution_shaped \
  --max-tasks 3
```

What good looks like:

- known-good completions outscore the bad completion
- no all-zero reward collapse

### 3. Updated multi-arm comparison run

The shell script still needs to be pointed at the desired dataset/reward mode when launching the real code-GRPO comparison. If using the Python runner directly, use:

```bash
python dream/scripts/run_dream_comparison.py \
  --phase grpo_lora_baseline \
  --device cuda \
  --dataset dream/data/code_grpo_train.sample.jsonl \
  --dataset-split train \
  --reward execution_shaped \
  --lora \
  --max-tree-nodes 8 \
  --max-new-tokens 96
```

## Current Recommendation

The next best checkpoint is:

1. run the single-step dataset-backed Dream smoke test on GPU
2. run a small comparison phase with execution-backed reward
3. then choose and build the external evaluation harness

Once those are done, the main open frontier becomes evaluation, not core GRPO plumbing.
