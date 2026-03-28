# Dream Implementation Status

Updated: 2026-03-26

## Documentation Map

| Document | Purpose |
|----------|---------|
| `dream/README.md` | How to run scripts, CLI flags, file layout |
| `dream/STATUS.md` | This file — what's done, what's next |
| `dream/PLAN_01_CORE_MIGRATION.md` | Original Dream migration plan (Steps 1-10, core stack) |
| `dream/PLAN_02_GRPO_EXTENSION.md` | Code-GRPO extension plan (Steps 11-18) |
| `dream/PLAN_03_ENVIRONMENT_SCALEUP.md` | **Active plan** — AceCode data, container sandbox, EvalPlus eval |
| `dream/HPC_SYNC.md` | Stale-clone troubleshooting for HPC |
| `dream/docs/EVAL_PROTOCOL.md` | HumanEval/MBPP evaluation procedure |
| `dream/docs/WANDB_METRICS.md` | W&B metric interpretation guide |
| `research_decisions.md` | All research decisions (OPEN/DECIDED/DEFERRED) |
| `literature_reference.md` | Papers and repos reference |

## What Is Implemented

### Core stack (PLAN_01, Steps 1-10)

- Dream-specific adapter, entropy weighting, interval-aware time weighting, adaptive branching, LoRA, grouped tree loss
- Flat GRPO (`BaselineGRPOTrainer`) and entropy-tree GRPO (`EntropyMCTSTrainer`) in `dream/src/trainer.py`
- Branch diversity / trajectory observability in `dream/src/observability.py`

### Code-GRPO infrastructure (PLAN_02, Steps 11-13)

- Task schema and registry: `dream/src/task_registry.py` (supports `test_format: assertion` and `args_expected`)
- Prompt/code normalization: `dream/src/formatting.py`
- Execution-first rewards: `dream/src/rewards.py` (factory via `build_reward_function`)
- Dataset-aware runners: `dream/scripts/run_dream_comparison.py`, `dream/scripts/single_step_dream.py`
- Sample datasets: `dream/data/code_grpo_{train,dev}.sample.jsonl`

### Environment scale-up (PLAN_03, Steps 1-5)

- **Container sandbox** (Step 1): `dream/sandbox/Dockerfile` + `entrypoint.py` — dual assertion/args_expected modes. Image pushed to `scottbiggs2001/dream-sandbox:latest`. Verified locally (Docker) and on HPC (Apptainer).
- **AceCode-89K converter** (Step 2): `dream/scripts/convert_acecode.py` — DiffuCoder-aligned filtering, `--limit` flag for partial downloads. Verified on HPC: 972 train / 51 dev tasks (with `--limit 5000`).
- **HumanEval/MBPP converters** (Step 3): `dream/scripts/convert_humaneval.py` (164 tasks), `dream/scripts/convert_mbpp.py` (378 tasks, merges `test_list` from sanitized MBPP). Verified on HPC.
- **Evaluation protocol** (Step 4): `dream/docs/EVAL_PROTOCOL.md`
- **DiffuCoder-aligned EvalPlus harness**: `dream/src/eval_prompts.py` (paper Tables 5–6 templates), `dream/src/eval_generate.py` (shared `diffusion_generate` loop + JSONL writer), `dream/scripts/eval_humaneval.py`, `dream/scripts/eval_mbpp.py` (optional `--run-evalplus`). See `research_decisions.md` **D-031** for training vs eval prompt mismatch.
- **Execution backends** (Step 5): `dream/src/execution_backends.py` — `ExecutionBackend` ABC with `SubprocessBackend` and `ContainerBackend` (Docker + Apptainer). Wired into `rewards.py` via optional `backend` param. Training scripts accept `--execution-backend` and `--sandbox-image` flags.

## Verified Locally

- `python -m pytest dream/tests -q` — **57 tests pass** (task registry, formatting, rewards, observability, dataset conversion, execution backends incl. live Docker, DiffuCoder-aligned eval prompts)
- `python dream/scripts/check_reward_pipeline.py` on sample dataset
- Container sandbox: Docker build + assertion/args_expected/failure modes all correct

## Verified on GPU / HPC

1. **Single-step tree GRPO** — `single_step_dream.py` with sample JSONL: Dream 7B loads, diversity metrics present, execution-shaped rewards nonzero
2. **Flat GRPO baseline** — `run_dream_comparison.py --phase grpo_lora_baseline`: finite metrics, `workload_source=dataset`
3. **Data converters** — AceCode, HumanEval, MBPP all produce correct JSONL on HPC
4. **Apptainer sandbox** — `dream-sandbox.sif` pulled and tested on Explorer

If `--dataset` / `--reward` flags are unrecognized, see `dream/HPC_SYNC.md`.

## What's Next (PLAN_03)

### Still to implement

- End-to-end training validation at AceCode scale (100-task subset)

### Next HPC execution: training with container backend

```bash
# Using AceCode data with container execution on HPC:
python dream/scripts/run_dream_comparison.py \
  --phase grpo_lora_baseline --device cuda \
  --dataset /scratch/biggs.s/dream_data/acecode_hard_train.jsonl \
  --dataset-split train --reward execution_shaped \
  --execution-backend apptainer \
  --sandbox-image /scratch/biggs.s/containers/dream-sandbox.sif \
  --lora --max-tasks 10 --num_epochs 1 \
  --max_tree_nodes 8 --max_new_tokens 128 --no_wandb

# Or with subprocess backend (no container, simpler):
python dream/scripts/run_dream_comparison.py \
  --phase grpo_lora_baseline --device cuda \
  --dataset /scratch/biggs.s/dream_data/acecode_hard_train.jsonl \
  --dataset-split train --reward execution_shaped \
  --lora --max-tasks 10 --num_epochs 1 \
  --max_tree_nodes 8 --max_new_tokens 128 --no_wandb
```

See `dream/PLAN_03_ENVIRONMENT_SCALEUP.md` for full details, failure modes, and delegation guide.

## Reference Commands (GPU)

### Single-step tree smoke

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python dream/scripts/single_step_dream.py \
  --lora --profile-memory \
  --dataset dream/data/code_grpo_train.sample.jsonl \
  --dataset-split train --task-index 0 \
  --reward execution_shaped \
  --max-tree-nodes 5 --max-new-tokens 96 --steps-per-expansion 12
```

### Flat GRPO smoke

```bash
python dream/scripts/run_dream_comparison.py \
  --phase grpo_lora_baseline --device cuda \
  --dataset dream/data/code_grpo_train.sample.jsonl \
  --dataset-split train --reward execution_shaped \
  --lora --max-tasks 2 --num_epochs 1 \
  --max_tree_nodes 8 --max_new_tokens 96 --no_wandb
```

### Reward pipeline (no GPU)

```bash
python dream/scripts/check_reward_pipeline.py \
  --dataset dream/data/code_grpo_train.sample.jsonl \
  --dataset-split train --reward execution_shaped --max-tasks 3
```

### EvalPlus (DiffuCoder-aligned prompts; GPU)

HumanEval+ / MBPP+ JSONL for `evalplus.evaluate` — use converted `humaneval.jsonl` / `mbpp.jsonl` (see `convert_humaneval.py` / `convert_mbpp.py`). `starter_code` / `instruction` slots match DiffuCoder’s `{prompt}` usage.

```bash
# Smoke (2 tasks), pass@1-style defaults
python dream/scripts/eval_humaneval.py \
  --model Dream-org/Dream-v0-Instruct-7B \
  --dataset /path/to/humaneval.jsonl \
  --output /path/to/humaneval_completions.jsonl \
  --max-tasks 2 --n-samples 1 --temperature 0.2

# Full run + EvalPlus scoring (requires evalplus on PATH)
python dream/scripts/eval_mbpp.py \
  --dataset /path/to/mbpp.jsonl \
  --output /path/to/mbpp_completions.jsonl \
  --n-samples 10 --temperature 0.4 \
  --run-evalplus
```
