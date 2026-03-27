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

### Environment scale-up (PLAN_03, Steps 1-4 partial)

- Container sandbox: `dream/sandbox/Dockerfile` + `entrypoint.py` (dual assertion/args_expected modes)
- AceCode-89K converter: `dream/scripts/convert_acecode.py` (DiffuCoder-aligned filtering)
- HumanEval/MBPP converters: `dream/scripts/convert_humaneval.py`, `dream/scripts/convert_mbpp.py`
- Shared conversion logic: `dream/src/eval_dataset_convert.py` (assertion extraction, `candidate` rewrite)
- Evaluation protocol doc: `dream/docs/EVAL_PROTOCOL.md`

## Verified Locally

- `python -m pytest dream/tests -q` — 17 tests pass (task registry, formatting, rewards, observability, dataset conversion)
- `python dream/scripts/check_reward_pipeline.py` on sample dataset
- Dataset-backed reward lookup with execution against starter code

## Verified on GPU / HPC

1. **Single-step tree GRPO** — `single_step_dream.py` with sample JSONL: Dream 7B loads, diversity metrics present, execution-shaped rewards nonzero
2. **Flat GRPO baseline** — `run_dream_comparison.py --phase grpo_lora_baseline`: finite metrics, `workload_source=dataset`

If `--dataset` / `--reward` flags are unrecognized, see `dream/HPC_SYNC.md`.

## What's Next (PLAN_03)

### Remaining steps for HPC execution

**Step 1 — Run AceCode converter** (needs network for HuggingFace download):

```bash
pip install datasets numpy
python dream/scripts/convert_acecode.py --output-dir dream/data/ --difficulty hard --dev-frac 0.05
# Expect: ~15K train tasks, ~750 dev tasks in dream/data/acecode_hard_{train,dev}.jsonl
```

**Step 2 — Run HumanEval/MBPP converters** (needs `evalplus`):

```bash
pip install evalplus
python dream/scripts/convert_humaneval.py --output dream/data/humaneval.jsonl
python dream/scripts/convert_mbpp.py --output dream/data/mbpp.jsonl
# Expect: 164 HumanEval tasks, ~378 MBPP tasks, all split=eval
```

**Step 3 — Build and test container sandbox** (local Docker):

```bash
docker build -t dream-sandbox:latest dream/sandbox/
# Test assertion mode:
echo '{"code":"def add(a,b): return a+b","test_cases":["assert add(1,2)==3"],"test_format":"assertion"}' \
  | docker run -i --rm --network=none dream-sandbox:latest
# Should print: 1.0
```

**Step 4 — Pull sandbox on HPC** (Apptainer):

```bash
srun --constraint=ib -p short --pty /bin/bash
cd /projects/$GROUP/container_images
mkdir -p cache tmp
export APPTAINER_CACHEDIR=$(pwd)/cache APPTAINER_TMPDIR=$(pwd)/tmp
apptainer pull dream-sandbox.sif docker://YOUR_DOCKERHUB/dream-sandbox:latest
```

### Still to implement (code changes needed)

- `dream/src/execution_backends.py` — pluggable backend ABC (`SubprocessBackend`, `ContainerBackend`)
- `dream/scripts/eval_humaneval.py`, `dream/scripts/eval_mbpp.py` — generate completions + run EvalPlus
- Wire `--execution-backend` flag into training scripts
- End-to-end validation at AceCode scale (100-task subset)

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
