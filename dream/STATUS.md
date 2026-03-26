# Dream Implementation Status

Updated: 2026-03-26

## What Is Implemented

- Dream-specific adapter, entropy weighting fix, interval-aware time weighting, adaptive branching, LoRA support, and grouped tree loss are implemented.
- Flat GRPO and entropy-tree GRPO both exist in `dream/src/trainer.py`.
- Dream code-GRPO infrastructure now includes:
  - task schema and registry via `dream/src/task_registry.py`
  - prompt/code normalization via `dream/src/formatting.py`
  - execution-first reward options via `dream/src/rewards.py` (`build_reward_function`, including `execution_shaped`)
  - dataset-aware comparison runner and single-step script via `dream/scripts/run_dream_comparison.py` and `dream/scripts/single_step_dream.py`
  - sample task datasets in `dream/data/`
  - branch diversity / trajectory observability in `dream/src/observability.py` (metrics merged into trainer outputs)
- Docs: `dream/HPC_SYNC.md` (stale clone vs current CLI), `dream/scripts/extract_dataset_prompt.py` (legacy `--prompt` workflows).

## Verified Locally

- `python -m pytest dream/tests -q` (includes task registry, formatting, rewards, observability, trainer minimal, etc.)
- `python dream/scripts/check_reward_pipeline.py` on the sample dataset.
- Dataset-backed reward lookup off `canonical_prompt` with execution against starter code.

## Verified on GPU / HPC (bring-up complete)

The following have been run successfully on a cloud GPU with `--dataset`, `--reward execution_shaped`, and LoRA:

1. **Single-step tree GRPO** — `dream/scripts/single_step_dream.py` with `dream/data/code_grpo_train.sample.jsonl`: loads Dream 7B, prints dataset task metadata, finite `Metrics:` (including diversity fields), execution-shaped rewards nonzero.
2. **Flat GRPO baseline** — `dream/scripts/run_dream_comparison.py --phase grpo_lora_baseline` with the same dataset: two prompts, finite metrics, `workload_source=dataset`.

If your cluster rejects `--dataset` / `--reward`, the checkout is behind; see **`dream/HPC_SYNC.md`**.

## Next: HumanEval and MBPP (primary focus)

Core training plumbing for code GRPO is in place; the **next milestone** is **standard benchmark evaluation** and (as needed) **training data at benchmark scale**, per `dream/FULL_GRPO_EXTENSION_PLAN.md` Step 16 and `research_decisions.md` (e.g. D-011, D-018, D-024).

### Implementation order (suggested)

1. **Choose harness** — Dream-org eval, Apple/DiffuCoder-style export, or `dllm` / OpenAI `human-eval` + MBPP scripts; keep **prompt formatting** aligned with `dream/src/formatting.py` / `build_code_task_prompt` where possible.
2. **Export + eval scripts** in `dream/scripts/` (e.g. load checkpoint → generate completions → extract code → run official or EvalPlus tests): HumanEval first (small, canonical), then MBPP (subset for speed).
3. **Task conversion** — converters from HumanEval/MBPP JSON into `CodeTask` JSONL (or documented bridge to `data/execution_lite.json`) so **training rewards** and **eval** share one schema where feasible.
4. **Train/dev policy** — explicit held-out split for tuning; keep benchmark test sets **eval-only** unless you intentionally train on them (document in `research_decisions.md`).
5. **Optional hardening** (can parallelize with eval): flat/tree logging parity, run metadata (git hash, dataset fingerprint), `dream/docs/EVAL_PROTOCOL.md`.

### Still open (not blocking MBPP/HumanEval scripts)

- Pluggable remote execution backends (`execution_backends.py` in the extension plan) — local subprocess path is enough for standard harnesses initially.
- LLM-as-a-judge — remains optional / out of headline path.

### Good delegation targets

- MBPP/HumanEval JSONL converters and `dream/data/` staging files
- Formatting edge-case tests (fenced code, chatty completions)
- Per-run artifact logging (best/worst completion per step) for debugging

## Reference commands (GPU)

### Single-step tree smoke (dataset + execution-shaped reward)

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

### Flat GRPO smoke (comparison runner uses **underscore** tree/token flags)

```bash
python dream/scripts/run_dream_comparison.py \
  --phase grpo_lora_baseline \
  --device cuda \
  --dataset dream/data/code_grpo_train.sample.jsonl \
  --dataset-split train \
  --reward execution_shaped \
  --lora \
  --max-tasks 2 \
  --num_epochs 1 \
  --max_tree_nodes 8 \
  --max_new_tokens 96 \
  --no_wandb
```

### Reward pipeline (no GPU model)

```bash
python dream/scripts/check_reward_pipeline.py \
  --dataset dream/data/code_grpo_train.sample.jsonl \
  --dataset-split train \
  --reward execution_shaped \
  --max-tasks 3
```

See **`dream/README.md`** for CLI naming (`single_step` vs `run_dream_comparison`) and **`dream/HPC_SYNC.md`** if flags mismatch.
