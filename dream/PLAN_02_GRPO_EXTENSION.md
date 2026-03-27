# Dream Full GRPO Extension Plan

**Project**: Entropy-Guided MCTS-GRPO for Diffusion Language Models  
**Scope**: Extend the current `dream/` stack from corrected tree mechanics plus light code smoke tests into a fuller, DiffuCoder-aligned GRPO setting for code generation.  
**Status**: Steps 11–13 **implemented** in code (schema, formatting, execution-first rewards, dataset-aware runners); GPU smoke tests passed (tree single-step + flat GRPO with `execution_shaped`). **Container sandbox (partial):** `dream/sandbox/Dockerfile` and `dream/sandbox/entrypoint.py` are in place per `dream/PLAN_03_ENVIRONMENT_SCALEUP.md` Step 1; `dream/src/execution_backends.py` and reward wiring are **not** done yet — see **`dream/PLAN_03_ENVIRONMENT_SCALEUP.md` → Step 1 implementation status and verification**. Steps 14–18 **partial or pending** — see **Current-State Summary** below. Research choices remain in `research_decisions.md`.

---

## Why This Document Exists

The current `dream/` stack is already beyond a toy prototype in several important ways:

- Dream-specific adapter, loss, time weighting, entropy normalization, LoRA, and adaptive branching are implemented.
- There is both a tree trainer and a flat trajectory GRPO baseline.
- Cloud validation scripts exist for Dream loading, tree building, and one-step training.

As of 2026-03, the training loop can run **execution-shaped, dataset-backed** GRPO on GPU (`--dataset`, `--reward execution_shaped`), but a **full research comparison** still needs larger task sources, held-out discipline, and **external** HumanEval/MBPP eval (Step 16).

- `run_dream_comparison.py` / `single_step_dream.py` support CLI-selected rewards (not syntax-only by default when configured).
- Task loading can use JSONL + `task_registry` or legacy `execution_lite`-compatible data; sample files live under `dream/data/`.
- **Remaining gap**: standard **benchmark** evaluation (HumanEval, MBPP, optional EvalPlus) as separate scripts from training — see Step 16 and `dream/STATUS.md` next steps.
- Optional LLM-as-a-judge remains a placeholder; keep it out of the headline path until execution-only results are stable.

This document defines a concrete plan for extending `dream/PLAN_01_CORE_MIGRATION.md` and for identifying which new open questions should be sent to `research_decisions.md`.

---

## Planning Principles

### 1. Keep the main experimental variable narrow

The cleanest initial claim is:

`same Dream-family model + same prompt format + same train tasks + same reward stack + same eval harness; only the rollout structure differs (flat GRPO vs entropy-tree GRPO)`

That means:

- align reward framing and prompt formatting as closely as practical with DiffuCoder;
- do **not** introduce coupled-sampling, judge-heavy rewards, or unrelated architecture changes in the first full comparison;
- treat MCTS-style tree structure and entropy-guided branching as the main independent variables.

### 2. Prefer execution-based rewards over heuristic rewards

DiffuCoder centers code RL around executable test-based rewards. To reduce confounding, the Dream full-GRPO stack should do the same. `SyntaxReward` should remain only for local smoke tests and early debugging.

### 3. Preserve a staged path from local to cloud

We still want weaker agents to contribute safely. The implementation should proceed in layers:

1. local schema and formatting work,
2. registry-backed execution reward,
3. cloud GRPO training on a modest code dataset,
4. external benchmark evaluation,
5. optional judge extensions.

### 4. Defer research choices explicitly

If a choice affects the scientific interpretation, do not bury it in code defaults alone. Instead:

- add a clear default for implementation,
- keep the decision `OPEN` in `research_decisions.md`,
- record the rationale and the expected ablation.

---

## Current-State Summary

### Already in place

- `dream/src/config.py`: Dream-aware config with adaptive stepping, LoRA, grouped loss, and corrected normalization switches.
- `dream/src/trainer.py`: `EntropyMCTSTrainer` and `BaselineGRPOTrainer` (diversity metrics from `observability.py`).
- `dream/src/loss.py`: interval-aware time weighting and corrected entropy weighting.
- `dream/src/rewards.py`: layered rewards (`CodeFormatReward`, execution stack, `build_reward_function`), `SyntaxReward` for smoke tests, judge placeholder.
- `dream/src/task_registry.py`, `dream/src/formatting.py`: canonical tasks + prompt/code extraction.
- `dream/scripts/validate_dream.py`, `validate_dream_tree.py`, `single_step_dream.py`: smoke-test ladder; single-step supports `--dataset` / `--reward`.
- `dream/scripts/run_dream_comparison.py`: dataset-aware phases (tree + flat GRPO arms), `--dataset`, `--reward`, etc.
- `dream/scripts/check_reward_pipeline.py`, `dream/data/*.sample.jsonl`: local verification without full model load.
- **GPU validated**: tree one-step + flat `grpo_lora_baseline` with `execution_shaped` on sample JSONL (see `dream/STATUS.md`).
- **Container image sources (bring-up)**: `dream/sandbox/Dockerfile` and `dream/sandbox/entrypoint.py` implement the dual test formats (`assertion` strings vs `args_expected` tuples) described in `dream/PLAN_03_ENVIRONMENT_SCALEUP.md` Step 1. **Verification** (local Docker, HPC Apptainer, caveats) lives in **`dream/PLAN_03_ENVIRONMENT_SCALEUP.md` → [Step 1 implementation status and verification](PLAN_03_ENVIRONMENT_SCALEUP.md#step-1-implementation-status-and-verification)**.

### Main gaps to close

- **Step 16 (next priority)**: external evaluation harness for **HumanEval** and **MBPP** (export completions, extract code, run official or EvalPlus tests); optional `dream/docs/EVAL_PROTOCOL.md`.
- **Data at scale**: train/dev JSONL beyond samples; explicit eval-only holdout for benchmark tasks (research decision in `research_decisions.md`).
- **Steps 14–15**: deeper flat baseline / tree parity logging, degenerate-batch handling, optional dedicated tests (`test_baseline_grpo_code.py`, etc.).
- **Step 17**: run metadata (git commit, dataset hash), `EXPERIMENT_MATRIX.md`-style documentation.
- **Step 13 (remaining)**: `dream/src/execution_backends.py`, optional `dream/sandbox/README.md`, and reward integration so training can call the container backend — the **Docker/Apptainer image layer** exists, but the **Python `ExecutionBackend` abstraction and `rewards.py` plumbing** are still pending (see `dream/PLAN_03_ENVIRONMENT_SCALEUP.md` Step 1 “needs careful attention”). Optional E2B or other remote services remain future options.
- **Step 18**: judge integration — stub only; keep opt-in.

---

## What Should Change In `dream/PLAN_01_CORE_MIGRATION.md`

The existing `dream/PLAN_01_CORE_MIGRATION.md` should remain the source of truth for the already-implemented Dream migration and corrected-loss stack. The extension should be additive, not a rewrite.

### Recommended structure change

Keep Steps 1-10 as the "core Dream stack" foundation, then replace the current high-level reward/eval tail with a fuller GRPO section:

1. Keep current Steps 1-10 largely as-is.
2. Replace the current Step 11-13 tail with a new "Full Code GRPO" sequence.
3. Add one appendix that states the controlled-comparison principle: prompting, reward, and eval must match across flat and tree arms.

### Recommended new sections to add

- `Step 11: Code Task Schema and Data Pipeline`
- `Step 12: Prompt Formatting and Completion Extraction`
- `Step 13: Execution-First Reward Stack`
- `Step 14: Flat GRPO Baseline Hardening`
- `Step 15: Tree GRPO Parity on the Same Task Source`
- `Step 16: External Evaluation Harness`
- `Step 17: Reproducible Comparison Runner`
- `Step 18: Optional Judge and Hybrid Reward Extensions`

The rest of this document spells out those additions in executable detail.

---

## Execution Plan

Each step below is designed so a weaker coding agent can execute it without needing to infer the whole research agenda.

## Step 11: Code Task Schema and Data Pipeline

**Objective**: Move from ad hoc prompt lists and the legacy `execution_lite` shape to a Dream-specific GRPO task schema that can support training, dev evaluation, and external benchmark export.

**Why first**:

- reward logic,
- formatting,
- flat/tree parity,
- and evaluation

all depend on having a stable task representation.

**Files to create or modify**:

- `dream/data/README.md`
- `dream/data/code_grpo_train.sample.jsonl`
- `dream/data/code_grpo_dev.sample.jsonl`
- `dream/src/task_registry.py`
- `dream/tests/test_task_registry.py`

**Target schema**:

Each JSONL row should be a single code task with fields like:

```json
{
  "task_id": "humaneval_000_example",
  "source": "HumanEval",
  "split": "train",
  "prompt_type": "chat_code",
  "instruction": "Write a Python function ...",
  "starter_code": "def foo(x):\n    ...",
  "canonical_prompt": "... fully formatted user-facing prompt ...",
  "entry_point": "foo",
  "public_tests": ["assert foo(1) == 2"],
  "private_tests": [],
  "language": "python",
  "metadata": {
    "difficulty": "unknown",
    "tags": ["arrays"]
  }
}
```

**Implementation tasks**:

1. Define a minimal dataclass or typed dict for a code task.
2. Add loader utilities that:
   - read JSONL,
   - validate required fields,
   - expose train/dev splits,
   - optionally convert the legacy `data/execution_lite.json` into the new in-memory schema.
3. Add a helper to export prompt-plus-tests back into the old execution-lite style when needed.

**Verification**:

```bash
python -m pytest dream/tests/test_task_registry.py -q
python -c "from dream.src.task_registry import load_code_tasks; print(len(load_code_tasks('dream/data/code_grpo_train.sample.jsonl')))"
```

**Exit criteria**:

- sample train and dev JSONL files load successfully;
- malformed rows fail loudly with readable errors;
- legacy `execution_lite` data can be wrapped into the new schema without manual edits.

**Dependencies**: none.

---

## Step 12: Prompt Formatting and Completion Extraction

**Objective**: Standardize how Dream prompts are constructed and how generated code is extracted for reward/eval.

**Why this matters**:

DiffuCoder uses a consistent code-oriented prompting style. If prompt formatting differs across tree vs flat baselines, or between training and evaluation, results become hard to interpret.

**Files to create or modify**:

- `dream/src/formatting.py`
- `dream/src/rewards.py`
- `dream/src/trainer.py`
- `dream/tests/test_formatting.py`

**Implementation tasks**:

1. Add one formatter module with functions such as:
   - `build_chat_code_prompt(task, tokenizer)`
   - `extract_python_code(text)`
   - `normalize_completion_for_reward(text)`
2. Support the output shapes most likely to appear:
   - bare function body,
   - full function definition,
   - markdown fenced code,
   - chatty answer with explanation plus code.
3. Ensure both `BaselineGRPOTrainer` and `EntropyMCTSTrainer` use the same formatter entry points.
4. Ensure `ExecutionLiteReward` and any future execution reward score the same normalized code string.

**Verification**:

```bash
python -m pytest dream/tests/test_formatting.py -q
```

Tests should include:

- extraction from fenced code blocks,
- extraction from full assistant-style answers,
- preservation of valid bare function completions.

**Exit criteria**:

- one canonical prompt path for both flat and tree trainers;
- one canonical completion-normalization path for all reward functions.

**Dependencies**: Step 11.

---

## Step 13: Execution-First Reward Stack

**Objective**: Make execution-based code reward the primary Dream GRPO reward path, with syntax shaping only as a fallback or auxiliary signal.

**Design target**:

Be close to DiffuCoder in spirit:

- code reward is primarily execution-based;
- formatting checks are auxiliary;
- judge reward remains optional and secondary.

**Files to create or modify**:

- `dream/src/rewards.py`
- `dream/src/execution_backends.py` *(pending — pluggable backend; see `PLAN_03_ENVIRONMENT_SCALEUP.md`)*
- `dream/sandbox/Dockerfile` *(present — `python:3.11-slim`, entrypoint as `ENTRYPOINT`)*
- `dream/sandbox/entrypoint.py` *(present — JSON on stdin or `argv[1]`; `test_format` `assertion` / `args_expected`)*
- `dream/sandbox/README.md` *(optional; not required for bring-up)*
- `dream/scripts/check_reward_pipeline.py`
- `dream/tests/test_rewards_execution_full.py`

**Implementation tasks**:

1. Split reward concerns into explicit layers:
   - `CodeFormatReward`
   - `ExecutionReward`
   - `ExecutionShapedReward`
   - optional `JudgeAugmentedReward`
2. Add backend abstraction so the same reward interface can use:
   - local subprocess sandbox,
   - E2B,
   - Piston or other remote execution service.
3. Make the reward config explicit in code and logging:
   - reward name,
   - weights,
   - timeout,
   - backend,
   - number of retries.
4. Keep `SyntaxReward` only for:
   - local smoke tests,
   - early model-load validation,
   - unit tests without sandbox access.

**Verification**:

```bash
python dream/scripts/check_reward_pipeline.py --dataset dream/data/code_grpo_dev.sample.jsonl --backend local
python -m pytest dream/tests/test_rewards_execution_full.py -q
```

**Container sandbox** (after `execution_backends.py` exists): follow `dream/PLAN_03_ENVIRONMENT_SCALEUP.md` Step 1 ([implementation status and verification](PLAN_03_ENVIRONMENT_SCALEUP.md#step-1-implementation-status-and-verification)) and automatic tests there.

`check_reward_pipeline.py` should:

- run several known-good and known-bad completions,
- print raw execution fraction,
- print shaped reward,
- fail nonzero if all rewards collapse to zero.

**Exit criteria**:

- the Dream comparison runner can switch reward sources by CLI/config rather than code edits;
- execution-backed reward works end-to-end on a small dev set.

**Dependencies**: Steps 11-12.

---

## Step 14: Flat GRPO Baseline Hardening

**Objective**: Upgrade the flat GRPO baseline from a mechanics sanity check to a true comparison arm for code RL.

**Why this is critical**:

If the flat baseline is weaker because of data plumbing or reward mismatch, the tree result will be uninterpretable.

**Files to create or modify**:

- `dream/src/trainer.py`
- `dream/src/config.py`
- `dream/tests/test_baseline_grpo_code.py`
- `dream/scripts/run_dream_comparison.py`

**Implementation tasks**:

1. Make `BaselineGRPOTrainer` task-aware instead of raw-prompt-only.
2. Log the following for every step:
   - reward source,
   - task id,
   - completion extraction mode,
   - tokens scored,
   - average execution fraction,
   - reward variance across samples.
3. Add safeguards for degenerate GRPO batches:
   - all rewards equal,
   - all completions unparsable,
   - all execution calls fail.
4. Add a small per-step sample artifact:
   - one prompt,
   - best completion,
   - worst completion,
   - raw reward breakdown.

**Verification**:

```bash
python -m pytest dream/tests/test_baseline_grpo_code.py -q
python dream/scripts/run_dream_comparison.py --phase grpo_lora_baseline --dataset dream/data/code_grpo_dev.sample.jsonl --reward execution_shaped --no_wandb
```

**Exit criteria**:

- flat GRPO runs on the same task objects used by tree GRPO;
- reward and logging are identical in structure to the tree arm where applicable.

**Dependencies**: Steps 11-13.

---

## Step 15: Tree GRPO Parity On The Same Task Source

**Objective**: Ensure the entropy-tree arm consumes exactly the same task objects, formatting path, reward stack, and evaluation hooks as the flat baseline.

**Files to create or modify**:

- `dream/src/trainer.py`
- `dream/src/tree_builder.py`
- `dream/scripts/run_dream_comparison.py`
- `dream/tests/test_tree_grpo_code_path.py`

**Implementation tasks**:

1. Move all task formatting and completion normalization out of trainer-specific code paths.
2. Ensure the tree trainer logs the same reward breakdown fields as the flat baseline.
3. Record tree-specific fields separately:
   - node count,
   - leaf count,
   - branch depths,
   - unique completions,
   - mean entropy at branching,
   - mean step interval.
4. Confirm that when `branch_width=1` and tree budget is minimized, the tree arm behaves as close as possible to a flat rollout path.

**Verification**:

```bash
python -m pytest dream/tests/test_tree_grpo_code_path.py -q
python dream/scripts/run_dream_comparison.py --phase baseline_train --dataset dream/data/code_grpo_dev.sample.jsonl --reward execution_shaped --no_wandb
```

**Exit criteria**:

- both arms consume the same data and reward abstractions;
- the only substantive algorithmic differences are branching and advantage aggregation.

**Dependencies**: Steps 11-14.

---

## Step 16: External Evaluation Harness

**Objective**: Add benchmark-facing evaluation that is separate from the training reward.

**Why separate train reward from evaluation**:

- training reward should be robust and cheap enough for GRPO;
- final claims should be reported on standard external benchmarks.

**Files to create or modify**:

- `dream/scripts/export_eval_tasks.py`
- `dream/scripts/eval_humaneval.py`
- `dream/scripts/eval_mbpp.py`
- `dream/docs/EVAL_PROTOCOL.md`

**Implementation tasks**:

1. Add a script that exports sampled model completions in the exact format needed by the chosen evaluator.
2. Add evaluator wrappers for:
   - HumanEval,
   - MBPP,
   - optional EvalPlus.
3. Keep evaluation prompts and formatting identical to the training formatter where appropriate.
4. Save evaluation artifacts:
   - raw completions,
   - extracted code,
   - pass/fail summaries,
   - evaluator version info.

**Verification**:

```bash
python dream/scripts/eval_humaneval.py --model_ckpt path/to/ckpt --max_tasks 5
python dream/scripts/eval_mbpp.py --model_ckpt path/to/ckpt --max_tasks 10
```

Each wrapper should print:

- number of tasks evaluated,
- number of valid completions,
- pass@1 or exact pass fraction,
- path to saved outputs.

**Exit criteria**:

- post-training evaluation can run without modifying training code;
- benchmark results are reproducible from saved outputs.

**Dependencies**: Steps 11-15.

---

## Step 17: Reproducible Comparison Runner

**Objective**: Turn `dream/scripts/run_dream_comparison.py` into the primary controlled-comparison entry point for full code GRPO experiments.

**Files to create or modify**:

- `dream/scripts/run_dream_comparison.py`
- `run_dream_comparison.sh`
- `dream/docs/EXPERIMENT_MATRIX.md`

**Implementation tasks**:

1. Add explicit CLI args for:
   - dataset path,
   - dataset split,
   - reward mode,
   - execution backend,
   - number of sampled completions,
   - eval-on-end toggles.
2. Replace the fixed prompt list default with dataset-driven task loading.
3. Standardize run metadata:
   - git commit,
   - config dump,
   - reward config dump,
   - dataset fingerprint,
   - tokenizer/model identifier,
   - LoRA vs dense status.
4. Define a first-pass experiment matrix:
   - Dream base eval only,
   - flat GRPO + LoRA,
   - entropy-tree GRPO fixed-step + LoRA,
   - entropy-tree GRPO adaptive + LoRA.
5. Keep full-finetune runs optional and clearly marked as separate from the primary comparison.

**Verification**:

```bash
python dream/scripts/run_dream_comparison.py --phase grpo_lora_baseline --dataset dream/data/code_grpo_dev.sample.jsonl --reward execution_shaped --no_wandb
python dream/scripts/run_dream_comparison.py --phase adaptive_default --dataset dream/data/code_grpo_dev.sample.jsonl --reward execution_shaped --no_wandb
```

**Exit criteria**:

- all primary comparison arms can be launched from one runner;
- run artifacts are sufficient to audit what changed between arms.

**Dependencies**: Steps 11-16.

---

## Step 18: Optional Judge And Hybrid Reward Extensions

**Objective**: Add LLM-as-a-judge only as a clearly isolated extension after the execution-first setup is working.

**Guiding constraint**:

Do not let judge reward become part of the "main result" until the execution-only comparison is stable and reported.

**Files to create or modify**:

- `dream/src/judge.py`
- `dream/src/rewards.py`
- `dream/scripts/eval_judge_agreement.py`
- `dream/docs/JUDGE_PROTOCOL.md`

**Implementation tasks**:

1. Add a judge interface that accepts:
   - prompt,
   - extracted code,
   - optional reference tests or rubric,
   - returns a scalar score and optional textual rationale.
2. Add a hybrid reward wrapper:
   - `R_total = alpha_exec * R_exec + alpha_judge * R_judge`
3. Log judge/execution agreement statistics.
4. Require explicit CLI opt-in so judge use cannot happen accidentally.

**Verification**:

```bash
python dream/scripts/eval_judge_agreement.py --dataset dream/data/code_grpo_dev.sample.jsonl --max_tasks 20
```

The script should report:

- correlation between judge score and execution reward,
- judge false-positive examples,
- judge false-negative examples.

**Exit criteria**:

- judge reward is operational but cleanly separated from the default experimental path.

**Dependencies**: Steps 11-17.

---

## Minimum Viable Full-GRPO Milestone

The stack should count as "full GRPO ready" only when all of the following are true:

- Dream flat GRPO and entropy-tree GRPO train on the same code task dataset.
- Both use the same formatter and code extraction path.
- Both use execution-based reward as the primary reward.
- A held-out dev split exists and is not mixed into training.
- HumanEval or MBPP evaluation can be run from saved checkpoints.
- `run_dream_comparison.py` can launch the primary comparison arms without manual code edits.

**As of 2026-03-26**: the first, third, and last bullets are **satisfied for bring-up** (same sample dataset, shared formatter/reward path, CLI-driven runner). The **dev-split discipline**, **HumanEval/MBPP eval from checkpoints**, and **scaled training data** are **not** done — that is the current frontier before calling the milestone “complete.”

---

## Proposed Additions For `research_decisions.md`

These should remain open until explicitly decided by the user. They are listed here so an implementation agent knows what to surface, not silently choose.

### D-018: Primary Training Task Source

**Keep OPEN**. Candidate defaults:

- small internal registry converted from `execution_lite` for bring-up;
- HumanEval-style curated train/dev set for first real comparison;
- AceCode-hard or a comparable larger code corpus for scaling.

### D-019: Primary Reward Stack

**Keep OPEN**. Candidate default:

- execution-based reward as primary,
- small format/syntax shaping bonus,
- no judge in the main result.

### D-020: Execution Backend

**Keep OPEN**. Candidate default:

- local subprocess backend for development,
- Docker/Apptainer container using `dream/sandbox/` (image + entrypoint implemented; Python `ExecutionBackend` wiring still pending),
- E2B or equivalent remote sandbox for heavier runs.

### D-021: Prompt Template Lock

**Keep OPEN**. Candidate default:

- single Qwen/Dream-style code chat template used in both train and eval where compatible.

### D-022: Coupled Sampling In Scope Or Not

**Keep OPEN**. Candidate default:

- out of scope for the first entropy-tree comparison,
- later ablation inspired by DiffuCoder after the baseline matrix is stable.

### D-023: Judge Usage

**Keep OPEN**. Candidate default:

- no judge in the headline result,
- optional exploratory hybrid reward after execution-only parity.

### D-024: Primary Benchmark Claim

**Keep OPEN**. Candidate default:

- held-out execution reward for development,
- HumanEval pass@1 as primary reported metric,
- MBPP as secondary,
- EvalPlus optional once the harness is stable.

---

## Suggested Implementation Order

If multiple weaker agents work on this plan, the safest order is:

1. `Step 11` task schema
2. `Step 12` formatting and extraction
3. `Step 13` reward stack
4. `Step 14` flat baseline hardening
5. `Step 15` tree parity
6. `Step 16` evaluation harness
7. `Step 17` comparison runner
8. `Step 18` optional judge

This order minimizes wasted work because data and formatting choices propagate into every later step.

---

## Recommended First Deliverable Set

If you want the smallest useful extension to pursue first, target this bundle:

- `dream/src/task_registry.py`
- `dream/src/formatting.py`
- execution-first reward wiring in `dream/src/rewards.py`
- dataset-aware `dream/scripts/run_dream_comparison.py`
- a tiny train/dev JSONL sample in `dream/data/`

That bundle is enough to shift the Dream comparison stack from "syntax-weighted smoke tests" to "real code GRPO plumbing" without yet committing to larger benchmark infrastructure.

---

## External Alignment Notes

This plan is intentionally closest to DiffuCoder on the following dimensions:

- code generation domain,
- execution-centered reward framing,
- Dream-style diffusion inference assumptions,
- low-temperature code sampling,
- keeping larger objective changes isolated.

It intentionally does **not** adopt coupled-sampling as the first extension, because that would create a second major algorithmic delta on top of tree search. That choice should remain explicit in `research_decisions.md`.

Relevant references:

- In-repo environment / sandbox scale-up: `dream/PLAN_03_ENVIRONMENT_SCALEUP.md` (container Step 1 implementation status, automatic tests, local/HPC verification, data, EvalPlus).
- DiffuCoder repository: <https://github.com/apple/ml-diffucoder>
- DiffuCoder paper: <https://arxiv.org/abs/2506.20639>
- Dream model: <https://huggingface.co/Dream-org/Dream-v0-Instruct-7B>

