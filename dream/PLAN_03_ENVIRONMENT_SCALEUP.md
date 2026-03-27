# GRPO Coding Environment Scale-up Plan

**Project**: Entropy-Guided MCTS-GRPO for Diffusion Language Models  
**Scope**: Scale Dream GRPO from 3-task smoke test to production-grade training + evaluation  
**Created**: 2026-03-26  
**Status**: In progress — Step 1 **partial**: `dream/sandbox/Dockerfile` and `dream/sandbox/entrypoint.py` implemented (delegation slice); `dream/sandbox/README.md`, `dream/src/execution_backends.py`, reward wiring, and `dream/tests/test_execution_backends.py` **pending**. Step 4 **partial** (delegation): `dream/docs/EVAL_PROTOCOL.md` **done**; `dream/scripts/eval_humaneval.py` and `dream/scripts/eval_mbpp.py` **pending** (see [Needs careful attention](#needs-careful-attention-do-yourself-or-review-closely)). See [Step 1 implementation status and verification](#step-1-implementation-status-and-verification).

---

## Table of Contents

1. [Situation Assessment](#situation-assessment)
2. [Architecture Overview](#architecture-overview)
3. [Step 1: Containerized Execution Sandbox](#step-1-containerized-execution-sandbox)
4. [Step 2: AceCode-89K Data Ingestion](#step-2-acecode-89k-data-ingestion)
5. [Step 3: HumanEval and MBPP Task Conversion](#step-3-humaneval-and-mbpp-task-conversion)
6. [Step 4: EvalPlus Evaluation Harness](#step-4-evalplus-evaluation-harness)
7. [Step 5: Wire Container Backend into Training Loop](#step-5-wire-container-backend-into-training-loop)
8. [Step 6: End-to-End Training Validation at Scale](#step-6-end-to-end-training-validation-at-scale)
9. [Transferability: Open-R1 and CoT](#transferability-open-r1-and-cot)
10. [Research Decisions](#research-decisions)
11. [Delegation Guide](#delegation-guide)
12. [External Resources](#external-resources)
13. [Step 1 implementation status and verification](#step-1-implementation-status-and-verification)

---

## Situation Assessment

### Current state

- **Training data**: 3 sample tasks in `dream/data/code_grpo_train.sample.jsonl`, 15 legacy tasks in `data/execution_lite.json`
- **Execution**: Bare `subprocess` + `exec()` in `scripts/run_execution_sandbox.py` -- no container isolation, no resource limits, 2s timeout
- **Evaluation**: None. No HumanEval/MBPP scripts exist
- **Reward stack**: Working `ExecutionShapedReward` in `dream/src/rewards.py` that calls `src/execution.py:run_tests()`

### Target state

- **Training data**: AceCode-89K hard split (~15K tasks with tests) in `CodeTask` JSONL format
- **Execution**: Containerized sandbox (Docker locally, Apptainer on Northeastern Explorer HPC) with resource limits, network isolation, and configurable timeouts
- **Evaluation**: EvalPlus-based HumanEval+ and MBPP+ evaluation from checkpoints
- **Reward stack**: Same `ExecutionShapedReward` interface, backed by pluggable container execution

### Literature alignment check

This plan was validated against:

- **DiffuCoder** ([apple/ml-diffucoder](https://github.com/apple/ml-diffucoder)): Our primary code-GRPO reference. They use E2B for execution, AceCode-89K hard split for training data, and EvalPlus/qwencoder-eval for evaluation. Our approach matches on data and eval; we diverge on execution backend (Docker/Apptainer vs E2B) and GRPO algorithm (entropy-tree vs coupled sampling).
- **Open-R1** ([huggingface/open-r1](https://github.com/huggingface/open-r1)): The upstream of DiffuCoder. Uses TRL `GRPOTrainer` with pluggable reward functions (`completions, **kwargs -> list[float]`). Our `ExecutionBackend` ABC is designed to be wrappable into this signature for future integration.
- **d1** ([dllm-reasoning/d1](https://github.com/dllm-reasoning/d1)): First diffu-GRPO for reasoning on LLaDA. Uses mean-field log-prob estimation. Our task registry, execution backend, and data pipeline are domain-agnostic and transfer to math/reasoning tasks with a different reward function.
- **AceCoder** ([TIGER-AI-Lab/AceCoder](https://github.com/TIGER-AI-Lab/AceCoder)): Source of AceCode-89K. Their filtering logic (bottom 20% accuracy, top 60% std-dev among `qwen_coder_2.5` and `llama3_instruct`) is what DiffuCoder's `process_data.py` implements.

### Key architectural insight: test case format divergence

**Critical detail**: AceCode-89K test cases are **assertion strings** (e.g., `"assert add(1, 2) == 3"`), not `[args, expected]` pairs. DiffuCoder's reward runs these via `subprocess.run(["python3", "-c", f"{code}\n{case}"])`. Our current `run_execution_sandbox.py` expects `[args, expected]` and calls `fn(*args) == expected`.

**Decision**: The container `entrypoint.py` must support **both** formats:
1. **Assertion-string mode** (for AceCode): exec `code + "\n" + assertion` directly
2. **Args-expected mode** (for legacy/HumanEval): call `fn(*args) == expected`

This is the single most important compatibility detail in the entire plan.

---

## Architecture Overview

```
AceCode-89K (HuggingFace)          HumanEval / MBPP (EvalPlus)
        |                                    |
  convert_acecode.py               convert_humaneval.py / convert_mbpp.py
        |                                    |
  acecode_hard_{train,dev}.jsonl      humaneval.jsonl / mbpp.jsonl
        |                                    |
  task_registry.py (CodeTask)        task_registry.py (CodeTask, split=eval)
        |                                    |
  [Training Loop]                    [Evaluation Harness]
        |                                    |
  ExecutionBackend ABC               eval_humaneval.py / eval_mbpp.py
   /          \                              |
SubprocessBack  ContainerBack          EvalPlus (pip)
                 /       \                   |
            Docker    Apptainer        pass@1, pass@10
           (local)   (Explorer HPC)
```

---

## Step 1: Containerized Execution Sandbox

### Goal

Build a Docker image that runs untrusted Python code safely, callable from the reward function during training. Make it work with Apptainer on Northeastern's Explorer HPC.

### Files to create

- `dream/sandbox/Dockerfile` *(done — see [implementation status](#step-1-implementation-status-and-verification))*
- `dream/sandbox/entrypoint.py` *(done)*
- `dream/sandbox/README.md` *(pending)*
- `dream/src/execution_backends.py` *(pending)*

### Docker image design

Base: `python:3.11-slim` (small, no extras beyond stdlib).

`entrypoint.py` design (must support both test formats):

```python
# Mode 1: assertion strings (AceCode-style)
# config.json: {"code": "...", "test_cases": ["assert add(1,2)==3", ...]}
# -> exec(code); for each assertion: exec(assertion) in same namespace

# Mode 2: args-expected pairs (legacy/HumanEval-style)  
# config.json: {"code": "...", "function_name": "add", "tests": [[1,2,3], [0,0,0]]}
# -> exec(code); fn = globals()[function_name]; fn(*args) == expected
```

Resource limits enforced at runtime:
- Docker: `docker run --memory=256m --cpus=0.5 --network=none --read-only --tmpfs /tmp:size=16m`
- Apptainer: `apptainer exec --contain --net --network none --no-home -B /tmp/task:/task:ro sandbox.sif python /entrypoint.py /task/config.json`

### `ExecutionBackend` ABC

Located in `dream/src/execution_backends.py`:

```python
class ExecutionBackend(ABC):
    @abstractmethod
    def run_tests(self, code: str, function_name: str, tests: list,
                  starter_code: str = "", timeout: float = 5.0,
                  test_format: str = "args_expected") -> float:
        """Returns fraction of tests passed in [0, 1]."""
        ...

class SubprocessBackend(ExecutionBackend):
    """Wraps existing src/execution.run_tests(). No container."""

class ContainerBackend(ExecutionBackend):
    """Runs code inside Docker (local) or Apptainer (HPC)."""
    def __init__(self, image: str = "dream-sandbox:latest",
                 runtime: str = "docker",  # or "apptainer"
                 sif_path: str | None = None,
                 memory_limit: str = "256m",
                 cpu_limit: str = "0.5"): ...
```

The `runtime` field auto-detects or can be overridden via env var `DREAM_SANDBOX_RUNTIME`. On HPC, set `DREAM_SANDBOX_RUNTIME=apptainer` and `DREAM_SANDBOX_SIF=/path/to/sandbox.sif`.

### Reward integration

`dream/src/rewards.py`: `ExecutionShapedReward.__init__` gains optional `backend: ExecutionBackend = None`. If `None`, falls back to current subprocess path. `build_reward_function()` accepts optional `backend` kwarg.

### HPC-specific: Apptainer on Explorer

Explorer uses **Apptainer** (system-wide, no module load needed). Docker is **not** available. Workflow:

```bash
# On a compute node (srun or sbatch), pull once:
srun --constraint=ib -p short --pty /bin/bash
cd /projects/$GROUP/container_images
mkdir -p cache tmp
export APPTAINER_CACHEDIR=$(pwd)/cache
export APPTAINER_TMPDIR=$(pwd)/tmp
apptainer pull dream-sandbox.sif docker://yourdockerhub/dream-sandbox:latest

# In training sbatch script:
export DREAM_SANDBOX_RUNTIME=apptainer
export DREAM_SANDBOX_SIF=/projects/$GROUP/container_images/dream-sandbox.sif
```

Reference: [Apptainer on Explorer](https://rc-docs.northeastern.edu/en/explorer-main/containers/apptainer.html)

### Automatic test conditions

```bash
# Test 1: Docker image builds
docker build -t dream-sandbox:latest dream/sandbox/ && echo "PASS: image builds"

# Test 2: Correct result on known-good code (assertion mode)
RESULT=$(echo '{"code": "def add(a,b): return a+b", "test_cases": ["assert add(1,2)==3", "assert add(0,0)==0"], "test_format": "assertion"}' | docker run -i --rm --network=none dream-sandbox:latest)
[ "$RESULT" = "1.0" ] && echo "PASS: assertion mode" || echo "FAIL: expected 1.0, got $RESULT"

# Test 3: Correct result on known-good code (args_expected mode)
RESULT=$(echo '{"code": "def add(a,b): return a+b", "function_name": "add", "tests": [[1,2,3],[0,0,0]], "test_format": "args_expected"}' | docker run -i --rm --network=none dream-sandbox:latest)
[ "$RESULT" = "1.0" ] && echo "PASS: args_expected mode" || echo "FAIL"

# Test 4: Timeout on infinite loop
RESULT=$(echo '{"code": "import time\ndef f(): time.sleep(999)", "function_name": "f", "tests": [[]], "test_format": "args_expected"}' | timeout 10 docker run -i --rm --network=none dream-sandbox:latest)
[ "$RESULT" = "0.0" ] && echo "PASS: timeout handled" || echo "FAIL"

# Test 5: Network isolation
RESULT=$(echo '{"code": "import urllib.request; urllib.request.urlopen(\"http://example.com\")", "test_cases": ["assert True"], "test_format": "assertion"}' | docker run -i --rm --network=none dream-sandbox:latest)
[ "$RESULT" = "0.0" ] && echo "PASS: network blocked" || echo "FAIL"

# Test 6: Unit test for ExecutionBackend
python -m pytest dream/tests/test_execution_backends.py -q
```

### Step 1 implementation status and verification

#### Implementation status

| Artifact | Status |
|---|---|
| `dream/sandbox/Dockerfile` | **Done** — `python:3.11-slim`, `ENTRYPOINT` runs `python -u /entrypoint.py`. |
| `dream/sandbox/entrypoint.py` | **Done** — JSON from **stdin** (no args) or **file path** in `sys.argv[1]` (Apptainer-friendly). `test_format`: **`assertion`** (`test_cases`) and **`args_expected`** (`function_name`, `tests`; row shape matches `scripts/run_execution_sandbox.py`). Infers format from keys if `test_format` omitted. Per-test timeout: `DREAM_SANDBOX_TEST_TIMEOUT` (default `5` seconds). |
| `dream/sandbox/README.md` | **Pending** |
| `dream/src/execution_backends.py` | **Pending** — `SubprocessBackend` / `ContainerBackend`; see [Delegation Guide](#delegation-guide) (“needs careful attention”). |
| `dream/src/rewards.py` integration | **Pending** — optional `backend` on `ExecutionShapedReward` / `build_reward_function()`. |
| `dream/tests/test_execution_backends.py` | **Pending** — blocked until `execution_backends.py` exists. |

#### Verification on a machine with Docker (local or remote login node)

From the repository root, with the Docker daemon running:

1. **Build** the image:

   ```bash
   docker build -t dream-sandbox:latest dream/sandbox/
   ```

2. **Run** the [Automatic test conditions](#automatic-test-conditions) block above (Tests 1–5). Interpret results as follows:

   - **Tests 2–3** (known-good code): stdout must be exactly **`1.0`** (single line).
   - **Test 4** (`"tests": [[]]`): expect **`0.0`** without hanging — the inner row is skipped under the same rules as `run_execution_sandbox.py`, so this is a weak “timeout” smoke check unless the payload is extended to invoke a hanging function with valid rows.
   - **Test 5** (network): expect **`0.0`** only when **`--network=none`** is used. If the host allows outbound HTTP, `exec(code)` may succeed and **`1.0`** can appear instead — that is **not** a failed build, only weaker isolation than Docker provides.
   - **Test 6**: passes only after `execution_backends.py` and tests exist.

3. **macOS:** GNU **`timeout`** is often missing; install **`gtimeout`** (Homebrew `coreutils`) or run Test 4 without an outer `timeout` (the `[[]]` case should still return quickly).

4. **Without Docker** (sanity check that `entrypoint.py` runs on the host Python):

   ```bash
   echo '{"code":"def add(a,b): return a+b","test_cases":["assert add(1,2)==3"],"test_format":"assertion"}' | python dream/sandbox/entrypoint.py
   ```

   Expect **`1.0`**.

#### Verification on HPC (Apptainer; Docker usually absent)

Build or pull a **`.sif`** from the same image (e.g. push `dream-sandbox:latest` to a registry from a machine with Docker, then `apptainer pull` on the cluster). Use project or scratch for `APPTAINER_CACHEDIR` / `APPTAINER_TMPDIR` per site docs ([Explorer Apptainer](https://rc-docs.northeastern.edu/en/explorer-main/containers/apptainer.html)).

If **stdin** to the container is unreliable under `apptainer exec --contain`, pass a **bind-mounted JSON file** (matches `entrypoint.py` `argv[1]` mode):

```bash
mkdir -p /tmp/task
echo '{"code":"def add(a,b): return a+b","test_cases":["assert add(1,2)==3"],"test_format":"assertion"}' > /tmp/task/config.json
apptainer exec --contain --net --network none --no-home \
  -B /tmp/task:/task:ro /path/to/dream-sandbox.sif \
  python /entrypoint.py /task/config.json
```

Expect **`1.0`**. If `--network none` is unsupported on a node, use the fallback in [Likely failure modes and fallbacks](#likely-failure-modes-and-fallbacks). Training jobs can set `DREAM_SANDBOX_RUNTIME=apptainer` and `DREAM_SANDBOX_SIF=...` once `ContainerBackend` exists ([HPC-specific](#hpc-specific-apptainer-on-explorer)).

### Likely failure modes and fallbacks

| Failure mode | Symptom | Fallback |
|---|---|---|
| Docker not installed locally | `docker: command not found` | Use `SubprocessBackend` (default). Container is optional for dev. |
| Apptainer `--network none` not supported on some kernels | `FATAL: network namespace not supported` | Drop `--network none`, rely on no-internet job partition. Document in `dream/sandbox/README.md`. |
| `.sif` file too large for `/home` quota | Apptainer pull fails with disk quota | Store in `/projects/$GROUP/` or `/scratch/`. Set `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR` accordingly. See [Explorer storage quotas](https://rc-docs.northeastern.edu/en/explorer-main/best-practices/homequota.html). |
| Container startup latency too high for training loop | >2s per execution call, training becomes impractical | Batch multiple test executions per container invocation. Or fall back to `SubprocessBackend` for training, use `ContainerBackend` only for eval. |
| `stdin` pipe doesn't work with Apptainer `--contain` | Empty input to entrypoint | Switch to file-based I/O: write code to a temp file, bind-mount it, read from file inside container. |

---

## Step 2: AceCode-89K Data Ingestion

### Goal

Convert AceCode-89K hard split into `CodeTask` JSONL format for training.

### Files to create

- `dream/scripts/convert_acecode.py`
- `dream/data/acecode_hard_train.jsonl` (gitignored, generated)
- `dream/data/acecode_hard_dev.jsonl` (gitignored, generated)
- Update `dream/data/.gitignore` to exclude large generated files

### AceCode-89K schema (from HuggingFace)

The dataset ([TIGER-Lab/AceCode-87K](https://huggingface.co/datasets/TIGER-Lab/AceCode-87K), also loadable as `TIGER-Lab/AceCode-89K`) contains:

- `id` (str): unique identifier
- `source` (str): origin dataset
- `question` (str): coding instruction (rewritten by GPT-4o-mini)
- `test_cases` (list[str]): assertion strings like `"assert add(1, 2) == 3"`
- `context_messages` (list[dict]): chat messages with `content` and `role`
- `inferences` (list[dict]): model completions with `model_name`, `completion`, `pass_rate`, `test_results`

**There is no `starter_code` or `entry_point` field.** DiffuCoder's `process_data.py` extracts function signatures from the best-inference completion. We must do the same or infer from test cases.

### Conversion strategy

Following DiffuCoder's [`recipes/process_data.py`](https://github.com/apple/ml-diffucoder/blob/main/recipes/process_data.py):

1. Load from HuggingFace: `datasets.load_dataset("TIGER-Lab/AceCode-89K", split="train")`
2. Compute per-row accuracy for `qwen_coder_2.5` and `llama3_instruct` models
3. Filter "hard" split: bottom 20th percentile by mean accuracy AND above 40th percentile by std-dev
4. For each row, optionally transform the question into a `def func(...):\n"""question"""` format (DiffuCoder duplicates both formats; we should at least support the structured version)
5. Extract `entry_point` by parsing the function name from `test_cases[0]` (e.g., `"assert add(1,2)==3"` -> `"add"`) or from the best completion's first `def` line
6. Store `test_cases` as assertion strings (new `test_format: "assertion"` field in CodeTask)
7. Build `canonical_prompt` via `build_code_task_prompt()`
8. Apply deterministic 95/5 train/dev split (seed=42)
9. Write JSONL files

### CodeTask schema extension

The existing `CodeTask` dataclass needs a new optional field:

```python
test_format: str = "args_expected"  # or "assertion"
```

When `test_format == "assertion"`, `tests` contains `list[str]` (assertion strings) instead of `list[list[Any]]` (args + expected). The execution backend must dispatch accordingly.

### Automatic test conditions

```bash
# Test 1: Converter runs without error
python dream/scripts/convert_acecode.py --output-dir dream/data/ --difficulty hard --dev-frac 0.05
echo "PASS: converter completed"

# Test 2: Output files exist and have expected scale
TRAIN_COUNT=$(wc -l < dream/data/acecode_hard_train.jsonl)
DEV_COUNT=$(wc -l < dream/data/acecode_hard_dev.jsonl)
echo "Train: $TRAIN_COUNT tasks, Dev: $DEV_COUNT tasks"
[ "$TRAIN_COUNT" -gt 5000 ] && echo "PASS: train scale OK" || echo "FAIL: expected >5000 train tasks"

# Test 3: Tasks load through registry
python -c "
from dream.src.task_registry import load_code_tasks
tasks = load_code_tasks('dream/data/acecode_hard_train.jsonl')
assert len(tasks) > 5000, f'Expected >5000, got {len(tasks)}'
assert all(t.split == 'train' for t in tasks)
assert all(t.tests for t in tasks), 'Some tasks have empty tests'
print(f'PASS: {len(tasks)} train tasks loaded, all have tests')
"

# Test 4: Reward pipeline works on sample
python dream/scripts/check_reward_pipeline.py \
  --dataset dream/data/acecode_hard_train.jsonl --dataset-split train \
  --reward execution_shaped --max-tasks 5
```

### Likely failure modes and fallbacks

| Failure mode | Symptom | Fallback |
|---|---|---|
| HuggingFace download fails (network, auth) | `ConnectionError` or `datasets` timeout | Download manually and pass `--local-path`. Document offline loading in script. |
| Filtering produces fewer tasks than expected | `<5000` hard tasks | Relax percentile thresholds or use `medium` difficulty. Log filter counts. |
| `entry_point` extraction fails for some rows | Tasks with unparseable test assertions | Skip those rows with a warning, log count. Extraction should try: regex on `assert func(`, `def func(` in completion, and `context_messages` function name. |
| `test_cases` contain non-assertion formats | Runtime errors when executing | Validate assertion format (`assert ...`) at conversion time; skip rows that don't match. |
| Memory issues loading full 87K dataset | OOM | Use `datasets.load_dataset(..., streaming=True)` or process in chunks. |

---

## Step 3: HumanEval and MBPP Task Conversion

### Goal

Convert HumanEval (164 tasks) and MBPP (378+ tasks after v0.2.0 cleanup) into `CodeTask` JSONL for **eval-only** use.

### Files to create

- `dream/scripts/convert_humaneval.py`
- `dream/scripts/convert_mbpp.py`
- `dream/src/eval_dataset_convert.py` (shared parsing + `CodeTask` row builders)
- `dream/data/humaneval.jsonl` (gitignored)
- `dream/data/mbpp.jsonl` (gitignored)

**Status**: Implemented — converters default to EvalPlus (`pip install evalplus`) or accept `--input` (HumanEval/MBPP JSONL). Offline unit tests: `pytest dream/tests/test_convert_eval_datasets.py`.

### Sources

- **HumanEval**: Available via `evalplus` package or [openai/human-eval](https://github.com/openai/human-eval). Schema: `task_id`, `prompt` (function signature + docstring), `entry_point`, `canonical_solution`, `test` (assertion code block).
- **MBPP**: Available via `evalplus` package or [google-research/mbpp](https://github.com/google-research/google-research/tree/master/mbpp). Schema: `task_id`, `text` (instruction), `code` (solution), `test_list` (list of assertion strings), `test_setup_code`.

### Conversion notes

- **Split must be `eval`** for all rows. This is the contamination barrier.
- HumanEval `test` field is a code block with assertions, not individual strings. Parse into individual assertion strings for the `assertion` test format.
- MBPP `test_list` is already individual assertion strings.
- Both datasets have `entry_point` natively.

### Automatic test conditions

```bash
# Test 1: HumanEval conversion
python dream/scripts/convert_humaneval.py --output dream/data/humaneval.jsonl
python -c "
from dream.src.task_registry import load_code_tasks
tasks = load_code_tasks('dream/data/humaneval.jsonl')
assert len(tasks) == 164, f'Expected 164, got {len(tasks)}'
assert all(t.split == 'eval' for t in tasks), 'Not all eval split'
assert all(t.entry_point for t in tasks), 'Missing entry_point'
print('PASS: 164 HumanEval tasks, all eval split')
"

# Test 2: MBPP conversion
python dream/scripts/convert_mbpp.py --output dream/data/mbpp.jsonl
python -c "
from dream.src.task_registry import load_code_tasks
tasks = load_code_tasks('dream/data/mbpp.jsonl')
assert len(tasks) >= 374, f'Expected >=374, got {len(tasks)}'
assert all(t.split == 'eval' for t in tasks)
print(f'PASS: {len(tasks)} MBPP tasks, all eval split')
"
```

### Laptop vs HPC verification

Development machines often lack `evalplus` or a full GPU stack; **Explorer HPC** is the authoritative environment for training and long eval runs. Use this split:

| Where | What to run |
|---|---|
| **Laptop / CI** | `pytest dream/tests/test_convert_eval_datasets.py` (no Hub, no `evalplus`). Optionally `pip install evalplus` and run the conversion commands above to confirm **164** HumanEval rows and **≥374** MBPP rows. |
| **HPC (rigorous)** | In the project venv used for training: `pip install evalplus`, re-run `convert_humaneval.py` / `convert_mbpp.py`, assert counts as in the snippets above. Optionally spot-check execution: load a few tasks from `humaneval.jsonl` and run `dream/scripts/check_reward_pipeline.py` (or the container backend from Step 1 once wired) so assertion strings execute under the same Python/sandbox as production. |

If laptop and HPC counts diverge, compare `evalplus` versions (`pip show evalplus`) and ensure both sides use the same `--input` file when not pulling from the EvalPlus API.

### Likely failure modes and fallbacks

| Failure mode | Symptom | Fallback |
|---|---|---|
| `evalplus` package not installed | `ModuleNotFoundError` | Install with `pip install evalplus`. Or download raw JSON from [human-eval repo](https://github.com/openai/human-eval/blob/master/data/HumanEval.jsonl.gz). |
| HumanEval `test` block has multi-line assertions that are hard to split | Incomplete assertion list | Keep as a single test block string; add `test_format: "test_block"` where the entire block is exec'd and pass = no exception. |
| MBPP v0.2.0 task count mismatch | 399 vs 378 tasks | Use `evalplus` MBPP+ which is explicitly 378. Document version. |

---

## Step 4: EvalPlus Evaluation Harness

### Goal

Build evaluation scripts that generate completions from a Dream checkpoint and score them with EvalPlus.

### Files to create

- `dream/scripts/eval_humaneval.py`
- `dream/scripts/eval_mbpp.py`
- `dream/docs/EVAL_PROTOCOL.md` *(done — normative prompt/generation/EvalPlus contract aligned with `formatting.py`, `task_registry.py`, `validate_dream.py`)*

### Design

```
[Dream Model + optional LoRA checkpoint]
         |
    load & prepare
         |
    For each HumanEval/MBPP task:
         |
    build_code_task_prompt(task)   <-- same formatter as training
         |
    model.diffusion_generate(
        input_ids, max_new_tokens=512,
        steps=512, temperature=0.2,
        top_p=0.95, alg="entropy", alg_temp=0.0
    )                              <-- Dream's generation API
         |
    extract_python_code(output)    <-- same extractor as training
         |
    Repeat N times per task (N=10 for pass@1/pass@10)
         |
    Save as {task_id: [completion_1, ..., completion_N]}
         |
    Run EvalPlus evaluation
         |
    Report pass@1, pass@10
```

### Sampling parameters

Following DiffuCoder's inference examples:
- `temperature=0.2` for pass@1 (low temp for greedy-ish)
- `temperature=0.4` for pass@10 (more diversity)
- `top_p=0.95`
- `alg="entropy"`, `alg_temp=0.0`
- `max_new_tokens=512`, `steps=512` (TOKEN_PER_STEP=1)

These should be CLI-configurable and documented in `EVAL_PROTOCOL.md`.

### Verification (laptop vs HPC)

Step 4 work is often drafted on a **laptop** without loading Dream 7B or scoring full benchmarks. That is sufficient to align prompts and APIs with source code, **not** to validate reportable pass@1 / pass@10. Before treating numbers as production-ready, run the **HPC / GPU rigorous verification** checklist in `dream/docs/EVAL_PROTOCOL.md` **§10** (environment, `validate_dream.py` smoke test, full task lists, EvalPlus backend, optional alignment with the training container). The automatic tests below assume a GPU-capable machine when they invoke the eval scripts.

### EvalPlus integration

Two options:

1. **CLI**: Generate completions to a file, run `evalplus.evaluate --dataset humaneval --samples completions.jsonl`
2. **Python API**: `from evalplus.data import get_human_eval_plus; from evalplus.evaluate import evaluate`

Option 1 is simpler and more robust. The completions file format is:

```jsonl
{"task_id": "HumanEval/0", "completion": "    return sorted(numbers)\n"}
```

Reference: [evalplus/evalplus](https://github.com/evalplus/evalplus)

### Prompt formatting alignment

DiffuCoder uses Qwen-style chat template:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
```

Dream also uses `<|im_start|>/<|im_end|>` tokens. Our `build_code_task_prompt()` should match this template. Any deviation must be documented in `EVAL_PROTOCOL.md` with rationale.

### Automatic test conditions

```bash
# Test 1: Eval script runs on base model with 2 tasks
python dream/scripts/eval_humaneval.py \
  --model Dream-org/Dream-v0-Instruct-7B \
  --max-tasks 2 --n-samples 1 \
  --output-dir dream/eval_results/test/
ls dream/eval_results/test/humaneval_completions.jsonl && echo "PASS: completions saved"

# Test 2: Completions are valid JSONL with expected fields
python -c "
import json
with open('dream/eval_results/test/humaneval_completions.jsonl') as f:
    for line in f:
        row = json.loads(line)
        assert 'task_id' in row and 'completion' in row
print('PASS: completions format valid')
"

# Test 3: EvalPlus scoring runs (may need GPU or pre-generated completions)
evalplus.evaluate --dataset humaneval --samples dream/eval_results/test/humaneval_completions.jsonl
```

### Likely failure modes and fallbacks

| Failure mode | Symptom | Fallback |
|---|---|---|
| Dream generation produces garbage at low temperature | All completions fail to parse | Increase `temperature` to 0.4; check prompt formatting matches what model was trained on. |
| Dream's `diffusion_generate` API differs from assumed interface | `AttributeError` or wrong output format | Read Dream model's `generate` method signature from HuggingFace model card. May need `model.generate()` instead. Inspect `Dream-org/Dream-v0-Instruct-7B` code. |
| EvalPlus expects specific `task_id` format | Mismatched IDs cause zero results | Use exact `"HumanEval/0"`, `"HumanEval/1"`, etc. Check EvalPlus docs for expected format. |
| OOM generating 10 completions per task on 7B model | CUDA OOM | Generate sequentially (batch_size=1), use gradient checkpointing off, `torch.no_grad()`. Or reduce `max_new_tokens` to 256. |
| EvalPlus sandboxed execution fails on HPC | Permission errors, missing Docker | EvalPlus can run without Docker (uses subprocess). Set `EVALPLUS_BACKEND=unsafe` for HPC (document security implications). |

---

## Step 5: Wire Container Backend into Training Loop

### Goal

Make training scripts optionally use containerized execution.

### Modifications

- `dream/scripts/run_dream_comparison.py`: add `--execution-backend` flag (`subprocess` | `container`)
- `dream/scripts/single_step_dream.py`: same flag
- `dream/src/rewards.py`: `build_reward_function()` accepts optional `backend` kwarg
- `run_dream_comparison.sh`: add Apptainer setup block

### Environment variables for HPC

```bash
# In sbatch script:
export DREAM_SANDBOX_RUNTIME=apptainer
export DREAM_SANDBOX_SIF=/projects/$GROUP/container_images/dream-sandbox.sif
```

### Automatic test conditions

```bash
# Test 1: Subprocess backend still works (backward compat)
python dream/scripts/check_reward_pipeline.py \
  --dataset dream/data/code_grpo_train.sample.jsonl --dataset-split train \
  --reward execution_shaped --execution-backend subprocess --max-tasks 3
# Should print nonzero rewards

# Test 2: Container backend works locally (requires Docker)
python dream/scripts/check_reward_pipeline.py \
  --dataset dream/data/code_grpo_train.sample.jsonl --dataset-split train \
  --reward execution_shaped --execution-backend container --max-tasks 3
# Should print same rewards as subprocess

# Test 3: Results match between backends
python dream/tests/test_backend_parity.py
# Runs same tasks through both backends, asserts rewards are equal
```

### Likely failure modes and fallbacks

| Failure mode | Symptom | Fallback |
|---|---|---|
| Container startup adds >2s latency per reward call | Training 10x slower | Batch execution: send multiple tasks to one container invocation. Or pool containers. |
| Apptainer bind-mount permissions | `EACCES` on temp files | Use `/scratch/$USER/` for temp files, ensure bind mounts are read/write. |
| Different Python versions inside/outside container | Assertion results differ | Pin `python:3.11-slim` in Dockerfile, document Python version requirement. |

---

## Step 6: End-to-End Training Validation at Scale

### Goal

Validate that training works with AceCode at realistic scale before committing to multi-day runs.

### Validation matrix

| Run | Data | Tasks | Epochs | Backend | Expected outcome |
|---|---|---|---|---|---|
| Flat GRPO baseline | AceCode hard train | 100 | 1 | container | Non-zero mean reward, reward improves over steps |
| Tree GRPO | AceCode hard train | 100 | 1 | container | Diversity metrics nonzero, reward comparable to flat |
| HumanEval eval (base) | HumanEval | 164 | N/A | EvalPlus | Baseline pass@1 number |
| HumanEval eval (post-flat) | HumanEval | 164 | N/A | EvalPlus | pass@1 >= baseline |

### Automatic test conditions

```bash
# Run 1: Flat GRPO
python dream/scripts/run_dream_comparison.py \
  --phase grpo_lora_baseline \
  --dataset dream/data/acecode_hard_train.jsonl --dataset-split train \
  --reward execution_shaped --execution-backend container \
  --max-tasks 100 --num_epochs 1 --lora --no_wandb
# Check: stdout shows reward progression, no NaN, no OOM

# Run 2: Tree GRPO  
python dream/scripts/single_step_dream.py \
  --lora --dataset dream/data/acecode_hard_train.jsonl --dataset-split train \
  --reward execution_shaped --execution-backend container \
  --max-tree-nodes 8 --max-new-tokens 128 --task-index 0
# Check: diversity metrics in output, nonzero rewards

# Run 3: Eval pipeline
python dream/scripts/eval_humaneval.py \
  --model Dream-org/Dream-v0-Instruct-7B \
  --max-tasks 10 --n-samples 1 \
  --output-dir dream/eval_results/baseline/
# Check: produces pass@1 numbers
```

### Likely failure modes and fallbacks

| Failure mode | Symptom | Fallback |
|---|---|---|
| OOM with 7B model + AceCode + LoRA + tree | CUDA OOM | Reduce `max_tree_nodes` to 3, `max_new_tokens` to 64. Or use gradient checkpointing. |
| All AceCode rewards are 0.0 | Assertion execution fails systematically | Check that assertion-string mode works in the container. Run `check_reward_pipeline.py` on 5 AceCode tasks first. |
| Training loss is NaN after first epoch | Gradient explosion | Check advantage clipping (D-014), reduce learning rate, verify reward normalization. |
| Wall-clock time prohibitive (>1h for 100 tasks) | Training too slow | Profile: is it execution latency or model forward? If execution, batch more aggressively or use subprocess backend. |

---

## Transferability: Open-R1 and CoT

### Transferability to Open-R1

Our infrastructure is designed to eventually plug into the [Open-R1](https://github.com/huggingface/open-r1) / TRL `GRPOTrainer` ecosystem:

**What transfers directly**:
- `ExecutionBackend` ABC can be wrapped into Open-R1's reward function signature: `(completions: list[list[dict]], **kwargs) -> list[float]`
- `CodeTask` JSONL is close to Open-R1's dataset format (they use `context_messages` as the key)
- Container sandbox is runtime-agnostic; Open-R1 already supports E2B and Piston

**What would need adaptation**:
- Open-R1 uses TRL `GRPOTrainer` which expects HuggingFace `PreTrainedModel`. Dream's diffusion generation is non-standard. DiffuCoder solved this by extending `GRPOTrainer` in `coupled_grpo.py`.
- Our `EntropyMCTSTrainer` is a custom trainer, not a TRL subclass. To run entropy-tree GRPO inside Open-R1, we'd need to either (a) port tree logic into a TRL trainer subclass, or (b) keep our custom trainer and just share the reward/data infrastructure.

**Recommended path**: Share reward/data/eval infrastructure. Keep trainers separate until entropy-tree results justify the engineering investment of a TRL integration.

### Transferability to CoT / Reasoning (d1-style)

The [d1 project](https://github.com/dllm-reasoning/d1) demonstrates diffu-GRPO for reasoning tasks on LLaDA. Our infrastructure transfers to CoT with these changes:

**What transfers directly**:
- `ExecutionBackend` ABC (replace code execution with answer verification)
- `CodeTask` schema (generalize to `Task` with a `domain` field)
- `task_registry.py` loaders (same JSONL format, different content)
- Container sandbox (reuse for any code-based verification)
- EvalPlus-style evaluation scripts (swap for math evaluation)

**What needs new implementation**:
- **Reward function**: Replace `ExecutionShapedReward` with `VerifierReward` that checks final-answer correctness (see `research_decisions.md` D-025 through D-030)
- **Task sources**: GSM8K, MATH500, Countdown (d1's evaluation suite)
- **Prompt template**: CoT prompts need `<think>` / `<answer>` tags (Open-R1 style) or equivalent
- **Log-probability estimation**: d1 uses mean-field approximation for masked dLLMs; our Dream adapter computes log-probs differently. This is a deep integration point.

**Key design principle for transferability**: Keep the `ExecutionBackend` / reward interface generic. The reward function should accept `(completion: str, task: Task) -> float` regardless of domain. Domain-specific logic lives in reward subclasses, not in the backend.

### Design for future extensibility

```python
# Generic interface that works for code AND reasoning:
class RewardFunction(ABC):
    def __call__(self, completion: str, prompt: str, task: Optional[Task] = None) -> float: ...

class ExecutionCodeReward(RewardFunction):  # Current
    """Runs code completion against test cases."""

class VerifierReward(RewardFunction):  # Future, for CoT
    """Checks final answer against ground truth."""

class JudgeReward(RewardFunction):  # Future, optional
    """LLM-as-a-judge scoring."""
```

This is already close to the existing `dream/src/rewards.py` ABC. The main work is ensuring `Task` is generic enough (not `CodeTask`-only) and that data loaders handle multiple domains.

---

## Research Decisions

The following should be updated in `research_decisions.md`:

### D-018: Primary Training Task Source

**Resolution**: AceCode-89K hard split (~15K tasks after filtering).  
**Rationale**: Direct alignment with DiffuCoder's training data. Well-tested, diverse, large enough for meaningful GRPO.  
**Future options**: APPS train split, CodeContests, or a union of multiple sources.

### D-020: Execution Backend

**Resolution**: Docker container (local) / Apptainer (Explorer HPC).  
**Rationale**: Free, good isolation, works on HPC without API keys. E2B remains a fallback if throughput is insufficient.  
**HPC constraint**: Explorer cluster supports Apptainer (system-wide), not Docker. See [Explorer Apptainer docs](https://rc-docs.northeastern.edu/en/explorer-main/containers/apptainer.html).

### D-011 / D-024: Evaluation Benchmarks

**Resolution**: HumanEval+ and MBPP+ via EvalPlus as primary. Report pass@1 (low temp) and pass@10 (higher temp).  
**Rationale**: Standard in the field. DiffuCoder reports EvalPlus results. Direct comparability.

---

## Delegation Guide

### Safe for weaker agents (well-scoped, independently verifiable)

| Step | Task | Input | Output | Verification |
|---|---|---|---|---|
| 1 (partial) | Write `Dockerfile` + `entrypoint.py` | This document, section on Docker image design | `dream/sandbox/Dockerfile`, `dream/sandbox/entrypoint.py` | `docker build` succeeds; test commands in Step 1 pass |
| 2 | Write `convert_acecode.py` | This document, DiffuCoder's `process_data.py`, AceCode HF page | `dream/scripts/convert_acecode.py` | Automatic tests in Step 2 pass |
| 3 | Write `convert_humaneval.py` and `convert_mbpp.py` | This document, EvalPlus docs | Converters + `eval_dataset_convert.py`; `pytest dream/tests/test_convert_eval_datasets.py`; full JSONL counts on HPC with `evalplus` (see Step 3 “Laptop vs HPC”) |
| 4 (partial) | Write `EVAL_PROTOCOL.md` | This document, DiffuCoder README | `dream/docs/EVAL_PROTOCOL.md` | **Done** — reviewed against `dream/src/formatting.py`, `task_registry.py`, `validate_dream.py` |

### Needs careful attention (do yourself or review closely)

| Step | Task | Why |
|---|---|---|
| 1: `execution_backends.py` + rewards integration | Core interface change, backward compat, dual test-format support | 
| 2: `CodeTask` schema extension for `test_format` | Affects all downstream consumers of `task_registry.py` |
| 4: `eval_humaneval.py` generation loop | Dream's `diffusion_generate` API is non-standard; need to verify exact call signature |
| 5: Wiring container into training | Latency/throughput implications for GRPO; Slurm script changes |
| 6: Scale validation and interpretation | Requires GPU; results determine if pipeline works end-to-end |

---

## External Resources

### Primary references

- **DiffuCoder repo**: https://github.com/apple/ml-diffucoder
  - `recipes/process_data.py` — AceCode filtering logic
  - `src/open_r1/rewards.py` — `code_reward()` and `get_code_format_reward()`
  - `src/open_r1/utils/code_providers.py` — `CodeExecutionProvider` ABC, `E2BProvider`
  - `README.md` — full setup, data prep, training commands
- **DiffuCoder paper**: https://arxiv.org/abs/2506.20639
- **AceCode-89K dataset**: https://huggingface.co/datasets/TIGER-Lab/AceCode-87K
- **AceCoder paper**: https://arxiv.org/abs/2502.01718
- **AceCoder repo**: https://github.com/TIGER-AI-Lab/AceCoder

### Evaluation

- **EvalPlus**: https://github.com/evalplus/evalplus — HumanEval+ and MBPP+ with 80x/35x more tests
- **EvalPlus PyPI**: https://pypi.org/project/evalplus/ (v0.3.1, CLI and Python API)
- **bigcode-evaluation-harness**: https://github.com/bigcode-project/bigcode-evaluation-harness — broader code eval with Docker
- **Qwen2.5-Coder eval**: https://github.com/QwenLM/Qwen2.5-Coder/tree/main/qwencoder-eval — DiffuCoder's cited eval tool
- **HumanEval**: https://github.com/openai/human-eval (164 tasks)
- **MBPP**: https://github.com/google-research/google-research/tree/master/mbpp

### GRPO frameworks

- **Open-R1**: https://github.com/huggingface/open-r1 — DiffuCoder's upstream; TRL `GRPOTrainer`
- **d1**: https://github.com/dllm-reasoning/d1 — diffu-GRPO for reasoning on LLaDA
- **TRL GRPOTrainer docs**: https://huggingface.co/docs/trl/v0.21.0/grpo_trainer

### Execution sandboxing

- **E2B**: https://e2b.dev — cloud code sandbox (DiffuCoder's choice)
- **Piston**: https://github.com/engineer-man/piston — self-hosted execution engine
- **Apptainer on Explorer**: https://rc-docs.northeastern.edu/en/explorer-main/containers/apptainer.html
- **Explorer storage quotas**: https://rc-docs.northeastern.edu/en/explorer-main/best-practices/homequota.html

### Models

- **Dream 7B Instruct**: https://huggingface.co/Dream-org/Dream-v0-Instruct-7B
- **DiffuCoder 7B cpGRPO**: https://huggingface.co/apple/DiffuCoder-7B-cpGRPO

### Related datasets for future scaling

- **APPS**: https://huggingface.co/datasets/codeparrot/apps — large code dataset with tests
- **CodeContests**: https://github.com/google-deepmind/code_contests — competitive programming
- **LiveCodeBench**: temporal/contamination-aware code eval

---

## Implementation Order

```
Step 1: Containerized Execution Sandbox
  |
  v
Step 2: AceCode-89K Data Ingestion        Step 3: HumanEval/MBPP Conversion
  |                                                |
  v                                                v
Step 5: Wire Container into Training       Step 4: EvalPlus Evaluation Harness
  |                                                |
  v                                                v
Step 6: End-to-End Training Validation at Scale
  |
  v
[Full GRPO training runs]
  |
  v
[Transfer to CoT/reasoning -- future phase]
```

Steps 2+3 and Steps 4+5 can be parallelized. Step 1 should be completed first as both training and eval benefit from the container.
