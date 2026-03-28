# Evaluation protocol — HumanEval+ / MBPP+ (EvalPlus)

**Project**: Entropy-Guided MCTS-GRPO for diffusion language models (Dream stack)  
**Purpose**: Single reference for how we generate completions from a Dream checkpoint and score them with [EvalPlus](https://github.com/evalplus/evalplus), aligned with training (`dream/src/formatting.py`, `dream/src/task_registry.py`) and with DiffuCoder-style reporting.

**Related**: `dream/PLAN_03_ENVIRONMENT_SCALEUP.md` (Step 4), `research_decisions.md` (D-011, D-024, **D-031** training vs DiffuCoder-eval prompt mismatch), `dream/scripts/validate_dream.py` (reference `diffusion_generate` usage), `dream/src/eval_prompts.py` (DiffuCoder Tables 5–6 templates for literature-aligned runs).

**Verification scope**: This protocol was **authored and reviewed against static code** (formatting, task registry, `validate_dream.py`) on a **typical laptop**, without loading Dream 7B or running full HumanEval+/MBPP+ EvalPlus jobs. Treat §10 as the checklist for **rigorous validation on the target GPU cluster** (e.g. Northeastern Explorer).

---

## 1. Metrics

| Metric | Meaning |
|--------|---------|
| **pass@1** | Fraction of tasks where at least one completion passes all tests, with **one** sample per task (low-temperature generation). |
| **pass@10** | Same, with **ten** samples per task (higher temperature for diversity). |

Primary benchmarks: **HumanEval+** and **MBPP+** via EvalPlus (expanded test suites vs. original benchmarks). This matches DiffuCoder’s evaluation story and keeps numbers comparable to the literature.

---

## 2. End-to-end pipeline

1. Load tasks as `CodeTask` rows (e.g. `humaneval.jsonl` / `mbpp.jsonl` from the conversion scripts; see `dream/PLAN_03_ENVIRONMENT_SCALEUP.md` Step 3).
2. Choose a **prompting regime** (see §3):
   - **Training-aligned** (debugging / reward parity): build the user string with `build_code_task_prompt()` + `tokenizer.apply_chat_template(..., add_generation_prompt=True)`.
   - **DiffuCoder-aligned** (literature comparison): use `dream/src/eval_prompts.py` — full string from system through assistant prefill ending in `` ```python\n`` — and tokenize that string **directly** (no `apply_chat_template`), matching DiffuCoder / `qwencoder-eval` (see `research_decisions.md` D-031).
3. For DiffuCoder-aligned HumanEval, the `{prompt}` slot is **`task.starter_code`** (EvalPlus `prompt` field). For MBPP, it is **`task.instruction`** (task description text).
4. Call Dream’s **`diffusion_generate`** on the tokenized prompt (see §4).
5. Decode only the **new** tokens after the prompt; strip chat special tokens if needed.
6. Extract executable Python: for DiffuCoder-aligned runs use **`extract_diffucoder_completion`** in `eval_prompts.py` (fenced `` ```python `` block); for training-style prompts use **`extract_python_code`** / **`normalize_completion_for_reward`** as in rewards code.
7. Write **JSONL** completions in EvalPlus format (see §5).
8. Run **EvalPlus** (`evalplus.evaluate` CLI or Python API) and record pass@1 / pass@10.

**Driver scripts** (DiffuCoder-aligned): `dream/scripts/eval_humaneval.py`, `dream/scripts/eval_mbpp.py` — shared loop in `dream/src/eval_generate.py`. Optional `--run-evalplus` invokes `evalplus.evaluate` after writing JSONL.

---

## 3. Prompt formatting

### 3.1 Training / reward stack (match `formatting.py`)

Training and reward pipelines build the **user** content with `build_code_task_prompt()` in `dream/src/formatting.py`:

- If `starter_code` is non-empty:  
  `instruction` + a short suffix + fenced block `` ```{language} `` … starter … `` ``` ``.
- If empty: `instruction` only.

Dataset rows should set **`canonical_prompt`** to this string (or rely on `task_registry` to compute it from `instruction` + `starter_code` + `language`).

### 3.2 Chat template (Qwen-style / Dream) — training default

Dream Instruct uses a **chat template** with `<|im_start|>` / `<|im_end|>` (Qwen-style).

**Canonical pattern** (same as `dream/scripts/validate_dream.py` and the tree builder):

```python
messages = [{"role": "user", "content": task.canonical_prompt}]
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    return_dict=True,
    add_generation_prompt=True,
)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs.get("attention_mask")
if attention_mask is None:
    attention_mask = torch.ones_like(input_ids, device=device)
else:
    attention_mask = attention_mask.to(device)
```

### 3.3 DiffuCoder-aligned eval (literature comparison)

For **comparable numbers to DiffuCoder** (paper Tables 5–6), do **not** use `apply_chat_template` alone. Instead build the **full** string with `build_humaneval_prompt` / `build_mbpp_prompt` in `dream/src/eval_prompts.py`:

- **System**: `You are a helpful assistant.`
- **User**: HumanEval includes `Please complete the following problem:` plus a fenced copy of `{prompt}`; MBPP is only a fenced `{prompt}` (task text).
- **Assistant prefill** (part of the **input** tokens): `Here is the code to solve this problem:\n` + `` ```python\n`` so generation continues inside the fence.

Tokenize with `tokenizer(full_prompt_string, return_tensors="pt", add_special_tokens=False)` (see `eval_generate.py`).

**Important**: This differs from §3.1–§3.2 training prompts. See `research_decisions.md` **D-031** — training may omit the system message and use a different user layout; document which regime each experiment uses.

---

## 4. Generation API (`diffusion_generate`)

Dream exposes **`model.diffusion_generate`** (not HuggingFace `generate` for the main diffusion path). Reference implementation: `dream/scripts/validate_dream.py`.

Typical keyword arguments:

| Argument | Role | Default for eval (plan) |
|----------|------|-------------------------|
| `max_new_tokens` | Response length cap | **512** (TOKEN_PER_STEP=1 ⇒ `steps` matches; see below) |
| `steps` | Denoising steps | **512** (match `max_new_tokens` for standard Dream usage) |
| `temperature` | Sampling temperature | **0.2** for pass@1; **0.4** for pass@10 |
| `top_p` | Nucleus sampling | **0.95** |
| `alg` | Denoising schedule / algorithm flag | ** `"entropy"` ** |
| `alg_temp` | Auxiliary temperature for `alg` | **0.0** |
| `attention_mask` | Same length as `input_ids` | Required / use tokenizer output |
| `return_dict_in_generate` | Get `.sequences` | **True** |

Example:

```python
with torch.no_grad():
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=512,
        steps=512,
        temperature=0.2,
        top_p=0.95,
        alg="entropy",
        alg_temp=0.0,
        return_dict_in_generate=True,
    )
prompt_len = input_ids.shape[1]
gen_ids = output.sequences[0][prompt_len:]
text = tokenizer.decode(gen_ids, skip_special_tokens=True)
```

**LoRA / PEFT**: If evaluating an adapter, load base weights + adapter per your training checkpoint convention and run the same call on the merged or wrapped model.

**Config alignment**: `dream/src/config.py` `MCTSConfig` uses `temperature=0.2`, `top_p=0.95`, `alg="entropy"`, `alg_temp=0.0`, with `max_new_tokens` / `total_denoising_steps` often 256 for *training* trees. For **EvalPlus parity with the environment plan**, prefer **512 / 512** for full-benchmark runs unless you are doing a quick smoke test (then document the shorter settings).

---

## 5. Code extraction

- **DiffuCoder-aligned eval** (`eval_prompts.py`): use **`extract_diffucoder_completion`** — matches DiffuCoder-style fenced `` ```python ... ``` `` extraction (handles unclosed fences and special tokens like `<|im_end|>`, `<|dlm_pad|>`).
- **Training / reward-style prompts**: use **`extract_python_code`** in `dream/src/formatting.py` — prefer fenced blocks; else slice from the first line that looks like top-level Python (`def`, `class`, `import`, etc.).

For rewards that need a single function body, **`normalize_completion_for_reward(..., entry_point=task.entry_point)`** trims to `def <entry_point>(` when possible.

---

## 6. EvalPlus: completions file and scoring

### 6.1 JSONL format

Each line is one JSON object. Minimal fields:

```json
{"task_id": "HumanEval/0", "completion": "    def ...\n"}
```

- **`task_id`**: Must match EvalPlus expectations, e.g. `"HumanEval/0"` … `"HumanEval/163"` for HumanEval+; MBPP+ uses its own IDs (see EvalPlus docs / dataset).
- **`completion`**: Typically the **function body or full solution string** as produced after extraction — follow EvalPlus examples for the exact convention (often indentation matters).

### 6.2 CLI (recommended in the environment plan)

After `pip install evalplus`:

```bash
evalplus.evaluate --dataset humaneval --samples path/to/humaneval_completions.jsonl
# MBPP:
evalplus.evaluate --dataset mbpp --samples path/to/mbpp_completions.jsonl
```

Use `evalplus.evaluate --help` for backend flags and dataset names for your installed version.

**Full coverage**: `evalplus.evaluate` expects the samples JSONL to include **every** task in that benchmark (e.g. all 164 HumanEval+ tasks for pass@1, or 164×10 lines for pass@10). A partial file (smoke test) raises `AssertionError: Missing problems in samples`. For subset evaluation, EvalPlus supports overriding the problem set via an environment variable (see [evalplus#21](https://github.com/evalplus/evalplus/issues/21)); otherwise use the full-batch driver `eval_base_dream_evalplus.sbatch` at the repo root.

### 6.3 Python API

```python
from evalplus.data import get_human_eval_plus  # or MBPP
from evalplus.evaluate import evaluate
# See EvalPlus repository for the exact evaluate() signature in your version.
```

### 6.4 Execution backend (local vs HPC)

EvalPlus can run tests in a sandbox. On clusters where Docker is unavailable or permissions are tight, you may need **`EVALPLUS_BACKEND=unsafe`** (or the flag your EvalPlus version documents). **Security**: `unsafe` runs code as the current user without container isolation — acceptable only on dedicated eval nodes with resource limits; document any use in run logs.

---

## 7. Reproducibility checklist

- [ ] Record: model id, checkpoint path, LoRA path (if any), commit hash, `pip freeze` or env name.
- [ ] Record: `max_new_tokens`, `steps`, `temperature`, `top_p`, `alg`, `alg_temp`.
- [ ] Confirm which prompt regime you used: **training-aligned** (`canonical_prompt` + `apply_chat_template`) vs **DiffuCoder-aligned** (`eval_prompts.py` + direct tokenization).
- [ ] Save JSONL completions next to EvalPlus stdout / JSON results.
- [ ] Note EvalPlus package version (`pip show evalplus`).

---

## 8. Operational notes

- **GPU memory**: Use `torch.no_grad()`, batch size 1, sequential tasks if you hit OOM; reduce `max_new_tokens` / `steps` only for smoke tests and note it.
- **HuggingFace cache**: Scripts such as `validate_dream.py` set `HF_HOME` under scratch when available; use the same on HPC to avoid home-quota issues.
- **Failures**: If pass rates are near zero, verify prompt template first, then temperature; see “Likely failure modes” in `dream/PLAN_03_ENVIRONMENT_SCALEUP.md` Step 4.

---

## 9. File map (code)

| Item | Location |
|------|----------|
| Training user prompt | `dream/src/formatting.py` — `build_code_task_prompt` |
| DiffuCoder-aligned eval templates | `dream/src/eval_prompts.py` — `build_humaneval_prompt`, `build_mbpp_prompt`, `extract_diffucoder_completion` |
| Shared eval generation + JSONL | `dream/src/eval_generate.py` |
| Tasks | `dream/src/task_registry.py` — `CodeTask`, `load_code_tasks` |
| Reference generation | `dream/scripts/validate_dream.py` |
| Eval drivers | `dream/scripts/eval_humaneval.py`, `dream/scripts/eval_mbpp.py` |

Normative for **DiffuCoder-aligned** runs: §3.3, §4, §5, and `eval_generate.py`. Training-aligned smoke tests follow §3.1–§3.2.

---

## 10. Laptop limits and rigorous HPC verification (later)

### 10.1 What a laptop can and cannot prove

| Aspect | On a laptop (typical) | On HPC / large GPU |
|--------|------------------------|-------------------|
| Protocol correctness vs **code** | Yes — compare this doc to `formatting.py`, `task_registry.py`, `validate_dream.py` | Same |
| **End-to-end** Dream 7B + `diffusion_generate` + JSONL | Often **no** — VRAM and wall time block full benchmarks | Yes — interactive or batch job |
| **EvalPlus** on all tasks | May be slow or impractical without a GPU; optional CPU smoke | Run full HumanEval+ / MBPP+ scoring |
| **Numerical parity** with published pass@k | No — needs the real eval run | Yes — record metrics and seeds |

Do not treat “doc matches source” as “eval pipeline is bug-free”; reserve that for §10.2.

### 10.2 HPC verification checklist (recommended before trusting numbers)

Run these on the **same environment** you use for training when possible (CUDA, PyTorch, `transformers`, `evalplus` versions pinned).

1. **Environment**
   - [ ] Activate the project conda/venv; `pip freeze` or `conda env export` saved next to results.
   - [ ] Set `HF_HOME` (and related caches) to **scratch** or project storage — see `dream/HPC_SYNC.md` and `validate_dream.py`-style cache setup — so downloads do not fill home quota.

2. **Model load smoke test**
   - [ ] `python dream/scripts/validate_dream.py` (or equivalent) completes: `diffusion_generate` runs without error on at least one prompt.

3. **Eval driver smoke test** (once `eval_humaneval.py` / `eval_mbpp.py` exist)
   - [ ] `--max-tasks 2 --n-samples 1` produces valid JSONL per §5 and §6.
   - [ ] `evalplus.evaluate` exits 0 on that file.

4. **Full benchmark** (reportable numbers)
   - [ ] Full task list for HumanEval+ and/or MBPP+ (no `--max-tasks` cap unless documented as a subset run).
   - [ ] pass@1 run: `temperature=0.2` (or the value you lock for the paper).
   - [ ] pass@10 run: `temperature=0.4`, `n-samples=10`, separate completions file or documented sampling.
   - [ ] Store: JSONL completions, EvalPlus stdout/log, git commit, and CLI flags.

5. **Execution backend for EvalPlus on cluster**
   - [ ] Confirm EvalPlus test execution works (Docker, subprocess, or `EVALPLUS_BACKEND=unsafe` per your site policy). If using `unsafe`, run only on **dedicated batch nodes**, note it in the experiment log, and see §6.4.

6. **Optional alignment with training sandbox**
   - [ ] If you also report execution-backed rewards from the **container** path (`dream/PLAN_03_ENVIRONMENT_SCALEUP.md` Step 1), you may spot-check a few tasks with both EvalPlus scores and `ExecutionShapedReward` — they measure different things, but gross mismatches flag extraction or `task_id` bugs.

### 10.3 Suggested batch job shape (Explorer-style)

- Request a GPU partition and adequate time for 164+ tasks × multiple samples (pass@10 is much slower than pass@1).
- Use `srun`/`sbatch` with the same modules you use for GRPO training.
- Prefer **sequential** task loops with `batch_size=1` for generation unless you implement batched diffusion (reduces OOM risk).

Document the actual Slurm script path or example command in your experiment notes when you add HPC automation; this protocol stays environment-agnostic.
