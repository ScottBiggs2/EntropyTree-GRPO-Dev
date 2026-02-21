# EntropyTree-GRPO

Entropy-guided MCTS for GRPO training of diffusion language models (Qwen2.5 0.5B Instruct MDLM). See `scaffold_plan.md` and `research_decisions.md` for design and choices.

**Implementation note:** Node entropy is the mean Shannon entropy over **masked positions only** (same-position logits). Tree expansion caps unmasking per chunk so each child keeps at least one masked token; otherwise entropy would be 0 and frontier selection would be uninformative.

## Environment setup

**If you see `zsh: command not found: pip`** — the conda env was likely created without Python, or `pip` isn’t on PATH. Use the steps below so the env has Python and use `python -m pip` instead of `pip`.

### Option 1: No dllm package (recommended — avoids circular import)

Install only the stack; load the model from a **local directory** that uses our dllm-free custom code so the HuggingFace loader never resolves `import dllm`. From the repo root:

```bash
conda create -n EntropyTreeGRPO_env python=3.10 -y   # or 3.11
conda activate EntropyTreeGRPO_env
cd /path/to/EntropyTree-GRPO
python -m pip install -r requirements.txt
python scripts/prepare_local_model.py --out-dir ./model_cache
```

Then run validation with the local path:

```bash
USE_LOCAL_MODEL_CODE=1 LOCAL_MODEL_PATH=./model_cache python scripts/validate_model.py
```

Our `model_custom_code/modeling_qwen2.py` is a copy of the model's custom code with the `if __name__ == "__main__"` block removed (no `import dllm`). The script above downloads the model and overwrites the modeling file with this copy. For `entropy_profile.py`, use the same env vars. For `tree_viz.py` and `single_step_train.py`, set `model_name_or_path` to your local path (e.g. `model_cache`) via config or CLI.

### Option 2: Clone dllm and install editable (Option A + D)

If you need the full dllm package (e.g. for dllm samplers or training recipes), use the [dLLM GitHub](https://github.com/ZHZisZZ/dllm) recommended setup and avoid the yanked transformers pin:

```bash
conda create -n EntropyTreeGRPO_env python=3.10 -y
conda activate EntropyTreeGRPO_env
python -m pip install -r requirements.txt
git clone https://github.com/ZHZisZZ/dllm.git
python -m pip install -e ./dllm --no-deps
```

Do **not** use `pip install git+https://github.com/ZHZisZZ/dllm.git`: that pulls `transformers==4.57.0` (yanked) and the full training stack.

### Option 3: Fresh conda env (minimal)

```bash
conda deactivate
conda env remove -n EntropyTreeGRPO_env 2>/dev/null || true
conda create -n EntropyTreeGRPO_env python=3.10 -y
conda activate EntropyTreeGRPO_env
python -m pip install -r requirements.txt
python -c "import torch; import transformers; print('OK')"
```

Then use Option 1 (prepare_local_model + USE_LOCAL_MODEL_CODE) to run scripts that load the model.

### Why not `pip install dllm` from git?

The HuggingFace model's custom code contains `import dllm` (inside an `if __name__` block); the loader still resolves it and can hit a circular import. There is no dllm on PyPI—only the [dLLM GitHub repo](https://github.com/ZHZisZZ/dllm). We avoid that by either (1) loading from a local dir with our dllm-free modeling file (Option 1), or (2) installing dllm from a clone with `pip install -e ./dllm --no-deps` (Option 2).

### NumPy note

We pin `numpy>=1.24,<2` in `requirements.txt` to avoid conflicts with older sklearn/pandas. If you still see NumPy 2.x errors, force reinstall: `python -m pip install "numpy>=1.24,<2" --force-reinstall`.

## Run tests (no model download)

From the repo root (with the conda env activated):

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

You should see 40 tests pass. No model download required (tests use mocks where needed).

## Scripts (require model download)

- **Prepare local model (Option 1)**: `python scripts/prepare_local_model.py --out-dir ./model_cache` — downloads the model and overwrites the modeling file with our dllm-free copy.
- **Validate model load + generate**: `python scripts/validate_model.py` with Option 1, use `USE_LOCAL_MODEL_CODE=1 LOCAL_MODEL_PATH=./model_cache python scripts/validate_model.py`
- **Terminal chat**: `USE_LOCAL_MODEL_CODE=1 LOCAL_MODEL_PATH=./model_cache python scripts/chat.py` — interactive chat in the terminal (type "quit" or "exit" to stop).
- **Entropy profile**: `python scripts/entropy_profile.py`
- **Tree visualization**: `python scripts/tree_viz.py`
- **Single training step**: `python scripts/single_step_train.py`
- **Phase 8 experiment**: `python scripts/run_experiment.py --method baseline --num_epochs 2` or `--method entropy_mcts`. Logs to WandB if `WANDB_API_KEY` is set; use `--no_wandb` to disable. Checkpoints go to `checkpoints/baseline_grpo/` and `checkpoints/entropy_mcts_grpo/` (see CONTRIBUTING / .gitignore).

Run from repo root. Model: `dllm-collection/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1`, or a local path after running `prepare_local_model.py`.

## Real-model verification (Phase 7.5)

Passing the 40 tests is not the same as running with the real HuggingFace model. Before Phase 8, run these four scripts with the model downloaded and confirm each outcome. Fix any shape/device/tokenizer bugs here.

From repo root with env activated and `PYTHONPATH=.` set (or run from repo root and use `PYTHONPATH=. python ...`):

| Step | Command | Expect |
|------|---------|--------|
| 1 | `python scripts/validate_model.py` (or with Option 1: `USE_LOCAL_MODEL_CODE=1 LOCAL_MODEL_PATH=./model_cache python scripts/validate_model.py`) | No errors; param count ~630M; logits shape printed; `mask_token_id` set; generated snippet looks like code. |
| 2 | `python scripts/entropy_profile.py` | Entropy at step 0 high; entropy decreases after 32/64/128 steps. |
| 3 | `python scripts/tree_viz.py` | Tree builds; node count and leaves printed; no OOM/crash. |
| 4 | `python scripts/single_step_train.py` | Completes; loss finite and non-NaN; metrics printed; time &lt; 5 min on M1. |

One-shot: `python scripts/verify_real_model.py` runs all four steps and reports pass/fail (exits non-zero if any step fails).

## Layout

- `src/` — config, entropy, time weight, tree node, tree builder, rewards, advantages, loss, trainer
- `tests/` — unit and integration tests (mock model where possible)
- `scripts/` — validation, entropy profile, tree viz, single-step train, verify_real_model (Phase 7.5 one-shot)
- `model_custom_code/` — dllm-free modeling file used when loading from a local model dir

## Before pushing to a remote

- **Do not commit:** model weights, `model_cache/`, `.env` or any files with API keys or secrets, virtual/conda env dirs, or large generated outputs. These are ignored via `.gitignore`.
- **Keep the repo light:** Clone the model separately with `prepare_local_model.py` after cloning the repo. Do not add `dllm/` unless you need it for development; it is gitignored by default.
- After adding a remote: `git remote add origin <url>`, then `git push -u origin main` (or your branch). Run `python scripts/verify_real_model.py` locally before pushing to confirm tests and real-model steps pass.

---

## Quick repro (copy-paste)

```bash
conda deactivate
conda env remove -n EntropyTreeGRPO_env 2>/dev/null || true
conda create -n EntropyTreeGRPO_env python=3.10 -y
conda activate EntropyTreeGRPO_env
cd /path/to/EntropyTree-GRPO
python -m pip install -r requirements.txt
PYTHONPATH=. python -m pytest tests/ -v
```
