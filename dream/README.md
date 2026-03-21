## Dream Subdirectory Overview

This `dream/` folder contains a **self-contained substack** for running
entropy-guided MCTS-GRPO on **Dream 7B** (and MDLM) without disturbing the
original toy MDLM implementation in `src/`.

**Starting point**: [Dream-org/Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) — the instruct-tuned 7B diffusion LLM (no RL). We apply our entropy-MCTS-GRPO on top for code; see the [model card](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) and [Dream blog](https://hkunlp.github.io/blog/2025/dream/) for details.

Use this directory when:

- you want to migrate the method to Dream 7B
- you want corrected entropy/time weighting and adaptive branching
- you are running on a **cloud GPU** (for real Dream weights)

The original MDLM stack in `src/` remains the reference baseline.

---

```bash 
conda create -n EntropyTreeGRPO_Dream_env python=3.10 -y   # Must be 3.10 for dLLM compatability
conda activate EntropyTreeGRPO_Dream_env
cd /path/to/EntropyTree-GRPO-Dream
python -m pip install -r requirements.txt
```

Scratch: 
```bash
conda create -n EntropyTreeGRPO_Dream_env python=3.10 -y   # Must be 3.10 for dLLM compatability

conda create --prefix /scratch/biggs.s/conda_envs/EntropyTreeGRPO_Dream_env --clone /home/biggs.s/miniconda/envs/EntropyTreeGRPO_Dream_env

conda activate /scratch/biggs.s/conda_envs/EntropyTreeGRPO_Dream_env

# Get a quick interactive GPU session:
srun --partition=gpu --nodes=1 --pty --gres=gpu:1 --ntasks=1 --mem=32GB --time=1:00:00 /bin/bash

```
---
### Layout

```text
dream/
├── DEVELOPMENT_PLAN.md   # Detailed step-by-step migration plan
├── README.md             # This file
├── src/
│   ├── __init__.py
│   ├── config.py         # MCTSConfig (Dream + MDLM, adaptive branching)
│   ├── entropy.py        # Corrected entropy normalization
│   ├── time_weight.py    # Interval-aware time weighting
│   ├── tree_node.py      # Nodes + transitions, track step intervals
│   ├── model_adapter.py  # Dream vs MDLM API adapter
│   ├── tree_builder.py   # Entropy-guided tree, optional adaptive stepping
│   ├── advantages.py     # BranchGRPO-style advantage computation
│   ├── loss.py           # Corrected weighted GRPO loss
│   ├── rewards.py        # Syntax / execution-lite / judge placeholder
│   ├── trainer.py        # Minimal EntropyMCTSTrainer (one-step loop)
│   └── utils.py          # Device, model loading, LR scheduler
├── scripts/
│   ├── validate_dream.py      # Phase 0: load Dream 7B, forward + entropy checks
│   ├── validate_dream_tree.py # Small real tree (fixed or adaptive stepping; optional LoRA)
│   ├── single_step_dream.py   # One training step (LoRA, adaptive flags, etc.)
│   └── run_dream_comparison.py # WandB comparison arms (initial eval, baseline train, adaptive variants)
└── tests/
    ├── __init__.py
    ├── test_entropy_corrected.py
    ├── test_time_weight_interval.py
    ├── test_model_adapter.py
    └── test_trainer_minimal.py
```

---

### Local Development (Laptop)

On the local laptop you should **not** load large models. All current
tests use tiny mock models and run quickly.

Run tests from the **repo root** (the directory that contains `dream/`), so that `dream.src` imports resolve:

```bash
# From repo root (e.g. EntropyTree-GRPO or EntropyTree-GRPO-Dream):
python -m pytest dream/tests -q
```

If your shell is **inside** `dream/` (e.g. `pwd` shows `.../dream`), go up one level then run the same command:

```bash
cd ..
python -m pytest dream/tests -q
```

This validates:

- corrected entropy normalization (`H_masked_mean / log(V)`)
- interval-aware time weighting
- basic ModelAdapter behavior for MDLM-style mocks
- minimal `EntropyMCTSTrainer` wiring with a tiny mock model

You can freely extend the Dream stack in `dream/src/` following the
guidelines in `DEVELOPMENT_PLAN.md` without touching the original `src/`
module.

---

### Cloud GPU — test checklist (run in order)

Use **repo root** (`cd` to the directory that contains `dream/`). Install deps once:

```bash
pip install -r dream/requirements.txt
```

| Step | Command | What “good” looks like | Status |
|------|---------|------------------------|--------|
| **1 — Unit tests** (no GPU) | `python -m pytest dream/tests -q` | All pass | All pass! |
| **2 — Phase 0: load + logits** | `python dream/scripts/validate_dream.py` | Model loads, entropy in `[0, log(V)]`, no device errors | Looks great! |
| **3 — Tree, fixed stepping** | `python dream/scripts/validate_dream_tree.py --max-tree-nodes 5 --branch-width 2 --steps-per-expansion 16 --max-new-tokens 128` | `Tree summary`, `Entropy summary`, `leaves >= 1` | All looks good! |
| **4 — Tree, adaptive stepping** | `python dream/scripts/validate_dream_tree.py --adaptive-stepping --branch-threshold 0.65 --min-steps-per-expansion 8 --max-steps-per-expansion 48 --max-tree-nodes 5 --branch-width 2 --max-new-tokens 128` | Same as above **plus** `steps_in_edge` line with **unique** values (not always identical) when adaptive triggers | Working. Smart threshold next?|
| **5 — Tree + LoRA** (optional) | `python dream/scripts/validate_dream_tree.py --lora --lora-r 8 --lora-alpha 16` | `[LoRA] trainable params: …`, tree still builds | If `AttributeError: module 'torch.distributed' has no attribute 'tensor'`, upgrade PEFT: `pip install --upgrade 'peft>=0.18.0'` |  Looks good! |
| **6 — One training step (~32GB)** | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python dream/scripts/single_step_dream.py --lora --profile-memory --max-tree-nodes 5 --max-new-tokens 96 --steps-per-expansion 12` | prints `Metrics:` with finite `loss`; `n_loss_forwards` ≤ `n_transitions` | Good |
| **7 — Training + adaptive** (optional) | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python dream/scripts/single_step_dream.py --lora --profile-memory --max-tree-nodes 5 --max-new-tokens 96 --adaptive-stepping --branch-threshold 0.65 --min-steps-per-expansion 8 --max-steps-per-expansion 48` | completes; branching uses same config as `validate_dream_tree` | Good |
| **8 — Light WandB comparison** | `export WANDB_API_KEY=...`; ensure **`logs/` exists** (repo has `logs/.gitkeep`); then `sbatch run_dream_comparison.sh` | One WandB **group** with four runs: `initial_eval` (no train, adaptive tree), `baseline_train` (fixed steps), `adaptive_default`, `adaptive_alt_hp` | Use for pipeline sanity + side-by-side curves |

**Step 8 details (repo root):** `run_dream_comparison.sh` runs, in order:

1. `python dream/scripts/run_dream_comparison.py --phase initial_eval ...` — **pre-train** metrics (`eval_step`, no optimizer).
2. `--phase baseline_train` — **MCTS-GRPO with fixed** `steps_per_expansion` (`adaptive_stepping=False`).
3. `--phase adaptive_default` — **adaptive stepping** + default `branch_threshold` / `alpha_entropy`.
4. `--phase adaptive_alt_hp` — adaptive + **alternate** `alpha_entropy=1.0`, `branch_threshold=0.55` (smoke test that HP changes show up in WandB).

**WandB layout:** metrics are logged with **shared names** across arms (`loss`, `avg_reward`, `wall_sec_step`, `epoch_mean_*`, …), similar to `scripts/run_experiment_2.py`, so in the UI you can select the whole **group** and compare runs on the **same charts**. Per-step logs include all numeric trainer fields; job logs print the full metric line. Default prompt list has **10** short code tasks; override with `--prompts_file`. Use `--wandb_prefixed_keys` if you prefer separate `phase/metric` charts.

Shared flags: `--wandb_group` (script sets to `dream_cmp_$SLURM_JOB_ID`), `--run_name`, `--lora`, tree limits, `checkpoints/dream_comparison/<run_name>/`. Override budget via env vars in the shell script (`NUM_EPOCHS`, `MAX_TREE_NODES`, …). Dry run without WandB: add `--no_wandb` to each Python invocation (edit script or call Python directly).

**Slurm / conda:** `run_dream_comparison.sh` must activate the **Dream** env (Python **3.10** + pinned torch). It prefers `conda activate /scratch/$USER/conda_envs/EntropyTreeGRPO_Dream_env` when that prefix exists, else `conda activate EntropyTreeGRPO_Dream_env`. Override with `export DREAM_CONDA_PREFIX=/path/to/env`. The script uses **`python -m pip`** so installs go to the same interpreter. If you see `torchvision::nms` / wrong Python 3.13 in the traceback, conda activation failed and you hit the wrong env.

**Adaptive stepping sanity:** `H/log(V)` is usually **≤ 1**; a threshold **> 1** (e.g. the old `1.1` default) **never** early-stops on that test. If step 4 shows identical `steps_in_edge`, try lowering `--branch-threshold` (e.g. `0.5`) or widening `[min,max]`; see `DEVELOPMENT_PLAN.md` Appendix C.

---

### LoRA size (what to tune)

LoRA is configured in **`MCTSConfig`** and on the CLI for scripts:

| Knob | Typical role | Notes |
|------|----------------|--------|
| **`lora_r`** (`--lora-r`) | Rank of the low-rank update | **Larger** → more trainable parameters, more capacity, more VRAM for optimizer/grads. |
| **`lora_alpha`** (`--lora-alpha`) | Scaling in PEFT | Effective scale is often thought of as **α/r**; e.g. `r=8, α=16` ⇒ scale 2. |
| **`lora_dropout`** (`--lora-dropout`) | Regularization on adapters | Default `0.0`; increase for heavier regularization. |

Target modules are **fixed** in `dream/src/utils.py` (`apply_lora_to_dream_model`): Dream’s attention + MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), not `lm_head`.

**Practical rule:** start with **`r=8`, `alpha=16`**; increase `r` (e.g. 16, 32) if you need more capacity and VRAM allows.

---

### LoRA vs adaptive branching

- **Adaptive branching** is controlled only by tree **`MCTSConfig`**: `adaptive_stepping`, `branch_threshold`, `min_steps_per_expansion`, `max_steps_per_expansion`, plus `branch_width` / `max_tree_nodes`. It does **not** read LoRA-specific fields.
- **LoRA changes the forward** (base weights + adapters). So entropy at each node, early-stop in `denoise_chunk_adaptive`, and frontier ordering **follow the same code** but with **different logits** than the base model. After a lot of LoRA training, you may want to **re-tune** `branch_threshold` (see Appendix C in `DEVELOPMENT_PLAN.md`).
- **Recommendation:** validate **adaptive stepping on the base model** first (step 4 above); then enable **`--lora`** for training (step 6) and compare behavior if needed.

---

### Cloud GPU: Dream 7B full setup

On a cloud machine with sufficient GPU memory (e.g. A100 40GB+):

1. **Environment** (isolate from main R&D):

   ```bash
   conda create -n dream-entropy python=3.10 -y   # 3.10 for compatibility
   conda activate dream-entropy
   cd /path/to/EntropyTree-GRPO-Dream   # or your repo root
   pip install -r dream/requirements.txt
   ```

1b. **HuggingFace cache hardening (important for quotas)**:

   The provided scripts (`validate_dream.py`, `validate_dream_tree.py`, `single_step_dream.py`, `run_dream_comparison.py`) automatically set `HF_HOME` to `/scratch/<user>/hf_home` when `HF_HOME` is not already defined, to avoid home-dir disk quota / lockfile failures.

2. **Validate Dream 7B** (Phase 0 — load, forward pass, entropy checks):

   From repo root:

   ```bash
   python dream/scripts/validate_dream.py
   ```

   You should see: model load, a short generated snippet, logits shape `[1, L, V]`, and entropy in `[0, log(V)]`. Fix any import or device errors before continuing.

3. **Tree + training (summary)** — see **Cloud GPU — test checklist** above for `validate_dream_tree.py` and `single_step_dream.py` (including **`--lora`** and **`--adaptive-stepping`**).

   **Full fine-tune vs ~32GB:** full **7B** fine-tune needs ~14GB weights + ~14GB grads (bf16) before activations. Use **`--lora`** on ~32GB GPUs. The loss groups **sibling edges**; metrics include `n_loss_forwards` ≤ `n_transitions`. See **`dream/DEVELOPMENT_PLAN.md` Appendix D**. Also: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--profile-memory`, smaller `--max-new-tokens` if needed.

4. **Use Dream in your own training loop**: load with `dream.src.utils.load_model_and_tokenizer`
   and `MCTSConfig(model_type="dream", model_name_or_path="Dream-org/Dream-v0-Instruct-7B")`.

5. **Choose a reward setup for GRPO on code**:

   ```python
   from dream.src.rewards import SyntaxReward, ExecutionLiteReward

   # Fast debugging / smoke tests (no sandbox execution):
   reward_fn = SyntaxReward()

   # Execution-based reward (fraction of tests passed from registry):
   # reward_fn = ExecutionLiteReward(registry_path="data/execution_lite.json")
   ```

   You can later experiment with an LLM-as-a-judge by wrapping an
   external scorer and combining it with these concrete rewards
   (see `DEVELOPMENT_PLAN.md`, Step 11). The trainer only requires
   a callable `reward_fn(completion, prompt) -> float`.

6. **Construct the trainer and run training**:

   ```python
   import torch
   from dream.src.config import MCTSConfig
   from dream.src.utils import load_model_and_tokenizer
   from dream.src.trainer import EntropyMCTSTrainer
   from dream.src.rewards import SyntaxReward

   cfg = MCTSConfig(model_type="dream")
   model, tokenizer = load_model_and_tokenizer(cfg)
   optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

   reward_fn = SyntaxReward()
   trainer = EntropyMCTSTrainer(model, tokenizer, cfg, reward_fn, optimizer)
   metrics = trainer.train_step("Write a function to check if a number is prime.")
   print(metrics)
   ```

7. For full experimental details (baselines, evaluation, baseline GRPO), follow
   the step-by-step instructions in `DEVELOPMENT_PLAN.md`.

---

### Design Goals

- **Isolation**: everything Dream-specific lives under `dream/`.
- **Correctness first**: entropy and time weighting are internally
  consistent with the conceptual review.
- **Cloud-ready**: the same trainer and adapter code can be reused on
  a GPU machine by simply swapping in the real Dream checkpoint.

