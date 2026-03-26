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
├── STATUS.md             # What is implemented vs what still needs GPU/eval work
├── HPC_SYNC.md           # If HPC rejects --dataset/--reward: stale clone; how to sync
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
│   ├── rewards.py        # Syntax / execution / execution-shaped / judge placeholder
│   ├── formatting.py     # Prompt formatting + code extraction
│   ├── task_registry.py  # Task schema + legacy execution-lite compatibility
│   ├── trainer.py        # Minimal EntropyMCTSTrainer (one-step loop)
│   └── utils.py          # Device, model loading, LR scheduler
├── data/
│   ├── README.md
│   ├── code_grpo_train.sample.jsonl
│   └── code_grpo_dev.sample.jsonl
├── scripts/
│   ├── validate_dream.py      # Phase 0: load Dream 7B, forward + entropy checks
│   ├── validate_dream_tree.py # Small real tree (fixed or adaptive stepping; optional LoRA)
│   ├── single_step_dream.py   # One training step (LoRA, adaptive flags, etc.)
│   ├── check_reward_pipeline.py # Task/reward smoke test (no Dream weights)
│   ├── extract_dataset_prompt.py # Print canonical_prompt for legacy --prompt workflows
│   └── run_dream_comparison.py # WandB arms: initial eval, MCTS-GRPO, flat LoRA / optional dense GRPO, adaptive
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
- code-task schema and dataset loading
- prompt formatting and code extraction
- execution-backed reward plumbing

For a concise running summary of what is implemented and what still needs GPU work,
see `dream/STATUS.md`.

You can freely extend the Dream stack in `dream/src/` following the
guidelines in `DEVELOPMENT_PLAN.md` without touching the original `src/`
module.

---

### Cloud GPU — test checklist (run in order)

Use **repo root** (`cd` to the directory that contains `dream/`). Install deps once:

```bash
pip install -r dream/requirements.txt
```

#### Stale clone check (read this first on HPC)

If you see `unrecognized arguments: --dataset ...` or `--reward ...`, your **`single_step_dream.py` / `run_dream_comparison.py` are an older revision** than the code-GRPO CLI. **No flag spelling will fix that** — update the repo or copy the current `dream/scripts/` and `dream/src/` files from the machine that has the latest commits. Step-by-step: `dream/HPC_SYNC.md`.

Sanity check:

```bash
python dream/scripts/single_step_dream.py -h 2>&1 | grep -E 'dataset|reward' || echo "STALE: missing dataset/reward CLI — sync per dream/HPC_SYNC.md"
```

Always call scripts as `python dream/scripts/...` (a bare `dream/scripts/foo.py` may hit “Permission denied” if the file is not executable).

#### CLI flag naming (after you have a current `-h`)

`argparse` names differ between entrypoints. Do **not** copy tree/budget flags from `single_step_dream.py` into `run_dream_comparison.py` (or vice versa):

- **`dream/scripts/single_step_dream.py`**, **`validate_dream_tree.py`**: hyphenated flags, e.g. `--max-tree-nodes`, `--max-new-tokens`, `--steps-per-expansion`.
- **`dream/scripts/run_dream_comparison.py`**: underscore flags for the same knobs, e.g. `--max_tree_nodes`, `--max_new_tokens`, `--steps_per_expansion`, `--branch_width`, `--num_epochs`.

Dataset and reward flags exist only on **current** scripts: `--dataset`, `--dataset-split`, `--reward`, `--max-tasks` (comparison runner only), `--reward-timeout`.

If anything still fails, run `python dream/scripts/<script>.py -h` on **that** checkout and copy flags verbatim from the help text.

#### Legacy fallback (old `-h` without `--dataset` / `--reward`)

You can still smoke-test the **training loop** with the **default prompt** or a **file-backed prompt**, but **`execution_shaped` and dataset-backed rewards require the updated scripts**.

1. Put one prompt per line in a text file (or use `dream/scripts/extract_dataset_prompt.py` to print one `canonical_prompt` from a JSONL row).
2. Single-step style (only flags your `-h` actually lists), e.g.:

```bash
python dream/scripts/extract_dataset_prompt.py dream/data/code_grpo_train.sample.jsonl 0 --split train > /tmp/dream_prompt.txt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python dream/scripts/single_step_dream.py \
  --lora --profile-memory \
  --max-tree-nodes 5 --max-new-tokens 96 --steps-per-expansion 12 \
  --prompt "$(cat /tmp/dream_prompt.txt)"
```

3. Comparison runner without `--dataset`: use `--prompts_file path/to/prompts_one_per_line.txt` plus the underscore tree flags your `-h` shows. You will **not** get execution-shaped scoring until you sync.

**Example — flat LoRA GRPO smoke (no WandB):**

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

| Step | Command | What “good” looks like | Status |
|------|---------|------------------------|--------|
| **1 — Unit tests** (no GPU) | `python -m pytest dream/tests -q` | All pass | All pass! |
| **1b — Reward pipeline** (no Dream weights) | `python dream/scripts/check_reward_pipeline.py --dataset dream/data/code_grpo_train.sample.jsonl --reward execution_shaped --max-tasks 3` | known-good completions outscore bad completion | New |
| **2 — Phase 0: load + logits** | `python dream/scripts/validate_dream.py` | Model loads, entropy in `[0, log(V)]`, no device errors | Looks great! |
| **3 — Tree, fixed stepping** | `python dream/scripts/validate_dream_tree.py --max-tree-nodes 5 --branch-width 2 --steps-per-expansion 16 --max-new-tokens 128` | `Tree summary`, `Entropy summary`, `leaves >= 1` | All looks good! |
| **4 — Tree, adaptive stepping** | `python dream/scripts/validate_dream_tree.py --adaptive-stepping --branch-threshold 0.65 --min-steps-per-expansion 8 --max-steps-per-expansion 48 --max-tree-nodes 5 --branch-width 2 --max-new-tokens 128` | Same as above **plus** `steps_in_edge` line with **unique** values (not always identical) when adaptive triggers | Working. Smart threshold next?|
| **5 — Tree + LoRA** (optional) | `python dream/scripts/validate_dream_tree.py --lora --lora-r 8 --lora-alpha 16` | `[LoRA] trainable params: …`, tree still builds | If `AttributeError: module 'torch.distributed' has no attribute 'tensor'`, upgrade PEFT: `pip install --upgrade 'peft>=0.18.0'` |  Looks good! |
| **6 — One training step (~32GB)** | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python dream/scripts/single_step_dream.py --lora --profile-memory --dataset dream/data/code_grpo_train.sample.jsonl --dataset-split train --task-index 0 --reward execution_shaped --max-tree-nodes 5 --max-new-tokens 96 --steps-per-expansion 12` | prints selected dataset task and finite `Metrics:`; `n_loss_forwards` ≤ `n_transitions` | New recommended path |
| **7 — Training + adaptive** (optional) | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python dream/scripts/single_step_dream.py --lora --profile-memory --max-tree-nodes 5 --max-new-tokens 96 --adaptive-stepping --branch-threshold 0.65 --min-steps-per-expansion 8 --max-steps-per-expansion 48` | completes; branching uses same config as `validate_dream_tree` | Good |
| **8 — Light WandB comparison** | `export WANDB_API_KEY=...`; `sbatch run_dream_comparison.sh` (from repo root). Slurm writes **`dream_comparison_<jobid>.out`** in that directory (no `logs/` folder required). | One WandB **group** with up to **six** runs: `initial_eval`, **`baseline_train`** (**MCTS / tree** GRPO — not “dense”), **`grpo_lora_baseline`** (flat GRPO + **LoRA**), optional **`grpo_dense_baseline`** (flat GRPO + **full fine-tune**), `adaptive_default`, `adaptive_alt_hp` | **Tree vs flat LoRA** vs **dense full-FT** isolates search vs adapter vs capacity |

**Step 8 details (repo root):** `run_dream_comparison.sh` runs, in order:

1. `python dream/scripts/run_dream_comparison.py --phase initial_eval ...` — **pre-train** metrics (`eval_step`, no optimizer).
2. `--phase baseline_train` — **MCTS / tree** GRPO with fixed `steps_per_expansion` (`adaptive_stepping=False`). This is **not** dense or flat GRPO — it always builds a search tree.
3. `--phase grpo_lora_baseline` — **flat trajectory GRPO** (no tree); **LoRA** matches `--lora-r` / `--lora-alpha` / `--lora` like the MCTS arms. Override K with `--num-baseline-samples` or **`NUM_BASELINE_SAMPLES`**.
4. **`grpo_dense_baseline`** (optional in shell) — **flat trajectory GRPO, full model fine-tune** (no LoRA; the script **ignores `--lora`** for this phase). Enable with **`RUN_GRPO_DENSE_BASELINE=1`** before `sbatch` — needs a **large GPU** (often ~80GB class for 7B + grads).
5. `--phase adaptive_default` — **adaptive stepping** + default `branch_threshold` / `alpha_entropy`.
6. `--phase adaptive_alt_hp` — adaptive + **alternate** `alpha_entropy=1.0`, `branch_threshold=0.55` (smoke test that HP changes show up in WandB).

**WandB layout:** metrics are logged with **shared names** across arms (`loss`, `avg_reward`, `wall_sec_step`, `epoch_mean_*`, …), similar to `scripts/run_experiment_2.py`, so in the UI you can select the whole **group** and compare runs on the **same charts**. Per-step logs include all numeric trainer fields; job logs print the full metric line. Default prompt list has **10** short code tasks; override with `--prompts_file`. Use `--wandb_prefixed_keys` if you prefer separate `phase/metric` charts.

Shared concepts: `--wandb_group` (script sets to `dream_cmp_$SLURM_JOB_ID`), `--run_name`, `--lora`, `--dataset`, `--dataset-split`, `--reward`. Tree limits on **`run_dream_comparison.py`** use underscores (`--max_tree_nodes`, `--max_new_tokens`, …); see **CLI flag naming** above. In the Slurm wrapper, set `DATASET=...`, `DATASET_SPLIT=...`, `REWARD=execution_shaped` as needed. **Checkpoints are off by default** (W&B metrics only). To save `.pt` files, pass **`--save-checkpoints`** and optionally **`--checkpoint-dir`** (Slurm: `SAVE_CHECKPOINTS=1`, default root **`/scratch/biggs.s/entropy_tree_grpo_dream`**, override with `CHECKPOINT_DIR`). Periodic saves: `SAVE_EVERY` / `--save-every-steps`. Override budget via env vars (`NUM_EPOCHS`, `MAX_TREE_NODES`, …). Dry run without WandB: add `--no_wandb` to each Python invocation (edit script or call Python directly).

**Slurm / conda:** `run_dream_comparison.sh` uses the **same conda pattern as `run_experiment_2.sh`**: source `$HOME/miniconda/etc/profile.d/conda.sh` (then `miniconda3`, then `anaconda3`), then **`conda activate EntropyTreeGRPO_Dream_env`**. Your `conda info --base` is often **`~/miniconda`**, not `~/miniconda3` — that path is checked first. Then **`python -m pip install -r dream/requirements.txt`**. If you see `torchvision::nms` or Python **3.13**, the wrong interpreter was used.

**Memory (OOM):** Training uses far more VRAM than `initial_eval` (no backward). If **`CUDA out of memory`** appears on the **second** prompt, the first `train_step` filled the GPU; the script now runs **`gc` + `torch.cuda.empty_cache()`** between training steps, and defaults use **`MAX_NEW_TOKENS=96`**, **`MAX_TREE_NODES=8`**, **`BRANCH_WIDTH=2`**. If WandB shows **only one step** for a training phase, check **`dream_comparison_<jobid>.err`** for OOM — then lower tree limits or `NUM_EPOCHS` / raise GPU memory.

**W&B metrics (flat lines, identical `n_loss_forwards`, etc.):** See **`dream/docs/WANDB_METRICS.md`**. Short version: `mean_w_time` / `mean_w_ent` are **raw** weights ( **`alpha` does not scale them in the log** ); `mean_weight` uses \(\alpha_{\text{time}}\) and \(\alpha_{\text{entropy}}\). `entropy_weight_min` (default **0.08** in comparison runs) **clamps** low-entropy edges; watch **`frac_entropy_clamped_low`**. **`mean_reward`**, **`min_reward`**, **`max_reward`** summarize reward spread. `phase_idx` is **arm id**, not time. **`cfg_*` scalars** are logged each step for verification.

**Adaptive stepping sanity:** `H/log(V)` is usually **≤ 1**; a threshold **> 1** (e.g. the old `1.1` default) **never** early-stops on that test. If step 4 shows identical `steps_in_edge`, try lowering `--branch-threshold` (e.g. `0.5`) or widening `[min,max]`; see `DEVELOPMENT_PLAN.md` Appendix C.

---

### Diversity observability

For diffusion trees, **distinct exploration** does **not** require distinct final text. Two leaves can decode to the same code while still representing different denoising trajectories:

- different response positions were committed at different times,
- different sibling edges updated different masked regions,
- the same final tokens were reached through different macro-step paths.

To make this visible, Dream now logs several diversity metrics from `dream/src/observability.py`.

#### What to look at

These metrics appear in:

- `python dream/scripts/validate_dream_tree.py ...`
- `EntropyMCTSTrainer.eval_step()` / `train_step()`
- `dream/scripts/run_dream_comparison.py` logs and WandB rows

Key metrics:

| Metric | What it measures | Why it matters |
|--------|------------------|----------------|
| `leaf_text_unique_frac` | Fraction of leaves with unique final response token sequences | Output diversity only |
| `leaf_path_unique_frac` | Fraction of leaves with unique parent→child edge signatures | Detects distinct trajectories even if final text matches |
| `leaf_schedule_unique_frac` | Fraction of leaves with unique token-commit schedules | Best answer to “same tokens, different denoising order?” |
| `avg_pairwise_leaf_hamming` | Average normalized Hamming distance between leaf responses | Surface-level output spread |
| `avg_pairwise_schedule_distance` | Average distance between token-commit schedules | Trajectory-level spread |
| `mean_sibling_position_overlap` | Average overlap in positions updated by sibling edges | Low overlap suggests broader local exploration |
| `mean_sibling_token_agreement` | On shared updated positions, how often sibling edges commit the same token | Distinguishes “same region, same token” vs real divergence |
| `unique_steps_in_edge_count` | Number of distinct `steps_in_edge` values in the tree | Quick sanity check for adaptive branching variation |

#### Recommended checks

1. Run `validate_dream_tree.py` on a real prompt.
2. Inspect `leaf_text_unique_frac` and `leaf_schedule_unique_frac` together.
3. If `leaf_text_unique_frac` is low but `leaf_schedule_unique_frac` is high, the tree is exploring different **trajectories** even when it converges to similar outputs.
4. If both are low, the tree may be collapsing.
5. If `mean_sibling_position_overlap` is near 1.0 and `mean_sibling_token_agreement` is also high, sibling branches are barely diversifying.

#### Example command

```bash
python dream/scripts/validate_dream_tree.py \
  --max-tree-nodes 5 \
  --branch-width 2 \
  --steps-per-expansion 16 \
  --max-new-tokens 128
```

Look for the `Diversity summary:` block in the output.

#### Practical interpretation

- High `leaf_text_unique_frac`, high `leaf_schedule_unique_frac`:
  broad exploration in both output and trajectory space.
- Low `leaf_text_unique_frac`, high `leaf_schedule_unique_frac`:
  trajectories differ, but the model is converging to the same answer. This still counts as meaningful diffusion-space exploration.
- Low `leaf_text_unique_frac`, low `leaf_schedule_unique_frac`:
  probable branch collapse.
- High `mean_sibling_position_overlap` with low token disagreement:
  branches are mostly repeating the same local updates.

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
   # reward_fn = ExecutionLiteReward(registry_path="dream/data/code_grpo_train.sample.jsonl")
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

