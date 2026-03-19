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
│   ├── validate_dream.py   # Phase 0: load Dream 7B, forward + entropy checks
│   └── single_step_dream.py # One training step with Dream 7B (small tree)
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

   The provided scripts (`validate_dream.py`, `validate_dream_tree.py`, `single_step_dream.py`) automatically set `HF_HOME` to `/scratch/<user>/hf_home` when `HF_HOME` is not already defined, to avoid home-dir disk quota / lockfile failures.

2. **Validate Dream 7B** (Phase 0 — load, forward pass, entropy checks):

   From repo root:

   ```bash
   python dream/scripts/validate_dream.py
   ```

   You should see: model load, a short generated snippet, logits shape `[1, L, V]`, and entropy in `[0, log(V)]`. Fix any import or device errors before continuing.

3. **Single training step** (one entropy-MCTS-GRPO step with Dream 7B):

   From repo root:

   ```bash
   python dream/scripts/single_step_dream.py --prompt "Write a Python function to check if a number is prime."
   ```

   Optional: `--max-tree-nodes 5 --max-new-tokens 128 --steps-per-expansion 16` to reduce VRAM. Uses `SyntaxReward` by default; for execution-based reward see Step 4 below.

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

