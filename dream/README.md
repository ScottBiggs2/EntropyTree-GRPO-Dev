## Dream Subdirectory Overview

This `dream/` folder contains a **self-contained substack** for running
entropy-guided MCTS-GRPO on Dream 7B (and MDLM) without disturbing the
original toy MDLM implementation in `src/`.

Use this directory when:

- you want to migrate the method to Dream 7B
- you want corrected entropy/time weighting and adaptive branching
- you are running on a **cloud GPU** (for real Dream weights)

The original MDLM stack in `src/` remains the reference baseline.

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
│   ├── utils.py          # Device, model loading, LR scheduler
│   └── ... (future: execution, scripts)
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

From the repo root:

```bash
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

### Cloud GPU Usage (Dream 7B)

On a cloud machine with sufficient GPU memory:

1. **Create a fresh, isolated environment** for Dream to satisfy pinned deps
   (keep it separate from the lightweight MDLM R&D environment):

   ```bash
   conda create -n dream-entropy python=3.11 -y
   conda activate dream-entropy
   pip install -r dream/requirements.txt
   ```

2. **Load the real Dream model** using `dream.src.utils.load_model_and_tokenizer`
   with `MCTSConfig(model_type="dream", model_name_or_path="Dream-org/Dream-v0-Instruct-7B")`.

3. **Choose a reward setup for GRPO on code**:

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

4. **Construct the trainer and run a single Dream training step**:

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

5. For full experimental details (validation scripts, baselines,
   evaluation, and baseline GRPO), follow the step-by-step instructions
   in `DEVELOPMENT_PLAN.md`.

---

### Design Goals

- **Isolation**: everything Dream-specific lives under `dream/`.
- **Correctness first**: entropy and time weighting are internally
  consistent with the conceptual review.
- **Cloud-ready**: the same trainer and adapter code can be reused on
  a GPU machine by simply swapping in the real Dream checkpoint.

