# Dream Migration: Development Plan

**Project**: Entropy-Guided MCTS-GRPO for Diffusion Language Models — Dream 7B  
**Created**: 2026-03-16  
**Scope**: Migrate from toy MDLM 0.5B to Dream 7B while fixing the three conceptual issues identified in the consistency review (entropy normalization, scale balance, interval-aware time weighting) and implementing adaptive branching.

---

## How This Plan Works

This plan is designed for coding agents to execute step-by-step. Each step has:

- **Objective**: What to accomplish
- **Files**: Exact paths to create or modify
- **Specification**: What the code must do (with snippets)
- **Verification**: How to confirm the step is done correctly
- **Dependencies**: Which prior steps must be complete

**All work happens inside `dream/`** — the existing `src/` directory is the toy MDLM stack and must not be modified.

**Environment**: Steps 1–7 can be done entirely on a local machine (no GPU, no Dream model weights). Steps 8+ require a cloud GPU with the Dream model.

---

## Architecture Overview

```
dream/
├── DEVELOPMENT_PLAN.md          # This file
├── requirements.txt             # Dream-specific pinned dependencies
├── src/
│   ├── __init__.py
│   ├── config.py                # MCTSConfig with Dream defaults + new fields
│   ├── model_adapter.py         # NEW: ModelAdapter abstraction (MDLM + Dream)
│   ├── entropy.py               # CORRECTED: fixed normalization
│   ├── time_weight.py           # CORRECTED: interval-aware weighting
│   ├── tree_builder.py          # UPDATED: adapter integration + adaptive branching
│   ├── tree_node.py             # UPDATED: step interval tracking on edges
│   ├── loss.py                  # UPDATED: corrected weighting + adapter integration
│   ├── advantages.py            # MOSTLY SAME: copied from parent
│   ├── rewards.py               # ADAPTED: Dream evaluation support
│   ├── trainer.py               # UPDATED: Dream training loop
│   ├── utils.py                 # UPDATED: Dream loading + categorical sampling
│   └── execution.py             # COPIED: execution sandbox (unchanged)
├── tests/
│   ├── __init__.py
│   ├── test_entropy_corrected.py
│   ├── test_time_weight_interval.py
│   ├── test_model_adapter.py
│   ├── test_tree_builder_adaptive.py
│   ├── test_loss_corrected.py
│   └── test_integration_dream.py
└── scripts/
    ├── validate_dream.py        # Phase 0 cloud validation
    ├── single_step_dream.py     # Single training step verification
    └── run_experiment_dream.py  # Full experiment runner
```

---

## Dependency Graph

```
Step 1: Config
   ↓
Step 2: Entropy Fix ←────────────────────────────┐
   ↓                                              │
Step 3: Time Weight Fix                           │
   ↓                                              │
Step 4: Tree Node Update                          │
   ↓                                              │
Step 5: Model Adapter ───────────────────────────→│
   ↓                                              │
Step 6: Tree Builder (adaptive branching) ←───────┘
   ↓
Step 7: Loss Module (corrected weighting)
   ↓
Step 8: Cloud Validation (Dream forward pass)
   ↓
Step 9: Integration Tests (Dream tree building)
   ↓
Step 10: Training Pipeline
   ↓
Step 11: Baseline GRPO
   ↓
Step 12: Evaluation
```

---

## Step 1: Config Module

**Objective**: Create the configuration dataclass with all new fields for Dream, adaptive branching, and corrected weighting.

**File**: `dream/src/config.py`

**Specification**:

Copy the structure from the parent `src/config.py` and add these new fields:

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class MCTSConfig:
    # --- Model ---
    model_type: str = "dream"  # "mdlm" or "dream"
    model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"
    device: Optional[str] = None

    # --- Tree construction ---
    max_tree_nodes: int = 15
    branch_width: int = 3
    steps_per_expansion: int = 32  # used only when adaptive_stepping=False

    # --- Adaptive stepping (NEW) ---
    adaptive_stepping: bool = False
    min_steps_per_expansion: int = 8
    max_steps_per_expansion: int = 48
    branch_threshold: float = 0.65  # H_masked_mean / log(V); must be <~1 to ever early-stop

    # --- Sampling ---
    temperature: float = 0.2  # Dream recommends low temp for code
    top_p: float = 0.95
    remasking: str = "low_confidence"  # MDLM default
    alg: str = "entropy"  # Dream's token ordering algorithm
    alg_temp: float = 0.0

    # --- Loss weighting ---
    alpha_time: float = 1.0
    alpha_entropy: float = 0.5
    entropy_weight_min: float = 0.5
    entropy_weight_max: float = 2.0
    advantage_clip: float = 2.0

    # --- Entropy normalization mode (NEW) ---
    # "analytic": w_ent = H_masked_mean / log(V)
    # "stage_aware": w_ent = H_masked_mean / E[H_masked_mean | r]  (requires calibration)
    entropy_norm_mode: str = "analytic"

    # --- Time weight normalization (NEW) ---
    # "sum_to_one": current behavior, each w(t) ~ O(1/T) — backward compatible
    # "mean_to_one": rescaled so mean w(t) = 1.0, each w(t) ~ O(1)
    time_weight_norm: str = "mean_to_one"

    # --- Generation ---
    total_denoising_steps: int = 256
    max_new_tokens: int = 256

    # --- Training ---
    batch_size: int = 1
    learning_rate: float = 5e-6  # conservative for 7B
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.05
    gradient_checkpointing: bool = True

    # --- Experiment ---
    num_epochs: int = 2
    num_baseline_samples: int = 4
    run_name: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    wandb_project: str = "entropy-tree-grpo-dream"
    save_every_steps: Optional[int] = None

    def __post_init__(self) -> None:
        if self.device is None:
            from dream.src.utils import get_device
            self.device = get_device()
```

Key additions vs parent:
- `model_type`: selects MDLM or Dream adapter behavior
- `adaptive_stepping`, `min_steps_per_expansion`, `max_steps_per_expansion`, `branch_threshold`: control adaptive branching
- `entropy_norm_mode`: selects corrected entropy normalization
- `time_weight_norm`: selects corrected time weight scaling
- `top_p`, `alg`, `alg_temp`: Dream-specific sampling parameters
- `gradient_checkpointing`: required for 7B training

**Verification**:

```bash
cd dream && python -c "from dream.src.config import MCTSConfig; c = MCTSConfig(); print(c)"
```

Should print a config with all fields and their defaults.

**Dependencies**: None.

---

## Step 2: Corrected Entropy Computation

**Objective**: Fix the entropy normalization convention so the denominator matches the numerator's aggregation level.

**File**: `dream/src/entropy.py`

### The Problem (from concept_consistency_review.md)

The current code computes:
- **Numerator**: `H_masked_mean` = mean entropy over masked positions only (range `[0, log(V)]`)
- **Denominator**: `masking_ratio * log(V)` = sequence-averaged upper bound (different aggregate)

This inflates the entropy ratio as `masking_ratio` decreases.

### The Fix

Provide two normalization modes:

1. **Analytic**: `w_ent = H_masked_mean / log(V)` — simple, no calibration needed, consistent because both numerator and denominator are in `[0, log(V)]`

2. **Stage-aware**: `w_ent = H_masked_mean / E[H_masked_mean | r]` — better captures "more uncertain than expected at this stage" but requires either an analytic estimate or empirical calibration data

### Specification

```python
class EntropyComputer:

    @staticmethod
    def compute_token_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
        """Shannon entropy per token from pre-computed logits.
        logits: [batch, seq_len, vocab_size]
        Returns: [batch, seq_len]
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        return -(probs * log_probs).sum(dim=-1)

    @staticmethod
    @torch.no_grad()
    def compute_token_entropy(model, input_ids, attention_mask=None):
        """Forward pass then Shannon entropy per token.
        For backward compat — calls model directly (MDLM style).
        For Dream, use compute_token_entropy_from_logits with adapter logits.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        return EntropyComputer.compute_token_entropy_from_logits(logits)

    @staticmethod
    def aggregate_entropy(token_entropy, mask_positions=None, method="mean"):
        """Unchanged from parent — aggregate over positions."""
        # ... (copy from parent src/entropy.py, identical logic)

    @staticmethod
    def compute_entropy_weight(
        measured_masked_mean: float,
        vocab_size: int,
        masking_ratio: float = 0.0,
        mode: str = "analytic",
        stage_baseline: float = 0.0,
        eps: float = 1e-6,
    ) -> float:
        """Corrected entropy weight.

        mode="analytic":
            w_ent = H_masked_mean / log(V)
            Both numerator and denominator are in [0, log(V)].
            Result is in [0, 1] for well-behaved models.

        mode="stage_aware":
            w_ent = H_masked_mean / stage_baseline
            stage_baseline = E[H_masked_mean | masking_ratio] from calibration.
            Result > 1 means "more uncertain than typical at this stage".
        """
        if mode == "analytic":
            log_v = math.log(vocab_size)
            if log_v < eps:
                return 0.0
            return measured_masked_mean / log_v
        elif mode == "stage_aware":
            if stage_baseline < eps:
                return 0.0
            return measured_masked_mean / stage_baseline
        else:
            raise ValueError(f"Unknown entropy norm mode: {mode}")

    @staticmethod
    def expected_entropy(masking_ratio: float, vocab_size: int) -> float:
        """DEPRECATED for loss weighting. Kept for backward compat / diagnostics.
        This is the sequence-averaged upper bound, not the masked-position mean baseline.
        """
        if masking_ratio <= 0:
            return 0.0
        return masking_ratio * math.log(vocab_size)
```

### Critical change summary

| Aspect | Old (parent) | New (dream) |
|--------|-------------|-------------|
| `compute_entropy_weight` denominator | `masking_ratio * log(V)` | `log(V)` (analytic) or `E[H|r]` (stage-aware) |
| New method | — | `compute_token_entropy_from_logits` (takes logits, not model) |
| `expected_entropy` | Primary normalization | Deprecated for loss weighting, kept for diagnostics |

**Verification**:

Create `dream/tests/test_entropy_corrected.py`:

```python
def test_analytic_entropy_weight_bounded():
    """Analytic w_ent should be in [0, 1] for entropy in [0, log(V)]."""
    V = 50000
    log_v = math.log(V)
    # At maximum entropy
    w = EntropyComputer.compute_entropy_weight(log_v, V, mode="analytic")
    assert abs(w - 1.0) < 1e-6
    # At half entropy
    w = EntropyComputer.compute_entropy_weight(log_v / 2, V, mode="analytic")
    assert abs(w - 0.5) < 1e-6
    # At zero entropy
    w = EntropyComputer.compute_entropy_weight(0.0, V, mode="analytic")
    assert abs(w) < 1e-6

def test_analytic_weight_independent_of_masking_ratio():
    """The key fix: w_ent should NOT change with masking_ratio for same H_masked_mean."""
    V = 50000
    h = 5.0  # some fixed masked-mean entropy
    w_high_r = EntropyComputer.compute_entropy_weight(h, V, masking_ratio=0.9, mode="analytic")
    w_low_r = EntropyComputer.compute_entropy_weight(h, V, masking_ratio=0.1, mode="analytic")
    assert abs(w_high_r - w_low_r) < 1e-6  # should be identical

def test_old_normalization_inflates_at_low_masking():
    """Demonstrate the old bug: old method inflates at low masking ratio."""
    V = 50000
    h = 5.0
    # Old method: h / (r * log(V))
    old_high_r = h / (0.9 * math.log(V))
    old_low_r = h / (0.1 * math.log(V))
    assert old_low_r > old_high_r * 5  # old method: ratio explodes as r → 0
```

Run: `cd /path/to/EntropyTree-GRPO && python -m pytest dream/tests/test_entropy_corrected.py -v`

**Dependencies**: Step 1.

---

## Step 3: Interval-Aware Time Weighting

**Objective**: Fix the time weighting module so it supports both point-lookup (backward-compatible) and interval-mass computation (required for adaptive branching).

**File**: `dream/src/time_weight.py`

### The Problem (from concept_consistency_review.md)

The current time weighter returns `w(t_parent)` — a single-point lookup. When all edges span the same number of steps, this is a proportional proxy. But under adaptive branching, an edge spanning 8 steps and one spanning 40 steps would get the same weight at the same start time, which underweights the longer edge.

Additionally, the current normalization (`weights.sum() = 1`) makes each `w(t) ~ O(1/T)`, while entropy weights are `O(1)`. This creates a scale mismatch where entropy dominates numerically.

### The Fix

1. Add `get_interval_weight(start, end)` that sums the base schedule over `[start, end)`.
2. Add a `"mean_to_one"` normalization mode where the average weight is 1.0, making each `w(t) ~ O(1)` and comparable to entropy weights.

### Specification

```python
class TimeWeighter:
    """Precomputed weights w(t) = (1 - t/T)^2 with configurable normalization."""

    def __init__(self, total_steps: int, norm_mode: str = "mean_to_one"):
        """
        norm_mode:
            "sum_to_one": weights sum to 1.0 (original behavior, each w(t) ~ O(1/T))
            "mean_to_one": weights have mean 1.0 (each w(t) ~ O(1), comparable to entropy weights)
        """
        self.total_steps = total_steps
        self.norm_mode = norm_mode
        self._precompute()

    def _precompute(self) -> None:
        T = self.total_steps
        timesteps = torch.arange(T, dtype=torch.float32)
        raw = (1.0 - timesteps / T) ** 2
        if self.norm_mode == "sum_to_one":
            self.weights = raw / (raw.sum() + 1e-10)
        elif self.norm_mode == "mean_to_one":
            self.weights = raw / (raw.mean() + 1e-10)
        else:
            raise ValueError(f"Unknown norm_mode: {self.norm_mode}")

    def get_weight(self, step_index: int) -> float:
        """Point weight for a single step index. Backward-compatible API."""
        if step_index >= self.total_steps or step_index < 0:
            return 0.0
        return self.weights[step_index].item()

    def get_interval_weight(self, start: int, end: int) -> float:
        """Sum of weights over interval [start, end).
        This is the correct weighting for a macro-edge spanning multiple steps.
        For fixed-step branching: equivalent to get_weight(start) * num_steps (approximately).
        For adaptive branching: correctly accumulates mass over the actual interval.
        """
        start = max(0, start)
        end = min(end, self.total_steps)
        if start >= end:
            return 0.0
        return self.weights[start:end].sum().item()
```

### Scale comparison

With `T = 256`, `mean_to_one` normalization:
- `w(0) = (1 - 0/256)^2 / mean ≈ 1.0 / 0.337 ≈ 2.97` (early steps — high weight)
- `w(128) = (1 - 0.5)^2 / mean ≈ 0.25 / 0.337 ≈ 0.74` (midpoint)
- `w(255) = (1 - 255/256)^2 / mean ≈ 0.0 / 0.337 ≈ 0.0` (late steps)

This is now O(1), comparable to entropy weights clamped to [0.5, 2.0].

**Verification**:

Create `dream/tests/test_time_weight_interval.py`:

```python
def test_mean_to_one_normalization():
    """Under mean_to_one, average weight should be 1.0."""
    tw = TimeWeighter(256, norm_mode="mean_to_one")
    avg = tw.weights.mean().item()
    assert abs(avg - 1.0) < 1e-5

def test_sum_to_one_backward_compat():
    """Under sum_to_one, weights should sum to 1.0 (matches parent behavior)."""
    tw = TimeWeighter(256, norm_mode="sum_to_one")
    assert abs(tw.weights.sum().item() - 1.0) < 1e-5

def test_interval_weight_sums_correctly():
    """Interval weight should equal sum of point weights."""
    tw = TimeWeighter(256, norm_mode="mean_to_one")
    interval_w = tw.get_interval_weight(10, 42)
    point_sum = sum(tw.get_weight(t) for t in range(10, 42))
    assert abs(interval_w - point_sum) < 1e-5

def test_interval_weight_longer_edge_gets_more_weight():
    """A longer edge should receive more time weight than a shorter one at the same start."""
    tw = TimeWeighter(256, norm_mode="mean_to_one")
    short = tw.get_interval_weight(10, 18)   # 8 steps
    long = tw.get_interval_weight(10, 50)    # 40 steps
    assert long > short * 3  # much more weight for much longer edge

def test_scale_comparable_to_entropy():
    """Time weights should now be O(1), not O(1/T)."""
    tw = TimeWeighter(256, norm_mode="mean_to_one")
    w_early = tw.get_weight(0)
    assert w_early > 0.5, f"Early time weight should be O(1), got {w_early}"
    w_mid = tw.get_weight(128)
    assert w_mid > 0.1, f"Mid time weight should be O(1), got {w_mid}"
```

Run: `python -m pytest dream/tests/test_time_weight_interval.py -v`

**Dependencies**: Step 1.

---

## Step 4: Updated Tree Node

**Objective**: Extend `MCTSNode` and `TreeTransition` to track step intervals (not just a single `step_index`) so that interval-aware weighting can be applied.

**File**: `dream/src/tree_node.py`

### Specification

Copy from parent `src/tree_node.py` with these additions:

```python
@dataclass
class MCTSNode:
    state: torch.Tensor
    attention_mask: torch.Tensor
    prompt_len: int
    step_index: int  # cumulative steps from root to reach this node
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    mask_id: Optional[int] = None

    entropy: Optional[float] = None
    token_entropy: Optional[torch.Tensor] = None
    reward: Optional[float] = None
    fused_reward: Optional[float] = None
    advantage: Optional[float] = None
    sampling_prob: float = 1.0
    depth: int = 0
    is_completed: bool = False

    # NEW: track how many steps this edge took (for interval-aware time weighting)
    steps_in_edge: Optional[int] = None  # steps from parent to this node

    # ... existing methods unchanged: _response_len, num_masked_tokens, masking_ratio, is_leaf


@dataclass
class TreeTransition:
    parent_state: torch.Tensor
    child_state: torch.Tensor
    parent_attention_mask: torch.Tensor
    child_attention_mask: torch.Tensor
    step_index: int           # parent's step_index (start of edge)
    child_step_index: int     # NEW: child's step_index (end of edge)
    advantage: float
    entropy: float
    time_weight: float
    entropy_weight: float
```

The key addition is `child_step_index` on `TreeTransition` (so the loss module can compute interval weight) and `steps_in_edge` on `MCTSNode` (set by the tree builder).

**Verification**: Unit test that `TreeTransition` can be constructed with both step indices.

**Dependencies**: None (no logic, just data structures).

---

## Step 5: Model Adapter

**Objective**: Create a thin abstraction layer that encapsulates the differences between MDLM and Dream model APIs. This is the key architectural piece that lets the tree builder, entropy computer, and loss module remain model-agnostic.

**File**: `dream/src/model_adapter.py`

### Specification

```python
import torch
import torch.nn.functional as F
import torch.distributions as dists
from typing import Tuple, Optional


class ModelAdapter:
    """Encapsulates model-specific forward pass and sampling.
    The rest of the pipeline interacts with this adapter, not the raw model.
    """

    def __init__(self, model, tokenizer, model_type: str = "dream"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.mask_id = tokenizer.mask_token_id
        self.vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
        self.device = next(model.parameters()).device

    def forward_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return logits [B, L, V] with model-specific transforms applied.
        For Dream: applies the right-shift correction.
        For MDLM: direct forward pass.
        """
        if self.model_type == "dream":
            tok_idx = self._compute_tok_idx(input_ids)
            logits = self.model(
                input_ids, attention_mask="full", tok_idx=tok_idx
            ).logits
            # Dream right-shift: logits[i] predicts position i+1 in raw output
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            return logits
        else:
            return self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

    def sample_and_confidence(
        self,
        logits: torch.Tensor,
        mask_positions: torch.Tensor,
        temperature: float,
        top_p: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample tokens and return (x0_pred, confidence) for all positions.
        x0_pred: [L] predicted tokens (unchanged at non-mask positions)
        confidence: [L] softmax probability of predicted token (-inf at non-mask)

        For Dream: Categorical sampling with top-p.
        For MDLM: Gumbel-Max trick.
        """
        if self.model_type == "dream":
            return self._dream_sample(logits, mask_positions, temperature, top_p)
        else:
            return self._mdlm_sample(logits, mask_positions, temperature)

    def transfer_count(
        self, n_masked: int, step: int, total_steps: int, eps: float = 1e-3
    ) -> int:
        """How many tokens to unmask at this step.
        For Dream: schedule-based (1 - s/t) from linear timesteps.
        For MDLM: uniform n_masked // steps_left.
        """
        if self.model_type == "dream":
            timesteps = torch.linspace(1, eps, total_steps + 1)
            t, s = timesteps[step].item(), timesteps[step + 1].item()
            return max(1, int(n_masked * (1 - s / t)))
        else:
            steps_left = total_steps - step
            return max(1, n_masked // max(steps_left, 1))

    def _dream_sample(self, logits, mask_positions, temperature, top_p):
        L = logits.shape[0]
        if temperature > 0:
            logits_t = logits / temperature
            logits_t = self._top_p_filter(logits_t, top_p)
            probs = F.softmax(logits_t, dim=-1)
            x0 = dists.Categorical(probs=probs).sample()
        else:
            x0 = torch.argmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
        conf = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        conf = torch.where(mask_positions, conf, torch.full_like(conf, -1e9))
        return x0, conf

    def _mdlm_sample(self, logits, mask_positions, temperature):
        from dream.src.utils import add_gumbel_noise
        logits_n = add_gumbel_noise(logits, temperature)
        x0 = torch.argmax(logits_n, dim=-1)
        probs = F.softmax(logits, dim=-1)
        conf = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        conf = torch.where(mask_positions, conf, torch.full_like(conf, -1e9))
        return x0, conf

    def _top_p_filter(self, logits, top_p):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[mask] = -float("inf")
        return sorted_logits.scatter(-1, sorted_indices, sorted_logits)

    def _compute_tok_idx(self, input_ids):
        B, L = input_ids.shape
        return torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
```

**Implementation note (matches Dream upstream)**: The live `dream/src/model_adapter.py` uses `_dream_top_p_logits`, copied from Hugging Face `Dream-org/Dream-v0-Instruct-7B` `generation_utils.top_p_logits` (shifted `sorted_indices_to_remove` mask + scatter + `torch.finfo(dtype).min`, not `-inf`). `_dream_sample` applies softmax and `Categorical.sample()` in **float32** so bf16 logits do not underflow to all-zero probs on CUDA; on any sampling failure it falls back like upstream `sample_tokens` (`probs.max`). Only masked rows are sampled; confidences at non-mask positions stay large negative for the tree builder.

### Test specification

Create `dream/tests/test_model_adapter.py`:

```python
class MockMDLMModel(nn.Module):
    """Mock that returns random logits like MDLM."""
    def __init__(self, vocab_size=100):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # so .parameters() works
        self.vocab_size = vocab_size
    def forward(self, input_ids, attention_mask=None, **kwargs):
        B, L = input_ids.shape
        logits = torch.randn(B, L, self.vocab_size)
        return type('Output', (), {'logits': logits})()


def test_mdlm_adapter_returns_correct_shape():
    model = MockMDLMModel(vocab_size=100)
    tokenizer = MockTokenizer(mask_token_id=99, vocab_size=100)
    adapter = ModelAdapter(model, tokenizer, model_type="mdlm")
    ids = torch.randint(0, 100, (1, 32))
    attn = torch.ones(1, 32, dtype=torch.long)
    logits = adapter.forward_logits(ids, attn)
    assert logits.shape == (1, 32, 100)

def test_mdlm_sample_returns_valid_tokens():
    adapter = ModelAdapter(MockMDLMModel(100), MockTokenizer(99, 100), "mdlm")
    logits = torch.randn(32, 100)
    mask = torch.zeros(32, dtype=torch.bool)
    mask[10:20] = True
    x0, conf = adapter.sample_and_confidence(logits, mask, temperature=0.8)
    assert x0.shape == (32,)
    assert conf.shape == (32,)
    assert (conf[mask] >= 0).all()
    assert (conf[~mask] < -1e8).all()

def test_transfer_count_mdlm_uniform():
    adapter = ModelAdapter(MockMDLMModel(100), MockTokenizer(99, 100), "mdlm")
    k = adapter.transfer_count(n_masked=100, step=0, total_steps=10)
    assert k == 10  # 100 // 10

def test_transfer_count_dream_schedule():
    adapter = ModelAdapter(MockMDLMModel(100), MockTokenizer(99, 100), "dream")
    k = adapter.transfer_count(n_masked=100, step=0, total_steps=10)
    assert k >= 1
    # Dream unmaskes more early, fewer late
    k_early = adapter.transfer_count(100, 0, 10)
    k_late = adapter.transfer_count(100, 8, 10)
    assert k_early >= k_late
```

**Verification**: `python -m pytest dream/tests/test_model_adapter.py -v`

**Dependencies**: Step 1, `dream/src/utils.py` for `add_gumbel_noise` (copy from parent).

---

## Step 6: Tree Builder with Adaptive Branching

**Objective**: Migrate the tree builder to use ModelAdapter and implement entropy-threshold adaptive branching (gated by `config.adaptive_stepping`).

**File**: `dream/src/tree_builder.py`

### Key Changes from Parent

1. **Replace `self.model` with `self.adapter`**: All forward passes and sampling go through ModelAdapter.
2. **Add `_denoise_chunk_adaptive`**: New method that monitors entropy during denoising and stops at high-uncertainty states.
3. **Set `child.steps_in_edge`**: Record actual steps taken per edge.
4. **Route expansion**: `_expand_node` calls either `_denoise_chunk` (fixed) or `_denoise_chunk_adaptive` based on config.

### Specification

```python
class EntropyGuidedTreeBuilder:

    def __init__(self, adapter: ModelAdapter, tokenizer, config: MCTSConfig,
                 entropy_computer: EntropyComputer):
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.config = config
        self.entropy_computer = entropy_computer
        self.device = config.device or adapter.device
        self.mask_id = tokenizer.mask_token_id
        self.vocab_size = adapter.vocab_size

    # build_tree: same structure as parent, but calls updated _expand_node
    # _create_root: same as parent
    # _compute_node_entropy: CHANGED — uses adapter.forward_logits + compute_token_entropy_from_logits

    def _compute_node_entropy(self, node):
        ids = node.state.unsqueeze(0)
        attn = node.attention_mask.unsqueeze(0)
        logits = self.adapter.forward_logits(ids, attn)
        token_entropy = self.entropy_computer.compute_token_entropy_from_logits(logits)
        node.token_entropy = token_entropy[0]
        mask_pos = (node.state == self.mask_id) & (node.attention_mask.bool())
        if mask_pos.any():
            node.entropy = self.entropy_computer.aggregate_entropy(
                token_entropy, mask_pos.unsqueeze(0), method="mean"
            )
        else:
            node.entropy = 0.0

    def _expand_node(self, node):
        children = []
        temp = self._node_temperature(node)
        for _ in range(self.config.branch_width):
            if self.config.adaptive_stepping:
                child = self._denoise_chunk_adaptive(
                    node, self.config.min_steps_per_expansion,
                    self.config.max_steps_per_expansion,
                    self.config.branch_threshold, temp
                )
            else:
                child = self._denoise_chunk(node, self.config.steps_per_expansion, temp)
            child.sampling_prob = 1.0 / self.config.branch_width
            child.depth = node.depth + 1
            children.append(child)
        return children

    def _denoise_chunk(self, node, num_steps, temperature):
        """Fixed-step denoising — adapter-based version of parent."""
        state = node.state.clone()
        attn = node.attention_mask.clone()
        prompt_len = node.prompt_len
        max_new = self.config.max_new_tokens
        response_region = torch.zeros_like(state, dtype=torch.bool)
        response_region[prompt_len : prompt_len + max_new] = True

        steps_taken = 0
        with torch.no_grad():
            for step in range(num_steps):
                mask_now = (state == self.mask_id) & response_region
                n_masked = mask_now.sum().item()
                if n_masked == 0:
                    break
                steps_left = num_steps - step
                k = max(1, n_masked // max(steps_left, 1))

                logits = self.adapter.forward_logits(
                    state.unsqueeze(0), attn.unsqueeze(0)
                )[0]
                x0_pred, conf = self.adapter.sample_and_confidence(
                    logits, mask_now, temperature, self.config.top_p
                )
                x0_pred = torch.where(mask_now, x0_pred, state)
                _, sel = torch.topk(conf, k=min(k, n_masked))
                state[sel] = x0_pred[sel]
                steps_taken += 1

        child = MCTSNode(
            state=state, attention_mask=attn, prompt_len=prompt_len,
            step_index=node.step_index + steps_taken,
            parent=node, mask_id=self.mask_id,
        )
        child.steps_in_edge = steps_taken
        return child

    def _denoise_chunk_adaptive(self, node, min_steps, max_steps,
                                 branch_threshold, temperature):
        """Entropy-threshold adaptive stepping.
        Run denoising steps. After min_steps, check if the model is
        more uncertain than expected — if so, stop (good branch point).
        Falls back to max_steps if threshold is never exceeded.
        """
        state = node.state.clone()
        attn = node.attention_mask.clone()
        prompt_len = node.prompt_len
        max_new = self.config.max_new_tokens
        response_region = torch.zeros_like(state, dtype=torch.bool)
        response_region[prompt_len : prompt_len + max_new] = True
        response_len = max_new

        steps_taken = 0
        with torch.no_grad():
            for step in range(max_steps):
                mask_now = (state == self.mask_id) & response_region
                n_masked = mask_now.sum().item()
                if n_masked == 0:
                    break

                logits = self.adapter.forward_logits(
                    state.unsqueeze(0), attn.unsqueeze(0)
                )[0]
                probs = F.softmax(logits, dim=-1)

                # Entropy check uses the same logits — zero extra cost
                token_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                masked_entropy = token_entropy[mask_now].mean().item()

                # Sample and unmask
                x0_pred, conf = self.adapter.sample_and_confidence(
                    logits, mask_now, temperature, self.config.top_p
                )
                k = self.adapter.transfer_count(
                    n_masked, step, max_steps
                )
                x0_pred = torch.where(mask_now, x0_pred, state)
                _, sel = torch.topk(conf, k=min(k, n_masked))
                state[sel] = x0_pred[sel]
                steps_taken += 1

                # Entropy-threshold check (after min_steps)
                if steps_taken >= min_steps:
                    log_v = math.log(self.vocab_size)
                    if log_v > 1e-6:
                        uncertainty_ratio = masked_entropy / log_v
                        if uncertainty_ratio > branch_threshold:
                            break  # high-entropy state → good branch point

        child = MCTSNode(
            state=state, attention_mask=attn, prompt_len=prompt_len,
            step_index=node.step_index + steps_taken,
            parent=node, mask_id=self.mask_id,
        )
        child.steps_in_edge = steps_taken
        return child
```

### Adaptive branching design notes

The entropy threshold uses the **corrected analytic normalization** (`H_masked_mean / log(V)`) consistently with Step 2. For Shannon entropy per token, \(H \le \log V\) (nats), so this ratio is typically **in \([0, 1]\)**. A default **`branch_threshold` > 1** therefore **never** fires the early-stop branch (bug we fixed: prior default `1.1`). Use **~0.5–0.8** after entropy profiling; `MCTSConfig.branch_threshold` default is **0.65** as a mid prior.

**Semantics:** after `min_steps`, we **break** (end the micro-chunk) when `H_\text{masked,mean} / \log(V) > \text{branch_threshold}` — i.e. stop when masked positions are **sufficiently uncertain** (good branch point). Tune `branch_threshold` against the empirical `H/log(V)` curve from `validate_dream.py`.

### Verification

Create `dream/tests/test_tree_builder_adaptive.py` with a mock model:

```python
def test_fixed_step_branching_produces_uniform_deltas():
    """With adaptive_stepping=False, all edges should be steps_per_expansion."""
    config = MCTSConfig(model_type="mdlm", adaptive_stepping=False,
                        steps_per_expansion=8, max_tree_nodes=5, branch_width=2)
    # ... build tree with mock ...
    # Check: all child.steps_in_edge == 8

def test_adaptive_branching_can_stop_early():
    """With adaptive_stepping=True, edges can be shorter than max_steps."""
    # Use a mock model that returns very high entropy logits (near uniform)
    config = MCTSConfig(model_type="mdlm", adaptive_stepping=True,
                        min_steps_per_expansion=2, max_steps_per_expansion=16,
                        branch_threshold=0.3, max_tree_nodes=5, branch_width=2)
    # ... build tree ...
    # Check: at least one child.steps_in_edge < max_steps_per_expansion

def test_adaptive_branching_respects_min_steps():
    """Should never stop before min_steps even if entropy is high."""
    config = MCTSConfig(model_type="mdlm", adaptive_stepping=True,
                        min_steps_per_expansion=4, max_steps_per_expansion=16,
                        branch_threshold=0.0, max_tree_nodes=5, branch_width=2)
    # ... build tree with high-entropy mock ...
    # Check: all child.steps_in_edge >= 4
```

**Dependencies**: Steps 1, 2, 4, 5.

---

## Step 7: Corrected Loss Module

**Objective**: Update the loss module to use corrected entropy normalization, interval-aware time weighting, and ModelAdapter for forward passes.

**File**: `dream/src/loss.py`

### Key Changes from Parent

1. **Entropy weight uses corrected normalization**: calls `EntropyComputer.compute_entropy_weight` with `mode=config.entropy_norm_mode`
2. **Time weight uses interval mass**: calls `time_weighter.get_interval_weight(parent_step, child_step)` instead of `get_weight(parent_step)`
3. **Forward pass through adapter**: `_log_prob_transition` uses `adapter.forward_logits` instead of `model(...)`

### Specification

```python
class WeightedGRPOLoss:

    def __init__(self, config, entropy_computer, time_weighter, mask_id, adapter=None):
        self.config = config
        self.entropy_computer = entropy_computer
        self.time_weighter = time_weighter
        self.mask_id = mask_id
        self.adapter = adapter  # NEW: used for forward passes

    def _collect_transitions(self, root, vocab_size):
        out = []

        def go(node):
            for c in node.children:
                # --- CORRECTED TIME WEIGHT ---
                child_step = c.step_index
                parent_step = node.step_index
                tw = self.time_weighter.get_interval_weight(parent_step, child_step)

                # --- CORRECTED ENTROPY WEIGHT ---
                ew = self.entropy_computer.compute_entropy_weight(
                    measured_masked_mean=node.entropy or 0.0,
                    vocab_size=vocab_size,
                    masking_ratio=node.masking_ratio(),
                    mode=self.config.entropy_norm_mode,
                )
                # Clamp for stability (D-014)
                ew = max(self.config.entropy_weight_min,
                         min(self.config.entropy_weight_max, ew))

                trans = TreeTransition(
                    parent_state=node.state,
                    child_state=c.state,
                    parent_attention_mask=node.attention_mask,
                    child_attention_mask=c.attention_mask,
                    step_index=parent_step,
                    child_step_index=child_step,  # NEW
                    advantage=c.advantage or 0.0,
                    entropy=node.entropy or 0.0,
                    time_weight=tw,
                    entropy_weight=ew,
                )
                out.append(trans)
                go(c)

        go(root)
        return out

    def _log_prob_transition(self, model_or_adapter, parent_state, child_state,
                              parent_attention_mask):
        """Log p(child | parent) via adapter or raw model."""
        changed = (parent_state != child_state) & (parent_state == self.mask_id)
        if not changed.any():
            return torch.tensor(0.0, device=parent_state.device)

        if self.adapter is not None:
            logits = self.adapter.forward_logits(
                parent_state.unsqueeze(0),
                parent_attention_mask.unsqueeze(0),
            )[0]
        else:
            logits = model_or_adapter(
                parent_state.unsqueeze(0),
                attention_mask=parent_attention_mask.unsqueeze(0),
            ).logits[0]

        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, child_state.unsqueeze(-1)).squeeze(-1)
        return (token_lp * changed.float()).sum()
```

### Verification

Create `dream/tests/test_loss_corrected.py`:

```python
def test_interval_time_weight_used():
    """Transitions should have interval-based time weights, not point weights."""
    # Build a tree with known step indices
    # Parent step=10, child step=42
    # Verify trans.time_weight == time_weighter.get_interval_weight(10, 42)

def test_corrected_entropy_weight():
    """Entropy weight should use analytic normalization: H / log(V)."""
    # Parent with entropy=5.0, vocab_size=50000
    # Expected: 5.0 / log(50000) ≈ 0.462
    # NOT: 5.0 / (0.3 * log(50000)) ≈ 1.54 (old method with masking_ratio=0.3)

def test_scale_balance():
    """Time and entropy weights should be on comparable scales."""
    # With mean_to_one normalization and T=256:
    # w_time for interval [0, 32) should be roughly O(10) — 32 steps * ~O(1) each
    # w_ent should be in [0.5, 2.0] after clamping
    # After alpha weighting: alpha_time * w_time + alpha_entropy * w_ent
    # Both terms should contribute meaningfully
```

**Dependencies**: Steps 1, 2, 3, 4, 5.

---

## Step 7b: Supporting Files (Copy + Adapt)

**Objective**: Copy remaining modules from parent `src/` into `dream/src/`, making minimal adaptations.

**Files to copy with minimal changes**:

| Source | Target | Changes |
|--------|--------|---------|
| `src/advantages.py` | `dream/src/advantages.py` | Change imports to `dream.src.*` |
| `src/rewards.py` | `dream/src/rewards.py` | Change imports to `dream.src.*` |
| `src/execution.py` | `dream/src/execution.py` | No changes |
| `src/utils.py` | `dream/src/utils.py` | Add Dream model loading, keep `add_gumbel_noise`, update imports |

### `dream/src/utils.py` additions

```python
def load_model_and_tokenizer(config):
    """Load model and tokenizer. Supports both MDLM and Dream."""
    device = config.device or get_device()

    if config.model_type == "dream":
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True,
        )
    else:  # mdlm
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        model = AutoModelForMaskedLM.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True,
        )

    return model, tokenizer
```

**Dependencies**: Steps 1–7.

---

## Step 7c: Trainer Module

**Objective**: Update both trainers to use ModelAdapter and corrected weighting.

**File**: `dream/src/trainer.py`

### Key changes

1. `EntropyMCTSTrainer.__init__` takes `adapter: ModelAdapter` instead of `model` directly
2. Tree builder receives adapter
3. Loss module receives adapter
4. `BaselineGRPOTrainer` updated for Dream's `diffusion_generate` when `model_type="dream"`

### Dream-specific baseline generation

For Dream, the baseline GRPO path can use `model.diffusion_generate(output_history=True)` to get intermediate states without building a tree:

```python
def _dream_generate_with_transitions(self, prompt):
    """Dream-specific: use diffusion_generate with output_history."""
    messages = [{"role": "user", "content": prompt}]
    inputs = self.tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_ids = inputs.input_ids.to(self.adapter.device)
    attention_mask = inputs.attention_mask.to(self.adapter.device)
    prompt_len = input_ids.shape[1]

    output = self.adapter.model.diffusion_generate(
        input_ids, attention_mask=attention_mask,
        max_new_tokens=self.config.max_new_tokens,
        steps=self.config.total_denoising_steps,
        temperature=self.config.temperature,
        top_p=self.config.top_p,
        alg=self.config.alg, alg_temp=self.config.alg_temp,
        output_history=True, return_dict_in_generate=True,
    )
    transitions = []
    for i in range(len(output.history) - 1):
        transitions.append((output.history[i], output.history[i + 1], attention_mask))
    completion = self.tokenizer.decode(
        output.sequences[0][prompt_len:], skip_special_tokens=True
    )
    return completion, transitions
```

**Dependencies**: Steps 1–7b.

---

## Step 8: Cloud Validation — Dream Forward Pass

**Objective**: Verify that Dream 7B loads, runs forward passes, produces correct entropy, and the right-shift correction works. This step requires a cloud GPU.

**File**: `dream/scripts/validate_dream.py`

**Prerequisite**: Access to a machine with >= 40GB GPU VRAM and the Dream model weights downloaded.

### Specification

The script should run three tests:

1. **Basic generation**: Load Dream IT, run `diffusion_generate`, verify coherent output.
2. **Raw forward + right-shift + entropy**: Manually create a masked canvas, run forward pass, apply right-shift, compute entropy, verify it is in `[0, log(V)]`.
3. **Entropy decreases**: Run partial denoising (0, 32, 64, 128 steps), verify mean masked-position entropy decreases monotonically.

See the parent `dream_migration_plan.md` § Phase 0 for the exact script body. Adapt it to use `ModelAdapter` from Step 5.

### Verification Checklist

- [ ] Model loads without error on GPU
- [ ] `diffusion_generate` produces coherent code for a simple prompt
- [ ] `ModelAdapter.forward_logits` returns shape `[1, L, V]` with right-shift applied
- [ ] Entropy over masked positions is in range `[0, log(V)]` where `V ≈ 151,936`
- [ ] Entropy decreases after partial denoising
- [ ] Peak GPU memory < 20GB for inference only
- [ ] Note the empirical entropy range at various masking ratios — this calibrates `branch_threshold`

### Calibration Output

The script should print a table like:

```
masking_ratio | mean_masked_entropy | entropy / log(V)
    1.00      |       X.XXX        |      X.XXX
    0.75      |       X.XXX        |      X.XXX
    0.50      |       X.XXX        |      X.XXX
    0.25      |       X.XXX        |      X.XXX
    0.10      |       X.XXX        |      X.XXX
```

This is critical for setting `branch_threshold` in Step 6. If the typical `entropy / log(V)` at 50% masking is 0.4, then `branch_threshold=0.5` would trigger branching only at unusually uncertain states.

**Dependencies**: Steps 1, 2, 5.

---

## Step 9: Cloud Integration — Dream Tree Building

**Objective**: Build a small tree (5 nodes) with the real Dream model, verifying that all components work end-to-end.

**File**: `dream/scripts/validate_dream_tree.py`

### Specification

```python
config = MCTSConfig(
    model_type="dream",
    model_name_or_path="Dream-org/Dream-v0-Instruct-7B",
    max_tree_nodes=5,
    branch_width=2,
    steps_per_expansion=16,
    max_new_tokens=128,
    temperature=0.2,
    adaptive_stepping=False,  # test fixed first
)
model, tokenizer = load_model_and_tokenizer(config)
adapter = ModelAdapter(model, tokenizer, model_type="dream")
entropy_computer = EntropyComputer()
builder = EntropyGuidedTreeBuilder(adapter, tokenizer, config, entropy_computer)

root, leaves = builder.build_tree("Write a Python function to check if a number is prime.")
```

### Verification Checklist

- [ ] Tree builds without error
- [ ] `len(leaves) >= 2`
- [ ] All nodes have `entropy is not None`
- [ ] Leaves produce decodable, non-empty text
- [ ] Frontier selection picked highest-entropy nodes (print tree structure to verify)
- [ ] Wall-clock time < 60s on A100

Then repeat with `adaptive_stepping=True`:

- [ ] At least some edges have `steps_in_edge` different from others
- [ ] No crash from the entropy threshold logic

**Dependencies**: Steps 1–8.

---

## Step 10: Cloud Training Pipeline

**Objective**: Verify one complete forward-backward-update cycle with Dream.

**File**: `dream/scripts/single_step_dream.py`

### Specification

1. Load Dream model with `gradient_checkpointing_enable()`
2. Build a small tree (5 nodes)
3. Compute rewards with syntax reward function
4. Compute BranchGRPO advantages
5. Compute corrected weighted GRPO loss
6. Run backward pass
7. Clip gradients, optimizer step
8. Verify model parameters changed

### Verification Checklist

- [ ] Forward pass through tree completes without OOM
- [ ] Loss is finite and in reasonable range (0.01 to 100)
- [ ] Gradients exist on model parameters
- [ ] Gradient norm is finite
- [ ] Optimizer step completes
- [ ] Model parameters changed (compare checksums before/after)
- [ ] Peak GPU memory logged and within budget
- [ ] Wall-clock time logged

### Key training considerations for 7B

**Memory**: Full fine-tuning on **32GB** may OOM during **loss backward** even when the tree build succeeds — see **Appendix D** (per-edge forward+backward + optimizer, not “tree graph size”). Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--profile-memory`, smaller `max_new_tokens`, or `--optimizer sgd` for smoke tests before committing to AdamW / LoRA / 8-bit optim / larger GPU.

```python
model.gradient_checkpointing_enable()

with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    loss, metrics = loss_computer.compute_loss(adapter, root, leaves, prompt, vocab_size)

loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
optimizer.step()
```

**Dependencies**: Steps 1–9.

---

## Step 11: GRPO Setup for Code (Rewards & Benchmarks)

**Objective**: Design a concrete GRPO configuration for code generation, inspired by DiffuCoder and compatible with both Dream and the entropy-tree method. This covers reward design, datasets, and judge options (including an optional LLM-as-a-judge with fallbacks).

### 11.1 Lessons from DiffuCoder

From the DiffuCoder paper and repo:

- **Domain & model**:
  - Uses a 7B Qwen2.5-based masked diffusion code model, architecturally close to Dream-7B.
  - Evaluates on HumanEval, MBPP, BigCodeBench variants, and GSM8K.
- **GRPO design** (Coupled-GRPO):
  - Constructs *coupled* mask noise patterns for pairs of completions to reduce variance in token log-likelihood estimates.
  - Uses execution-based rewards on code benchmarks (fraction of unit tests passed).
  - Observes that higher diffusion sampling temperature diversifies both token **choices** and **generation order**, effectively enlarging the exploration space.
  - Achieves +4.4% on EvalPlus-style code benchmarks vs baselines.
- **Relevance for Dream stack**:
  - We do **not** adopt coupled sampling immediately (different focus: entropy-tree credit assignment and branching), but the reward and evaluation framing for code is highly aligned:
    - code benchmarks with test suites
    - execution-based rewards as primary signals
    - diffusion-specific sampling temperature choices for RL exploration.

### 11.2 Reward hierarchy for Dream

We want a reward setup that:
- works for quick local debugging,
- scales to realistic code RL on a GPU,
- can optionally incorporate an LLM-as-a-judge while preserving a concrete fallback.

**Files**:
- `dream/src/rewards.py` — already added
- later: optional judge integration script (cloud-only)

**Reward options**:

- **SyntaxReward** (`dream.src.rewards.SyntaxReward`)
  - Fast heuristic: AST parse, presence of `def`/`return`, docstring.
  - Use for:
    - local smoke tests (`test_trainer_minimal.py`-style setups),
    - early Dream debugging where execution sandbox is not yet wired.

- **ExecutionLiteReward** (`dream.src.rewards.ExecutionLiteReward`)
  - Wraps the existing `src.execution.run_tests` and `data/execution_lite.json`.
  - Reward:
    - `tests_passed_fraction * 0.8` (primary signal)
    - + shaping bonuses (AST parse, `def` + `return`, function name, control flow, indentation).
  - Use for:
    - serious GRPO runs on Dream in the cloud,
    - benchmarks aligned with our existing execution-lite dataset (subset of HumanEval-like problems).

- **LLMEvalReward** (`dream.src.rewards.LLMEvalReward`)
  - Placeholder class that returns 0.0 by default.
  - Intended design:
    - Wrap an external judge model (e.g., AR code LLM or distilled small judge) that returns a score in [0, 1].
    - Combine with ExecutionLiteReward:
      - `R_total = α_exec * R_exec + α_judge * R_judge`
    - **Must remain optional**:
      - all experiments must be reproducible without external judge (e.g., when offline or debugging).
      - the trainer should accept any callable `RewardFunction`, so judge usage is opt-in.

**Recommended default stack**:

1. **Phase A (local / early)**: `SyntaxReward`
2. **Phase B (cloud, primary)**: `ExecutionLiteReward`
3. **Phase C (exploratory)**: `0.7 * ExecutionLiteReward + 0.3 * LLMEvalReward`

### 11.3 Benchmarks and datasets

**Benchmarks (from literature_reference.md and DiffuCoder)**:

- **HumanEval**:
  - 164 Python programming problems, canonical for code RL.
  - Use pass@1 and pass@10 as primary metrics.
- **MBPP**:
  - 974 small Python problems.
  - Use subset (e.g., 100–200 tasks) for faster eval.
- **Execution-lite internal dataset**:
  - Already wired via `data/execution_lite.json`.
  - Use for training and dev evaluation (cheaper than full HumanEval harness).

**Prompting templates**:

- Align with DiffuCoder / Qwen2.5-Coder templates where possible:
  - For HumanEval-like tasks:
    - System: `"You are a helpful assistant."`
    - User: include the description and function signature in a fenced block.
    - Assistant: "Here is the code to solve this problem: ```python ...```"
  - Use a single, consistent chat template across baseline GRPO and tree-based GRPO to remove template as a confound.

**Plan**:

- Implement a small "registry" builder that can convert a subset of HumanEval-style tasks into the `execution_lite` JSON format, so ExecutionLiteReward can be reused with Dream.
- For full HumanEval/MBPP:
  - use external evaluation harnesses (OpenAI human-eval repo, EvalPlus) *after* training; keep training-time reward based on the internal registry for simplicity.

### 11.4 Trainer wiring

Update or extend `dream/src/trainer.py`:

- Allow passing any `RewardFunction` from `dream.src.rewards`.
- For Dream code runs on GPU:

```python
from dream.src.config import MCTSConfig
from dream.src.utils import load_model_and_tokenizer
from dream.src.rewards import ExecutionLiteReward, SyntaxReward
from dream.src.trainer import EntropyMCTSTrainer

cfg = MCTSConfig(model_type="dream")
model, tokenizer = load_model_and_tokenizer(cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

reward_fn = ExecutionLiteReward(registry_path="data/execution_lite.json")
# or: reward_fn = SyntaxReward()

trainer = EntropyMCTSTrainer(model, tokenizer, cfg, reward_fn, optimizer)
```

**Verification checklist**:

- [ ] SyntaxReward works as a drop-in for the existing minimal trainer test (already covered by `test_trainer_minimal.py`-style usage).
- [ ] ExecutionLiteReward can be imported and instantiated in a small cloud smoke test (no local heavy execution).
- [ ] Trainer signature accepts any `RewardFunction` from `dream.src.rewards`.
- [ ] For cloud GRPO experiments, reward source is clearly recorded in run configs (e.g., `cfg.run_name`, WandB tags).

**Dependencies**: Steps 1–10 (for full Dream usage); for local-only work, just Steps 1–7 and this reward design.

---

## Step 12: Baseline GRPO on Dream

**Objective**: Get standard trajectory-level GRPO (no tree) working on Dream as the comparison baseline.

**File**: Part of `dream/src/trainer.py` (BaselineGRPOTrainer, to be added later)

### Specification

Use Dream's `diffusion_generate` with `output_history=True` to extract transitions. This is the "fair comparison" baseline — same model, same reward, same sampling; only difference is tree vs no tree.

### Verification

- [ ] Baseline GRPO runs for 10 steps without crash
- [ ] Loss decreases over 10 steps
- [ ] Generated completions are coherent code

**Dependencies**: Steps 1–11.

---

## Step 13: Full Training + Evaluation

**Objective**: Multi-epoch training on code generation prompts, evaluated on HumanEval/MBPP.

### Experimental Conditions

| Condition | Method | Config |
|-----------|--------|--------|
| A | Dream IT (untrained baseline) | No training |
| B | Dream IT + standard GRPO | `BaselineGRPOTrainer` |
| C | Dream IT + entropy-MCTS GRPO (fixed step) | `adaptive_stepping=False` |
| D | Dream IT + entropy-MCTS GRPO (adaptive) | `adaptive_stepping=True` |

### Metrics

- pass@1, pass@10 on HumanEval
- pass@1, pass@10 on MBPP
- Training loss curves
- Diversity of tree leaves (number of unique completions)
- Compute cost per training step (wall-clock time, peak GPU memory)

### Reference Numbers

From Dream paper:
- Dream IT: HumanEval baseline in paper
- Dream-Coder IT: HumanEval/MBPP numbers from paper (their GRPO result)

**Dependencies**: Steps 1–11.

---

## Appendix A: Requirements

**File**: `dream/requirements.txt`

```
transformers==4.46.2
torch==2.5.1
wandb
deepspeed
numpy
```

Dream requires pinned `transformers==4.46.2` and `torch==2.5.1` for `SdpaAttention` compatibility. Use a separate virtual environment for Dream work.

---

## Appendix B: What the Three Fixes Actually Change

### Before (parent `src/`)

**Entropy weight in loss.py line 93-99**:
```python
expected_h = self.entropy_computer.expected_entropy(
    node.masking_ratio(), vocab_size=vocab_size
)
ew = (node.entropy or 0.0) / expected_h
# Result: H_masked_mean / (masking_ratio * log(V))
# Problem: inflates as masking_ratio → 0
```

**Time weight in loss.py line 92**:
```python
tw = self.time_weighter.get_weight(node.step_index)
# Result: single point weight, normalized so sum = 1 → each w(t) ~ O(1/T)
# Problem 1: doesn't account for edge length
# Problem 2: O(1/T) scale vs O(1) entropy scale
```

### After (dream `src/`)

**Entropy weight**:
```python
ew = self.entropy_computer.compute_entropy_weight(
    measured_masked_mean=node.entropy or 0.0,
    vocab_size=vocab_size,
    mode="analytic",
)
# Result: H_masked_mean / log(V)
# Range: [0, 1] for well-behaved models
# Independent of masking ratio
```

**Time weight**:
```python
tw = self.time_weighter.get_interval_weight(parent_step, child_step)
# Result: sum of w(t) over [parent_step, child_step)
# With mean_to_one normalization: each w(t) ~ O(1)
# Longer edges get proportionally more weight
```

---

## Appendix C: Adaptive Branching Threshold Tuning Guide

The `branch_threshold` parameter controls when denoising stops to create a branch point. It is compared against `H_masked_mean / log(V)`.

**Expected ranges** (to be calibrated in Step 8):
- Fully masked (`r = 1.0`): `H/log(V)` typically 0.3–0.6 (model is uncertain but has some structure from the prompt)
- Half masked (`r = 0.5`): `H/log(V)` typically 0.1–0.3
- Nearly complete (`r = 0.1`): `H/log(V)` typically 0.02–0.1

**Tuning strategy**:
1. Run Step 8 validation to get the empirical entropy profile
2. Set `branch_threshold` slightly above the typical `H/log(V)` at the 50th percentile masking ratio
3. This means branching happens when uncertainty is higher than usual, targeting decision points
4. If too many branches happen (tree depth is shallow): increase threshold
5. If too few branches happen (always hits max_steps): decrease threshold

A good starting point is `branch_threshold` = median of the empirical `H/log(V)` profile + 0.1.

---

## Appendix D: Tree forwards vs GRPO loss — GPU memory (deep dive)

This appendix explains **why `loss_backward_per_transition` alone may not fix OOM** on ~32GB GPUs with Dream 7B, and lists **optimization options** in priority order. It mirrors the actual call graph in `dream/src/tree_builder.py`, `dream/src/loss.py`, and `dream/src/trainer.py`.

### D.1 Two phases, two different memory stories

| Phase | Code | `grad` / mode | What runs |
|--------|------|---------------|-----------|
| **A — Tree** | `EntropyGuidedTreeBuilder.build_tree` | `torch.no_grad()`, `model.eval()` | Many **full-sequence** forwards: length \(L \approx\) `prompt_len + max_new_tokens`. Per expansion: up to `branch_width` × `steps_per_expansion` adapter forwards; completion (`_denoise_to_completion`) adds more. Entropy uses additional forwards. |
| **B — Loss** | `WeightedGRPOLoss.compute_loss` | `model.train()`, autograd **on** | **One forward per distinct parent node** (default `loss_group_by_parent=True`): siblings share `parent_state`, so we run `adapter.forward_logits` once per parent, add all child log-prob terms, then **one backward per parent group**. Metric `n_loss_forwards` ≤ `n_transitions`. |

**Retained after phase A (GPU):**

- Model weights (bf16 ~14GB for 7B).
- **Every** `MCTSNode.state` and `attention_mask` (`LongTensor` on device) — size \(\mathcal{O}(\text{nodes} \times L)\); negligible vs weights.
- **No** autograd graph from phase A.

**Phase B peak (per parent group, sequential):**

- Weights + **gradients for all trainable parameters** (full fine-tune: same order as weights in bf16, ~14GB each → ~28GB before activations).
- **Optimizer state** (AdamW adds two fp32 moments per param if used — **if SGD still OOMs, optimizer is not the binding constraint**).
- **One** full transformer forward + backward with **gradient checkpointing** (Dream sets `use_cache=False` when checkpointing is on).
- Final **logits** buffer \([L, V]\) — scales with \(L \times V\).

**Per-group backward** (default): stitches only **sibling** edges that share a parent into one subgraph before backward; peak through the trunk is still **one** full 7B backward. **LoRA** shrinks trainable params so weight+grad footprint drops to \(\mathcal{O}(\text{LoRA params})\).

### D.2 Scaling (order of magnitude)

Let \(T\) = `n_transitions`, \(G\) = number of distinct parent tensors = `n_loss_forwards` (≤ \(T\), often ≈ internal nodes with ≥1 child).

- **Wall time** for loss: **\(G\)** forwards + **\(G\)** backwards (grouped path), not \(T\) of each.
- **Peak VRAM**: still \(\mathcal{O}(1)\) in \(T\) — one backward at a time; grouping saves **time** and **autograd graph size**, not the asymptotic **single-backward** activation peak through 7B.
- **Sequence length** \(L\): linear in activations through layers; **vocab** \(V\): logits row is \(\mathcal{O}(L \times V)\).

Concrete sanity check (not exact): \(L=512\), \(V \approx 1.5\times10^5\) → logits alone in fp16 \(\approx 512 \times 152064 \times 2 \,\mathrm{B} \approx 150\,\mathrm{MB}\) per tensor life; full layer stack is much larger.

### D.2b Full fine-tune vs ~32GB (empirical lesson)

Rough accounting for **7B** in bf16:

- **Weights** \(\approx 14\) GB.
- **Gradients** (one backward) \(\approx 14\) GB if stored in bf16 for all params.
- **Total \(\approx 28\) GB** before any activation / checkpoint / cudnn workspace — **already near a 32GB cap**.

So if **`--optimizer sgd` still OOMs**, that **confirms** the limit is **weights + grads + backward workspace**, not Adam. The practical fix on one GPU is **PEFT LoRA** (or multi-GPU sharding), not more tree micro-optimizations alone.

### D.3 Why this is a “deeper” optimization problem

1. **Loss requires \(p(\text{child}\mid\text{parent})\)** under the **current** policy. That is computed by a **full forward** on the parent diffusion state. There is no implemented shortcut that only runs a subset of layers without Dream API support.
2. **Tree-shaped win we *can* use**: **siblings share one parent state** — mathematically \(\sum_i \nabla \log p(c_i\mid p) = \nabla \sum_i \log p(c_i\mid p)\) when logits are shared. Implemented as **`loss_group_by_parent`** (default `True` in `MCTSConfig`).
3. **Caching KV from tree build** for loss is **not** automatically valid for training: build runs under `no_grad`; recomputing with grad is required for policy-gradient correctness unless you adopt an explicit off-policy / replay objective (research).
4. **Sparsifying `log_softmax` over \(V\)** (e.g. only at masked indices) still needs hidden states at those positions from the stack — memory win is possible with custom kernels / model hooks, not yet in this repo.

### D.4 Mitigation tiers (recommended order)

**Tier 0 — Measure (verifiable)**

1. Run `single_step_dream.py` with `--profile-memory` and note `peak CUDA allocated`, `n_transitions`, and **`n_loss_forwards`** (should be ≤ `n_transitions` when grouping is on).
2. Set env `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and retry (reduces fragmentation OOMs that look like “need 130MB”).
3. Confirm failure phase: tree-only script (`validate_dream_tree.py`) succeeding while `single_step_dream.py` fails implies bottleneck is **phase B + weights/grads**, not tree sampling.

**Tier 1 — Knobs (no new algorithms)**

- Reduce `--max-new-tokens` (linear effect on \(L\)).
- Reduce `--max-tree-nodes` / `--steps-per-expansion` (fewer phase-A forwards; **\(T\)** may still be \(\sim\) edges in final tree).
- `MCTSConfig.cuda_empty_cache_after_tree=True` (default): `torch.cuda.empty_cache()` after advantages, before loss (helps allocator fragmentation, not mathematical peak).

**Tier 2 — Optimizer / training mode**

- **`--lora`** in `single_step_dream.py` (requires `peft` in `dream/requirements.txt`): **primary fix for ~32GB** — trains adapter matrices only; base Dream weights stay frozen; gradients are tiny.
- **`--optimizer sgd`**: if this **still** OOMs, Adam was never the issue; proceed to LoRA or more VRAM.
- **8-bit / paged optimizers** (e.g. `bitsandbytes`): lower optimizer footprint once trainable set is already small.
- **Multi-GPU**: FSDP / DeepSpeed ZeRO for sharded full fine-tune.

**Tier 3 — Algorithm / implementation (future work)**

- **CPU-side tree states**: store `MCTSNode.state` on CPU; move one edge at a time to GPU — saves little VRAM (states are small) but can help fragmentation.
- **Parent-grouped loss** (implemented): `MCTSConfig.loss_group_by_parent` — one 7B forward per parent for all children; fewer passes and smaller graph than per-edge forwards.
- **Approximate objectives** / **windowed logits** (would need theory + Dream changes).

### D.5 Verifiable checklist (copy-paste)

```bash
# V1 — Peak memory + transition count (metrics line includes n_transitions)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python dream/scripts/single_step_dream.py --profile-memory \
  --max-tree-nodes 5 --max-new-tokens 128 --steps-per-expansion 16

# V2 — SGD (if this still OOMs, optimizer was never the bottleneck)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python dream/scripts/single_step_dream.py --optimizer sgd \
  --max-tree-nodes 5 --max-new-tokens 96 --steps-per-expansion 12

# V3 — LoRA (expected to work on ~32GB interactive)
pip install 'peft>=0.13.0'   # if not already from dream/requirements.txt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python dream/scripts/single_step_dream.py --lora --profile-memory \
  --max-tree-nodes 5 --max-new-tokens 96 --steps-per-expansion 12
```

**Success criteria:** V3 should complete on typical 32GB GPUs. Compare `n_loss_forwards` vs `n_transitions` to see sibling grouping. For **full** fine-tune on one GPU, plan on **40GB+** or ZeRO/FSDP.

---

## Appendix E: Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| OOM during tree building | Medium | High | Gradient checkpointing; reduce `max_tree_nodes` to 10; reduce `max_new_tokens` to 128 |
| OOM during loss / backward (32GB, 7B) | High | High | **`--lora`** (PEFT); Appendix D; SGD still OOM ⇒ use LoRA or 40GB+ / ZeRO |
| Right-shift logits applied incorrectly | Medium | High | Phase 8: compare adapter entropy to Dream's internal `alg="entropy"` ordering |
| `transformers==4.46.2` version conflict | High | Medium | Separate conda/venv for Dream |
| Branch threshold poorly calibrated | Medium | Medium | Phase 8 entropy profiling; start conservative |
| Corrected time weighting changes training dynamics | Low | Medium | Compare fixed-step with old vs new weighting first |
| Training instability at 7B scale | Medium | Medium | Conservative LR (5e-6), advantage clipping, gradient clipping |

---

## Appendix F: Quick Reference — What Goes Where

| Conceptual piece | File | Key function/class |
|---|---|---|
| Corrected entropy normalization | `dream/src/entropy.py` | `EntropyComputer.compute_entropy_weight(mode="analytic")` |
| Interval-aware time weighting | `dream/src/time_weight.py` | `TimeWeighter.get_interval_weight(start, end)` |
| Scale balance (time vs entropy) | `dream/src/time_weight.py` | `TimeWeighter(norm_mode="mean_to_one")` |
| Adaptive branching | `dream/src/tree_builder.py` | `_denoise_chunk_adaptive()` |
| Dream model abstraction | `dream/src/model_adapter.py` | `ModelAdapter.forward_logits()` |
| Dream right-shift | `dream/src/model_adapter.py` | Inside `forward_logits` for `model_type="dream"` |
| Dream categorical sampling | `dream/src/model_adapter.py` | `_dream_sample()` |
| Edge step tracking | `dream/src/tree_node.py` | `MCTSNode.steps_in_edge`, `TreeTransition.child_step_index` |
| Sibling-grouped loss forwards | `dream/src/loss.py` | `_parent_groups()`, `loss_group_by_parent` |
| Dream + LoRA load | `dream/src/utils.py` | `apply_lora_to_dream_model()`, `MCTSConfig.use_lora` |
