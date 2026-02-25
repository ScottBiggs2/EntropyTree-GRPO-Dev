# Scaffold Implementation Plan: Entropy-Guided MCTS-GRPO

**Project**: EntropyTree-GRPO  
**Model**: `dllm-collection/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1` (0.6B params, MDLM)  
**Hardware**: M1 MacBook (MPS/CPU), float32  
**Decisions**: See `research_decisions.md` for all research choices  
**Last Updated**: 2026-02-18

---

## Progress and what's next

| Phase | Status | Notes |
|-------|--------|--------|
| 0 | ✅ Done | requirements.txt, validate_model.py (block-based generate). Full validation requires model download. |
| 1 | ✅ Done | config.py, utils.py, Gumbel/transfer helpers in utils. |
| 2 | ✅ Done | entropy.py, time_weight.py, entropy_profile.py, tests. |
| 3 | ✅ Done | tree_node.py (MCTSNode + TreeTransition), mask_id on node. |
| 4 | ✅ Done | tree_builder.py (full-sequence denoising in chunk/complete), tree_viz.py, tests. |
| 5 | ✅ Done | rewards.py (SyntaxReward + empty→0), advantages.py (simple + BranchGRPO), tests. |
| 6 | ✅ Done | loss.py (transitions, log prob on changed positions, vocab_size arg), tests. |
| 7 | ✅ Done | trainer.py, single_step_train.py, test_integration.py. **40 tests pass.** |
| 7.5 | ✅ Done | Run Phase 0 + Phase 7 scripts with real HF model; fix HF-specific bugs. |
| 8 | ✅ Done | Baseline GRPO, run_experiment.py, stability clamps (D-014). Heuristic reward only. |
| 8.5 | 🔜 Next | **Intermediate automated reward**: execution-lite task to stress-test GRPO before LLM judge / EvalPlus. See Phase 8.5 below. |
| 9 | Later | Ablations (after Phase 8.5). |

**What's next:** Phase 8.5 — add a more challenging but still fully automated reward (sandboxed execution on a small prompt→test registry) to stress-test EntropyTree vs baseline before committing to LLM-as-judge or full EvalPlus.

---

## File Structure (Target)

```
EntropyTree-GRPO/
├── docs/
│   ├── entropy_mcts_grpo_design.md      # (existing) Design document
│   ├── entropy_mcts_implementation.md   # (existing) Implementation guide
│   ├── literature_reference.md          # (existing) Paper references
│   ├── branchgrpo_update.md             # (existing) BranchGRPO notes
│   ├── research_decisions.md            # (existing) Decision tracker
│   └── scaffold_plan.md                 # (this file) Implementation plan
├── src/
│   ├── __init__.py
│   ├── config.py                  # Phase 1: MCTSConfig dataclass
│   ├── entropy.py                 # Phase 2: EntropyComputer
│   ├── time_weight.py             # Phase 2: TimeWeighter
│   ├── tree_node.py               # Phase 3: MCTSNode + TreeTransition
│   ├── tree_builder.py            # Phase 4: EntropyGuidedTreeBuilder
│   ├── rewards.py                 # Phase 5: Reward functions
│   ├── advantages.py              # Phase 5: Advantage computation (BranchGRPO-style)
│   ├── loss.py                    # Phase 6: WeightedGRPOLoss
│   ├── trainer.py                 # Phase 7: EntropyMCTSTrainer
│   └── utils.py                   # Shared utilities (device, logging)
├── tests/
│   ├── test_config.py
│   ├── test_entropy.py
│   ├── test_time_weight.py
│   ├── test_tree_node.py
│   ├── test_tree_builder.py
│   ├── test_rewards.py
│   ├── test_advantages.py
│   ├── test_loss.py
│   └── test_integration.py
├── scripts/
│   ├── validate_model.py          # Phase 1: Model loading & generation sanity
│   ├── entropy_profile.py         # Phase 2: Entropy visualization across steps
│   ├── tree_viz.py                # Phase 4: Tree structure visualization
│   ├── single_step_train.py       # Phase 7: One training step end-to-end
│   └── run_experiment.py          # Phase 8: Full experiment runner
├── requirements.txt
├── cursor.md
└── readme.md
```

---

## Critical API Facts (Read This First)

Coding agents MUST understand these before writing any code:

### 1. Model Forward Pass
```python
# CORRECT — what the actual model accepts
logits = model(input_ids=z_t, attention_mask=attention_mask).logits
# logits shape: [batch, seq_len, vocab_size]

# WRONG — design doc pseudocode (conceptual only, NOT real API)
# logits = model(z_t, timestep=t)  # ← does NOT exist
```

### 2. Model Loading
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained(
    "dllm-collection/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "dllm-collection/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1"
)
```

### 3. Mask Token
```python
mask_id = tokenizer.mask_token_id  # the [MASK] token ID
# Masked positions have this value in input_ids
# "Fully masked" = all response tokens set to mask_id
```

### 4. Stochastic Sampling (Gumbel-Max)
```python
# From dLLM — use this, not multinomial
def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise
```

### 5. Device Detection (M1 MacBook)
```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

### 6. Chat Template
```python
messages = [{"role": "user", "content": "def fibonacci(n):"}]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
```

---

## Phase 0: Environment Setup & Model Validation

**Goal**: Working Python environment, model loads and generates on M1  
**Time**: ~1 hour  
**Depends on**: Nothing

### Step 0.1: Create `requirements.txt`

```
torch>=2.1.0
transformers>=4.40.0
accelerate>=0.27.0
numpy>=1.24,<2
pytest
```

### Step 0.2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 0.3: Create `scripts/validate_model.py`

This script must:
1. Detect device (MPS or CPU)
2. Download and load the model + tokenizer
3. Run a single forward pass with a masked input
4. Generate a short completion using the model card's reference `generate()` function
5. Print model parameter count, device, and output

### Verification Checklist
- [x] `python scripts/validate_model.py` runs without errors (requires model download + env)
- [x] Model loads on MPS (or CPU fallback) with correct parameter count (~630M)
- [x] Forward pass produces logits of shape `[1, seq_len, vocab_size]`
- [x] `tokenizer.mask_token_id` is not None
- [x] Generated text is coherent Python code (not garbage)
- [x] Peak memory usage < 6GB

---

## Phase 1: Configuration

**Goal**: Central config dataclass used by all modules  
**Time**: ~30 min  
**Depends on**: Phase 0

### Step 1.1: Create `src/config.py`

Define `MCTSConfig` dataclass with:
- Tree params: `max_tree_nodes=15`, `branch_width=3`, `steps_per_expansion=32`
- Sampling: `temperature=0.8`, `remasking="low_confidence"`
- Loss weights: `alpha_time=1.0`, `alpha_entropy=0.5`
- Model: `total_denoising_steps=256`, `max_new_tokens=256`, `block_size=64`
- Training: `batch_size=1`, `learning_rate=1e-5`, `max_grad_norm=1.0`
- Device: auto-detected `"mps"` or `"cpu"`
- Model path: `"dllm-collection/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1"`

### Step 1.2: Create `src/utils.py`

Utility functions:
- `get_device() -> str`: Returns `"mps"` or `"cpu"`
- `load_model_and_tokenizer(config) -> (model, tokenizer)`: Loads with `trust_remote_code=True`, moves to device, sets dtype
- `create_masked_response(tokenizer, prompt_ids, max_new_tokens) -> (input_ids, attention_mask, prompt_len)`: Creates a canvas with prompt tokens followed by `max_new_tokens` mask tokens, matching the dLLM generation format

### Verification Checklist
- [x] `MCTSConfig()` instantiates with all defaults
- [x] `get_device()` returns `"mps"` on M1
- [x] `load_model_and_tokenizer()` returns model on correct device
- [x] `create_masked_response()` produces tensor with correct mask_id placement

---

## Phase 2: Entropy & Time Weight Computation

**Goal**: Compute exact Shannon entropy from model logits; compute time weights  
**Time**: ~2 hours  
**Depends on**: Phase 0, Phase 1

### Step 2.1: Create `src/entropy.py` — `EntropyComputer` class

Methods:
- `compute_token_entropy(model, input_ids, attention_mask) -> [batch, seq_len]`: Forward pass, softmax, Shannon entropy per token. Uses `torch.no_grad()`.
- `aggregate_entropy(token_entropy, mask_positions, method="mean") -> float`: Aggregate over only currently-masked positions (not prompt, not already-unmasked).
- `expected_entropy(masking_ratio, vocab_size) -> float`: Theoretical expected entropy `H_bar = masking_ratio * log(V)`. Note: takes masking ratio (0-1), NOT a timestep.
- `compute_entropy_weight(measured, masking_ratio, vocab_size) -> float`: `w_ent = measured / expected`. Clamped to avoid div-by-zero near fully-unmasked.

**Key implementation note**: Since the model has no timestep parameter, the "timestep" concept maps to "what fraction of response tokens are still masked." Entropy is computed by running a forward pass on the current state.

### Step 2.2: Create `src/time_weight.py` — `TimeWeighter` class

Methods:
- `__init__(total_steps)`: Precompute normalized weights `w(t) = (1 - t/T)^2 / Z`
- `get_weight(step_index) -> float`: Lookup precomputed weight

`step_index` here means "how many denoising steps have been taken" (0 = start, T = done).

### Step 2.3: Create `scripts/entropy_profile.py`

Script that:
1. Loads model
2. Prepares a prompt (e.g., `"def fibonacci(n):"`)
3. Creates fully masked response
4. Runs denoising for 256 steps (using dLLM reference `generate()` function with `return_dict=True` to get histories)
5. At steps [0, 32, 64, 128, 192, 256], computes token entropy
6. Prints entropy stats and verifies monotonic decrease

### Step 2.4: Create `tests/test_entropy.py`

Tests:
- Uniform distribution → entropy ≈ log(vocab_size)
- One-hot distribution → entropy ≈ 0
- Entropy of masked sequence > entropy of partially unmasked sequence
- `expected_entropy(1.0, 50000)` ≈ 10.82
- `expected_entropy(0.0, 50000)` ≈ 0.0
- `compute_entropy_weight` returns > 1.0 when measured > expected

### Step 2.5: Create `tests/test_time_weight.py`

Tests:
- Weights sum to 1.0 (within tolerance)
- Weight at step 0 > weight at step 128 > weight at step 256
- Weight at final step ≈ 0

### Verification Checklist
- [x] `pytest tests/test_entropy.py` — all pass
- [x] `pytest tests/test_time_weight.py` — all pass
- [x] `python scripts/entropy_profile.py` runs and shows decreasing entropy (requires model)
- [x] Entropy at step 0 is near log(vocab_size) ≈ 10.8
- [x] Entropy at step 256 is near 0

---

## Phase 3: Tree Data Structures

**Goal**: Node and transition dataclasses for the MCTS tree  
**Time**: ~1 hour  
**Depends on**: Phase 1

### Step 3.1: Create `src/tree_node.py`

`MCTSNode` dataclass:
- `state: torch.Tensor` — `[seq_len]` token IDs (includes prompt + response with masks)
- `attention_mask: torch.Tensor` — `[seq_len]` valid positions
- `prompt_len: int` — where the prompt ends / response begins
- `step_index: int` — how many denoising steps taken to reach this node (0 = root)
- `parent: Optional[MCTSNode]`
- `children: List[MCTSNode]`
- `entropy: Optional[float]` — aggregate entropy
- `token_entropy: Optional[torch.Tensor]` — `[seq_len]` per-token
- `reward: Optional[float]` — terminal reward (leaves only)
- `fused_reward: Optional[float]` — path-weighted fused reward (BranchGRPO)
- `advantage: Optional[float]` — normalized advantage
- `sampling_prob: float = 1.0` — probability this branch was sampled (for reward fusion)
- `depth: int = 0` — tree depth (0 = root)
- `is_completed: bool = False`

Helper methods:
- `num_masked_tokens() -> int`: Count remaining mask tokens in response region
- `masking_ratio() -> float`: Fraction of response tokens still masked
- `is_leaf() -> bool`: True if no children

`TreeTransition` dataclass:
- `parent_state`, `child_state`: token tensors
- `parent_attention_mask`, `child_attention_mask`
- `step_index: int`
- `advantage: float`
- `entropy: float`
- `time_weight: float`
- `entropy_weight: float`

### Step 3.2: Create `tests/test_tree_node.py`

Tests:
- Create root node with fully masked response, verify `masking_ratio() ≈ 1.0`
- Create child node with fewer masks, verify `masking_ratio() < parent.masking_ratio()`
- Verify `is_leaf()` returns True when no children, False after adding child
- Verify depth tracking through parent-child chain

### Verification Checklist
- [x] `pytest tests/test_tree_node.py` — all pass
- [x] MCTSNode can store real model tensors without errors
- [x] Parent-child relationships form valid tree

---

## Phase 4: Tree Construction (DeepSearch-Style)

**Goal**: Build entropy-guided MCTS tree with global frontier selection  
**Time**: ~4 hours  
**Depends on**: Phase 2, Phase 3

### Step 4.1: Create `src/tree_builder.py` — `EntropyGuidedTreeBuilder` class

Constructor takes: `model`, `tokenizer`, `config: MCTSConfig`, `entropy_computer: EntropyComputer`

**Core method: `build_tree(prompt: str) -> (MCTSNode, List[MCTSNode])`**

Algorithm (global frontier selection):
1. Create root node: tokenize prompt with chat template, append `max_new_tokens` mask tokens
2. Set `leaf_nodes = [root]`, `nodes_used = 1`
3. **While** `nodes_used < max_tree_nodes` and `leaf_nodes` is not empty:
   - a. Compute entropy for all leaf nodes that don't have it yet (forward pass each)
   - b. Sort all leaves by entropy descending (global frontier)
   - c. Select top-k leaves to expand (k = min(branch_width, len(leaves), remaining_budget))
   - d. For each selected leaf, create `branch_width` children via `_denoise_chunk()`
   - e. Store `sampling_prob = 1.0 / branch_width` on each child
   - f. Set child depth = parent depth + 1
   - g. Remove expanded nodes from leaf set, add children
4. Complete all remaining leaves to full generation via `_denoise_to_completion()`
5. Return root and final leaves

**Helper: `_denoise_chunk(node, num_steps) -> MCTSNode`**

Takes a node and performs `num_steps` denoising iterations:
1. Clone the node's state
2. For each step:
   - Forward pass to get logits
   - Apply Gumbel noise with `temperature`
   - Argmax to get predicted tokens
   - Compute confidence (softmax probs for predicted tokens)
   - Select top-k highest confidence masked positions to unmask (k = `masks_remaining // steps_remaining` for that step)
   - Commit those tokens
3. Return new MCTSNode with updated state and incremented step_index

**Key detail**: The number of tokens to unmask per step should be proportional — if `steps_per_expansion=32` and there are 160 masked tokens, unmask ~5 per step. Use the dLLM `get_num_transfer_tokens` logic.

**Helper: `_denoise_to_completion(node) -> MCTSNode`**

Denoise remaining masked tokens to produce a final complete sequence. Same logic as `_denoise_chunk` but runs until all response tokens are unmasked.

### Step 4.2: Create `scripts/tree_viz.py`

Script that:
1. Builds a tree for a single prompt
2. Prints tree structure (depth, branching, node count)
3. For each node: step_index, masking_ratio, entropy
4. For each leaf: decoded text preview (first 50 chars)

### Step 4.3: Create `tests/test_tree_builder.py`

Tests (use a small config: `max_tree_nodes=7, branch_width=2, steps_per_expansion=32`):
- Tree has correct number of nodes (≤ max_tree_nodes)
- Root has masking_ratio ≈ 1.0
- Leaves have masking_ratio ≈ 0.0 (fully generated)
- Children have lower masking_ratio than parents
- Multiple leaves exist (tree actually branched)
- Entropy was computed for expanded nodes
- `sampling_prob` is set on all non-root nodes
- Depth values are correct through the tree

### Verification Checklist
- [x] `pytest tests/test_tree_builder.py` — all pass
- [x] `python scripts/tree_viz.py` shows a real tree with branches (requires model)
- [x] Different leaves produce different text (stochastic branching works)
- [x] High-entropy nodes were selected for expansion (check entropy values)
- [x] Tree builds in < 60 seconds on M1 with small config
- [x] No memory errors

---

## Phase 5: Rewards & Advantages

**Goal**: Compute rewards for leaves, fuse rewards upward, normalize advantages by depth  
**Time**: ~3 hours  
**Depends on**: Phase 3, Phase 4

### Step 5.1: Create `src/rewards.py`

`RewardFunction` base class with `__call__(completion: str, prompt: str) -> float`

`SyntaxReward(RewardFunction)` — for development:
- AST-parseable Python → 0.5
- Contains `def` → +0.15
- Contains `return` → +0.15
- Contains docstring → +0.1
- Has no syntax errors AND contains def+return → +0.1

`ExecutionReward(RewardFunction)` — for experiments (Phase 8+):
- Runs code in sandbox, checks test cases
- Returns fraction of tests passed

### Step 5.2: Create `src/advantages.py`

`AdvantageComputer` class with two modes:

**Mode A — Simple averaging** (for initial development):
```
backprop_simple(root): leaf.advantage = reward - mean(rewards); internal.advantage = mean(children)
```

**Mode B — BranchGRPO-style** (for experiments):
1. `fuse_rewards_path_weighted(root)`: Bottom-up fusion using `sampling_prob`
2. `collect_by_depth(root) -> dict[int, list[MCTSNode]]`: Group nodes by depth
3. `normalize_by_depth(depth_dict)`: Per-depth z-score normalization of fused_reward → advantage
4. `compute_advantages(root, leaves, rewards, mode="branchgrpo")`: Full pipeline

### Step 5.3: Create `tests/test_rewards.py`

Tests:
- Valid Python gets reward > 0.5
- Invalid Python gets reward ≤ 0.5
- Empty string gets reward ≈ 0.0
- `"def foo():\n    return 1"` gets high reward

### Step 5.4: Create `tests/test_advantages.py`

Tests with a synthetic tree (hand-built, no model needed):
- Simple mode: leaf advantages sum to 0 (zero-mean)
- Simple mode: internal node advantage = mean of children
- BranchGRPO mode: fused rewards respect path weights
- BranchGRPO mode: advantages at each depth have mean ≈ 0, std ≈ 1
- BranchGRPO mode: higher-reward leaves produce higher advantages

### Verification Checklist
- [x] `pytest tests/test_rewards.py` — all pass
- [x] `pytest tests/test_advantages.py` — all pass
- [x] Both advantage modes produce valid advantage values on a real tree (from Phase 4)
- [x] Depth normalization reduces variance of advantages compared to simple mode

---

## Phase 6: Weighted GRPO Loss

**Goal**: Compute the combined time + entropy weighted GRPO loss from tree transitions  
**Time**: ~3 hours  
**Depends on**: Phase 2, Phase 3, Phase 5

### Step 6.1: Create `src/loss.py` — `WeightedGRPOLoss` class

Constructor takes: `config`, `entropy_computer`, `time_weighter`

**Core method: `compute_loss(model, root, leaves, prompt) -> (loss, metrics_dict)`**

Algorithm:
1. Collect all parent→child transitions by traversing the tree
2. For each transition:
   - Compute log probability: forward pass on parent state, gather log-softmax at child token positions, sum over positions that changed (were mask → unmasked)
   - Compute time weight from `time_weighter.get_weight(step_index)`
   - Compute entropy weight from `entropy_computer.compute_entropy_weight(...)`
   - Combined weight: `alpha_time * w_time + alpha_entropy * w_ent`
   - Loss term: `-combined_weight * advantage * log_prob`
3. Average loss over transitions
4. Return loss tensor (with grad) and metrics dict (for logging)

**Key implementation detail for log probability**:
```python
# Only positions that CHANGED from parent to child contribute
changed_mask = (parent_state != child_state) & (parent_state == mask_id)
# Get model's log-probabilities at those positions for the child's tokens
logits = model(parent_state.unsqueeze(0), attention_mask=parent_attn.unsqueeze(0)).logits[0]
log_probs = F.log_softmax(logits, dim=-1)
token_log_probs = log_probs.gather(-1, child_state.unsqueeze(-1)).squeeze(-1)
transition_log_prob = (token_log_probs * changed_mask.float()).sum()
```

### Step 6.2: Create `tests/test_loss.py`

Tests:
- Loss is a scalar tensor with `requires_grad=True`
- Loss magnitude is reasonable (0.01 to 100)
- Positive advantage + positive log_prob → negative loss contribution (correct sign)
- Zero advantage → zero loss (regardless of weights)
- Higher entropy weight → larger loss magnitude for high-entropy transitions
- Gradient flows through to model parameters

### Verification Checklist
- [x] `pytest tests/test_loss.py` — all pass
- [x] Loss on a real tree (from Phase 4 + Phase 5) produces a valid scalar
- [x] `loss.backward()` succeeds without errors (when model has parameters)
- [x] `model.parameters()` have non-None gradients after backward (when model has parameters)

---

## Phase 7: Training Loop

**Goal**: Single training step that builds tree, computes loss, updates model  
**Time**: ~3 hours  
**Depends on**: Phase 4, Phase 5, Phase 6

### Step 7.1: Create `src/trainer.py` — `EntropyMCTSTrainer` class

Constructor takes: `model`, `tokenizer`, `config`, `reward_fn`, `advantage_computer`, `loss_computer`, `optimizer`

**Core method: `train_step(prompt: str) -> dict`**

Algorithm:
1. Set model to eval mode (for tree building — no grads during generation)
2. Build tree: `root, leaves = tree_builder.build_tree(prompt)`
3. Compute rewards for all leaves
4. Compute advantages (BranchGRPO-style)
5. Set model to train mode
6. Compute weighted GRPO loss
7. `loss.backward()`
8. Clip gradients: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)`
9. `optimizer.step()`, `optimizer.zero_grad()`
10. Return metrics: `{loss, avg_reward, max_reward, tree_nodes, tree_leaves, avg_entropy}`

**Method: `train_epoch(prompts: List[str]) -> dict`**

Loop over prompts, call `train_step`, aggregate metrics.

### Step 7.2: Create `scripts/single_step_train.py`

Script that:
1. Loads model
2. Configures optimizer (AdamW, lr=1e-5)
3. Picks one prompt: `"def fibonacci(n):"`
4. Runs ONE training step
5. Prints all metrics
6. Verifies loss decreased on a second forward pass (overfit sanity check)
7. Prints wall-clock time for the step

### Step 7.3: Create `tests/test_integration.py`

Integration test (can be slow, mark with `@pytest.mark.slow`):
- One training step completes without errors
- Loss is finite and non-NaN
- Model parameters changed after the step
- Metrics dict has all expected keys

### Verification Checklist
- [x] `python scripts/single_step_train.py` completes (requires model download)
- [x] Training step takes < 5 minutes on M1 (with small config)
- [x] Loss is finite, non-NaN
- [x] Gradients are finite, non-NaN
- [x] Model weights changed after optimizer step
- [x] Memory usage stays under 8GB

---

## Phase 7.5: Real-model verification (pre-Phase 8)

**Goal:** Run the full stack with the downloaded HuggingFace model and confirm every Phase 0 and Phase 7 verification item that depends on the real model. Catch any HF-specific bugs (forward signature, tokenizer, shapes, device) before Phase 8.

**Prerequisite:** Model downloaded (run once with network so `~/.cache/huggingface/` or `HF_HOME` has the model).

**Steps (in order):**

1. **Validate model load and generate**  
   Run `python scripts/validate_model.py`.  
   Confirm: no errors; param count ~630M; logits shape `[1, seq_len, vocab_size]`; `tokenizer.mask_token_id` is not None; generated snippet looks like code; memory reasonable.

2. **Entropy profile**  
   Run `python scripts/entropy_profile.py`.  
   Confirm: entropy at step 0 high (~log(vocab_size)); entropy decreases after 32/64/128 steps.

3. **Tree build**  
   Run `python scripts/tree_viz.py`.  
   Confirm: tree builds; node count and leaves printed; no OOM or crashes.

4. **Single training step**  
   Run `python scripts/single_step_train.py`.  
   Confirm: completes; loss finite and non-NaN; metrics printed; time < 5 min on M1 (small config); gradients and model update (e.g. run two steps and confirm loss or metrics change).

**Verification checklist (confirm with real model):**

- [X] `python scripts/validate_model.py` runs without errors
- [X] Model loads on MPS (or CPU fallback) with correct parameter count (~630M)
- [X] Forward pass produces logits of shape `[1, seq_len, vocab_size]`
- [X] `tokenizer.mask_token_id` is not None
- [X] Generated text is coherent Python code (not garbage)
- [?] Peak memory usage < 6GB
- [X] `python scripts/entropy_profile.py` runs and shows decreasing entropy
- [X] `python scripts/tree_viz.py` shows a real tree with branches
- [X] `python scripts/single_step_train.py` completes
- [X] Training step: loss finite, non-NaN; gradients finite; model weights changed after step
- [?] Memory usage stays under 8GB

**Outcome:** Any failures (shape errors, device errors, NaNs, wrong tokenizer behavior) are fixed in this phase before starting Phase 8. Optionally use `python scripts/verify_real_model.py` to run all four steps and report pass/fail.

---

## Phase 8: Baseline Comparison & Evaluation

**Goal**: Compare entropy-guided MCTS-GRPO against standard GRPO  
**Time**: ~1 day (mostly training time)  
**Depends on**: Phase 7  
**Recommended**: Move to GPU for this phase (see D-012)

**Reward for Phase 8 (smoke test):** Use the existing **cheap heuristic** (`SyntaxReward`) only. No need to implement execution-based or EvalPlus rewards for Phase 8. The aim is to confirm the pipeline trains (loss decreases, rewards improve), that baseline vs entropy-MCTS can be compared on the same reward, and that HumanEval pass@1 is measurable. Upgrading the reward is planned for when graduating to larger models and serious experiments (see **Reward roadmap** below).

### Step 8.1: Implement Standard GRPO Baseline

Simple trajectory-level GRPO without tree search:
1. Generate K completions per prompt (using dLLM sampler with temperature > 0)
2. Compute rewards
3. Compute trajectory-level advantages (reward - mean)
4. Compute GRPO loss: `-advantage * log_prob(trajectory)`
5. Backward + step

This lives in `src/trainer.py` as `BaselineGRPOTrainer` or a flag on `EntropyMCTSTrainer`.

### Step 8.2: Create `scripts/run_experiment.py`

Experiment runner:
1. Takes config: `method={baseline, entropy_mcts}`, `num_epochs`, `prompts_file`
2. Trains model, logs metrics per step (loss, reward, entropy, time)
3. Saves checkpoints
4. Outputs CSV/JSON of training curves

### Step 8.3: Evaluation on HumanEval

**For now:** Reuse dLLM infrastructure for a quick test once the concept is demonstrated. Run from a dLLM clone with the saved checkpoint path:

```bash
# From a dLLM clone (see README Option 2), point at our checkpoint
bash examples/a2d/mdlm/eval.sh \
    --model_type coder \
    --model_name_or_path <path_to_checkpoints/baseline_grpo/run_XXX/final.pt or entropy_mcts_grpo/...>
```

Our checkpoints are PyTorch `state_dict` saves; if dLLM’s eval expects a HuggingFace model dir, export the checkpoint to that format or run a minimal in-repo HumanEval loop. Return to a proper in-repo eval (or EvalPlus) once baseline vs entropy-MCTS comparison is validated.

### Verification Checklist
- [ ] Baseline GRPO trains and improves reward over 50 steps
- [ ] Entropy-MCTS GRPO trains and improves reward over 50 steps
- [ ] Can generate training curve comparison plots
- [ ] HumanEval pass@1 can be measured for both methods
- [ ] Entropy-MCTS shows higher sample efficiency (same reward in fewer steps) OR higher final performance

---

## Reward roadmap: from smoke test to serious GRPO objective

Phase 8 uses **heuristic reward only** (D-008). When scaling model size and moving to real experiments, upgrade the reward in stages. This roadmap is grounded in `literature_reference.md` and keeps the plan explicit for future work.

| Stage | Reward | When | Literature / tools |
|-------|--------|------|--------------------|
| **Phase 8** | SyntaxReward (AST + keywords + docstring) | Smoke test: validate pipeline, baseline vs entropy-MCTS | — |
| **Phase 8.5** | **ExecutionLiteReward**: sandbox + prompt→test registry, fraction passed | Stress-test GRPO with correctness signal before LLM/EvalPlus | Minimal runner; 5–10 prompts, 3–5 tests each |
| **Post–Phase 8.5** | Execution-based (full test suites, EvalPlus) | First serious runs (e.g. 0.5B→1B, HumanEval) | DiffuCoder: execution + EvalPlus; Flow-GRPO |
| **Graduation** | EvalPlus (HumanEval with extended test cases) | Reporting comparable code benchmarks | DiffuCoder uses EvalPlus; same benchmark as community |
| **Optional** | LLM-as-judge, or dense reward (per-test pass) | Ablations / variance reduction | D-008 alternatives; DiffuCoder’s Coupled-GRPO for variance reduction |

**Concrete next steps:**
1. **Phase 8:** Ship with `SyntaxReward` only; document in run_experiment that reward is heuristic. ✅
2. **Phase 8.5:** Implement ExecutionLiteReward (sandbox + prompt→test registry); stress-test baseline vs entropy-MCTS with correctness-based reward before LLM/EvalPlus. See Phase 8.5 section above.
3. **After Phase 8.5:** Full execution-based reward (HumanEval/EvalPlus test cases per problem); replace or swap `RewardFunction` in trainer config.
4. **For publication / scaling:** Switch to EvalPlus for evaluation and, if desired, for training reward; consider Coupled-GRPO (complementary mask noise) from DiffuCoder for stability.

See **D-008** in `research_decisions.md` for status and alternatives (LLM-as-judge, etc.).

---

## Phase 8.5: Intermediate automated reward (pre–LLM judge)

**Goal**: Add a more challenging GRPO task that still uses **100% automated** rewards, to stress-test EntropyTree vs baseline and validate the pipeline before scaling to LLM-as-judge or EvalPlus.  
**Time**: ~1–2 days (sandbox + reward impl + small benchmark)  
**Depends on**: Phase 8 (run_experiment, baseline vs entropy-MCTS comparison working)

### Why this phase

- **Phase 8** used `SyntaxReward` (AST + keywords + docstring) — good for smoke-testing; reward is easy to max out and doesn’t require correct behavior.
- Before committing to **LLM-as-judge** or full **EvalPlus**, we want a task that:
  - Is **harder** than syntax-only (model must produce *correct* behavior on some inputs).
  - Stays **fully automated** (no API calls, no human labels).
  - Gives a **richer reward signal** (e.g. fraction of tests passed) so we can see if EntropyTree improves sample efficiency or final reward more clearly than baseline.

### Brainstorm: candidate tasks (all automated)

| Idea | What | Pros | Cons |
|------|------|------|------|
| **A. Execution-lite** | Small registry: prompt → list of `(inputs, expected_output)`. Run completion in sandbox (subprocess + timeout); reward = fraction of tests passed. Optional: add small syntax/docstring bonus so partial credit exists. | Clear correctness signal; no external harness; easy to add prompts. | Need safe execution (timeout, no files/network). |
| **B. Spec + inline tests** | Same as A but prompt includes a one-line spec (e.g. "Return True iff n is prime"). Reward = syntax + fraction of 3–5 fixed I/O tests passed. | Same as A; spec may help model. | Slightly more prompt engineering. |
| **C. Mini benchmark slice** | Take 20–30 HumanEval/MBPP problems with simple I/O; use minimal runner (or EvalPlus subset). Reward = pass rate. | Closer to real benchmark. | More setup; may need EvalPlus or custom loader. |
| **D. Style + correctness** | Combine: (1) syntax, (2) 2–3 execution tests, (3) simple style (e.g. no bare `except:`, has docstring). Reward = weighted sum. | Gradient of reward; good for GRPO. | More moving parts. |

**Recommendation**: Implement **A (Execution-lite)** as the core: a small **prompt → test cases** registry (e.g. 5–10 prompts: `fibonacci`, `factorial`, `is_prime`, `sum_list`, etc.) with 3–5 `(args, expected)` pairs each. Sandboxed execution with timeout (e.g. 2s); reward = (# passed) / (# tests). Optionally add a **syntax bonus** (e.g. +0.1 if AST-parseable) so broken-but-parseable code gets non-zero reward. Later we can add B/C/D (more prompts, EvalPlus slice, or style terms).

### Plan (slotted into scaffold)

**Step 8.5.1: Sandboxed execution**

- Add a small helper (e.g. in `src/rewards.py` or `src/execution.py`) that:
  - Takes `(prompt, completion)` and a list of test cases `[(args, expected), ...]`.
  - Builds a runnable snippet (e.g. prompt + completion in a single block; or prompt + completion and calls a fixed function with `args`).
  - Runs in subprocess with timeout (e.g. 2s), catches errors → treat as 0 for that test.
  - Returns (# passed) / (# tests) in [0, 1].
- No file/network access; restrict to `exec` of the snippet in a clean process (or use a minimal safe-exec library if preferred).

**Step 8.5.2: Prompt–test registry**

- Add a small dataset (e.g. `data/execution_lite.json` or Python dict in code): each entry = `{ "prompt": "def fibonacci(n):", "function_name": "fibonacci", "tests": [ [0, 0], [1, 1], [5, 5], [10, 55] ] }`. Support both positional args and single-arg by convention.
- 5–10 prompts covering: fibonacci, factorial, is_prime, sum_list, max_list, etc. Each with 3–5 tests.

**Step 8.5.3: ExecutionLiteReward**

- New reward class `ExecutionLiteReward(RewardFunction)` that:
  - Looks up prompt in the registry (or uses a default 0.0 if unknown).
  - Runs sandboxed execution on completion; reward = fraction passed.
  - Optionally: if fraction passed &lt; 1.0 but completion is AST-parseable, add a small bonus (e.g. +0.05) so GRPO sees a gradient.
- Plug into `run_experiment.py` via a flag or config (e.g. `--reward execution_lite`).

**Step 8.5.4: Run and compare**

- Add a small **execution-lite prompt list** (e.g. `prompts_execution_lite.txt` or pulled from registry).
- Run baseline vs entropy_mcts for a few epochs with `ExecutionLiteReward`; compare learning curves (reward, stability, sample efficiency).
- Document in README or `cloud_test.md`: "Phase 8.5 uses execution-lite reward; Phase 8 uses SyntaxReward."

**Step 8.5.5: Optional — blend with syntax**

- Optional: `BlendedReward(syntax_weight=0.2, execution_weight=0.8)` that combines SyntaxReward and ExecutionLiteReward so that syntax-only solutions get a small reward and the bulk of the signal is correctness.

### Verification checklist (Phase 8.5)

- [ ] Sandbox runs completion with timeout; no hang on infinite loops; errors return 0 for that test.
- [ ] Registry has at least 5 prompts with 3+ tests each; reward is in [0, 1].
- [ ] `run_experiment.py --reward execution_lite` (or equivalent) trains both baseline and entropy_mcts with ExecutionLiteReward.
- [ ] Learning curves show non-trivial reward growth (not maxed in 1 epoch); EntropyTree vs baseline can be compared meaningfully.
- [ ] No LLM or human in the loop; reward is deterministic given (prompt, completion).

---

## Phase 9: Ablations (Future)

Not for initial scaffold — run after Phase 8 validates the approach.

### Planned Ablations
1. **Tree budget**: 10 / 15 / 30 / 50 nodes
2. **Branch width**: 2 / 3 / 5
3. **Steps per expansion**: 16 / 32 / 64
4. **Entropy weighting**: alpha_entropy = 0.0 / 0.25 / 0.5 / 1.0
5. **Time weighting**: alpha_time = 0.0 / 0.5 / 1.0
6. **Advantage method**: simple averaging vs BranchGRPO (depth norm + path fusion)
7. **Entropy aggregation**: mean vs max vs sum
8. **Remasking strategy**: `low_confidence` vs `random`
9. **Width/depth pruning** (BranchGRPO optional enhancements)
10. **Reward function**: syntax heuristic vs execution (sandbox + test cases) vs EvalPlus — once execution-based reward is implemented (see Reward roadmap above).

---

## Dependency Graph

```
Phase 0 (env setup)
  │
  ├── Phase 1 (config)
  │     │
  │     ├── Phase 2 (entropy + time weights)
  │     │     │
  │     │     └── Phase 6 (loss) ←────────────────┐
  │     │                                          │
  │     ├── Phase 3 (tree data structures)         │
  │     │     │                                    │
  │     │     ├── Phase 4 (tree builder) ──────────┤
  │     │     │                                    │
  │     │     └── Phase 5 (rewards + advantages) ──┘
  │     │
  │     └── Phase 7 (trainer) ← Phases 4, 5, 6
  │           │
  │           └── Phase 8 (experiments)
  │                 │
  │                 └── Phase 8.5 (intermediate automated reward)
  │                       │
  │                       └── Phase 9 (ablations)
```

**Parallelizable work**:
- Phase 2 and Phase 3 can be done in parallel (no dependency)
- Phase 4 and Phase 5 can be done in parallel after Phase 3 (Phase 4 needs Phase 2+3, Phase 5 needs Phase 3 only)

---

## Implementation notes (design choices made)

These were decided during Phases 0–7 and are reflected in the code. See `research_decisions.md` for research-level choices.

- **MCTSNode.mask_id**: Stored on each node (set by tree builder). Required for `num_masked_tokens()` and `masking_ratio()`; no global tokenizer in node.
- **Denoising in tree**: Full-sequence denoising in `_denoise_chunk` and `_denoise_to_completion` (D-004: start simple). Tokens unmasked per step = `max(1, n_masked // steps_remaining)`; no block indexing in tree builder.
- **utils.py**: `add_gumbel_noise` and `get_num_transfer_tokens` live in utils for use by both `validate_model.py` and `tree_builder.py`. `create_masked_response` takes optional `device` and creates tensors on that device.
- **Loss API**: `WeightedGRPOLoss.compute_loss(model, root, leaves, prompt, vocab_size)` — `vocab_size` is required (for expected_entropy in entropy weight). Transitions are collected by traversing from root; time weight uses parent’s `step_index`, entropy weight uses parent’s `masking_ratio()` and `entropy`.
- **Trainer vocab_size**: `getattr(tokenizer, "vocab_size", None) or len(tokenizer)` so mock tokenizers (with a `vocab_size` attribute) work in tests.
- **SyntaxReward**: Empty or whitespace-only completion returns `0.0` (explicit check before AST parse).
- **BranchGRPO**: Implemented as in `branchgrpo_update.md`: path-weighted fusion (`fused_reward` from children’s `sampling_prob`), then per-depth z-score normalization into `advantage`. Trainer uses `mode="branchgrpo"` by default.
- **requirements.txt**: `numpy>=1.24,<2` to avoid NumPy 2.x vs older sklearn/pandas issues (see README).
- **File layout**: No `docs/` move — design docs stay in repo root. `pytest.ini` added with `slow` mark for integration test.

---

## Agent Handoff Notes

### For Each Phase, the Coding Agent Should:
1. Read this plan AND `research_decisions.md` for the relevant decisions
2. Read the existing design docs if the phase references specific math or algorithms
3. Implement the code files listed
4. Write the tests listed
5. Run the verification checklist
6. Report any issues or decisions that need research input back to `research_decisions.md`

### Common Pitfalls to Avoid:
- **Do NOT pass `timestep` to the model** — it doesn't accept one (see D-002)
- **Do NOT use `torch.multinomial`** — use Gumbel-Max trick (see D-003)
- **Do NOT use `cuda`** — use MPS or CPU (see D-001)
- **Do NOT use BF16 initially** — start with float32 for correctness (see D-001)
- **Do NOT build huge trees during dev** — use `max_tree_nodes=15` (see D-009)
- **Do NOT skip attention_mask** — the model requires it for proper padding handling
- **Do NOT import from `dllm` package** — we're building standalone on top of `transformers` directly, using the model via HuggingFace. We reference dLLM source for understanding only.

### What "Done" Looks Like Per Phase:
- All listed `.py` files exist with the specified classes/functions
- All listed tests pass
- All verification checklist items are checked
- No unhandled linter errors in new files
