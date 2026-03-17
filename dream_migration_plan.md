# Migration Plan: MDLM 0.5B → Dream 7B

**Date**: 2026-03-09
**Goal**: Transfer the entropy-guided MCTS-GRPO pipeline from the toy MDLM setting to Dream 7B, including the entropy-threshold adaptive stepping extension.

---

## 1. Model Choice: Dream-v0-Instruct-7B

### Available Checkpoints

| Checkpoint | Description | SFT | RL | Code-tuned |
|---|---|---|---|---|
| `Dream-org/Dream-v0-Base-7B` | General base | No | No | No |
| **`Dream-org/Dream-v0-Instruct-7B`** | General instruct | **Yes** | **No** | No |
| `Dream-org/Dream-Coder-v0-Base-7B` | Code base | No | No | Yes (pretraining) |
| `Dream-org/Dream-Coder-v0-Instruct-7B` | Code instruct | Yes | Yes (GRPO) | Yes |

### Recommendation: `Dream-v0-Instruct-7B`

**Why not Dream-Coder IT**: Dream-Coder already has GRPO applied (with coupled sampling, clip-higher, no-entropy-loss). Applying our entropy-MCTS-GRPO on top creates GRPO-on-GRPO, confounding the attribution of improvements. We can't cleanly say whether gains come from our tree structure or from interacting with their RL training.

**Why not Dream-Coder Base**: No SFT — the model can't follow instructions. We'd need to do our own SFT first, which is a large separate engineering task and introduces another variable.

**Why Dream IT**: It has instruction-following (SFT) but no RL. Our entropy-MCTS-GRPO is the only reinforcement learning applied. This gives the cleanest experimental comparison:

- **Baseline**: Dream IT → standard GRPO (no tree) → evaluate on HumanEval/MBPP
- **Ours**: Dream IT → entropy-MCTS-GRPO (with adaptive stepping) → evaluate on HumanEval/MBPP
- **Reference**: Dream-Coder IT results from their paper (direct comparison of our tree-based approach vs. their coupled-GRPO on a comparable model)

The Dream team's blog reports that Dream IT already has reasonable coding ability from pretraining on code tokens — it's not starting from zero. And since their paper provides HumanEval/MBPP baselines for both Dream IT and Dream-Coder IT, we have ready-made comparison numbers.

---

## 2. Critical API Differences: MDLM vs. Dream

### 2.1 Model Loading

```python
# CURRENT (MDLM 0.5B)
from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.float32)

# DREAM (7B)
from transformers import AutoModel
model = AutoModel.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.bfloat16)
```

Key changes:
- `AutoModel` not `AutoModelForMaskedLM` (Dream has custom model code)
- `torch.bfloat16` not `torch.float32` (7B model requires it for memory; BF16 is stable on CUDA)
- Requires `transformers==4.46.2` and `torch==2.5.1`

### 2.2 Forward Pass

```python
# CURRENT (MDLM 0.5B)
logits = model(input_ids=z_t, attention_mask=mask).logits  # [B, L, V]

# DREAM (7B) — inside _sample()
logits = model(x, attention_mask, tok_idx).logits           # [B, L, V]
logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)  # RIGHT-SHIFT
```

**Critical difference: right-shifted logits.** Dream inherits an AR-style "predict next token" setup from its Qwen2.5 initialization. The logits at position $i$ predict position $i+1$ in the original output, so they must be shifted right by 1. This shift is baked into Dream's `_sample()` but NOT into the raw `model()` forward pass. **Our code must apply this shift manually whenever calling `model()` directly.**

`tok_idx` is a tensor of position IDs computed from the attention mask. When `attention_mask` has no padding (all 1s), `tok_idx` is just `torch.arange(L)`. When there IS padding, it's the cumulative sum of valid positions. For our tree builder (no batched padding), we can simplify:

```python
tok_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
```

### 2.3 Attention Mask Format

```python
# CURRENT (MDLM 0.5B) — 1D mask
attention_mask = torch.ones(seq_len)  # [L]

# DREAM (7B) — 4D bidirectional mask (when padding exists)
# shape [B, 1, L, L] — full bidirectional attention within valid tokens
attention_mask = torch.logical_and(
    mask.unsqueeze(1).unsqueeze(-2),
    mask.unsqueeze(1).unsqueeze(-1),
)
# Or simply "full" (string) when no padding
```

For tree building where we process one sequence at a time (no batched padding), we can pass `attention_mask="full"` and `tok_idx=None`, which Dream handles as full bidirectional attention.

### 2.4 Sampling Mechanism

```python
# CURRENT (MDLM 0.5B) — Gumbel-Max trick (D-003)
logits_n = add_gumbel_noise(logits, temperature)
x0_pred = torch.argmax(logits_n, dim=-1)

# DREAM (7B) — Categorical + top_p/top_k
# With temperature > 0:
logits_t = logits / temperature
logits_t = top_p_logits(logits_t, top_p=0.95)
probs = torch.softmax(logits_t, dim=-1)
x0 = dists.Categorical(probs=probs).sample()
confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
```

These are mathematically equivalent for sampling from a Categorical distribution. Gumbel-Max is one implementation; temperature+softmax+sample is another. We should **switch to Dream's Categorical sampling** to match the model's design and their evaluation protocol.

### 2.5 Timestep Schedule

```python
# CURRENT (MDLM 0.5B) — no explicit timestep; transfer count is n_masked // steps_left
k = min(k_cap, max(1, n_masked // steps_left))

# DREAM (7B) — linear timestep schedule from 1 → eps
timesteps = torch.linspace(1, eps, steps + 1)  # eps = 1e-3
t, s = timesteps[i], timesteps[i+1]
number_transfer_tokens = int(num_mask_token * (1 - s / t))
```

Dream's schedule unmaskes more tokens early and fewer late (since $(1 - s/t)$ increases as $t$ decreases). Our MDLM approach distributes unmaskings uniformly. For Dream, we should **adopt Dream's linear timestep schedule** for the denoising steps within `_denoise_chunk`, while our tree structure (branching, frontier selection) wraps around it.

### 2.6 Remasking / Ordering Strategy

Dream supports four `alg` options for deciding which tokens to commit:
- `origin`: random order
- `maskgit_plus`: highest softmax confidence first
- `topk_margin`: largest top1-top2 margin first
- `entropy`: lowest per-token entropy first (negative entropy as confidence)

Our MDLM code uses `low_confidence` (highest softmax probability first), which maps to `maskgit_plus`. Dream's recommended default is `alg="entropy"` — we should use that for consistency with their published results.

### 2.7 Summary of Interface Mapping

| Aspect | MDLM 0.5B (current) | Dream 7B (target) |
|---|---|---|
| Loader | `AutoModelForMaskedLM` | `AutoModel` |
| Dtype | `float32` | `bfloat16` |
| Forward | `model(input_ids, attention_mask)` | `model(x, attention_mask, tok_idx)` + right-shift |
| Sampling | Gumbel-Max | `Categorical(softmax(logits/T))` |
| Ordering | `low_confidence` (top softmax) | `alg="entropy"` (neg entropy) |
| Transfer count | uniform `n_masked // steps_left` | schedule-based `n_masked * (1 - s/t)` |
| Attention mask | `[B, L]` tensor, always 1s | `"full"` string or `[B, 1, L, L]` |
| Position IDs | Not used | `tok_idx` (cumsum of valid mask) |
| Context length | Unlimited (0.5B) | 2048 (Dream) |

---

## 3. Architecture: ModelAdapter Abstraction

Rather than scattering Dream-specific logic throughout the codebase, introduce a thin adapter layer that encapsulates the model-specific forward pass, sampling, and logit handling. The rest of the pipeline (entropy computation, tree builder, loss, advantages) interacts with the adapter.

```python
class ModelAdapter:
    """Encapsulates model-specific forward pass and sampling for tree building."""
    
    def __init__(self, model, tokenizer, model_type: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type  # "mdlm" or "dream"
        self.mask_id = tokenizer.mask_token_id
        self.device = next(model.parameters()).device
    
    def forward_logits(self, input_ids, attention_mask):
        """Return logits [B, L, V] with any model-specific transforms (e.g., right-shift) applied."""
        if self.model_type == "dream":
            tok_idx = self._compute_tok_idx(attention_mask)
            logits = self.model(input_ids, attention_mask="full", tok_idx=tok_idx).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)  # right-shift
            return logits
        else:  # mdlm
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits
    
    def sample_and_confidence(self, logits, mask_positions, temperature, top_p=0.95):
        """Sample tokens and return (x0_pred, confidence) for masked positions."""
        if self.model_type == "dream":
            return self._dream_sample(logits, mask_positions, temperature, top_p)
        else:  # mdlm
            return self._mdlm_sample(logits, mask_positions, temperature)
    
    def transfer_count(self, n_masked, step, total_steps, eps=1e-3):
        """How many tokens to unmask at this step."""
        if self.model_type == "dream":
            timesteps = torch.linspace(1, eps, total_steps + 1)
            t, s = timesteps[step].item(), timesteps[step + 1].item()
            return max(1, int(n_masked * (1 - s / t)))
        else:  # mdlm
            steps_left = total_steps - step
            return max(1, n_masked // steps_left)
```

This keeps the tree builder, entropy computer, and loss module model-agnostic.

---

## 4. Entropy Computation: What Changes

Shannon entropy is computed from `softmax(logits)`. For Dream, the only change is that **logits must be right-shifted before computing entropy**. Since `ModelAdapter.forward_logits()` handles the shift, the `EntropyComputer` class requires **zero changes** — it receives pre-shifted logits.

```python
# Unchanged — EntropyComputer.compute_token_entropy still works:
logits = adapter.forward_logits(input_ids, attention_mask)   # right-shift already applied
probs = F.softmax(logits, dim=-1)
entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)   # [B, L]
```

The entropy normalization (`expected_entropy = masking_ratio * log(V)`) also transfers directly — Dream's vocabulary comes from Qwen2.5, so $V \approx 151,936$, giving $\log V \approx 11.93$ (vs. $\approx 10.82$ for the MDLM tokenizer with $V \approx 50,000$). The `entropy_weight_min` and `entropy_weight_max` clamps (D-014) will handle any scaling differences.

---

## 5. Entropy-Threshold Adaptive Stepping

This is included as part of the migration, since it meshes well with Dream's schedule-based denoising and doesn't require persistent memory across expansions.

### Modified `_denoise_chunk` with Entropy Monitoring

```python
def _denoise_chunk_adaptive(self, node, min_steps, max_steps, branch_threshold, temperature):
    """Run denoising steps with entropy-threshold stopping.
    Stop when the current state is unusually uncertain (good branch point)."""
    state = node.state.clone()
    # ... setup response_region, etc. ...
    
    steps_taken = 0
    with torch.no_grad():
        for step in range(max_steps):
            mask_now = (state == self.mask_id) & response_region
            n_masked = mask_now.sum().item()
            if n_masked == 0:
                break
            
            # Forward pass — logits already right-shifted by adapter
            logits = self.adapter.forward_logits(state.unsqueeze(0), attn.unsqueeze(0))[0]
            probs = F.softmax(logits, dim=-1)
            
            # Entropy from these logits — essentially free
            token_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [L]
            masked_entropy = token_entropy[mask_now].mean().item()
            
            # Sample and unmask
            k = self.adapter.transfer_count(n_masked, step, max_steps)
            x0_pred, confidence = self.adapter.sample_and_confidence(logits, mask_now, temperature)
            _, sel = torch.topk(confidence, k=min(k, n_masked))
            state[sel] = x0_pred[sel]
            steps_taken += 1
            
            # Entropy-threshold check (after min_steps)
            if steps_taken >= min_steps:
                masking_ratio = mask_now.sum().item() / response_len
                expected_h = masking_ratio * math.log(self.adapter.vocab_size)
                if expected_h > 1e-6:
                    uncertainty_ratio = masked_entropy / expected_h
                    if uncertainty_ratio > branch_threshold:
                        break  # high-entropy state → good branch point
    
    child = MCTSNode(state=state, ..., step_index=node.step_index + steps_taken, ...)
    return child
```

Key detail: the entropy check uses `masked_entropy` computed from the *same logits* used for sampling — zero extra forward passes. The `uncertainty_ratio > branch_threshold` stops at states where the model is more uncertain than expected, placing nodes at decision points.

### New Config Parameters

```python
# Add to MCTSConfig:
adaptive_stepping: bool = False          # toggle; False = fixed steps (backward-compatible)
min_steps_per_expansion: int = 8         # minimum before entropy check
max_steps_per_expansion: int = 48        # fallback maximum
branch_threshold: float = 1.1            # uncertainty ratio to trigger branching
```

---

## 6. Infrastructure Requirements

### Hardware

| Resource | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 1× A100 40GB | 1× A100 80GB or 2× A100 40GB |
| System RAM | 32GB | 64GB |
| Storage | 30GB (model weights + checkpoints) | 100GB |

Dream 7B in BF16 is ~14GB for weights. During tree building, peak memory includes:
- Model weights: ~14GB
- Optimizer states (Adam): ~28GB (2× weights for momentum + variance)
- Forward pass activations: ~2-4GB per sequence
- Tree node states: negligible (~100KB for 30 nodes at seq_len=2048)

**For tree building only (no training)**: 1× A100 40GB is sufficient.
**For GRPO training**: 1× A100 80GB, or use gradient checkpointing + DeepSpeed ZeRO-2 on 2× A100 40GB.

### Software

```
transformers==4.46.2   # Dream requires this exact version
torch==2.5.1           # SdpaAttention compatibility
wandb                  # experiment tracking
deepspeed              # optional, for multi-GPU training
```

---

## 7. Migration Phases

### Phase 0: Validate Dream Forward Pass [1-2 days]

**Goal**: Confirm we can load Dream, run a forward pass, compute entropy, and generate text.

**Script**: `scripts/validate_dream.py`

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_path = "Dream-org/Dream-v0-Instruct-7B"
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = model.to("cuda").eval()

# Test 1: basic generation via diffusion_generate
messages = [{"role": "user", "content": "Write a Python function to compute fibonacci numbers."}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True)
input_ids = inputs.input_ids.to("cuda")
attention_mask = inputs.attention_mask.to("cuda")

output = model.diffusion_generate(
    input_ids, attention_mask=attention_mask,
    max_new_tokens=256, steps=256, temperature=0.2, top_p=0.95, alg="entropy", alg_temp=0.,
    return_dict_in_generate=True, output_history=True,
)
print("Generated:", tokenizer.decode(output.sequences[0][len(input_ids[0]):], skip_special_tokens=True))

# Test 2: raw forward pass + right-shift + entropy
mask_id = tokenizer.mask_token_id
L = input_ids.shape[1] + 128
x = torch.full((1, L), mask_id, dtype=torch.long, device="cuda")
x[0, :input_ids.shape[1]] = input_ids[0]

with torch.no_grad():
    logits = model(x, attention_mask="full").logits
    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)  # right-shift
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    mask_positions = (x[0] == mask_id)
    mean_entropy = entropy[0][mask_positions].mean().item()
    print(f"Mean entropy over {mask_positions.sum().item()} masked positions: {mean_entropy:.4f}")
    print(f"Expected max entropy (log V): {torch.log(torch.tensor(float(len(tokenizer)))):.4f}")

# Test 3: verify entropy decreases after partial denoising
# (run a few steps of Dream's internal _sample, then re-check entropy)
```

**Verification checklist**:
- [ ] Model loads without error on GPU
- [ ] `diffusion_generate` produces coherent code
- [ ] Raw forward pass returns logits of shape `[1, L, V]`
- [ ] Right-shifted entropy over masked positions is in expected range `[0, log(V)]`
- [ ] Entropy decreases after partial denoising (test at 0, 32, 64, 128 steps)
- [ ] Memory usage is within GPU budget

### Phase 1: ModelAdapter + EntropyComputer Integration [2-3 days]

**Goal**: Create `ModelAdapter` class that wraps Dream's model API, and verify `EntropyComputer` works through it.

**Files to create/modify**:
- `src/model_adapter.py` (new) — `ModelAdapter` class
- `src/utils.py` — update `load_model_and_tokenizer` to support `model_type="dream"`
- `src/config.py` — add `model_type: str = "mdlm"` field

**Test**: `tests/test_model_adapter.py`

```python
def test_dream_entropy_computation():
    """Entropy through ModelAdapter matches direct computation."""
    adapter = ModelAdapter(model, tokenizer, model_type="dream")
    logits = adapter.forward_logits(input_ids, attention_mask)
    # Verify right-shift was applied
    assert logits.shape == (1, L, V)
    # Verify entropy computation
    entropy = EntropyComputer.compute_token_entropy_from_logits(logits)
    assert entropy.shape == (1, L)
    assert entropy[0][mask_positions].mean() > 0
    assert entropy[0][mask_positions].mean() < math.log(len(tokenizer))

def test_dream_sample_and_confidence():
    """Sampling returns valid tokens and confidence scores."""
    adapter = ModelAdapter(model, tokenizer, model_type="dream")
    logits = adapter.forward_logits(input_ids, attention_mask)
    x0, conf = adapter.sample_and_confidence(logits[0], mask_positions, temperature=0.2, top_p=0.95)
    assert x0.shape == (L,)
    assert conf.shape == (L,)
    assert (conf[mask_positions] >= 0).all()
    assert (conf[mask_positions] <= 1).all()
```

**Verification checklist**:
- [ ] `ModelAdapter("dream")` produces correct logits shape
- [ ] Right-shift is applied (logits differ from raw model output)
- [ ] `EntropyComputer` produces valid entropy through adapter
- [ ] Sampling produces valid token IDs within vocabulary
- [ ] Confidence scores are in `[0, 1]` for masked positions
- [ ] MDLM adapter still works (backward-compatible)

### Phase 2: Tree Builder Migration [3-4 days]

**Goal**: `EntropyGuidedTreeBuilder` works with Dream via `ModelAdapter`.

**Files to modify**:
- `src/tree_builder.py` — replace direct `self.model(...)` calls with `self.adapter.forward_logits(...)` and `self.adapter.sample_and_confidence(...)`
- Replace Gumbel-noise sampling with adapter-based sampling
- Handle Dream's `tok_idx` / attention mask differences

**Key changes in `_denoise_chunk`**:

```python
# BEFORE (MDLM-specific):
logits = self.model(state.unsqueeze(0), attention_mask=attn.unsqueeze(0)).logits[0]
logits_n = add_gumbel_noise(logits, temperature)
x0_pred = torch.argmax(logits_n, dim=-1)
probs = F.softmax(logits, dim=-1)
conf = torch.gather(probs, -1, x0_pred.unsqueeze(-1)).squeeze(-1)

# AFTER (model-agnostic via adapter):
logits = self.adapter.forward_logits(state.unsqueeze(0), attn.unsqueeze(0))[0]
x0_pred, conf = self.adapter.sample_and_confidence(logits, mask_now, temperature, top_p)
```

**Also add**: entropy-threshold adaptive stepping (Section 5) as an option gated by `config.adaptive_stepping`.

**Test**: `tests/test_tree_builder_dream.py`

```python
def test_dream_tree_build_small():
    """Build a small tree (5 nodes) with Dream model."""
    config = MCTSConfig(
        model_type="dream",
        model_name_or_path="Dream-org/Dream-v0-Instruct-7B",
        max_tree_nodes=5, branch_width=2, steps_per_expansion=16,
        max_new_tokens=128, temperature=0.2,
    )
    builder = EntropyGuidedTreeBuilder(model, tokenizer, config, entropy_computer, adapter)
    root, leaves = builder.build_tree("Write a function to check if a number is prime.")
    
    assert len(leaves) >= 2
    # Entropy should be computed for all nodes
    assert root.entropy is not None
    for leaf in leaves:
        assert leaf.is_completed
    # Leaves should produce decodable text
    for leaf in leaves:
        text = tokenizer.decode(leaf.state[root.prompt_len:].tolist(), skip_special_tokens=True)
        assert len(text) > 0

def test_adaptive_stepping():
    """Adaptive stepping creates nodes at varying step_indices."""
    config = MCTSConfig(
        model_type="dream", adaptive_stepping=True,
        min_steps_per_expansion=4, max_steps_per_expansion=32,
        branch_threshold=1.1, max_tree_nodes=5, branch_width=2,
    )
    builder = EntropyGuidedTreeBuilder(model, tokenizer, config, entropy_computer, adapter)
    root, leaves = builder.build_tree("Write a fibonacci function.")
    
    # Verify nodes have non-uniform step_index deltas
    step_deltas = []
    def collect_deltas(node):
        for c in node.children:
            step_deltas.append(c.step_index - node.step_index)
            collect_deltas(c)
    collect_deltas(root)
    # At least some deltas should differ (adaptive = non-uniform)
    assert len(set(step_deltas)) > 1 or len(step_deltas) <= 1  # allow degenerate small tree
```

**Verification checklist**:
- [ ] Tree builds without error on Dream model
- [ ] All nodes have entropy computed
- [ ] Leaves produce decodable, non-empty text
- [ ] Global frontier selection picks highest-entropy nodes (log to verify)
- [ ] Adaptive stepping produces varying step deltas
- [ ] Memory usage stays within GPU budget during tree building
- [ ] Wall-clock time for 5-node tree is reasonable (<60s on A100)

### Phase 3: Loss + Advantage Computation [2-3 days]

**Goal**: `WeightedGRPOLoss` and `AdvantageComputer` work with Dream trees.

**Key concern**: `_log_prob_transition` must use the adapter for the forward pass:

```python
# BEFORE:
logits = model(parent_state.unsqueeze(0), attention_mask=...).logits[0]

# AFTER:
logits = self.adapter.forward_logits(parent_state.unsqueeze(0), parent_attention_mask.unsqueeze(0))[0]
```

**TimeWeighter** — needs adjustment for Dream's context. `total_denoising_steps` in config may differ (Dream's recommended is `steps=512` for 512 new tokens, or `steps=max_new_tokens`). Update config defaults for Dream: `total_denoising_steps=256` (matching `max_new_tokens` for 1 token/step).

**Test**: `tests/test_loss_dream.py`

```python
def test_dream_loss_computation():
    """Loss computes without NaN/Inf on Dream tree."""
    root, leaves = builder.build_tree("Write a prime checker.")
    rewards = [syntax_reward(leaf, prompt) for leaf in leaves]
    AdvantageComputer.compute_advantages(root, leaves, rewards, mode="branchgrpo")
    loss, metrics = loss_computer.compute_loss(model, root, leaves, prompt, vocab_size)
    
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert 0.01 < abs(loss.item()) < 100  # reasonable magnitude
    assert metrics["n_transitions"] > 0
```

**Verification checklist**:
- [ ] Log probabilities are finite for all transitions
- [ ] BranchGRPO advantages are computed and clipped correctly
- [ ] Time weights + entropy weights produce reasonable combined weights
- [ ] Loss magnitude is in expected range
- [ ] Backward pass produces gradients (check `model.parameters()` have `.grad != None`)
- [ ] Gradient norms are reasonable (not exploding)

### Phase 4: Single Training Step [2-3 days]

**Goal**: One complete forward-backward-update cycle on Dream.

**Considerations for 7B training**:
- Gradient checkpointing to reduce activation memory
- BF16 mixed precision (already using BF16 model weights; accumulate in FP32)
- Gradient clipping (already configured as `max_grad_norm=1.0`)

```python
# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Use BF16 autocast for forward/backward
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    loss, metrics = loss_computer.compute_loss(adapter, root, leaves, prompt, vocab_size)

# FP32 optimizer step
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
scaler.step(optimizer)
scaler.update()
```

**Test**: `scripts/single_step_dream.py`

```python
def main():
    # Load model
    config = MCTSConfig(model_type="dream", model_name_or_path="Dream-org/Dream-v0-Instruct-7B", ...)
    model, tokenizer = load_model_and_tokenizer(config)
    model.gradient_checkpointing_enable()
    
    # Build tree
    prompt = "Write a function that returns the nth prime number."
    root, leaves = builder.build_tree(prompt)
    
    # Compute rewards, advantages, loss
    rewards = [reward_fn(leaf) for leaf in leaves]
    AdvantageComputer.compute_advantages(root, leaves, rewards)
    loss, metrics = loss_computer.compute_loss(model, root, leaves, prompt, vocab_size)
    
    # Backward + step
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}, Grad norm: {grad_norm:.4f}")
    print(f"Metrics: {metrics}")
    
    # Verify model weights changed
    # (compare param checksums before/after)
```

**Verification checklist**:
- [ ] Forward pass through tree completes without OOM
- [ ] Loss backward completes (gradients exist)
- [ ] Gradient norm is finite and reasonable
- [ ] Optimizer step completes
- [ ] Model parameters changed after step
- [ ] Peak GPU memory usage logged and within budget
- [ ] Wall-clock time for one training step logged

### Phase 5: Baseline GRPO on Dream [3-4 days]

**Goal**: Standard trajectory-level GRPO (no tree) working on Dream, as the comparison baseline.

This uses `generate_one_trajectory` adapted for Dream — essentially wrapping Dream's `diffusion_generate` to return `(completion_str, transitions)`.

**Approach**: For baseline GRPO, we can use Dream's `diffusion_generate` with `output_history=True` to get intermediate states, then construct transitions from consecutive history snapshots.

```python
def dream_generate_with_transitions(adapter, prompt, config):
    """Generate one completion and extract transitions for baseline GRPO."""
    output = adapter.model.diffusion_generate(
        input_ids, attention_mask=attention_mask,
        max_new_tokens=config.max_new_tokens,
        steps=config.total_denoising_steps,
        temperature=config.temperature,
        top_p=0.95, alg="entropy", alg_temp=0.,
        output_history=True, return_dict_in_generate=True,
    )
    # Extract transitions from consecutive history snapshots
    transitions = []
    for i in range(len(output.history) - 1):
        parent = output.history[i]
        child = output.history[i + 1]
        transitions.append((parent, child, attention_mask))
    
    completion = tokenizer.decode(output.sequences[0][prompt_len:], skip_special_tokens=True)
    return completion, transitions
```

**Verification**: baseline GRPO training loop runs for 10 steps, loss decreases.

### Phase 6: Full Training + Evaluation [1-2 weeks]

**Goal**: Multi-epoch training on code generation dataset, evaluated on HumanEval/MBPP.

**Training data**: Same execution-lite prompts used in current MDLM experiments, extended with prompts from HumanEval training splits or similar.

**Evaluation**: Use the Dream team's eval harness (based on `lm-evaluation-harness`) adapted for our trained model:

```bash
cd eval
# Our entropy-MCTS-GRPO trained model:
bash eval_dream_gen.sh --model_path checkpoints/entropy_mcts_grpo/final/

# Baseline GRPO trained model:
bash eval_dream_gen.sh --model_path checkpoints/baseline_grpo/final/
```

**Metrics**: pass@1, pass@10 on HumanEval, MBPP. Compare against:
- Dream IT (untrained, the starting checkpoint)
- Dream IT + our baseline GRPO
- Dream IT + our entropy-MCTS-GRPO
- Dream-Coder IT (their GRPO — published numbers)

---

## 8. Config Defaults for Dream

```python
@dataclass
class MCTSConfig:
    # Model
    model_type: str = "dream"
    model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"
    device: Optional[str] = None
    
    # Tree construction
    max_tree_nodes: int = 15          # start small for 7B
    branch_width: int = 3
    steps_per_expansion: int = 32
    
    # Adaptive stepping
    adaptive_stepping: bool = True
    min_steps_per_expansion: int = 8
    max_steps_per_expansion: int = 48
    branch_threshold: float = 1.1
    
    # Sampling (Dream defaults)
    temperature: float = 0.2          # Dream recommends low temp for code
    top_p: float = 0.95
    alg: str = "entropy"              # Dream's entropy-based ordering
    alg_temp: float = 0.0
    
    # Loss weighting
    alpha_time: float = 1.0
    alpha_entropy: float = 0.5
    entropy_weight_min: float = 0.5
    entropy_weight_max: float = 2.0
    advantage_clip: float = 2.0
    
    # Generation
    total_denoising_steps: int = 256  # = max_new_tokens for 1 token/step
    max_new_tokens: int = 256
    
    # Training (conservative for 7B)
    batch_size: int = 1
    learning_rate: float = 5e-6       # more conservative than MDLM (larger model)
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    gradient_checkpointing: bool = True
```

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| OOM during tree building | Medium | High | Gradient checkpointing; reduce `max_tree_nodes` to 10; reduce `max_new_tokens` to 128 |
| Entropy scale different from MDLM | Low | Medium | Vocab is larger ($V \approx 152K$), so max entropy is higher ($\sim 11.9$). Normalization by `expected_entropy` handles this. Run Phase 0 to verify. |
| Right-shift logits applied incorrectly | Medium | High | Phase 0 verification: compare our entropy to Dream's internal `alg="entropy"` ordering. Both should agree on which positions are most/least certain. |
| Dream requires `transformers==4.46.2` (pinned) | High | Medium | Use a separate conda/venv for Dream work. Pin transformers version. |
| Training instability at 7B scale | Medium | Medium | Lower learning rate (5e-6), advantage clipping (D-014), gradient clipping. Monitor loss carefully. |
| Wall-clock time too slow for tree building | Medium | Medium | Each forward pass: ~50ms on A100 for 7B BF16. 30-node tree ≈ 30 × 32 × 50ms ≈ 48s per tree. Manageable for 1 prompt, but batch=4 = 3 min/step. May need to reduce tree budget or parallelize. |

---

## 10. Timeline

| Phase | Duration | Deliverable |
|---|---|---|
| 0: Validate Dream forward pass | 1-2 days | `scripts/validate_dream.py` passing all checks |
| 1: ModelAdapter + entropy | 2-3 days | `src/model_adapter.py`, tests passing |
| 2: Tree builder migration | 3-4 days | Tree builds on Dream, adaptive stepping works |
| 3: Loss + advantages | 2-3 days | Full loss computation, gradients flow |
| 4: Single training step | 2-3 days | One forward-backward-update verified |
| 5: Baseline GRPO | 3-4 days | Baseline trainer runs on Dream |
| 6: Full training + eval | 1-2 weeks | HumanEval/MBPP results for all conditions |
| **Total** | **~4-5 weeks** | Complete experimental results |
