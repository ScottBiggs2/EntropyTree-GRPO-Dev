# Future Extensions: Generalized Uncertainty & Adaptive Stepping

**Date**: 2026-03-09  
**Status**: Scoping / Proposal  
**Prerequisites**: Stable MDLM-based entropy-MCTS-GRPO pipeline (current system)

---

## Motivation

The current entropy-guided MCTS-GRPO system is tightly coupled to two assumptions:

1. **Entropy is computed as exact Shannon entropy over MDLM's categorical output** — which limits the method to masked discrete diffusion models.
2. **`steps_per_expansion` is a fixed hyperparameter** — which imposes a uniform temporal grid on the tree regardless of what the model is actually doing at each stage of denoising.
3. **Time weighting is attached to the parent node's single `step_index`** — which is only a faithful proxy while each edge spans the same number of denoising steps.

Both are perfectly reasonable starting points, but relaxing them could meaningfully expand the method's reach and efficiency. This document scopes out whether and how to pursue each.

---

## Extension 1: Generalized Uncertainty Proxy

### What We Have Now

The current `EntropyComputer` (in `src/entropy.py`) computes **exact Shannon entropy** from the model's softmax output:

$$H_i(z_t) = -\sum_{v=1}^{V} p_\theta(x_i = v \mid z_t) \log p_\theta(x_i = v \mid z_t)$$

This works because MDLM outputs a categorical distribution over the full vocabulary at every masked position. The entropy is:
- Exact (not approximated)
- Well-bounded: $H_i \in [0, \log V]$
- Monotonically decreasing as denoising progresses (fewer masked tokens → less total uncertainty)
- Computed via a single forward pass: `model(input_ids, attention_mask).logits → softmax → entropy`

The entropy serves two roles in the system:
1. **Node selection signal** (tree builder): high-entropy nodes are expanded first (DeepSearch-style global frontier)
2. **Loss weight** (GRPO): transitions at high-entropy nodes get up-weighted ($w_{\text{ent}} = H / \bar{H}$)

### Important Consistency Note

The current implementation stores node entropy as the **mean over masked positions only**:

$$H_{\text{masked-mean}} = \frac{1}{|M|}\sum_{i \in M} H_i$$

but often normalizes it using a quantity of the form:

$$\bar{H}(r) = r \log V$$

where $r$ is the masking ratio. Those are not the same statistic:
- $H_{\text{masked-mean}} \in [0, \log V]$
- $r \log V$ is a **sequence-averaged upper bound** that corresponds to averaging over *all* response positions with unmasked positions contributing zero

So if we keep masked-position mean entropy as the node score, the normalization baseline must be defined on that same statistic. There are two clean options:

1. **Analytic upper-bound normalization**: use $H_{\text{masked-mean}} / \log V$
2. **Stage-aware empirical normalization (recommended)**: estimate $\mathbb{E}[H_{\text{masked-mean}} \mid r]$ from calibration rollouts and use

$$w_{\text{ent}}(r) = \frac{H_{\text{masked-mean}}}{\mathbb{E}[H_{\text{masked-mean}} \mid r]}$$

The second option better matches the intended semantics of "more uncertain than expected at this denoising stage," which is the notion we want for both loss weighting and adaptive branch timing.

### The Generalization Question

What if we want to apply this method to:

| Model Type | Architecture | Output Space | Entropy Status |
|---|---|---|---|
| **BD3LM** | Block discrete diffusion | Discrete (per block) | Exact Shannon — identical computation |
| **Dream** | MDLM at 7B scale | Discrete (full sequence) | Exact Shannon — identical computation |
| **LLaDA / LLaDA2** | Masked LM (BERT-like) | Discrete | Exact Shannon — identical computation |
| **Flow matching** (e.g., Flow-GRPO models) | Continuous flow | Continuous embeddings | **Not directly available** |
| **EditFlow** | Flow-based editing | Continuous embeddings | **Not directly available** |
| **DDPM-style** continuous diffusion | Gaussian noise/score | Continuous | **Not directly available** |

The key observation: **for all discrete masked diffusion models, Shannon entropy transfers directly with zero mathematical change**. The interesting challenge is continuous diffusion.

### Analysis by Model Type

#### BD3LM (Block Discrete Diffusion) — Straightforward

BD3LM decomposes sequences into blocks of size $B$ and performs discrete diffusion *within* each block, while blocks are generated autoregressively (each block conditions on all previous blocks via KV-cached prefix).

**Entropy computation**: Within a block, the model produces logits over the full vocabulary for each position — exactly like MDLM. Shannon entropy is computed identically:

```
logits = model(x, attention_mask=block_causal_mask, position_ids=..., past_key_values=prefix_cache).logits
probs = softmax(logits, dim=-1)
entropy = -(probs * log(probs)).sum(dim=-1)
```

**What changes for tree building**: The current tree builder treats the response as a flat sequence and denoises positions anywhere in it. BD3LM requires block-awareness:

- **Denoising is confined to the current block**. You cannot unmask a token in block $k+1$ until block $k$ is complete.
- **Tree branching can happen at two levels**: (a) within a block (which masked positions to unmask, in what order — exactly our current approach), and (b) at block boundaries (different completions of block $k$ yield different conditioning for block $k+1$).
- **Block boundaries are natural branching points**. Completing a block "commits" to a prefix; branching before commitment gives the most diverse downstream options.

**Proposed tree construction for BD3LM**:

```
For each block k = 0, 1, ..., num_blocks:
    1. Inherit prefix from parent node (blocks 0..k-1 are fixed)
    2. Initialize block k: all masked
    3. Run denoising within block k for steps_per_block steps
    4. At branching points within block k: fork with stochastic sampling
    5. When block k completes: this becomes a tree node
    6. Repeat for block k+1 with different prefixes from different branches
```

This is essentially the current tree builder, but the "response region" per expansion is one block rather than the full sequence. The `_denoise_chunk` method already respects a `response_region` mask — we would narrow that mask to the current block.

**Estimated difficulty**: Low-Medium. Primary work is in `tree_builder.py` to add block-aware expansion logic and to handle the KV-cache prefix propagation. `EntropyComputer` requires zero changes.

#### Continuous Diffusion Models — Requires a Proxy

For flow matching and DDPM-style models, there is no discrete categorical distribution. The model predicts a continuous vector (velocity field, noise, or denoised embedding), and "entropy" in the Shannon sense is undefined.

**Candidate proxy metrics**, ordered by practicality:

**1. Prediction Variance Under Noise (Recommended)**

Run $K$ forward passes with different Gumbel noise / sampling noise at the same state, measure the variance of predictions:

$$\text{Uncertainty}(x_t, t) = \frac{1}{K} \sum_{k=1}^{K} \| \hat{x}_0^{(k)} - \bar{x}_0 \|^2$$

where $\hat{x}_0^{(k)}$ are sampled clean-data predictions and $\bar{x}_0$ is their mean.

- **Pro**: Model-agnostic, works for any diffusion model, measures exactly what we care about (how much does sampling noise change the outcome).
- **Con**: Requires $K$ forward passes per node (vs. 1 for Shannon entropy). With $K=4$, this is 4x the entropy computation cost.
- **Normalization**: Expected variance decreases with denoising progress, similar to expected entropy. Can compute $\bar{U}(t)$ empirically from a calibration run.

**2. Score Function Magnitude**

For models parameterized as score functions $s_\theta(x_t, t) \approx \nabla_{x_t} \log p(x_t)$:

$$\text{Uncertainty}(x_t, t) \propto \frac{1}{\| s_\theta(x_t, t) \|}$$

Low score magnitude → flat probability landscape → high uncertainty.

- **Pro**: Single forward pass, theoretically grounded.
- **Con**: Only applicable to score-parameterized models. The magnitude depends on the noise schedule in complex ways, making normalization non-trivial.

**3. Embedding-Space Discrete Entropy**

Project continuous predictions to nearest tokens in embedding space, then compute Shannon entropy over the resulting discrete distribution:

$$\hat{p}(v) = \text{softmax}(-\| \hat{x}_0 - e_v \|^2 / \tau)$$

where $e_v$ is the embedding of token $v$ and $\tau$ is a temperature.

- **Pro**: Produces a number directly comparable to Shannon entropy (same units, same normalization).
- **Con**: Requires access to the token embedding matrix. The quality depends heavily on $\tau$. Adds indirection.

**4. Top-K Probability Mass Concentration**

For models that can produce a distribution (even approximately):

$$\text{Uncertainty} = 1 - \sum_{v \in \text{top-}k} p(v)$$

If the top-$k$ tokens capture most of the mass, uncertainty is low.

- **Pro**: Very cheap, intuitive.
- **Con**: Only applies to models that produce discrete-ish distributions. Doesn't generalize to truly continuous models.

**Proposed abstraction**:

```python
class UncertaintyEstimator(ABC):
    """Abstract base for computing node uncertainty across model types."""
    
    @abstractmethod
    def compute_node_uncertainty(self, model, state, attention_mask, **kw) -> float:
        """Scalar uncertainty for one node."""
    
    @abstractmethod
    def expected_uncertainty(self, progress: float, **kw) -> float:
        """Expected uncertainty at this stage of denoising for normalization.
        progress: 0.0 (fully masked/noisy) → 1.0 (fully denoised/clean)."""
    
    def compute_weight(self, measured: float, progress: float, **kw) -> float:
        expected = self.expected_uncertainty(progress, **kw)
        if expected < 1e-6:
            return 0.0
        return measured / expected


class ShannonEntropyEstimator(UncertaintyEstimator):
    """Exact Shannon entropy for discrete masked diffusion (MDLM, BD3LM, Dream, LLaDA)."""
    # Wraps current EntropyComputer — zero behavior change for existing pipeline


class PredictionVarianceEstimator(UncertaintyEstimator):
    """Variance of K stochastic forward passes for continuous diffusion."""
    
    def compute_node_uncertainty(self, model, state, attention_mask, K=4, **kw):
        predictions = [model_forward_with_noise(model, state, ...) for _ in range(K)]
        mean_pred = torch.stack(predictions).mean(dim=0)
        variance = torch.stack([(p - mean_pred).pow(2).sum() for p in predictions]).mean()
        return variance.item()
```

The tree builder and loss module would interact with this through the abstract interface. `EntropyComputer` becomes one implementation, not the only option.

**Estimated difficulty**: Medium. The abstraction is clean. The hard part is calibrating normalization for continuous-model uncertainty proxies so that the $w_{\text{ent}}$ weights remain meaningful and stable.

### Feasibility Assessment for Extension 1

| Target | Feasibility | Entropy Change | Tree Builder Change | Notes |
|---|---|---|---|---|
| BD3LM | **High** | None (exact Shannon) | Medium (block-aware expansion) | Natural next step after MDLM validation |
| Dream 7B | **High** | None (exact Shannon) | Minimal (just scale) | Same as MDLM, only larger |
| LLaDA / LLaDA2 | **High** | None (exact Shannon) | Minimal | BERT-like masking, very similar to MDLM |
| Flow matching | **Medium** | Prediction variance proxy | Major (continuous state space) | Fundamentally different denoising mechanics |
| DDPM continuous | **Medium** | Score magnitude or prediction variance | Major | Different noise process, different branching semantics |

### Recommended Path

1. **Short-term** (after current MDLM experiments): Extend to BD3LM. This tests whether the *tree-building strategy* generalizes across denoising architectures while keeping the entropy computation identical. The `dllm` submodule already has `BD3LMSampler` with block-by-block generation that can serve as a reference for block-aware tree construction.

2. **Medium-term**: Extend to Dream-7B or LLaDA. This tests *scale* — does entropy-guided MCTS remain useful when the base model is already strong? (DeepSearch found that larger models need larger exploration constants to overcome confidence bias — this directly applies.)

3. **Long-term**: Continuous diffusion. This requires the prediction-variance proxy and a fundamentally different tree builder that operates on continuous states. The payoff is large (flow matching is the dominant paradigm for image diffusion, and extending to that community would be impactful), but the engineering is substantial. Worth pursuing only after the discrete case is published and validated.

---

## Extension 2: Dynamic / Adaptive Steps Per Expansion

### What We Have Now

The current `_expand_node` in `tree_builder.py` calls:

```python
child = self._denoise_chunk(node, self.config.steps_per_expansion, temp)
```

where `steps_per_expansion` is a fixed integer (default 32, tuned to 12-24 in ablations). Every expansion runs exactly this many inner denoising steps. Each inner step unmasks $k$ tokens where:

$$k = \min\left(k_{\text{cap}},\ \max\left(1,\ \left\lfloor \frac{n_{\text{masked}}}{\text{steps\_left}} \right\rfloor \right)\right)$$

with $k_{\text{cap}} = \max(1, \lfloor (n_{\text{masked\_initial}} - 1) / \text{num\_steps} \rfloor)$ to preserve at least one masked token for the child's entropy computation.

This means: the tree has a **uniform temporal grid**. Nodes are always separated by exactly `steps_per_expansion` denoising steps, regardless of whether those steps were trivially easy (model was 99% confident on all unmasked tokens) or critically hard (model was torn between two completely different completions).

### Why This Matters

Consider the denoising trajectory of a function completion:

```
Step 0:    [MASK MASK MASK MASK MASK MASK MASK MASK MASK MASK]  (high entropy everywhere)
Step 12:   [def  MASK MASK MASK MASK MASK MASK MASK MASK MASK]  (trivial first tokens)
Step 24:   [def  fibo MASK MASK MASK MASK MASK MASK MASK MASK]  (still trivial)  
Step 36:   [def  fibonacci(n MASK MASK MASK MASK MASK MASK)]    (model gaining confidence)
Step 48:   [def  fibonacci(n):  if  MASK MASK MASK MASK MASK]   ← CRITICAL DECISION
Step 60:   [def  fibonacci(n):  if  n <= 1 MASK MASK MASK]      (path committed)
Step 72:   [def  fibonacci(n):  if  n <= 1: return MASK MASK]   (straightforward)
Step 84:   [def  fibonacci(n):  if  n <= 1: return n   MASK]    (finishing)
```

At step 48, the model decides whether to use recursion, memoization, or iteration. With fixed `steps_per_expansion=32`, we branch at steps 0, 32, 64 — completely missing the critical decision at step 48. We waste a branching point at step 32 (still trivial) and another at step 64 (decision already committed).

With adaptive stepping, we could detect that the model is highly uncertain around step 48 (entropy at the remaining masked positions is elevated relative to what's expected at this denoising stage) and branch *there* instead — placing a node right at the decision point where stochastic exploration adds the most value.

### Proposed Approaches

#### Approach A: Entropy-Threshold Stopping — Branch at High Entropy (Recommended)

The core principle: **branch where entropy is high, because stochastic sampling at uncertain states produces maximally diverse children.** This is the same principle behind global frontier selection, now applied to *when* we stop denoising within a chunk.

The intuition: rush through low-entropy regions (model is confident, branching would yield similar children) and stop at high-entropy regions (model is uncertain, branching here maximizes diversity). A node created at a high-entropy state will be prioritized by global frontier selection and will produce divergent subtrees.

**Why NOT stop at entropy drops**: An entropy drop means a critical decision was just *committed*. The high-entropy state is now behind us — the node we've created has low entropy, won't be prioritized for expansion, and if expanded, its children will converge. Worse, entropy drops often cascade: once a key structural decision is made (e.g., "use recursion"), downstream tokens become predictable. Stopping after the drop means the entire information-rich part of the landscape is already traversed.

**Implementation** — the inner denoising loop in `_denoise_chunk` already computes `logits` and `probs` at every step (lines 184-189 of `tree_builder.py`). Computing entropy from these is essentially free:

```
for step in range(max_steps):
    logits = model(state, attention_mask)          # already doing this
    probs = softmax(logits, dim=-1)                # already doing this
    
    # Entropy from existing tensors — negligible additional cost
    masked_entropy_mean = -(probs * log(probs + 1e-10)).sum(-1)[mask_positions].mean()
    
    x0_pred = argmax(add_gumbel_noise(logits))     # use same logits for denoising
    # ... unmask top-k as usual ...
    
    if step >= min_steps:
        masking_ratio = n_masked / response_len
        baseline_h = expected_masked_mean_entropy(masking_ratio)  # calibrated offline
        uncertainty_ratio = masked_entropy_mean / max(baseline_h, 1e-6)
        if uncertainty_ratio > branch_threshold:
            break  # state is unusually uncertain → good branch point
```

This stops early when the model is *more uncertain than expected on the same masked-mean entropy scale*, placing nodes at the decision points themselves rather than after the decisions have been made.

**Design tension — what if entropy is always high?** If the model is struggling throughout (uncertainty ratio always above threshold), we'd stop at `min_steps` every time, creating very fine-grained branching everywhere. This has two possible interpretations:
- *Correct behavior*: The problem is genuinely hard and deserves dense exploration. Fine-grained branching in a hard region is exactly what the system should do.
- *Wasteful*: If entropy is *uniformly* high (no specific decision points, just general difficulty), all branches are equally uncertain and may not diverge meaningfully.

**Mitigation — combine with stagnation detection**: Branch when entropy is high AND changing slowly (the model is stuck at a decision point, not just generally uncertain):

```
if uncertainty_ratio > branch_threshold and abs(dH_per_step) < stagnation_eps:
    break  # high entropy + slow change = model stuck at a decision it can't resolve alone
```

Slow entropy change means the model isn't making progress on resolving the uncertainty through normal denoising — exactly the situation where branching (multiple stochastic samples) adds the most value.

**New hyperparameters**:
- `min_steps_per_expansion: int = 8` — minimum steps before allowing early stop (prevent degenerate tiny chunks)
- `max_steps_per_expansion: int = 48` — maximum steps if threshold is never hit (fallback)
- `branch_threshold: float = 1.1` — uncertainty ratio above which to branch (10% more uncertain than expected at this denoising stage)
- `stagnation_eps: float = 0.05` — (optional) maximum entropy change rate to qualify as "stuck"

#### Approach B: Token-Fraction-Based Expansion

Define expansion in terms of the fraction of remaining masked tokens to unmask, rather than number of steps:

```
_denoise_chunk_fraction(node, fraction=0.25):
    n_target = max(1, int(node.num_masked_tokens() * fraction))
    tokens_unmasked = 0
    while tokens_unmasked < n_target:
        # unmask step
        tokens_unmasked += k  # k tokens unmasked this step
    return child_node
```

**Why this works**: At the start of denoising (200 masked tokens), 25% = 50 tokens → many steps → fine-grained tree. At the end (20 masked tokens), 25% = 5 tokens → few steps → coarse tree. The granularity naturally adapts to the information density of each region.

**Advantages**:
- Extremely simple to implement (change 3-4 lines in `_expand_node`)
- No extra forward passes
- Tree depth becomes roughly $\lceil \log_{1/(1-f)}(n_{\text{initial}}) \rceil$ — logarithmic in sequence length, which is a natural scaling
- `step_index` still tracks total denoising steps taken; `masking_ratio()` remains the true progress measure

**Disadvantages**:
- Does not directly detect entropy-based decision points — just distributes branching proportionally to how many tokens are left
- Less principled than Approach A, but much cheaper

**New hyperparameters**:
- `expansion_fraction: float = 0.25` — fraction of remaining masked tokens to unmask per expansion
- `min_tokens_per_expansion: int = 4` — minimum tokens to unmask (prevent trivially small chunks)

#### Approach C: KL-Budget Expansion

Run step-by-step, accumulating the KL divergence between the model's distribution before and after each unmasking:

$$\text{KL}_{\text{step}} = D_{\text{KL}}\left( p_\theta(\cdot \mid z_{t}) \ \|\ p_\theta(\cdot \mid z_{t+1}) \right)$$

Stop when accumulated KL exceeds a budget. The node is created at the state *before* the KL spike (the uncertain state), not after (the committed state).

**Why this works**: KL divergence directly measures how much the model's "mind changed" after an unmasking. A high-KL step means an unmasking was highly consequential — the model's predictions for remaining tokens shifted dramatically. Crucially, the pre-unmasking state is the uncertain one, and the post-unmasking state is the committed one. By monitoring KL and stopping *when we detect an upcoming high-KL transition*, we place the node at the decision point itself.

**Implementation nuance**: Since we need the logits both before and after to compute KL, this naturally suggests saving the pre-unmasking logits (which we already have from the denoising step) and comparing to the post-unmasking logits (the next step's forward pass). When accumulated KL exceeds the budget, we return the *previous* state as the child node, not the current one.

**Cost**: The logits from the current step are already computed. The comparison is with the *next* step's logits, which is the normal denoising forward pass for that step. So KL can be computed between consecutive steps at zero additional forward-pass cost — just store the previous step's logits.

**When to prefer this**: When you want the most theoretically principled branching. The KL formulation has a clean information-theoretic interpretation: each expansion consumes a fixed "information budget," and branching happens when the budget is exhausted. This is closely related to Approach A (high entropy = likely to produce high KL transitions) but operates on the actual information change rather than a static snapshot.

### Impact on Existing System Components

The current codebase is surprisingly well-prepared for dynamic stepping:

| Component | Current Assumption | Impact of Dynamic Steps |
|---|---|---|
| `MCTSNode.step_index` | Increments by fixed `steps_per_expansion` | Would increment by variable amounts — **already supports this** (just an int field) |
| `MCTSNode.masking_ratio()` | Computed from actual masked count | **Already step-index-agnostic** — no change needed |
| `TimeWeighter.get_weight(step_index)` | Lookup in precomputed array | **Needs interval aggregation** once a node can span variable numbers of denoising steps |
| `EntropyComputer.compute_entropy_weight()` | Takes `masking_ratio`, not `step_index` | **Already step-index-agnostic** — no change needed |
| `AdvantageComputer` (BranchGRPO) | Groups by `depth` (tree depth) | Nodes at same depth may now span different step ranges — **this is correct** since depth measures tree structure, not temporal position |
| `WeightedGRPOLoss._collect_transitions()` | Uses `node.step_index` for time weight and `node.masking_ratio()` for entropy weight | **Needs change**: time weight should use the edge interval `[t_{\text{parent}}, t_{\text{child}})` rather than a single parent timestamp |
| `_denoise_to_completion()` | Runs until no masks remain | **Unaffected** — this is leaf completion, not expansion |

The changes are concentrated in two methods:
1. `_expand_node()` — replace fixed `self.config.steps_per_expansion` with adaptive logic
2. `_denoise_chunk()` — add optional early-stopping condition
3. `WeightedGRPOLoss._collect_transitions()` — make time weighting interval-aware

---

## Extension 3: Interval-Aware Time Weighting

### What We Have Now

The current loss assigns each edge a time weight using only the parent's timestamp:

$$w_{\text{edge}} = w(t_{\text{parent}})$$

This is acceptable as a proxy only while every edge spans the same fixed number of denoising steps, because the bias is then shared uniformly across the tree.

Once branching becomes dynamic, an edge may cover:
- a short interval, e.g. 8 denoising steps
- a long interval, e.g. 37 denoising steps

Treating both with the same point weight at $t_{\text{parent}}$ is no longer conceptually correct. The longer edge covers more denoising time and therefore should inherit more of the time-weight mass.

### Correct View: Time Weight Is a Density Over Denoising Progress

TempFlow-style weighting defines importance as a function over denoising time:

$$w(t) \propto (1 - t/T)^2$$

For macro-edges that cover an interval $[t_0, t_1)$, the natural discrete-time generalization is:

$$w_{\text{interval}}(t_0, t_1) = \sum_{t=t_0}^{t_1-1} w(t)$$

This is the right object if the macro-edge loss represents the aggregate policy update over all micro-steps inside that interval.

### Two Implementation Paths

**Path A: Minimal correction (recommended before large adaptive-stepping experiments)**

Keep macro-edges, but replace the point lookup with interval mass:

```python
interval_weight = 0.0
for t in range(parent.step_index, child.step_index):
    interval_weight += time_weighter.get_weight(t)
```

This fixes the most obvious bias introduced by irregular branch timing.

**Path B: Principled process supervision (preferred long-term)**

Log the micro-step transitions inside `_denoise_chunk()` and compute the loss over those individual denoising steps:

$$\mathcal{L} = -\sum_{t \in \text{visited steps}} w(t)\,A_{\text{edge}}\,\log \pi(a_t \mid s_t)$$

This aligns more closely with TreeRL's emphasis on dense, on-policy process supervision and with TreeGRPO's step-specific credit assignment.

### Feasibility Assessment

| Option | Difficulty | Correctness | Notes |
|---|---|---|---|
| Point lookup at parent step | None | Poor under variable-step edges | Fine only for fixed-step baseline |
| Interval-mass weighting | Low | Good | Best minimal fix for adaptive stepping |
| Micro-step logging | Medium | Best | Also resolves the "macro-edge as one action" approximation |

### Feasibility Assessment for Extension 2

| Approach | Difficulty | Extra Compute | Quality of Branch Placement | Compatibility |
|---|---|---|---|---|
| **A: Entropy-Threshold** | Medium | ~0% (reuse logits) | Best (directly targets high-uncertainty states) | Full |
| **B: Token-Fraction** | Low | 0% | Good (proportional, not targeted) | Full |
| **C: KL-Budget** | Medium | ~0% (store prev logits) | Best (information-theoretic) | Full |

### Recommended Path

1. **Immediate (low-hanging fruit)**: Implement Approach B (token-fraction). This is a ~10-line change in `_expand_node` and `_denoise_chunk`, requires no extra forward passes, and naturally adapts granularity to the denoising stage. Use as a simple-but-effective baseline for adaptive stepping.

2. **Next (after Approach B is ablated)**: Implement Approach A (entropy-threshold). Restructure `_denoise_chunk` to compute entropy from the logits it already has and stop when the current state is unusually uncertain. This directly places nodes at the high-entropy states where branching adds the most value.

3. **Compare**: Run the three-way comparison on the MDLM code completion task:
   - Fixed `steps_per_expansion=32` (current)
   - Token-fraction `expansion_fraction=0.25`
   - Entropy-threshold with `branch_threshold=1.1`
   
   Measure: tree diversity (number of unique leaf completions), reward variance across leaves (proxy for how well the tree explored), wall-clock time per tree, and final GRPO training metrics.

4. **Before adaptive-stepping conclusions**: switch time weighting from parent-point lookup to interval-mass weighting so that variable-length edges are not misweighted.

5. **Long-term**: If Approach A proves valuable, explore Approach C (KL-budget) as a more principled variant that places nodes just *before* high-information transitions rather than at static entropy snapshots, and consider micro-step transition logging for a more faithful process-level loss.

---

## How These Two Extensions Interact

The two extensions are orthogonal and can be pursued independently, but they are **synergistic**:

- **Generalized uncertainty (Ext 1)** changes *what signal* drives the tree → broadens the method to new model types
- **Adaptive stepping (Ext 2)** changes *when to branch* based on that signal → makes the tree structure itself smarter

Together, they complete a vision of **model-agnostic, signal-driven tree construction**: any diffusion model provides an uncertainty signal through the `UncertaintyEstimator` interface, and the tree builder uses that signal both for global frontier selection (which nodes to expand — already implemented) *and* for deciding how far to denoise before creating a branch point (when to expand — the new capability).

### Concrete Interaction Example

BD3LM with entropy-threshold stopping:

```
Block 0: [def fibonacci(n):]     ← prompt, no tree needed
Block 1: [MASK × 64]             ← high entropy, branch after 8 steps (entropy drops fast = easy tokens)
         → Branch A: [if n <= 1: return n ...]
         → Branch B: [if n < 2: return ...]
Block 1 continued:                ← lower entropy in both branches, run 20 more steps before branching
         → A1: [if n <= 1: return n; return fibonacci(n-1) + fibonacci(n-2)]
         → A2: [if n <= 1: return n; memo = {} ...]
         → B1: [if n < 2: return n; a, b = 0, 1 ...]
```

The block structure from BD3LM provides natural granularity boundaries. Entropy-threshold stopping within each block creates branch points at the high-uncertainty states where the model most needs diverse exploration. The combination is more powerful than either alone.

---

## Summary & Prioritization

| Extension | Priority | Effort | Impact | When |
|---|---|---|---|---|
| BD3LM support (Ext 1, discrete) | **High** | 1-2 weeks | Opens block diffusion models; validates generality | After current MDLM ablations complete |
| Token-fraction stepping (Ext 2B) | **High** | 2-3 days | Low-risk adaptive stepping; immediate comparison | Can be done in parallel with BD3LM |
| Entropy-threshold stepping (Ext 2A) | **Medium** | 1 week | Branch at high-uncertainty states where diversity is maximized | After token-fraction is baselined |
| Interval-aware time weighting (Ext 3) | **High** | 1-2 days | Required for correctness once adaptive stepping lands | Implement alongside adaptive stepping |
| Dream/LLaDA support (Ext 1, scale) | **Medium** | 1 week | Tests scaling; no new math, just engineering | After BD3LM validates approach |
| Continuous diffusion (Ext 1) | **Low (long-term)** | 3-4 weeks | Broadest impact but heaviest lift | After publication of discrete results |
| KL-budget stepping (Ext 2C) | **Low** | 1-2 weeks | Most principled; places nodes before high-information transitions | If entropy-threshold results are promising |

### Open Questions for Each

**BD3LM tree construction**:
- Should we branch at block boundaries only, or also within blocks?
- How should KV-cache prefix propagation work in the tree? (Each branch needs its own prefix cache — memory implications.)
- Does the BD3LM model's autoregressive inter-block structure change the advantage backpropagation?

**Entropy-threshold stopping**:
- What is the right `branch_threshold`? Likely depends on vocab size, sequence length, and model size. May need a calibration phase (run a few trees, measure the distribution of uncertainty ratios across all inner steps, set threshold at e.g. 75th percentile).
- If entropy is uniformly high (hard problem), should `min_steps` be increased to prevent overly fine-grained branching? Or is dense branching in hard regions actually correct?
- Does the stagnation criterion (slow entropy change) add value over the raw threshold, or is it unnecessary complexity?
- Does variable-depth tree structure affect BranchGRPO's depth-wise normalization quality? (Nodes at the same depth may span very different numbers of denoising steps.)

**Continuous diffusion uncertainty**:
- How many samples $K$ are needed for prediction variance to be a reliable signal?
- How do we normalize uncertainty for flow matching models where the noise schedule is learned?
- Is the prediction-variance proxy monotonically decreasing with denoising progress? (It should be, but needs empirical validation.)
