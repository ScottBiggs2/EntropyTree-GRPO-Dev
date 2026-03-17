# Research Decision Tracker

**Project**: Entropy-Guided MCTS-GRPO for Diffusion Language Models  
**Model**: `dllm-collection/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1`  
**Last Updated**: 2026-03-15

---

## How to Use This Document

This is the single source of truth for all research decisions. Each decision has:
- **Status**: `OPEN` (needs your input), `DECIDED` (locked in), `DEFERRED` (revisit later)
- **Default**: What the coding agents will use unless you override
- **Your Notes**: Space for you to write / paste chat responses

Coding agents: treat `DECIDED` entries as ground truth. For `OPEN` entries, use the default and flag for review.

---

## D-001: Target Hardware & Precision

| Field | Value |
|-------|-------|
| **Status** | `DECIDED` |
| **Decision** | Local M1 MacBook, MPS device with float32 fallback; **CUDA when available** (e.g. cloud GPU) |
| **Rationale** | 0.6B param model (~1.2GB BF16) fits comfortably. MPS has partial BF16 support but float32 is safer for correctness during development. **On cloud GPU, use CUDA (auto-detected via `get_device()`).** Switch to BF16 for speed once validated. |

**Details**:
- Device priority: **cuda** (if available) > **mps** (Apple) > **cpu**
- BF16 may produce NaN on some MPS ops; start float32, toggle later
- **MPS does not support float64**: `add_gumbel_noise` uses float32 on MPS, float64 elsewhere (see `src/utils.py`, `scripts/validate_model.py`).
- Batch size 1-2 for tree building (memory constrained during expansion)
- Tree node budget: start at 10-15 for dev, scale to 30 for experiments

**Your Notes**:
> _(paste responses here)_

---

## D-002: Model API — No Explicit Timestep

| Field | Value |
|-------|-------|
| **Status** | `DECIDED` |
| **Decision** | Model takes `(input_ids, attention_mask)` only. No timestep parameter. |
| **Rationale** | Confirmed from HuggingFace model card and dLLM source code. The MDLM model infers denoising state from the masking pattern itself. The design doc pseudocode showing `model(z_t, timestep=t)` is conceptual shorthand — the real call is `model(input_ids=z_t, attention_mask=mask).logits`. |

**Details**:
- Forward pass: `model(input_ids, attention_mask).logits` → `[batch, seq_len, vocab_size]`
- "Timestep" in our tree = denoising step index = proxy for how many tokens remain masked
- Entropy at a node depends on the masking pattern, not an explicit `t` value
- The `steps_per_expansion` config controls how many unmaskings happen between tree nodes

**Your Notes**:
> _(paste responses here)_

---

## D-003: Denoising Mechanism — Gumbel Noise vs Multinomial

| Field | Value |
|-------|-------|
| **Status** | `DECIDED` |
| **Decision** | Use the dLLM `add_gumbel_noise` + argmax for sampling, matching the reference implementation |
| **Rationale** | The HuggingFace model card and dLLM sampler both use Gumbel-Max trick (not temperature + multinomial). Gumbel noise with `temperature > 0` gives equivalent stochastic sampling but is more numerically stable. Using the same mechanism ensures our tree branches are consistent with how the model was designed to generate. |

**Details**:
```python
# dLLM's approach (what we'll use)
logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
x0 = torch.argmax(logits_with_noise, dim=-1)

# Design doc approach (conceptual, NOT what we'll use)
probs = F.softmax(logits / temperature, dim=-1)
sampled = torch.multinomial(probs, num_samples=1)
```

**Your Notes**:
> _(paste responses here)_

---

## D-004: Block-Based vs Full-Sequence Denoising

| Field | Value |
|-------|-------|
| **Status** | `CLOSED` |
| **Default** | Start with full-sequence denoising (simpler), add block-based as optimization |
| **Rationale** | The dLLM sampler uses block-based denoising (`block_size=64`) for efficiency. For our tree construction, full-sequence is conceptually simpler and matches the design docs. Block-based can be added later if memory or speed is an issue. |

**Tradeoffs**:
- Full-sequence: simpler code, matches design docs, may be slower
- Block-based: matches dLLM sampler exactly, more complex tree logic, potentially better quality (model was trained with blocks)
- Hybrid: full-sequence for tree branching, block-based for leaf completion

**Your Notes**:
> Let's begin with the MDLM system because it is simpler in implementation and the MDLM guarantees perfect complete sequence entropy over tokens at each timestep. We can experiment with generalizing the method to BD3LM and other systems once we have assessed the core capabilities of the idea in the toy MDLM setting. 

---

## D-005: Entropy Aggregation Method

| Field | Value |
|-------|-------|
| **Status** | `CLOSED` |
| **Default** | Mean entropy across masked positions only |
| **Alternatives** | Max, sum, position-weighted, attention-weighted |

**Rationale**: Mean is simplest and most interpretable. Computing over only-masked-positions (not already-unmasked tokens) gives a cleaner signal about remaining uncertainty. Max might be better for finding "one really hard token" but could be noisy.

**Implementation**: In `_denoise_chunk`, we cap tokens unmasked per step so each child retains at least one masked position; otherwise all children would have zero masked tokens and entropy would be 0, making entropy-based frontier selection useless.

**Your Notes**:
> For now let's proceed with mean entropy over masked positions for simplicity. We will flag this for a return to experiment with more complex entropy aggregation strategies. 

---

## D-006: Remasking Strategy During Tree Construction

| Field | Value |
|-------|-------|
| **Status** | `CLOSED` |
| **Default** | `low_confidence` (matches model card default) |
| **Alternatives** | `random` |

**Rationale**: The dLLM sampler supports two remasking strategies. `low_confidence` uses the model's own softmax probabilities to decide which tokens to commit (highest confidence first). `random` picks randomly. For tree diversity, `random` might produce more varied branches, but `low_confidence` produces higher quality individual trajectories.

**Your Notes**:
> Let's begin with 'low_confidence' to encourage high quality individual trajectories. We can ablate over this once a core is stable and running to evaluate the merits of this decision. 

---

## D-007: Advantage Computation Method

| Field | Value |
|-------|-------|
| **Status** | `DECIDED` |
| **Decision** | BranchGRPO-style: path-weighted reward fusion + depth-wise normalization |
| **Rationale** | BranchGRPO (Li et al., 2025) demonstrates that depth-wise normalization is critical for stability in tree-based GRPO. Simple averaging causes early high-variance steps to dominate gradients. Path-weighted fusion is more principled than uniform averaging. |

**Details**:
- Phase 1: implement simple averaging (get something working)
- Phase 2: upgrade to BranchGRPO-style (for stability)
- Ablation: compare simple vs BranchGRPO advantage computation

**Your Notes**:
> _(paste responses here)_

---

## D-008: Reward Function for Development

| Field | Value |
|-------|-------|
| **Status** | `OPEN` |
| **Default** | Simple heuristic reward (syntax check + keyword matching) for Phase 1-4, real execution-based reward for Phase 5+ |
| **Alternatives** | Sandbox code execution from day 1, LLM-as-judge |

**Rationale**: Getting code execution sandboxing right is a separate engineering task. For validating the tree/entropy/loss machinery, a simple heuristic reward is sufficient. The reward function is modular and can be swapped.

**Candidate heuristics** (for dev only):
1. Contains `def` + `return` → 0.5, passes basic syntax check → +0.3, has docstring → +0.2
2. Character-level similarity to reference solution
3. AST-parseable → 1.0, else 0.0

**Your Notes**:
> _(paste responses here)_

---

## D-009: Tree Budget and Branching Width

| Field | Value |
|-------|-------|
| **Status** | `OPEN` |
| **Default** | `max_tree_nodes=15, branch_width=3, steps_per_expansion=32` for dev; scale to `30/3/32` for experiments |
| **Rationale** | M1 MacBook memory is limited. Each node stores a `[seq_len]` tensor. With seq_len=512, that's ~2KB per node (int64). The real cost is the forward passes: 15 nodes × branch_width=3 = up to 45 forward passes per tree. At ~0.5s per forward on M1, that's ~22s per tree. Manageable for dev. |

**Ablation plan** (for experiments):
- `max_tree_nodes`: 10 / 15 / 30 / 50
- `branch_width`: 2 / 3 / 5
- `steps_per_expansion`: 16 / 32 / 64

**Your Notes**:
> _(paste responses here)_

---

## D-010: Loss Weight Balance (alpha_time, alpha_entropy)

| Field | Value |
|-------|-------|
| **Status** | `OPEN` |
| **Default** | `alpha_time=1.0, alpha_entropy=0.5` (from design doc) |
| **Rationale** | Starting with the design doc defaults. Time weighting is well-established (TempFlow-GRPO). Entropy weighting is our novel contribution — start at half weight to be conservative, ablate later. |

**Ablation plan**:
- `alpha_entropy`: 0.0 (no entropy) / 0.25 / 0.5 / 1.0
- `alpha_time`: 0.0 (no time) / 0.5 / 1.0

**Important Note (Concept Check, 2026-03-15)**:
- The current implementation normalizes `w_time` so the full schedule sums to 1, which makes each individual per-step time weight `O(1 / T)`.
- The current entropy weight is clamped around `O(1)`.
- Therefore the additive combination `alpha_time * w_time + alpha_entropy * w_ent` is not scale-balanced: with defaults, entropy weighting dominates unless time weights are rescaled or interpreted as interval mass over multiple steps.

**Your Notes**:
> _(paste responses here)_

---

## D-011: Evaluation Benchmarks

| Field | Value |
|-------|-------|
| **Status** | `OPEN` |
| **Default** | HumanEval (small, 164 problems) for primary metric; MBPP subset for secondary |
| **Rationale** | Model already has HumanEval/MBPP baselines on the model card (28.1 / 23.0). Small enough to run locally. EvalPlus would be ideal but is heavier. |

**Baseline numbers** (from model card):
| Model | HumanEval | MBPP |
|-------|-----------|------|
| Qwen2.5-Coder-0.5B-Instruct (AR) | 28.0 | 52.9 |
| MDLM v0.1 (our target, pre-training) | 28.1 | 23.0 |
| MDLM v1.1 (improved) | 41.5 | 33.6 |

**Your Notes**:
> _(paste responses here)_

---

## D-012: When to Move Off M1 to Cloud GPU

| Field | Value |
|-------|-------|
| **Status** | `OPEN` |
| **Default** | Stay on M1 through Phase 4 (single training step verified). Move to GPU for Phase 5+ (full training runs, ablations). |
| **Rationale** | M1 is fine for development, debugging, and single-step validation. Full training runs with tree construction will be too slow locally. |

**Your Notes**:
> _(paste responses here)_

---

## D-013: Baseline for Phase 8 Comparison

| Field | Value |
|-------|-------|
| **Status** | `DECIDED` |
| **Decision** | Single baseline: **trajectory-level GRPO** (no tree). Generate K completions per prompt with the same model and temperature; reward − mean advantage; GRPO loss on trajectory log-prob. |
| **Rationale** | Keeps comparison simple (same reward, same model, same sampling; only difference is tree vs no tree). No extra baselines (e.g. no “uniform tree” or “no entropy weight”) for initial Phase 8; can add ablations later. |

**Details**:
- Baseline: `BaselineGRPOTrainer` — K samples per prompt, trajectory log-prob summed over denoising steps, advantage = reward − mean(rewards).
- Entropy-MCTS: `EntropyMCTSTrainer` — tree with entropy-guided expansion, BranchGRPO advantages, time+entropy weighted loss.
- Checkpoints: `checkpoints/baseline_grpo/` and `checkpoints/entropy_mcts_grpo/` so both are clearly named.

**Your Notes**:
> _(paste responses here)_

---

## D-014: Stability Clamps (Advantage Clip, Entropy Weight Clamp)

| Field | Value |
|-------|-------|
| **Status** | `DECIDED` |
| **Decision** | Clip advantages to `[-advantage_clip, +advantage_clip]` (default 2.0); clamp entropy weight \(w_{\mathrm{ent}} = H/\bar{H}\) to `[entropy_weight_min, entropy_weight_max]` (default 0.5 and 2.0). |
| **Rationale** | Cloud test runs showed loss sign flips and unstable reward when (1) depth-wise advantages grew unbounded (few nodes per depth → large z-scores) and (2) entropy weight increased over training (model entropy at nodes rising), amplifying gradient scale. Clipping/clamping keeps the weighted GRPO signal bounded and stabilizes training. |

**Details**:
- **Advantage clip**: Applied in `AdvantageComputer` after depth-wise z-score; default `advantage_clip=2.0`. Config: `advantage_clip`.
- **Entropy weight clamp**: Applied in `WeightedGRPOLoss._collect_transitions` when setting `entropy_weight` on each transition; default `[0.5, 2.0]`. Config: `entropy_weight_min`, `entropy_weight_max`.
- Bug fix at same time: removed erroneous `+ 1e-8` on the advantage value in BranchGRPO (only std denominator keeps 1e-8).

**Your Notes**:
> _(paste responses here)_

---

## D-015: Adaptive Branch Timing

| Field | Value |
|-------|-------|
| **Status** | `OPEN` |
| **Default** | Keep fixed `steps_per_expansion` as the baseline implementation, but treat it as a bootstrap baseline rather than the target long-term design |
| **Alternatives** | Entropy-threshold branching, token-fraction branching, KL-budget branching |

**Rationale**: Fixed branch intervals are simple and reproducible, but they do not align branch points with the model's actual decision structure. [TreeRL](https://arxiv.org/abs/2506.11902) argues that branching from high-uncertainty intermediate states improves search efficiency under the same generation token budget. For our method, fixed-step branching is acceptable for a baseline but not the strongest conceptual endpoint.

**Current Read**:
- **Good enough now**: fixed-step branching is a reasonable scaffold while validating the tree, reward, and loss pipeline.
- **Likely better target**: uncertainty-triggered branching after a minimum number of denoising steps, with a maximum-step fallback.
- **Recommended comparison set**: fixed-step vs token-fraction vs entropy-threshold.

**Your Notes**:
> Concept check (2026-03-15): treat fixed-step branching as a baseline, not a settled research choice. Once baseline parity is established, prioritize uncertainty-triggered branching over schedule-based branching.

---

## D-016: Time Weighting for Variable-Length Expansions

| Field | Value |
|-------|-------|
| **Status** | `OPEN` |
| **Default** | For the fixed-step baseline, the current parent-step lookup is an acceptable proxy. If adaptive stepping is introduced, switch to **interval-aware** edge weighting |
| **Alternatives** | Parent-step lookup only, interval-mass weighting, full micro-step logging |

**Rationale**: The current loss attaches time weight to `node.step_index`, i.e. the parent timestamp of an edge. This is only faithful when every edge spans the same number of denoising steps. If branches happen at inconsistent times, different edges cover different intervals and should not all receive a single-point weight. The natural discrete-time generalization of TempFlow-style schedules is:

$$w_{\text{edge}}(t_0, t_1) = \sum_{t=t_0}^{t_1-1} w(t)$$

**Current Read**:
- **Minimal fix**: compute time weight from the whole edge interval `[t_parent, t_child)`.
- **Preferred long-term fix**: log intra-chunk micro-transitions and apply time weights step-by-step.
- **Blocking issue for adaptive stepping**: do not evaluate dynamic branch timing without updating the time-weight logic accordingly.

**Your Notes**:
> Concept check (2026-03-15): point lookup is only a fixed-step proxy. Dynamic branching should ship together with interval-aware time weighting.

---

## D-017: Entropy Normalization Convention

| Field | Value |
|-------|-------|
| **Status** | `OPEN` |
| **Default** | Keep **mean entropy over masked positions** for node ranking, but normalize it against a baseline defined on the same statistic |
| **Alternatives** | Sequence-averaged entropy, masked-position mean entropy with analytic bound, masked-position mean entropy with empirical stage baseline |

**Rationale**: The current implementation stores node entropy as the mean over masked positions only, but compares it against `masking_ratio * log(V)`. That denominator corresponds to a sequence-averaged upper bound, not the masked-position mean. Mixing the two artificially inflates the entropy ratio at later denoising stages and muddies the interpretation of both entropy weighting and adaptive branch thresholds.

**Recommended options**:
1. **Analytic fallback**: `w_ent = H_masked_mean / log(V)`
2. **Preferred stage-aware version**: `w_ent = H_masked_mean / E[H_masked_mean | masking_ratio]`

**Current Read**:
- Node ranking by raw masked-position mean entropy is fine.
- Loss weighting and dynamic branch thresholds should use a normalization baseline defined on the exact same aggregate.
- If we keep additive time+entropy weights, both terms should eventually be rescaled to comparable magnitudes so `alpha_*` remains interpretable.

**Your Notes**:
> Concept check (2026-03-15): the entropy signal itself is good, but the current normalization convention is not yet conceptually clean.

---

## Decision Log (Chronological)

| Date | ID | Decision | Decided By |
|------|-----|----------|------------|
| 2026-02-18 | D-001 | M1 MacBook, float32 initially | Scott + Claude |
| 2026-02-18 | D-002 | No timestep param in model API | Confirmed from dLLM source |
| 2026-02-18 | D-003 | Gumbel noise matching dLLM | Confirmed from dLLM source |
| 2026-02-18 | D-007 | BranchGRPO-style advantages | Scott (branchgrpo_update.md) |
| 2026-02-08 | D-004 | MDLM Only |Scott (Expand after initial verification) |
| 2026-02-08 | D-005 | Mean Aggregation |Scott (Explore other Entropy Aggregation methods later?) |
| 2026-02-08 | D-006 | `low_confidence` | Scott (Ablate these methods later) |
| 2026-02-18 | D-013 | Single baseline: trajectory GRPO (no tree) | Plan (Phase 8) |
| 2026-02-18 | D-014 | Stability clamps: advantage_clip=2, entropy_weight in [0.5, 2] | Cloud test analysis |
