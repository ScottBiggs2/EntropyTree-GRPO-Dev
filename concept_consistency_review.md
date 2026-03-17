# Concept Consistency Review

**Project**: Entropy-Guided MCTS-GRPO for Diffusion Language Models  
**Date**: 2026-03-15  
**Scope**: Concept check of the current method and the next-stage evolution, grounded in the current implementation plus the research notes in `literature_reference.md`, `future_extensions_plan.md`, and `research_decisions.md`

---

## Executive Verdict

The current method is **directionally strong but not yet fully self-consistent**.

1. **Entropy as the uncertainty signal is the right core choice for MDLM**. The model gives exact Shannon entropy, and using that for global frontier selection is conceptually well-supported by MDLM, DeepSearch, and the general entropy-guided MCTS literature.
2. **The current branching policy is only half aligned with the stated method**. The tree already uses entropy to decide **which node** to expand, but not yet **when** to branch within a denoising trajectory. Fixed `steps_per_expansion` is a useful scaffold, not the right end state.
3. **Time weighting is only a proxy under fixed-size macro-edges**. Once branch timing becomes dynamic, parent-point lookup is no longer conceptually correct. Time weight has to become interval-aware, or the loss will misweight edges that span different amounts of denoising time.

The highest-priority conceptual correction is:

**Make the weighting conventions internally consistent before drawing conclusions from adaptive branching experiments.**

That means:
- fix the entropy normalization convention
- fix the scale mismatch between time and entropy weights
- switch time weighting from point lookup to interval-aware weighting when branch lengths become variable

---

## Current Method Snapshot

As implemented today, the method is:

- **Tree construction**: global frontier selection over leaves using node entropy
- **Uncertainty signal**: exact Shannon entropy from the MDLM logits
- **Branch generation**: stochastic denoising with Gumbel noise, but each edge advances by a fixed number of denoising steps
- **Advantage assignment**: BranchGRPO-style path-weighted reward fusion plus depth-wise normalization
- **Loss weighting**: additive combination of time weight and entropy weight

This is a coherent baseline. The main issue is not that the ingredients are wrong, but that a few of them are still expressed on different scales or different temporal abstractions.

---

## Check 1: Are We Weighting Entropy Correctly?

## Short Answer

**Partly.** The entropy signal itself is correct. The current normalization and weight combination are not yet conceptually clean.

## What Is Correct

For an MDLM-style masked discrete diffusion model, exact Shannon entropy is the right uncertainty primitive:

$$H_i = -\sum_v p_\theta(x_i = v \mid z_t)\log p_\theta(x_i = v \mid z_t)$$

That part is strong:
- it is exact, not approximate
- it is available from the model in one forward pass
- it directly measures uncertainty over discrete token choices
- it is well-matched to branching decisions in text/code generation

Using entropy to rank frontier nodes is also conceptually sound. This is the cleanest part of the current design.

## What Is Not Yet Consistent

The current code stores node entropy as:

$$H_{\text{masked-mean}} = \frac{1}{|M|}\sum_{i \in M} H_i$$

that is, the **mean entropy over masked positions only**.

But the current normalization baseline is effectively:

$$\bar{H}(r) = r \log V$$

where $r$ is the masking ratio.

These are different aggregates:
- $H_{\text{masked-mean}}$ lives in $[0, \log V]$
- $r \log V$ is a sequence-averaged upper bound that corresponds to treating unmasked positions as zero-entropy contributors

So the ratio

$$\frac{H_{\text{masked-mean}}}{r \log V}$$

is not a clean "relative entropy" quantity. It will tend to inflate as $r$ gets small, even when the masked-position uncertainty itself is not unusually large.

That matters twice:
- for the loss weight
- for any future entropy-threshold branching rule

## Additional Scale Problem

The current combined weight is:

$$w = \alpha_{\text{time}} w_{\text{time}} + \alpha_{\text{ent}} w_{\text{ent}}$$

but the two terms are not on comparable scales:
- `w_time` is normalized so the whole time schedule sums to 1, so each individual per-step weight is roughly `O(1 / T)`
- `w_ent` is clamped around `O(1)`

So with the current defaults, the entropy term dominates numerically and the time term is mostly a small perturbation. That makes `alpha_time` and `alpha_entropy` less interpretable than the docs suggest.

## Recommended Fix

Split the problem into two separate design choices.

### 1. Node ranking signal

Keep:

$$H_{\text{rank}} = H_{\text{masked-mean}}$$

This is simple and well-matched to "which partially denoised state is still uncertain?"

### 2. Loss weighting signal

Use a baseline defined on the **same statistic**.

Two good options:

1. **Analytic fallback**

$$w_{\text{ent}} = \frac{H_{\text{masked-mean}}}{\log V}$$

2. **Preferred stage-aware version**

$$w_{\text{ent}} = \frac{H_{\text{masked-mean}}}{\mathbb{E}[H_{\text{masked-mean}} \mid r]}$$

The empirical stage-aware version is better because it captures what "more uncertain than expected at this denoising stage" actually means for the current model.

## Conclusion

**Entropy itself: yes. Current entropy normalization: not quite.**

The project is using the right uncertainty primitive, but the denominator should be redefined so the numerator and denominator describe the same object.

---

## Check 2: Are We Branching Properly?

## Short Answer

**Only partially.** The project already branches from the right *kind* of states globally, but not yet at the right *moments* locally.

## What Is Already Good

The current tree builder does several things well:

- uses **global frontier selection** instead of only local parent expansion
- prioritizes high-entropy leaves first
- uses stochastic sampling so children can actually diverge
- shares prefixes across branches, which is exactly where tree-based efficiency comes from

This lines up well with the broader tree-RL literature.

## Where It Falls Short

Each branch edge currently advances by a fixed `steps_per_expansion`.

That means:
- the tree decides **which leaf** is important using entropy
- but it still decides **when to stop denoising and create the next node** using a schedule

This is the main conceptual mismatch in the present method.

If the model's critical uncertainty spike happens at step 47, but the system only branches at steps 32 and 64, then the tree misses the decision point itself. The branch point is placed before the interesting moment or after the decision has already collapsed.

That is exactly what [TreeRL](https://arxiv.org/abs/2506.11902) is pushing against: branch from **high-uncertainty intermediate states** rather than from arbitrary or random intermediate positions.

## Why This Matters More for Diffusion Text Than It Might First Appear

In diffusion text generation, a small set of structural choices can collapse uncertainty across many remaining tokens:
- recursive vs iterative implementation
- base case phrasing
- variable naming and local structure
- control-flow template

Once one of those choices is committed, many downstream positions become easy.

So a good branch point is usually the state **just before** that collapse, not an arbitrary fixed offset along the denoising path.

## Recommended Evolution

Treat branch timing as its own policy.

The right next comparison is:

1. **Fixed-step branching**: current baseline
2. **Token-fraction branching**: cheap adaptive baseline
3. **Entropy-threshold branching**: preferred uncertainty-driven design

The entropy-threshold design should be defined using a consistent entropy baseline, not the current mixed statistic.

A practical version is:
- require a minimum number of denoising steps before early stopping
- stop when masked-position entropy is unusually high relative to a calibrated stage baseline
- optionally add a stagnation criterion so the system branches where the model is not making progress

## Conclusion

**The project is branching over the right frontier, but not yet at the right internal times.**

Fixed-step branching is a valid bootstrap baseline. It should not be treated as the conceptually preferred final form of the method.

---

## Check 3: Are We Weighting by Time Properly?

## Short Answer

**For the fixed-step baseline: approximately. For dynamic branching: no, not yet.**

## What the Current Code Assumes

The current loss attaches a time weight to each edge using only the parent node's `step_index`:

$$w_{\text{edge}} = w(t_{\text{parent}})$$

with a TempFlow-style schedule:

$$w(t) \propto (1 - t/T)^2$$

This is tolerable while all edges cover the same number of denoising steps, because the approximation error is shared fairly uniformly.

## Why It Breaks Under Dynamic Branching

If branch timing becomes adaptive, two different edges may cover very different denoising intervals:
- edge A: 8 steps
- edge B: 40 steps

If both receive the same single-point weight at their start time, the longer edge is underweighted relative to the amount of denoising process it represents.

The right interpretation is that `w(t)` is a density over denoising time, so an edge spanning `[t_0, t_1)` should inherit the mass of that whole interval:

$$w_{\text{interval}}(t_0, t_1) = \sum_{t=t_0}^{t_1-1} w(t)$$

That is the minimal correction required for variable branch timing.

## Even More Fundamental Issue

The current macro-edge loss also uses a coarse log-prob surrogate:
- an edge may summarize several denoising micro-steps
- the loss scores the parent-to-child token changes from the parent state in one shot

That is not the exact probability of the whole sequential denoising sub-trajectory. It is a useful surrogate for the fixed-step prototype, but once the project moves toward TreeRL-style process supervision, this approximation becomes the next conceptual bottleneck.

## Recommended Time-Weighting Path

### Minimal fix for adaptive stepping

Keep macro-edges, but weight them by interval mass:

$$w_{\text{edge}} = \sum_{t=t_{\text{parent}}}^{t_{\text{child}}-1} w(t)$$

### Preferred long-term fix

Store the micro-step transitions inside each expansion and compute:

$$\mathcal{L} = -\sum_t w(t)\,A_{\text{edge}}\,\log \pi(a_t \mid s_t)$$

This does three good things at once:
- handles irregular branch timing naturally
- aligns better with TreeRL's dense process supervision
- makes time weighting literally step-level instead of edge-level approximation

## Immediate Housekeeping Issue

There is also a horizon mismatch in the current project materials:
- runtime config uses `total_denoising_steps = 128`
- several docs and tests still assume `256`

That should be resolved before any interpretation of time-weight ablations, because the shape and support of the time schedule depend on `T`.

## Conclusion

**Time weighting is currently a fixed-step proxy, not a general solution.**

If dynamic branching lands, interval-aware time weighting should land with it.

---

## Recommended Development Trajectory

## Phase 1: Clean up the weighting semantics

Before major new experiments:

1. Make entropy normalization use a baseline defined on the same aggregate as the numerator
2. Put time and entropy weights on comparable scales so `alpha_time` and `alpha_entropy` are meaningful
3. Resolve the `128` vs `256` denoising-horizon inconsistency in docs/config/tests

## Phase 2: Upgrade branch timing

Then compare:

1. fixed-step branching
2. token-fraction branching
3. entropy-threshold branching

This is where [TreeRL](https://arxiv.org/abs/2506.11902) should inform the evolution most directly.

## Phase 3: Make time weighting interval-aware

The moment branch timing becomes variable:

1. switch from parent-point time lookup to interval-mass weighting
2. do not interpret adaptive-branching results without that fix

## Phase 4: Move toward process-level loss accounting

If the adaptive system is promising:

1. log micro-step transitions within `_denoise_chunk()`
2. apply time weights per micro-step
3. consider whether entropy should also be attached per micro-step or per branch state

That would move the method closer to the spirit of both [TreeRL](https://arxiv.org/abs/2506.11902) and [TreeGRPO](https://treegrpo.github.io/): shared prefixes, branch-level reuse, and step-aware credit assignment.

---

## Final Judgment

The project's conceptual center is good:
- exact entropy from MDLM
- global entropy-guided frontier selection
- tree-based reward backpropagation

The next stage should not be "add more machinery" blindly. It should be:

1. **make the current weighting conventions self-consistent**
2. **let uncertainty decide branch timing**
3. **upgrade time weighting when branch intervals become nonuniform**

That path preserves the originality of the method while bringing the math and implementation closer together.
