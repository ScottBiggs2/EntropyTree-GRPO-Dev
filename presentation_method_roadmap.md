# Presentation Roadmap: Current Method vs Next Steps

**Project**: Entropy-Guided MCTS-GRPO for Diffusion Language Models  
**Date**: 2026-03-15  
**Purpose**: Presentation-ready markdown summarizing the current method, where it is conceptually strong, where it is still provisional, and the next development steps grounded in both the codebase and the literature.

---

## One-Slide Executive Summary

**Core idea**: Train a masked diffusion language model with tree-structured GRPO, using entropy to decide which partial denoising states deserve more exploration.

**Current method**:
- exact Shannon entropy from MDLM
- global frontier selection over tree leaves
- fixed-size macro-step branching
- BranchGRPO-style advantage backpropagation
- additive time + entropy loss weighting

**Main conceptual gap**:
- the system uses entropy well to decide **which node** to expand
- but not yet to decide **when** to branch
- and the current weighting conventions are not fully internally aligned

**Next-step thesis**:
1. clean up the weighting math
2. move from fixed-step to uncertainty-triggered branching
3. make time weighting interval-aware when branch lengths become variable

---

## Suggested Slide Sequence

1. Problem framing
2. Why diffusion + tree RL
3. Current method overview
4. Current method mathematics
5. Current implementation mapping
6. What is already working conceptually
7. Limitation 1: entropy normalization
8. Limitation 2: fixed branch timing
9. Limitation 3: time weighting under variable intervals
10. Immediate next steps
11. Medium-term roadmap
12. Research positioning and literature support

---

## 1. Problem Framing

### Slide Message

Trajectory-level RL wastes structure in diffusion generation.

### Narrative

Masked diffusion language models generate by gradually resolving uncertainty over a set of masked tokens. A standard trajectory-level RL setup treats a whole denoising path as a single sample, which loses two useful forms of structure:

- **shared prefixes**: many candidate trajectories agree for a long time before diverging
- **intermediate uncertainty**: not all denoising states are equally informative

This motivates a tree-structured formulation:

- reuse common denoising prefixes
- explore several continuations from uncertain intermediate states
- propagate terminal rewards back through the shared tree

### Literature Positioning

- **TreeGRPO** shows that denoising can be recast as a tree and that tree-structured reward backpropagation improves efficiency and credit assignment: [TreeGRPO](https://treegrpo.github.io/)
- **TreeRL** shows that on-policy tree search with branching from high-uncertainty intermediate states can beat simple chain sampling under the same token budget: [TreeRL](https://arxiv.org/abs/2506.11902)

---

## 2. Why Diffusion + Tree RL

### Slide Message

Diffusion gives a natural uncertainty signal; tree RL gives a natural exploration structure.

### Key Observation

For MDLM-like masked discrete diffusion models, each denoising state produces a categorical distribution over tokens, so we get exact token-level uncertainty:

$$H_i(z_t) = -\sum_v p_\theta(x_i = v \mid z_t)\log p_\theta(x_i = v \mid z_t)$$

This is unusually useful:
- no separate uncertainty model is needed
- entropy is exact, not heuristic
- entropy can be used for both search and training

### Why a Tree

If two candidate completions share the first half of denoising, we should not pay the full cost twice. A tree lets us:

- keep a shared prefix state
- branch only where exploration matters
- assign credit to decision points rather than only to full trajectories

### Literature Positioning

- **MDLM** gives the exact entropy foundation for discrete diffusion: [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)
- **DeepSearch** supports global frontier selection over the whole tree rather than purely local child selection: [DeepSearch](https://arxiv.org/abs/2509.25454)

---

## 3. Current Method Overview

### Slide Message

The current system is a strong baseline: entropy-guided tree search with weighted GRPO over macro-step edges.

### Current Pipeline

1. Start from a prompt plus a fully masked response region.
2. Compute node entropy from the MDLM logits.
3. Rank active leaves globally by entropy.
4. Expand the top frontier nodes.
5. For each expansion, run stochastic denoising for a fixed number of steps.
6. Build multiple children from the same parent using Gumbel-noised sampling.
7. Complete the leaves to full responses.
8. Score the leaves with the reward function.
9. Backpropagate rewards using BranchGRPO-style fusion and depth-wise normalization.
10. Compute the weighted GRPO loss over parent-child edges.

### Codebase Mapping

- Entropy computation: `src/entropy.py`
- Tree construction and branching: `src/tree_builder.py`
- Advantage propagation: `src/advantages.py`
- Time weighting: `src/time_weight.py`
- Loss construction: `src/loss.py`

---

## 4. Current Method Mathematics

### 4.1 Node Uncertainty

The current node score is the mean entropy over the still-masked positions:

$$H_{\text{masked-mean}}(n) = \frac{1}{|M(n)|}\sum_{i \in M(n)} H_i(n)$$

where:
- $n$ is a node
- $M(n)$ is the set of masked response positions at that node

### 4.2 Frontier Selection

The tree uses **global frontier selection**:

$$n^\star = \arg\max_{n \in \mathcal{F}} H_{\text{masked-mean}}(n)$$

where $\mathcal{F}$ is the set of current leaf nodes.

### 4.3 Macro-Step Branch Expansion

Each chosen frontier node expands by running a fixed-size denoising chunk:

$$n \rightarrow c_j \quad \text{after} \quad \Delta t = \texttt{steps\_per\_expansion}$$

Each inner denoising step commits some number of masked tokens:

$$k = \min\left(k_{\text{cap}},\ \max\left(1,\left\lfloor \frac{n_{\text{masked}}}{\text{steps\_left}} \right\rfloor\right)\right)$$

This is the current approximation used in `tree_builder.py` to distribute unmasking over the chunk.

### 4.4 Advantage Computation

The current implementation uses BranchGRPO-style path-weighted reward fusion:

$$\bar{r}(n) = \sum_{c \in \text{children}(n)} p(c \mid n)\,\bar{r}(c)$$

with leaf initialization:

$$\bar{r}(\ell) = R(\ell)$$

Then it normalizes fused rewards depth-wise:

$$A_d(n) = \frac{\bar{r}(n) - \mu_d}{\sigma_d + \epsilon}$$

This is implemented in `src/advantages.py`.

### 4.5 Weighted GRPO Loss

The current edge-level loss is:

$$\mathcal{L} = - \frac{1}{|\mathcal{E}|} \sum_{e \in \mathcal{E}} \left(\alpha_{\text{time}} w_{\text{time}}(e) + \alpha_{\text{ent}} w_{\text{ent}}(e)\right) A(e)\,\log \pi_\theta(e)$$

where:
- $e$ is a parent-child transition
- $A(e)$ is the advantage attached to the child
- $\log \pi_\theta(e)$ is the log-prob surrogate of the child transition from the parent state

Current time schedule:

$$w_{\text{time}}(t) \propto (1 - t/T)^2$$

Current entropy ratio intention:

$$w_{\text{ent}} \approx \frac{H(n)}{\bar{H}(n)}$$

---

## 5. What Is Already Strong

### Slide Message

The method already has a solid conceptual center.

### Strengths

#### 1. Exact entropy is a real advantage

Unlike continuous diffusion systems, the project does not need a learned or approximate uncertainty proxy for MDLM.

#### 2. Global frontier selection is the right search abstraction

Rather than forcing all parents to expand equally, the tree focuses budget on the most uncertain current leaves.

#### 3. Tree-structured credit assignment is stronger than trajectory-level credit

BranchGRPO-style fused rewards and depth-wise normalization are the right stability layer for a branching method.

#### 4. Macro-step edges are a practical starting point

Using coarse expansions keeps the tree computationally manageable and matches the spirit of sparse planning ideas like **Fast-MCTD**: [Fast-MCTD](https://arxiv.org/abs/2506.09498)

---

## 6. Current Limitation 1: Entropy Normalization

### Slide Message

The entropy signal is right; the current normalization target is not yet matched to the actual statistic.

### Current State

The implementation stores:

$$H_{\text{masked-mean}} = \frac{1}{|M|}\sum_{i \in M} H_i$$

but often reasons using a baseline of the form:

$$\bar{H}(r) = r \log V$$

These are different objects:
- $H_{\text{masked-mean}}$ is a mean over masked positions only
- $r\log V$ is a sequence-averaged upper-bound style quantity

### Why This Matters

The ratio

$$\frac{H_{\text{masked-mean}}}{r\log V}$$

will grow as $r$ decreases, even if masked-position uncertainty is not especially unusual. That can distort:
- entropy weighting in the loss
- any future entropy-threshold branching rule

### Next-Step Correction

Use a denominator defined on the same statistic.

#### Option A: analytic upper-bound normalization

$$w_{\text{ent}} = \frac{H_{\text{masked-mean}}}{\log V}$$

#### Option B: stage-aware empirical normalization

$$w_{\text{ent}}(r) = \frac{H_{\text{masked-mean}}}{\mathbb{E}[H_{\text{masked-mean}} \mid r]}$$

### Recommendation

Use:
- raw masked-position mean entropy for **frontier ranking**
- stage-aware normalized masked-position mean entropy for **loss weighting** and **adaptive branch decisions**

---

## 7. Current Limitation 2: Fixed Branch Timing

### Slide Message

The current method knows which states are uncertain, but not yet when to stop and branch inside a denoising chunk.

### Current State

The current expansion rule in `src/tree_builder.py` is:

$$\Delta t = \texttt{steps\_per\_expansion}$$

So every edge spans the same number of denoising steps.

### Why This Is Not Ideal

Important decisions in code generation are often localized:
- recursion vs iteration
- variable binding structure
- control-flow pattern

The model may become sharply uncertain at a specific intermediate point, then rapidly become confident once one decision is committed.

Fixed-step branching can miss exactly that moment.

### Literature Support

**TreeRL** explicitly argues for branching from high-uncertainty intermediate states rather than arbitrary intermediate positions: [TreeRL](https://arxiv.org/abs/2506.11902)

### Next-Step Options

#### Option A: token-fraction branching

Stop after unmasking a fixed fraction of the remaining masked tokens:

$$n_{\text{target}} = \max\left(1,\left\lfloor f \cdot n_{\text{masked}} \right\rfloor\right)$$

This is the cheapest adaptive baseline.

#### Option B: entropy-threshold branching

Monitor uncertainty during denoising and stop when the state is more uncertain than expected:

$$u_t = \frac{H_{\text{masked-mean}}(t)}{\mathbb{E}[H_{\text{masked-mean}} \mid r_t]}$$

Branch when:

$$u_t > \tau_{\text{branch}}$$

Possible refinement:

$$u_t > \tau_{\text{branch}} \quad \text{and} \quad \left|\frac{dH}{dt}\right| < \epsilon_{\text{stagnation}}$$

This tries to branch at states where the model is uncertain and not making progress.

#### Option C: KL-budget branching

Accumulate stepwise KL shifts:

$$\text{KL}_{t \rightarrow t+1} = D_{\text{KL}}(p_\theta(\cdot \mid z_t)\,\|\,p_\theta(\cdot \mid z_{t+1}))$$

Branch when cumulative KL exceeds a budget.

### Recommendation

Presentation framing:
- fixed-step branching = strong baseline
- token-fraction = low-risk adaptive baseline
- entropy-threshold = research target
- KL-budget = principled later extension

---

## 8. Current Limitation 3: Time Weighting Under Variable Intervals

### Slide Message

Time weighting is currently only correct under the fixed-step abstraction.

### Current State

The current loss attaches time weight using the parent timestamp:

$$w_{\text{edge}} = w(t_{\text{parent}})$$

with:

$$w(t) \propto (1 - t/T)^2$$

This is acceptable only while every edge spans the same denoising interval.

### Why It Breaks Under Adaptive Branching

If one edge spans 8 steps and another spans 40 steps, a single point weight does not represent the same amount of denoising process.

### Correct Macro-Edge View

Treat time weighting as mass over an interval:

$$w_{\text{interval}}(t_0,t_1) = \sum_{t=t_0}^{t_1-1} w(t)$$

This is the minimal correction once edge lengths become variable.

### Preferred Long-Term View

Log the micro-step denoising transitions and weight them individually:

$$\mathcal{L}_{\text{process}} = -\sum_t w(t)\,A_t\,\log \pi_\theta(a_t \mid s_t)$$

This is more faithful to:
- TreeRL's dense process supervision
- TreeGRPO's step-specific credit philosophy

### Recommendation

Adaptive branching and interval-aware time weighting should ship together.

---

## 9. Immediate Next Steps

### Slide Message

The next moves should improve correctness before they increase complexity.

### Step 1: Clean up the weighting conventions

#### Goal

Make the math internally consistent before running major adaptive-branching experiments.

#### Tasks

1. Align entropy normalization with the actual node statistic.
2. Resolve the scale mismatch between `w_time` and `w_ent`.
3. Resolve the `total_denoising_steps` horizon mismatch across config/docs/tests.

#### Practical Deliverable

A new weighting formulation where:
- `w_ent` is computed from a consistent baseline
- `w_time` is interpretable relative to `w_ent`
- `alpha_time` and `alpha_entropy` actually mean what they say

### Step 2: Add adaptive branching

#### Goal

Let uncertainty decide branch timing, not just node ranking.

#### Tasks

1. Add token-fraction branching as a simple baseline.
2. Add entropy-threshold branching as the main research path.
3. Keep fixed-step branching as the controlled baseline.

### Step 3: Make time weighting interval-aware

#### Goal

Avoid misweighting edges once branch intervals are no longer uniform.

#### Tasks

1. Replace point lookup with interval-mass weighting at the edge level.
2. Only then interpret results from dynamic branching experiments.

---

## 10. Medium-Term Roadmap

### Slide Message

After the MDLM method is internally consistent, the next question is generality.

### Phase A: Validate on MDLM

Compare:
- baseline GRPO
- current fixed-step entropy tree
- adaptive branching tree

Track:
- reward
- diversity of leaves
- training stability
- compute cost per tree

### Phase B: Extend to BD3LM

Why:
- entropy stays exact
- tree logic becomes block-aware
- tests whether the method generalizes across discrete denoising structures

### Phase C: Scale to Dream / LLaDA-style systems

Why:
- same entropy mathematics
- different scale regime
- larger models may need different exploration settings

### Phase D: Continuous diffusion

Why:
- largest long-term scope expansion

What changes:
- no direct Shannon entropy
- need a proxy uncertainty estimator
- tree builder semantics change more substantially

---

## 11. Presentation-Ready Comparison Table

| Topic | Current Method | Next-Step Method |
|---|---|---|
| Uncertainty signal | Exact Shannon entropy over MDLM logits | Same core signal for MDLM; eventually generalized via uncertainty-estimator abstraction |
| Node ranking | Global frontier selection by masked-position mean entropy | Keep |
| Branch timing | Fixed `steps_per_expansion` | Token-fraction baseline, then entropy-threshold branching |
| Time weighting | Parent-step lookup | Interval-aware edge mass; eventually micro-step process weighting |
| Entropy weighting | Mixed normalization convention | Baseline matched to masked-position mean entropy |
| Credit assignment | BranchGRPO-style path fusion + depth norm | Keep |
| Transition granularity | Macro-edge surrogate | Macro-edge with corrected interval weighting, then micro-step logging |
| Research status | Strong baseline | Stronger, more self-consistent research version |

---

## 12. Literature Positioning

### MDLM

Supports the exact entropy foundation for masked discrete diffusion:
[Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)

### DeepSearch

Supports global frontier selection and the idea that search budget should be focused, not uniformly distributed:
[DeepSearch](https://arxiv.org/abs/2509.25454)

### TreeGRPO

Supports tree-structured denoising, shared prefix reuse, and step-sensitive credit assignment:
[TreeGRPO](https://treegrpo.github.io/)

### TreeRL

Supports branching from high-uncertainty intermediate states and motivates moving from fixed branch intervals toward uncertainty-triggered branching:
[TreeRL](https://arxiv.org/abs/2506.11902)

### TempFlow-GRPO

Supports the idea that denoising time matters and should be explicitly weighted:
[TempFlow-GRPO](https://arxiv.org/abs/2508.04324)

### Fast-MCTD

Supports macro-step planning as a practical computational approximation:
[Fast-MCTD](https://arxiv.org/abs/2506.09498)

---

## Closing Slide

### Final Message

The project already has the right conceptual core:
- exact uncertainty from MDLM
- tree-structured exploration
- structured reward backpropagation

The next version should not change the idea. It should **tighten the alignment between the idea, the mathematics, and the implementation**.

### Bottom Line

1. Keep entropy as the central uncertainty signal.
2. Move from fixed branch schedules to uncertainty-triggered branching.
3. Upgrade time weighting as soon as branch timing becomes irregular.
4. Treat weighting consistency as the prerequisite to the next round of experiments.

---

## Optional Appendix Slides

### Appendix A: Current Code Anchors

- `src/entropy.py`: token entropy, aggregation, expected entropy helper
- `src/tree_builder.py`: root creation, frontier ranking, fixed-step expansion, completion
- `src/advantages.py`: path-weighted reward fusion, depth-wise normalization
- `src/time_weight.py`: quadratic time schedule
- `src/loss.py`: edge collection, time/entropy weighting, log-prob surrogate

### Appendix B: Minimal Mathematical Upgrade Path

#### Current

$$w(e) = \alpha_{\text{time}} w(t_{\text{parent}}) + \alpha_{\text{ent}} \frac{H_{\text{masked-mean}}}{r\log V}$$

#### Proposed near-term

$$w(e) = \alpha_{\text{time}} \sum_{t=t_0}^{t_1-1} w(t) + \alpha_{\text{ent}} \frac{H_{\text{masked-mean}}}{\mathbb{E}[H_{\text{masked-mean}} \mid r]}$$

#### Proposed long-term

$$\mathcal{L} = -\sum_t \left(\alpha_{\text{time}}w(t) + \alpha_{\text{ent}}u_t\right) A_t \log \pi_\theta(a_t \mid s_t)$$

with:

$$u_t = \frac{H_{\text{masked-mean}}(t)}{\mathbb{E}[H_{\text{masked-mean}} \mid r_t]}$$
