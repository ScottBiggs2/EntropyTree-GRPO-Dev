# EntropyTree-GRPO: LaTeX Pseudocode

This document gives a concise mathematical description of the EntropyTree-GRPO process in LaTeX-style notation. It matches the implementation in `src/` (tree builder, entropy, advantages, loss).

---

## 1. Notation and Setup

- **Prompt** \( x_{\text{prompt}} \): token sequence (e.g. from chat template).
- **State** \( z_t \in \mathcal{V}^L \): full sequence at “time” \( t \); positions \( \ell \in [L_{\text{prompt}}, L) \) are the **response region**; some may be **masked** (token id \( \texttt{[MASK]} \)).
- **Model** \( f_\theta \): MDLM forward; \( f_\theta(z_t) \) returns logits \( \mathbf{o} \in \mathbb{R}^{L \times |\mathcal{V}|} \) (same-position prediction: \( \mathbf{o}_\ell \) is the distribution over \( \mathcal{V} \) at position \( \ell \)).
- **Config**: tree budget \( N_{\max} \), branch width \( B \), steps per expansion \( S \), temperature \( \tau \), total denoising steps \( T \), loss weights \( \alpha_{\text{time}}, \alpha_{\text{entropy}} \).

---

## 2. Entropy at a Node

**Per-token entropy** (Shannon, from logits at the current state):

\[
\mathbf{o} = f_\theta(z_t), \qquad
p_\ell(v) = \mathrm{softmax}(\mathbf{o}_\ell)_v, \qquad
H_\ell = -\sum_{v \in \mathcal{V}} p_\ell(v) \log p_\ell(v).
\]

**Masked aggregation** (mean over masked positions only; \( \mathcal{M}_t = \{ \ell : (z_t)_\ell = \texttt{[MASK]} \} \)):

\[
H(z_t) = \frac{1}{|\mathcal{M}_t|} \sum_{\ell \in \mathcal{M}_t} H_\ell, \qquad
\text{(if } |\mathcal{M}_t| = 0 \text{, set } H(z_t) = 0\text{).}
\]

**Expected entropy** (for normalization; \( \rho_t \) = fraction of response still masked):

\[
\bar{H}(\rho_t) = \rho_t \ln|\mathcal{V}|.
\]

**Entropy weight** (used in loss):

\[
w_{\text{ent}}(z_t) = \frac{H(z_t)}{\bar{H}(\rho_t) + \epsilon}, \qquad \epsilon \approx 10^{-6}.
\]

---

## 3. Tree Construction (Entropy-Guided MCTS)

**3.1 Root**

\[
z_0: \text{ prompt tokens } + L_{\text{new}} \text{ mask tokens in response region}.
\]

**3.2 Global frontier selection (DeepSearch-style)**

- Maintain a **frontier** \( \mathcal{F} \) of current leaf nodes.
- For each node in \( \mathcal{F} \), if entropy not yet computed: set \( H(\text{node}) = H(z_t) \) from §2.
- Sort \( \mathcal{F} \) by entropy **descending** (high uncertainty first).
- Select top-\( k \) nodes to expand:
  \[
  k = \min\left(B,\; |\mathcal{F}|,\; N_{\max} - \text{nodes\_used}\right).
  \]

**3.3 Expand node**

For each selected node with state \( z_t \):

- Run **\( S \) denoising steps** from \( z_t \) with Gumbel-Max (see §4) to get \( z_{t+S}^{(j)} \), \( j = 1, \ldots, B \).
- Create \( B \) children; set \( \text{sampling\_prob} = 1/B \) per child.
- Remove the expanded node from \( \mathcal{F} \) and add the \( B \) new nodes to \( \mathcal{F} \).
- Repeat until \( \text{nodes\_used} \geq N_{\max} \) or \( \mathcal{F} \) is empty.

**3.4 Complete leaves**

For each node \( n \in \mathcal{F} \) left at the end:

- If \( |\mathcal{M}| = 0 \): treat as final leaf.
- Else: run **denoise-to-completion** from \( n \) (repeated unmasking until no masks in response); append the completed node as a child and take it as the **final leaf**.

**3.5 Fill missing entropy**

DFS over the full tree; for any node with entropy still unset, compute \( H(z_t) \) as in §2.

---

## 4. One Denoising Step (Low-Confidence Unmasking)

Given state \( z \), response region \( \mathcal{R} \), and number of steps left \( S_{\text{left}} \):

- \( \mathcal{M} = \{ \ell \in \mathcal{R} : z_\ell = \texttt{[MASK]} \} \), \( n = |\mathcal{M}| \). If \( n = 0 \), stop.
- **Unmask count** (cap so at least one mask remains after the chunk):
  \[
  k_{\text{cap}} = \max\left(1,\; \left\lfloor \frac{n_0 - 1}{S_{\text{total}}} \right\rfloor \right), \qquad
  k = \min\left( k_{\text{cap}},\; \max\!\left(1, \left\lfloor \frac{n}{S_{\text{left}}} \right\rfloor\right) \right),
  \]
  where \( n_0 \) is the number of masks at the start of the current expansion chunk.
- Forward: \( \mathbf{o} = f_\theta(z) \). Gumbel noise (temperature \( \tau \)): \( \hat{x}_0 = \operatorname{argmax}_v \bigl( \text{Gumbel}(\mathbf{o}; \tau) \bigr) \).
- Confidence (only on \( \mathcal{M} \)): \( c_\ell = p_\ell(\hat{x}_{0,\ell}) \) for \( \ell \in \mathcal{M} \), else \( -\infty \).
- Unmask the **top-\( k \)** positions by confidence: update \( z_\ell \leftarrow \hat{x}_{0,\ell} \) for those \( \ell \).

---

## 5. Rewards and Advantages

**Rewards:** For each final leaf, decode the response to string \( c \); \( R(c) = \text{Reward}(c, \text{prompt}) \) (e.g. syntax heuristic or execution-based).

**BranchGRPO advantages:**

1. **Path-weighted fusion** (bottom-up):
   - Leaf: \( \tilde{R}(n) = R(n) \).
   - Internal node \( n \) with children \( n_1,\ldots,n_B \):
     \[
     \tilde{R}(n) = \sum_{j=1}^{B} q_j\, \tilde{R}(n_j), \qquad q_j = \text{sampling\_prob of } n_j \;\text{(e.g. } 1/B\text{)}.
     \]

2. **Depth-wise normalization:** For each depth \( d \), let \( \mathcal{N}_d \) be the set of nodes at depth \( d \). Then
   \[
   A(n) = \frac{ \tilde{R}(n) - \mu_d }{ \sigma_d + \epsilon }, \qquad
   \mu_d = \frac{1}{|\mathcal{N}_d|} \sum_{n' \in \mathcal{N}_d} \tilde{R}(n'), \quad
   \sigma_d = \sqrt{ \frac{1}{|\mathcal{N}_d|} \sum_{n' \in \mathcal{N}_d} (\tilde{R}(n') - \mu_d)^2 } + \epsilon.
   \]
   The **advantage** stored on each node (for the loss) is this \( A \) (child’s advantage is used on the parent→child transition).

---

## 6. Weighted GRPO Loss

**Time weight** (TempFlow-GRPO style; \( t \) = step index of the parent node):
\[
w_{\text{time}}(t) = \frac{ (1 - t/T)^2 }{ \sum_{s=0}^{T-1} (1 - s/T)^2 }.
\]

**Transition:** Each parent→child edge is one transition. Let \( (z_{\text{par}}, z_{\text{child}}) \) be the state pair, \( A_{\text{child}} \) the child’s advantage, \( H(z_{\text{par}}) \) the parent’s entropy, \( \rho_{\text{par}} \) the parent’s masking ratio.

**Log-probability** of the transition (only positions that changed from mask to a token):
\[
\mathcal{M}_{\text{ch}} = \bigl\{ \ell : (z_{\text{par}})_\ell = \texttt{[MASK]} \wedge (z_{\text{par}})_\ell \neq (z_{\text{child}})_\ell \bigr\},
\]
\[
\log \pi_\theta(z_{\text{child}} \mid z_{\text{par}}) = \sum_{\ell \in \mathcal{M}_{\text{ch}}} \log \mathrm{softmax}(f_\theta(z_{\text{par}})_\ell)_{(z_{\text{child}})_\ell}.
\]

**Entropy weight** for this transition:
\[
w_{\text{ent}} = \frac{ H(z_{\text{par}}) }{ \bar{H}(\rho_{\text{par}}) + \epsilon }.
\]

**Per-transition loss term:**
\[
\ell_{\text{trans}} = -\,\Bigl( \alpha_{\text{time}}\, w_{\text{time}}(t) + \alpha_{\text{entropy}}\, w_{\text{ent}} \Bigr)\, A_{\text{child}}\, \log \pi_\theta(z_{\text{child}} \mid z_{\text{par}}).
\]

**Total loss** (average over all transitions in the tree):
\[
\mathcal{L} = \frac{1}{|\mathcal{T}|} \sum_{\text{trans} \in \mathcal{T}} \ell_{\text{trans}}.
\]

---

## 7. One Training Step (Summary)

1. **Build tree:** Given prompt, run §3 → root and final leaves \( \{n_1,\ldots,n_K\} \).
2. **Rewards:** \( R_i = R(c_i) \) for each leaf completion \( c_i \).
3. **Advantages:** Run §5 (path-weighted fusion + depth normalization) → \( A(n) \) on all nodes.
4. **Loss:** Collect all parent→child transitions; for each compute \( w_{\text{time}}, w_{\text{ent}}, \log \pi_\theta \), and form \( \mathcal{L} \) as in §6.
5. **Update:** \( \mathcal{L}.{\texttt{backward}}() \), gradient step (e.g. AdamW) with gradient clipping.

---

## 8. Baseline GRPO (No Tree)

- Sample \( K \) independent **trajectories** from the same prompt (each trajectory = full denoising to completion, storing all state pairs).
- \( R_i = R(c_i) \), \( \bar{R} = \frac{1}{K}\sum_i R_i \), \( A_i = R_i - \bar{R} \).
- \( \log \pi_\theta(\text{traj}_i) = \sum_{\text{trans} \in \text{traj}_i} \log \pi_\theta(z_{\text{child}} \mid z_{\text{par}}) \).
- \( \mathcal{L}_{\text{base}} = -\frac{1}{K} \sum_{i=1}^{K} A_i \log \pi_\theta(\text{traj}_i) \).

No tree, no time/entropy weighting; used as the baseline in Phase 8.
