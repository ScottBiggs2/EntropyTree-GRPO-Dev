Baseline: 

python scripts/run_experiment.py --method baseline --num_epochs 10 --run_name 10_epoch_baseline_8
 --num_baseline_samples 8

[baseline] epoch 0 loss=0.10586023330688477 avg_reward=0.23749999999999996 max_reward=0.3 tree_nodes=0.0 tree_leaves=8.0 avg_entropy=0.0 epoch=0 method=basel
ine wall_sec=1.7425792776048183
[baseline] epoch 1 loss=-0.19257442156473795 avg_reward=0.28125 max_reward=0.3 tree_nodes=0.0 tree_leaves=8.0 avg_entropy=0.0 epoch=1 method=baseline wall_se
c=1.2302715554833412
[baseline] epoch 2 loss=0.00201342503229777 avg_reward=0.28750000000000003 max_reward=0.3 tree_nodes=0.0 tree_leaves=8.0 avg_entropy=0.0 epoch=2 method=basel
ine wall_sec=1.239667471498251
[baseline] epoch 3 loss=0.0 avg_reward=0.3 max_reward=0.3 tree_nodes=0.0 tree_leaves=8.0 avg_entropy=0.0 epoch=3 method=baseline wall_sec=1.2035185396671295
[baseline] epoch 4 loss=0.0 avg_reward=0.3 max_reward=0.3 tree_nodes=0.0 tree_leaves=8.0 avg_entropy=0.0 epoch=4 method=baseline wall_sec=1.242689322680235
[baseline] epoch 5 loss=-0.04191978772481283 avg_reward=0.2625 max_reward=0.3 tree_nodes=0.0 tree_leaves=8.0 avg_entropy=0.0 epoch=5 method=baseline wall_sec
=1.220770139247179
[baseline] epoch 6 loss=-0.08722400665283203 avg_reward=0.24375 max_reward=0.25 tree_nodes=0.0 tree_leaves=8.0 avg_entropy=0.0 epoch=6 method=baseline wall_s
ec=1.2077123410999775
[baseline] epoch 7 loss=-0.06338652968406677 avg_reward=0.24375 max_reward=0.25 tree_nodes=0.0 tree_leaves=8.0 avg_entropy=0.0 epoch=7 method=baseline wall_s
ec=1.3618386276066303
[baseline] epoch 8 loss=0.03645020226637522 avg_reward=0.23124999999999998 max_reward=0.25 tree_nodes=0.0 tree_leaves=8.0 avg_entropy=0.0 epoch=8 method=base
line wall_sec=1.203005563467741
[baseline] epoch 9 loss=-0.02622215449810028 avg_reward=0.25416666666666665 max_reward=0.2833333333333334 tree_nodes=0.0 tree_leaves=8.0 avg_entropy=0.0 epoc
h=9 method=baseline wall_sec=1.265282653272152

EntropyTree Test: 

python scripts/run_experiment.py --method entropy_mcts --num_epochs 10 --run_name 2_epoch_retest 
--max_tree_nodes 16 --branch_width 3 --steps_per_expansion 8 --num_baseline_samples 8 

[entropy diagnostic] depth=1 node: n_masked=8, token_entropy at masked: min=0.000172 mean=0.092603 max=0.141104
[entropy_mcts] epoch 0 loss=0.0 avg_reward=0.29999999999999993 max_reward=0.3 tree_nodes=16.0 tree_leaves=11.0 avg_entropy=0.2816251847479079 n_transitions=1
5.0 mean_abs_adv=0.0 mean_w_time=0.010938175581395626 mean_w_ent=0.23495190217161066 mean_weight=0.12841412666720095 mean_weighted_adv_logp=0.0 epoch=0 metho
d=entropy_mcts
[entropy_mcts] epoch 1 loss=-0.027441720167795818 avg_reward=0.31515151515151507 max_reward=0.3333333333333333 tree_nodes=16.0 tree_leaves=11.0 avg_entropy=0
.2794340451558431 n_transitions=15.0 mean_abs_adv=0.2514156914586358 mean_w_time=0.010938175581395626 mean_w_ent=0.48012746023322267 mean_weight=0.2510019056
9800696 mean_weighted_adv_logp=-0.027441774411340266 epoch=1 method=entropy_mcts
[entropy_mcts] epoch 2 loss=-0.8985458215077718 avg_reward=0.25454545454545446 max_reward=0.3 tree_nodes=16.0 tree_leaves=11.0 avg_entropy=0.2568994429376390
6 n_transitions=15.0 mean_abs_adv=0.251415726644107 mean_w_time=0.010938175581395626 mean_w_ent=0.38536101451900223 mean_weight=0.20361868284089676 mean_weig
hted_adv_logp=-0.898545631097464 epoch=2 method=entropy_mcts
[entropy_mcts] epoch 3 loss=0.014317730131248632 avg_reward=0.29090909090909084 max_reward=0.3 tree_nodes=16.0 tree_leaves=11.0 avg_entropy=0.222608659002516
 n_transitions=15.0 mean_abs_adv=0.43997739273835834 mean_w_time=0.010938175581395626 mean_w_ent=0.3717020148002848 mean_weight=0.19678918298153802 mean_weig
hted_adv_logp=0.0143177588697773 epoch=3 method=entropy_mcts
[entropy_mcts] epoch 4 loss=0.0279252752661705 avg_reward=0.3045454545454545 max_reward=0.3333333333333333 tree_nodes=16.0 tree_leaves=11.0 avg_entropy=0.184
47845511966285 n_transitions=15.0 mean_abs_adv=0.23756551027136744 mean_w_time=0.010938175581395626 mean_w_ent=0.3469692060652436 mean_weight=0.1844227786140
1743 mean_weighted_adv_logp=0.02792527938203539 epoch=4 method=entropy_mcts
[entropy_mcts] epoch 5 loss=0.5944836934407552 avg_reward=0.32424242424242417 max_reward=0.3333333333333333 tree_nodes=16.0 tree_leaves=11.0 avg_entropy=0.17
89755490091112 n_transitions=15.0 mean_abs_adv=0.2514156914586359 mean_w_time=0.010938175581395626 mean_w_ent=0.3996473971706915 mean_weight=0.21076187416674
133 mean_weighted_adv_logp=0.5944836197413707 epoch=5 method=entropy_mcts
[entropy_mcts] epoch 6 loss=-0.3252462263529499 avg_reward=0.2712121212121212 max_reward=0.3666666666666667 tree_nodes=16.0 tree_leaves=11.0 avg_entropy=0.18
653634521696305 n_transitions=15.0 mean_abs_adv=0.684033025525696 mean_w_time=0.010938175581395626 mean_w_ent=0.412859728626571 mean_weight=0.217368039894681
12 mean_weighted_adv_logp=-0.3252461529077508 epoch=6 method=entropy_mcts
[entropy_mcts] epoch 7 loss=0.22871280457577647 avg_reward=0.29545454545454547 max_reward=0.3166666666666667 tree_nodes=16.0 tree_leaves=11.0 avg_entropy=0.2
0416579643885294 n_transitions=15.0 mean_abs_adv=0.5028313300465653 mean_w_time=0.010938175581395626 mean_w_ent=0.5712911238098372 mean_weight=0.296583737486
3142 mean_weighted_adv_logp=0.22871273888112423 epoch=7 method=entropy_mcts
[entropy_mcts] epoch 8 loss=0.3205408801635106 avg_reward=0.3151515151515151 max_reward=0.35000000000000003 tree_nodes=16.0 tree_leaves=11.0 avg_entropy=0.23
071788416968453 n_transitions=15.0 mean_abs_adv=0.7319811602870755 mean_w_time=0.010938175581395626 mean_w_ent=0.6815199079464777 mean_weight=0.3516981295546
344 mean_weighted_adv_logp=0.32054088380814505 epoch=8 method=entropy_mcts
[entropy_mcts] epoch 9 loss=-0.12826283586828602 avg_reward=0.31363636363636366 max_reward=0.35000000000000003 tree_nodes=16.0 tree_leaves=11.0 avg_entropy=0
.2583866980340746 n_transitions=15.0 mean_abs_adv=0.2514157096064818 mean_w_time=0.010938175581395626 mean_w_ent=0.6251749791966068 mean_weight=0.32352566517
969905 mean_weighted_adv_logp=-0.128262766488941 epoch=9 method=entropy_mcts
[entropy_mcts] Saved final.pt and config.json to checkpoints/entropy_mcts_grpo/2_epoch_retest
---

I notice that the loss is sometimes negative, or flips between positive and negative.

The average reward is extremely unstable, as is the max. I also notice that the mean weight and mean entropy weights are increasing, while the time weight mean is constant! I think this may be the issue - let's be sure we're being careful about diversifying where the initial branch point (if it's a bug, not a feature? it smells funny though).  

---

## What this tells us

### 1. **Epoch 0: zero advantage → no learning**
- `mean_abs_adv=0.0` and `mean_weighted_adv_logp=0.0` in epoch 0 ⇒ **every node’s advantage was 0**, so loss = 0 and there was no gradient.
- Cause: all 11 leaves got the same reward (0.3), so path-weighted fusion gives the same `fused_reward` at every node. Depth-wise z-score is then \((x - \mu_d)/\sigma_d\); when all nodes at depth \(d\) are equal, \(\sigma_d \approx 0\) (plus epsilon), so we effectively get 0. So **when rewards don’t differ across leaves, we get no signal**.
- Implication: we need a baseline (e.g. subtract global mean reward or use a running average) so that even when rewards are identical, we still get a nonzero spread, or we need to avoid zero std in the advantage normalization.

### 2. **Constant mean_w_time is expected (not a bug)**
- Time weight depends only on **step_index** (which denoising step the parent node is at). The tree has fixed shape (16 nodes, 15 transitions) and step indices are determined by the tree (root=0, children = parent + steps_per_expansion, etc.). So the **set of step_indices** in the 15 transitions is the same every epoch ⇒ **mean_w_time is the same**. So constant mean_w_time is expected given fixed tree topology and same prompts per epoch.
- It can “smell funny” because we’re always emphasizing the same time slice; diversifying *which* nodes we expand (e.g. random tie-break in frontier selection) would change which step_indices appear and could spread the time weighting — but that’s an exploration choice, not a bug.

### 3. **Increasing mean_w_ent and mean_weight ⇒ instability**
- **mean_w_ent** goes from ~0.23 (epoch 0) to ~0.68 (epoch 9); **mean_weight** (α_time·w_time + α_ent·w_ent) grows similarly.
- Entropy weight is \(w_{\mathrm{ent}} = H/\bar{H}\) at each node; \(\bar{H}\) is fixed by mask ratio (tree structure). So the rise is from **model entropy \(H\)** increasing at those same nodes as training progresses ⇒ the loss is **up-weighting high-uncertainty nodes more over time**, so effective gradient scale grows ⇒ **feedback loop and unstable updates**.
- Fix: cap or anneal \(w_{\mathrm{ent}}\) (e.g. clamp to [0.5, 2] or decay α_entropy over epochs), or use a stable baseline for \(\bar{H}\).

### 4. **mean_abs_adv growing ⇒ large, flipping advantages**
- **mean_abs_adv** goes from 0 → ~0.25 → up to ~0.73. So advantage magnitudes grow; with depth-wise z-scores and few nodes per depth, \(\sigma_d\) can be small ⇒ large \(|A|\). Then loss = \(-w \cdot A \cdot \log p\); when \(A\) flips sign across batches, **loss flips sign** (as observed).
- So the flip-flop is from **high-variance, sometimes negative advantages** (depth-normalized) combined with negative log_probs. Stabilizing: clip advantages (e.g. ±2), increase epsilon in std, or use global advantage normalization instead of per-depth.

### 5. **Unstable avg/max reward**
- Tied to the above: noisy, sometimes wrong-signed gradients and increasing weights make policy updates inconsistent, so reward curves don’t improve steadily.

### Recommended next steps (in order)
1. **Fix zero advantage when all rewards equal:** e.g. subtract global mean reward (over leaves) so advantages are relative to batch mean, and ensure \(\sigma_d\) has a minimum (e.g. max(σ_d, 0.1)).
2. **Stabilize entropy weight:** cap \(w_{\mathrm{ent}}\) (e.g. min(max(w_ent, 0.5), 2)) or reduce α_entropy.
3. **Stabilize advantages:** clip \(A\) to [-2, 2] or similar, and/or use a single global mean/std over all nodes instead of per-depth.
4. **Optional:** add randomness to frontier selection (e.g. shuffle ties by entropy) so branch points and step_index mix vary more across runs.

---

## How to interpret tree_nodes vs tree_leaves

- **tree_nodes** = total number of nodes in the search tree (root + every expanded child). So it’s the size of the whole tree.
- **tree_leaves** = number of *final* leaves: the completion nodes we assign rewards to and use for the GRPO loss (after expansion stops and `_complete_leaves` runs).

So **leaves ≤ nodes** always. With branching, many nodes are internal (they have children); only the “tip” nodes at the end of expansion become the leaves we complete and score.

Your EntropyTree run: **10 nodes, 5 leaves** is expected. It means the tree has 10 vertices total; 5 of them are internal (root + 4 others that were expanded), and 5 are final leaves (the completions used for reward and advantages). So you get 5 completion trajectories per prompt, coming from a small tree of 10 nodes. Baseline correctly reports **0 nodes, 4 leaves** because it doesn’t build a tree—it just samples 4 trajectories (num_baseline_samples=4); the “leaves” count is the number of completions.