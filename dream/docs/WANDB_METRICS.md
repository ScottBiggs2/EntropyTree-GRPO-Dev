# Interpreting Dream comparison metrics (W&B)

This note explains **loss-side** scalars from `WeightedGRPOLoss` and **run metadata** from `run_dream_comparison.py`, with an eye on **stability** (clamps, normalization) and why some curves look “flat” or identical across runs.

## 1. `phase_idx` does not vary over training (by design)

`phase_idx` indexes `PHASE_ORDER` in `run_dream_comparison.py` (which arm this run is), not time. It is **constant within a run** and **should not** be used as an x-axis for learning curves. Use **`global_step`** / step index, or compare **separate runs** in a group.

## 2. `mean_w_time` and `mean_w_ent` are **raw** weights (before `alpha_*`)

For each transition, the loss uses:

\[
w = \alpha_{\text{time}} \cdot w_{\text{time}} + \alpha_{\text{entropy}} \cdot w_{\text{ent}}
\]

Logged **means**:

- **`mean_w_time`**: average of **`w_time`** = interval time weight from `TimeWeighter.get_interval_weight(parent_step, child_step)` (depends only on **tree topology** and `total_denoising_steps`, not on `alpha_entropy`).
- **`mean_w_ent`**: average of **`w_ent`** = entropy weight from `EntropyComputer.compute_entropy_weight`, then **clamped** to `[entropy_weight_min, entropy_weight_max]` (global `MCTSConfig` defaults **0.08–2.5**; `run_dream_comparison.py` / shell match unless overridden).

Changing **`alpha_entropy`** between `adaptive_default` and `adaptive_alt_hp` **does not change** `mean_w_time` or `mean_w_ent` — those are **pre-alpha** building blocks. What **should** change between those arms is **`mean_weight`** (average of \(w\)) and the **loss**.

So: **identical `mean_w_time` / `mean_w_ent` across hyperparams that only change `alpha_entropy` is expected** if the **tree** (same prompt, same caps) is the same.

## 2b. Private W&B projects

We cannot see your runs without logging in. Export a **CSV** or paste **scalar names + a few values** if you want help interpreting a specific chart.

## 3. Why `mean_w_ent` often looks constant (near the floor)

`entropy_weight` is computed from normalized entropy, then:

```text
ew = clamp(ew_raw, entropy_weight_min, entropy_weight_max)   # e.g. min=0.08, max=2.5
```

If many edges have **low** raw weight, **`ew_raw` is below `entropy_weight_min`** and the **clamp pins `w_ent` to the floor** on those edges. Averaging then yields **`mean_w_ent` stuck near `entropy_weight_min`** — tight floors **reduce variance** but can **flatten** learning signal when **`frac_entropy_clamped_low` is high** (e.g. >0.5).

**Implemented diagnostics** (logged every train step from `WeightedGRPOLoss`):

- **`mean_w_ent_raw`**: average **pre-clamp** \(H/\log V\) (or stage-aware) weight.
- **`frac_entropy_clamped_low`**: fraction of transitions with raw weight **below** `entropy_weight_min` (hitting the floor).
- **`frac_entropy_clamped_high`**: fraction hitting the ceiling.
- **`mean_edge_denoising_delta`**: mean `child_step_index - step_index` per edge — if **constant** across steps, **`mean_w_time`** often is too (same interval schedule on every edge).

The comparison script defaults to **`--entropy-weight-min 0.08`** (and **`max` 2.5**) so fewer transitions hit the floor; **raise** the floor slightly if gradients become unstable.

## 3a. Reward diagnostics (trainer)

Every step also logs:

- **`mean_reward`** (same value as **`avg_reward`**) — mean reward over tree leaves or baseline trajectories.
- **`min_reward`** / **`max_reward`** — spread across leaves/samples; large `max_reward - min_reward` with flat loss can indicate reward scale or advantage issues.

## 3b. “Cyclic” `avg_entropy` over training

**`avg_entropy`** (tree / node Shannon) can look **periodic or saw-tooth** even when the **loss weight** is flat, because:

1. **Diffusion / iterative unmasking** revisits similar masking densities across depth or chunks (entropy often tracks **how many positions are still masked**).
2. **Fixed tree policy** + **similar prompts** can repeat similar entropy **patterns** step-to-step.
3. That does **not** contradict clamped **`mean_w_ent`** being flat: the latter is a **normalized, clamped scalar** per edge for the loss, not a raw copy of node entropy.

Use **`mean_w_ent_raw`** vs **`mean_w_ent`** to see clamping vs signal.

## 4. Why `mean_weight` can match across “baseline” vs “adaptive” on the same step

If two phases build **different trees** (fixed-step vs adaptive), **`w_time` intervals differ** → **`mean_w_time`** and **`mean_weight`** should usually **differ** (your logs: baseline `mean_w_time≈6.86` vs adaptive `≈19.99`).

If two phases build the **same topology** for the same prompt (e.g. same random seed and same caps), **`mean_w_time` / `mean_w_ent` / `mean_weight`** can be **very close**.

## 5. `n_loss_forwards` and `epoch_mean_*` of it

With **`loss_group_by_parent=True`**, `n_loss_forwards` = number of **parent groups** (≤ `n_transitions`). For a **fixed tree shape** (same branching, same grouping), **`n_loss_forwards` is identical** step-to-step. So **`epoch_mean_n_loss_forwards` constant over epochs** is **normal** if every prompt yields the same tree size/shape.

## 6. `avg_entropy` (tree / node) vs `mean_w_ent` (loss weight)

- **`avg_entropy`** in trainer metrics: **Shannon entropy** (or related) from **tree nodes** — can **change over training** as logits change.
- **`mean_w_ent`**: **clamped, normalized weight** used in the loss — often **flat** near the clamp floor (see §3).

They measure **different things**; do not expect them to track each other one-to-one.

## 7. Short runs and W&B overlays

With **few steps** or **crashes after step 1**, many runs only show **one point**. **Grouped charts** then look like “only one setting works” — check **per-run step count** and **`.err` for OOM**.

## 8. What we log explicitly (recommended checks)

In W&B **config** (and per-step scalars where implemented):

- **`alpha_time`**, **`alpha_entropy`**, **`branch_threshold`**, **`adaptive_stepping`**
- **`entropy_weight_min` / `entropy_weight_max`** (clamps)

Compare **`mean_weight`** across runs when **`alpha_entropy`** changes; **`mean_w_ent`** alone will not show that ablation.
