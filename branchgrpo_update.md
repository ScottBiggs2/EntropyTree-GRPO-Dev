# BranchGRPO Integration Notes

## Critical Paper Addition

### BranchGRPO: Stable and Efficient GRPO with Structured Branching
**Paper**: [BranchGRPO: Stable and Efficient GRPO with Structured Branching in Diffusion Models](https://arxiv.org/abs/2509.06040)  
**Authors**: Li et al. (Sept 2025)  
**OpenReview**: https://openreview.net/forum?id=T2nP2IQasd

**Key Contributions**:
- Tree-structured rollouts with path-weighted reward fusion (propagates leaf rewards upward using path probabilities)
- Depth-wise advantage normalization (normalizes advantages separately at each tree depth for stability)
- Achieves 16% better alignment and 55% faster training on image/video diffusion

**Relevance**: Provides critical stability improvements (depth normalization, reward fusion) that we should adopt. Our entropy-guided exploration remains novel and complementary.

---

## Impact on Our Design

### Core Scaffold: ✅ Unchanged
The main implementation structure remains the same:
- Entropy computation
- Tree construction with global frontier selection (DeepSearch)
- Entropy-guided branching decisions
- Time + entropy weighted loss

### Required Updates: 🔧 Two Key Additions

#### 1. Depth-Wise Advantage Normalization (Critical for Stability)

**Problem**: Without depth normalization, early high-variance steps dominate gradients.

**Solution**: Normalize advantages separately at each tree depth.

**Add to `WeightedGRPOLoss._backprop_advantages()`**:

```python
def _backprop_advantages_with_depth_norm(self, root: MCTSNode):
    """
    Backprop with BranchGRPO-style depth-wise normalization
    
    Changes from original:
    1. First pass: compute raw fused rewards (path-weighted)
    2. Second pass: normalize by depth
    3. Assign normalized values as advantages
    """
    # Step 1: Fuse rewards upward (path-weighted)
    self._fuse_rewards_path_weighted(root)
    
    # Step 2: Collect all nodes by depth
    nodes_by_depth = {}
    self._collect_by_depth(root, nodes_by_depth)
    
    # Step 3: Normalize within each depth
    for depth, nodes in nodes_by_depth.items():
        rewards = [n.fused_reward for n in nodes]
        mean = np.mean(rewards)
        std = np.std(rewards) + 1e-8  # Epsilon for stability
        
        # Assign normalized advantages
        for node in nodes:
            node.advantage = (node.fused_reward - mean) / std
```

**Math**:
$$A_d(n) = \frac{\bar{r}(n) - \mu_d}{\sigma_d + \epsilon}$$

where $\mu_d = \text{mean}_{n \in N_d} \bar{r}(n)$ and $\sigma_d = \text{std}_{n \in N_d} \bar{r}(n)$, computed over all nodes $N_d$ at depth $d$.

#### 2. Path-Weighted Reward Fusion (More Principled Than Averaging)

**Problem**: Simple averaging treats all paths equally, ignoring sampling probabilities.

**Solution**: Weight leaf rewards by path probability.

**Add helper methods to `WeightedGRPOLoss`**:

```python
def _fuse_rewards_path_weighted(self, node: MCTSNode):
    """
    Fuse leaf rewards with path probability weighting (BranchGRPO)
    
    Math:
        r̄(n) = Σ_ℓ P(path(n→ℓ)) · R(ℓ)
    
    where sum is over all leaf descendants ℓ of node n
    """
    if node.is_leaf:
        node.fused_reward = node.reward  # Terminal reward
        return node.fused_reward
    
    # Recurse to ensure children are fused first
    for child in node.children:
        self._fuse_rewards_path_weighted(child)
    
    # Fuse from children (weighted by their sampling probability)
    fused = 0.0
    for child in node.children:
        # Path probability from this node to child
        # (stored during stochastic sampling in tree construction)
        path_prob = child.sampling_prob if hasattr(child, 'sampling_prob') else 1.0 / len(node.children)
        fused += path_prob * child.fused_reward
    
    node.fused_reward = fused
    return fused

def _collect_by_depth(self, node: MCTSNode, depth_dict: dict, current_depth: int = 0):
    """Collect all nodes organized by depth"""
    if current_depth not in depth_dict:
        depth_dict[current_depth] = []
    depth_dict[current_depth].append(node)
    
    for child in node.children:
        self._collect_by_depth(child, depth_dict, current_depth + 1)
```

**Implementation Detail**: During tree construction in `EntropyGuidedTreeBuilder._expand_node()`, store sampling probability:

```python
# In _expand_node() when creating children
for _ in range(self.config.branch_width):
    child_state = self._denoise_steps(...)
    child = MCTSNode(
        state=child_state,
        timestep=node.timestep + self.config.steps_per_expansion,
        parent=node,
        children=[]
    )
    
    # NEW: Store sampling probability for reward fusion
    child.sampling_prob = 1.0 / self.config.branch_width  # Uniform for now
    
    children.append(child)
```

---

## What Stays the Same

✅ **Entropy computation** - MDLM gives us exact Shannon entropy (they don't have this for continuous diffusion)  
✅ **Global frontier selection** - DeepSearch-style adaptive branching (they use fixed scheduled branches)  
✅ **Entropy weighting in loss** - Our contribution, orthogonal to BranchGRPO  
✅ **Time weighting** - TempFlow-GRPO approach, compatible with depth normalization  
✅ **Tree construction logic** - Entropy-guided > scheduled (better for discrete diffusion)

---

## Optional Enhancements (Phase 2)

### Width Pruning (Efficiency)
Only backprop through top-k highest-reward leaves:

```python
def _prune_leaves_by_reward(self, leaves: List[MCTSNode], keep_fraction: float = 0.7):
    """Keep only top fraction of leaves by reward"""
    k = max(1, int(len(leaves) * keep_fraction))
    return sorted(leaves, key=lambda l: l.reward, reverse=True)[:k]
```

### Depth Pruning (Efficiency)
Skip backprop at late depths where gradients are small:

```python
# In loss computation, skip certain depths
skip_depths = set(range(230, 256))  # Last ~25 steps
for trans in transitions:
    if trans.timestep in skip_depths:
        continue  # Skip gradient computation
    # ... compute loss
```

---

## Implementation Checklist Updates

### Phase 3: Loss Computation (Updated)
- [ ] Implement `_fuse_rewards_path_weighted()` (BranchGRPO addition)
- [ ] Implement `_collect_by_depth()` helper
- [ ] Replace `_backprop_advantages()` with `_backprop_advantages_with_depth_norm()`
- [ ] Store `sampling_prob` during tree construction
- [ ] Test on synthetic tree: verify depth normalization stabilizes gradients
- [ ] ~~Implement original simple averaging backprop~~ (replaced)

### Phase 5: Evaluation (Updated)
- [ ] Compare baseline GRPO vs entropy-guided MCTS
- [ ] **NEW**: Ablation on depth normalization (with/without)
- [ ] **NEW**: Ablation on reward fusion (path-weighted vs simple averaging)
- [ ] Ablation studies (alpha_time, alpha_entropy, tree budget)
- [ ] Visualize which timesteps get explored most

---

## Key Differences: BranchGRPO vs Our Approach

| Feature | BranchGRPO | Our Design |
|---------|------------|------------|
| **Domain** | Continuous (images/video) | Discrete (text/code) |
| **Branching Strategy** | Scheduled timesteps (fixed) | Entropy-guided (adaptive) |
| **Branch Selection** | All nodes at schedule | Global frontier (top-k entropy) |
| **Advantage Normalization** | Depth-wise ✓ | Depth-wise ✓ (adopted) |
| **Reward Fusion** | Path-weighted ✓ | Path-weighted ✓ (adopted) |
| **Entropy Weighting** | N/A (continuous) | Yes ✓ (our contribution) |
| **Time Weighting** | Not explicit | Yes ✓ (TempFlow-GRPO) |
| **Pruning** | Width + depth | Can add (Phase 2) |

---

## Why Our Approach Remains Distinct

1. **Entropy-Guided Branching**: Adaptive selection based on model uncertainty (BranchGRPO uses fixed schedules)
2. **Discrete Diffusion Focus**: Text/code domain with exact entropy (they focus on continuous image/video)
3. **Global Frontier Selection**: DeepSearch-style prioritization across entire tree (they branch all nodes at scheduled steps)
4. **Combined Weighting**: Time + entropy weighting in loss (they don't explicitly weight by time or entropy)

**Bottom Line**: BranchGRPO validates tree-based GRPO and provides critical stability techniques (depth normalization, reward fusion) that strengthen our approach. Our entropy-guided exploration for discrete diffusion remains a novel contribution.

---

## References

- **BranchGRPO**: https://arxiv.org/abs/2509.06040 (Li et al., Sept 2025)
- **TreeGRPO**: https://openreview.net/forum?id=3rZdp4TmUb (Ding & Ye, ICLR 2026)
- **DeepSearch**: https://arxiv.org/abs/2509.25454 (Global frontier selection)
- **TempFlow-GRPO**: https://arxiv.org/abs/2508.04324 (Time weighting)

