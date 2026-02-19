# Entropy-Guided MCTS for Diffusion Language Model Training

**Model Target**: `dllm-collection/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1`  
**Objective**: Improve GRPO training efficiency and sample quality through entropy-guided tree search during training

---

## 1. Core Problem Statement

Current GRPO for diffusion LMs uses trajectory-level rewards with uniform credit assignment across denoising steps. This is inefficient because:

1. **Not all denoising steps are equally important** - early decisions shape the solution space more than late refinement
2. **Random sampling explores inefficiently** - especially for sparse rewards (code correctness, math problems)
3. **No mechanism to identify critical decision points** - uniform sampling treats all timesteps equally

**Solution**: Use token-level entropy to guide MCTS-style exploration during training, focusing computational budget on high-uncertainty decision points.

---

## 2. Exploration Philosophy: DeepSearch vs McDiffuSE

### DeepSearch Approach (Recommended ✓)

**Core Principle**: Breadth over depth, global frontier selection

**Key Mechanisms**:
```python
# 1. Global Frontier Selection
def select_nodes_to_expand(all_nodes_in_tree, budget_k):
    """Select top-k highest entropy nodes across ENTIRE tree"""
    sorted_by_entropy = sorted(all_nodes_in_tree, 
                               key=lambda n: n.entropy, 
                               reverse=True)
    return sorted_by_entropy[:budget_k]

# 2. Wide, shallow trees
max_depth_per_branch = 3  # Shallow
branches_per_iteration = 5  # Wide
```

**Why This Works**:
- **Avoids local optima**: By considering all nodes globally, doesn't get stuck following one promising path
- **Efficient exploration**: Wide search discovers diverse solutions quickly
- **Natural for diffusion**: Each denoising step is relatively independent, no need for deep sequential planning
- **Proven results**: DeepSearch achieved 62.95% on AIME (vs ~50% for deep search methods)

**Critical Insight from Paper**:
> "Larger exploration constants, rather than increased simulations, are necessary to overcome model confidence biases"

Translation: Breadth (many branches) beats depth (long simulations) in diffusion models because:
- Model already has strong priors from pretraining
- Need to escape these priors, not follow them deeper
- Entropy tells us WHERE the model is uncertain → branch there

### McDiffuSE Approach (Alternative)

**Core Principle**: Lookahead simulation for slot ordering

**Key Mechanisms**:
```python
# 1. Local UCB selection with model priors
def select_next_slot(current_state):
    """Use UCB to balance model confidence vs exploration"""
    for slot in available_slots:
        ucb_score = (
            Q_value[slot] +  # Exploitation
            C * sqrt(log(N_parent) / N_slot)  # Exploration
        )
    return argmax(ucb_score)

# 2. Rollout-based evaluation
def simulate_completion(partial_state):
    """Complete remaining slots, evaluate trajectory"""
    return rollout_reward(partial_state)
```

**Why This Might Be Suboptimal for Training**:
- **Designed for inference-time ordering** (which slot to fill next)
- **Requires rollouts** (expensive during training)
- **Locality bias**: UCB focuses on local parent-child relationships
- **Sequential assumption**: Assumes slot ordering matters sequentially

### Why DeepSearch is Better for Training

| Aspect | DeepSearch | McDiffuSE |
|--------|-----------|-----------|
| **Scope** | Global tree-wide selection | Local parent-child |
| **Depth** | Shallow & wide | Deeper rollouts |
| **Cost** | Lower (no rollouts) | Higher (needs simulations) |
| **Goal** | Discover diverse solutions | Optimize ordering |
| **Parallelization** | Natural (independent branches) | Sequential (UCB path-dependent) |
| **Training fit** | Direct (batch of diverse examples) | Indirect (ordering not main goal) |

**Your Intuition is Correct**: DeepSearch's global frontier selection is better because:
1. **Training wants diversity** (for good GRPO advantage estimation)
2. **Diffusion steps are parallel** (global selection respects this)
3. **Entropy is the signal** (not simulated rollouts)
4. **Computational efficiency** (no expensive rollouts during training)

---

## 3. Precise Algorithm Design

### 3.1 Entropy Computation (MDLM)

For discrete masked diffusion, we get **exact Shannon entropy**:

```python
def compute_token_entropy(model, z_t, t):
    """
    z_t: [batch, seq_len] - current masked sequence
    t: scalar - timestep (0 = fully masked, 1 = clean)
    
    Returns: [batch, seq_len] - per-token entropy
    """
    with torch.no_grad():
        logits = model(z_t, timestep=t)  # [batch, seq_len, vocab_size]
        probs = F.softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]
        
        # Shannon entropy per token
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        # [batch, seq_len]
        
    return entropy
```

**Key Properties**:
- High entropy (near log(vocab_size) ≈ 11 for 50k vocab) → model very uncertain
- Low entropy (near 0) → model confident in prediction
- **This is exact**, not approximated (advantage of discrete diffusion)

### 3.2 Tree Construction with Global Frontier Selection

```python
class MCTSNode:
    def __init__(self, state, timestep, parent=None):
        self.state = state              # [batch, seq_len] masked tokens
        self.timestep = timestep        # int: current denoising step
        self.parent = parent            # MCTSNode or None
        self.children = []              # List[MCTSNode]
        self.entropy = None             # float: node's entropy score
        self.reward = None              # float: final reward (leaves only)
        self.advantage = None           # float: backprop'd advantage
        
class EntropyGuidedMCTS:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        
        # DeepSearch-style configuration
        self.max_tree_nodes = config.max_nodes  # Total budget (20-50)
        self.branch_width = config.branch_width  # New branches per expansion (3-5)
        self.steps_per_expansion = config.steps_per_expansion  # How many denoising steps between expansions (32-64)
        
    def build_tree(self, prompt, total_steps=256):
        """
        Build exploration tree using global frontier selection
        
        Args:
            prompt: str - input prompt
            total_steps: int - total denoising steps
            
        Returns:
            root: MCTSNode - tree root with all leaves
        """
        # Initialize root
        root = MCTSNode(
            state=self.initialize_fully_masked(prompt),
            timestep=0,
            parent=None
        )
        
        all_nodes = [root]
        leaf_nodes = [root]  # Active nodes that can be expanded
        nodes_used = 1
        
        # Expansion loop
        while nodes_used < self.max_tree_nodes and leaf_nodes:
            # === GLOBAL FRONTIER SELECTION (DeepSearch) ===
            
            # 1. Compute entropy for all leaf nodes
            for node in leaf_nodes:
                if node.entropy is None:
                    token_entropy = self.compute_token_entropy(
                        node.state, 
                        node.timestep
                    )
                    # Aggregate to single score (mean across sequence)
                    node.entropy = token_entropy.mean().item()
            
            # 2. Select top-k highest entropy nodes globally
            leaf_nodes.sort(key=lambda n: n.entropy, reverse=True)
            nodes_to_expand = leaf_nodes[:self.branch_width]
            
            # 3. Expand selected nodes
            newly_created = []
            for node in nodes_to_expand:
                if nodes_used >= self.max_tree_nodes:
                    break
                
                # Create branches by sampling different noise patterns
                for _ in range(self.branch_width):
                    if nodes_used >= self.max_tree_nodes:
                        break
                    
                    # Denoise for multiple steps (not just 1)
                    child_state = self.denoise_steps(
                        node.state,
                        start_t=node.timestep,
                        num_steps=self.steps_per_expansion
                    )
                    
                    child = MCTSNode(
                        state=child_state,
                        timestep=node.timestep + self.steps_per_expansion,
                        parent=node
                    )
                    
                    node.children.append(child)
                    newly_created.append(child)
                    all_nodes.append(child)
                    nodes_used += 1
                
                # Remove expanded node from leaf set
                leaf_nodes.remove(node)
            
            # Add new nodes to leaf set
            leaf_nodes.extend(newly_created)
        
        # Complete all remaining leaves to final generation
        final_leaves = []
        for node in leaf_nodes:
            if node.timestep < total_steps:
                final_state = self.denoise_to_completion(
                    node.state,
                    start_t=node.timestep,
                    end_t=total_steps
                )
                # Create final leaf
                leaf = MCTSNode(
                    state=final_state,
                    timestep=total_steps,
                    parent=node
                )
                node.children.append(leaf)
                final_leaves.append(leaf)
            else:
                final_leaves.append(node)
        
        return root, final_leaves
    
    def denoise_steps(self, z_t, start_t, num_steps):
        """
        Perform multiple denoising steps
        
        Key: Use stochastic sampling (not greedy) to create diversity
        """
        current = z_t
        for step in range(num_steps):
            t = start_t + step
            
            # Get model predictions
            logits = self.model(current, timestep=t)
            
            # Sample stochastically (creates branches)
            probs = F.softmax(logits / temperature, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, probs.size(-1)), 
                num_samples=1
            ).view(probs.shape[:-1])
            
            # Update masked positions
            mask_positions = (current == self.tokenizer.mask_token_id)
            current = torch.where(mask_positions, sampled, current)
        
        return current
```

### 3.3 Key Design Decisions

**Q: Why expand multiple steps at once (steps_per_expansion)?**

A: **Coarse-grained branching** (from TreeGRPO, Fast-MCTD):
- Fine-grained (1 step/branch) creates massive trees
- Coarse-grained (32-64 steps/branch) keeps tree manageable
- Still captures critical decision points (high entropy regions)

**Q: Why sample stochastically instead of greedily?**

A: **Exploration requires diversity**:
- Greedy decoding → all branches converge to same solution
- Stochastic sampling → diverse trajectories
- Entropy tells us WHERE to branch, temperature controls HOW much diversity

**Q: Why mean entropy across sequence?**

A: **Tractability**:
- Per-token entropy is [batch, seq_len] dimensional
- Need single score for node selection
- Could also use: max, sum, or weighted by position
- Mean is simplest, can refine later

---

## 4. GRPO Training Integration

### 4.1 Weighted GRPO Objective

```python
def compute_weighted_grpo_loss(tree, root, leaves, prompts):
    """
    Compute GRPO loss with time + entropy weighting
    
    Args:
        tree: all nodes in tree
        root: MCTSNode - tree root
        leaves: List[MCTSNode] - final completions
        prompts: List[str] - input prompts
        
    Returns:
        loss: torch.Tensor - weighted GRPO loss
    """
    # 1. Get rewards for all leaves
    completions = [decode_tokens(leaf.state) for leaf in leaves]
    rewards = torch.tensor([
        reward_function(completion, prompt)
        for completion, prompt in zip(completions, prompts)
    ])
    
    # 2. Compute advantages (group normalization)
    advantages = rewards - rewards.mean()
    
    # 3. Assign leaf advantages
    for leaf, adv in zip(leaves, advantages):
        leaf.advantage = adv
    
    # 4. Backpropagate advantages through tree
    backprop_advantages(root)
    
    # 5. Collect all transitions for loss computation
    transitions = []
    for node in tree:
        if node.parent is not None:
            transitions.append({
                'parent_state': node.parent.state,
                'child_state': node.state,
                'timestep': node.parent.timestep,
                'advantage': node.advantage,
                'entropy': node.parent.entropy,
            })
    
    # 6. Compute weighted loss
    loss = 0
    for trans in transitions:
        # Get log probability of transition
        log_prob = compute_log_prob(
            model,
            trans['parent_state'],
            trans['child_state'],
            trans['timestep']
        )
        
        # Time weight (early steps matter more)
        time_weight = get_time_weight(trans['timestep'], total_steps=256)
        
        # Entropy weight (high uncertainty matters more)
        entropy_weight = trans['entropy'] / expected_entropy(trans['timestep'])
        
        # Combined weight
        combined_weight = alpha_time * time_weight + alpha_entropy * entropy_weight
        
        # Weighted GRPO term
        loss -= combined_weight * trans['advantage'] * log_prob
    
    return loss / len(transitions)

def backprop_advantages(node):
    """
    Backpropagate advantages from leaves to root
    
    Standard MCTS backup: node value = average of children
    """
    if not node.children:  # Leaf
        return node.advantage
    
    # Internal node: average children advantages
    child_advantages = [backprop_advantages(child) for child in node.children]
    node.advantage = sum(child_advantages) / len(child_advantages)
    return node.advantage

def get_time_weight(timestep, total_steps):
    """
    TempFlow-GRPO style: early steps get higher weight
    
    Intuition: early decisions shape solution space
    """
    # Inverse linear (could also be exponential)
    return 1.0 - (timestep / total_steps)

def expected_entropy(timestep, total_steps=256, vocab_size=50000):
    """
    Expected entropy decreases as denoising progresses
    
    t=0 (fully masked): high entropy ~ log(vocab_size)
    t=T (clean): low entropy ~ 0
    """
    masking_ratio = 1.0 - (timestep / total_steps)
    # Rough approximation: entropy proportional to masking
    return masking_ratio * np.log(vocab_size)
```

### 4.2 Hyperparameters

```python
config = {
    # Tree construction (DeepSearch style)
    'max_tree_nodes': 30,           # Total budget
    'branch_width': 3,              # Branches per expansion
    'steps_per_expansion': 32,      # Coarse-grained steps
    
    # Weighting
    'alpha_time': 1.0,              # Time weight coefficient
    'alpha_entropy': 0.5,           # Entropy weight coefficient
    
    # Sampling
    'temperature': 0.8,             # Stochastic diversity
    
    # Training
    'batch_size': 4,                # Prompts per batch
    'learning_rate': 1e-5,          # Conservative for RL
}
```

**Ablation priorities**:
1. `max_tree_nodes`: 20 vs 30 vs 50 (cost/benefit)
2. `branch_width`: 2 vs 3 vs 5 (breadth importance)
3. `alpha_entropy`: 0.0 vs 0.5 vs 1.0 (entropy weighting value)
4. `steps_per_expansion`: 16 vs 32 vs 64 (granularity)

---

## 5. Why DeepSearch Over McDiffuSE: Mathematical Intuition

### DeepSearch's Global Frontier

**Optimization Problem**:
$$\max_{S \subset \mathcal{T}, |S| = k} \sum_{n \in S} H(n)$$

where $\mathcal{T}$ is all nodes in tree, $H(n)$ is node entropy, $k$ is branch budget.

**Solution**: Greedily select top-k highest entropy nodes **globally**.

**Why this works for training**:
- **Objective alignment**: GRPO wants diverse high-quality samples
- **Efficiency**: One global sort, no repeated evaluations
- **Exploration**: Naturally finds multiple distinct solution modes

### McDiffuSE's Local UCB

**Optimization Problem** (per node):
$$\arg\max_{a \in \text{actions}} \left[ Q(s,a) + C \sqrt{\frac{\log N(s)}{N(s,a)}} \right]$$

**Why this is suboptimal for training**:
- **Local scope**: Only considers parent-child relationship
- **Sequential bias**: Assumes path-dependent value
- **Computational cost**: Requires maintaining visit counts, Q-values
- **Exploration inefficiency**: UCB converges to exploitation over time

### Concrete Example

**Scenario**: 3 high-entropy regions in sequence

```
Position: [10,  50,  90]  (in sequence length 128)
Entropy:  [9.5, 9.2, 9.4]
```

**DeepSearch**:
- Sees all three globally
- Creates 3 parallel branches, one from each position
- Result: Explores all three simultaneously

**McDiffuSE**:
- Starts at position 0, uses UCB
- Likely follows first path (position 10) deeply
- May never discover positions 50, 90 if path 10 seems promising early
- Result: Less diverse exploration

**For GRPO Training**:
- Want: Multiple diverse solutions (for good advantage estimates)
- DeepSearch: ✓ Parallel exploration finds all modes
- McDiffuSE: ✗ Sequential search may miss modes

---

## 6. Implementation Roadmap

### Phase 1: Entropy Extraction & Validation (Week 1)
```python
# Validate entropy computation
model = load_model("dllm-collection/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1")
prompt = "def fibonacci(n):"
z_t = initialize_masked(prompt)

for t in [0, 64, 128, 192, 256]:
    entropy = compute_token_entropy(model, z_t, t)
    print(f"Step {t}: mean entropy = {entropy.mean():.2f}")
    # Expect: decreasing entropy as t increases
```

### Phase 2: Tree Construction (Week 2)
```python
# Build tree, visualize structure
tree, leaves = build_tree(prompt, total_steps=256)
print(f"Tree has {len(tree)} nodes, {len(leaves)} leaves")

# Verify global frontier selection
entropies = [n.entropy for n in tree if n.entropy is not None]
print(f"Entropy range: {min(entropies):.2f} - {max(entropies):.2f}")
```

### Phase 3: GRPO Integration (Week 3)
```python
# Train with baseline GRPO (no tree)
baseline_model = train_standard_grpo(prompts, num_epochs=1)

# Train with entropy-guided MCTS
mcts_model = train_with_entropy_mcts(prompts, num_epochs=1)

# Compare: sample efficiency, final performance
```

### Phase 4: Evaluation & Ablation (Week 4)
- Metrics: pass@k on HumanEval, MBPP
- Ablations: tree budget, entropy weighting, branching strategy
- Analysis: Which timesteps get branched? Does it correlate with reasoning?

---

## 7. Open Research Questions

### Q1: Optimal Tree Budget
- **Tradeoff**: More nodes = better exploration, higher cost
- **Hypothesis**: 30-50 nodes sufficient for 0.5B model
- **Experiment**: Ablate 10/20/30/50/100 nodes, measure cost vs. reward

### Q2: Entropy Aggregation
- **Current**: Mean entropy across sequence
- **Alternatives**: Max, sum, position-weighted, attention-weighted
- **Hypothesis**: Max might be better (focus on hardest tokens)
- **Experiment**: Compare aggregation methods on code completion

### Q3: Time vs. Entropy Weight Balance
- **Current**: $\alpha_{\text{time}} = 1.0, \alpha_{\text{entropy}} = 0.5$
- **Question**: Should entropy dominate? Or time?
- **Hypothesis**: Early training = more time weight (learn basics), late = more entropy (refine hard cases)
- **Experiment**: Adaptive scheduling of $\alpha$ values

### Q4: When Does Tree Search Help Most?
- **Hypothesis**: Sparse rewards (code correctness) benefit more than dense (fluency)
- **Experiment**: Compare tree vs. no-tree on multiple tasks
  - Code: correctness (sparse)
  - Math: answer accuracy (sparse)
  - Text: fluency (dense)

---

## 8. Expected Outcomes

**Success Criteria**:
1. **Efficiency**: 20-30% fewer training samples to reach same performance
2. **Quality**: +5-10% absolute on pass@1 metrics (HumanEval, MBPP)
3. **Interpretability**: Can visualize which denoising steps get explored (reasoning insights)

**Risks**:
1. **Overhead**: Tree construction might be too slow
2. **Instability**: RL with tree advantages might be high variance
3. **Hyperparameter sensitivity**: Might require careful tuning

**Mitigation**:
1. Start with small tree (20 nodes), scale up if successful
2. Use advantage clipping, gradient clipping (standard RL stability tricks)
3. Comprehensive ablation study, share negative results openly

