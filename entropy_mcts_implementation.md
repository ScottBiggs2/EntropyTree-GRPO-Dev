# Implementation Guide: Entropy-Guided MCTS-GRPO for dLLMs

**Target Model**: `dllm-collection/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1`  
**Framework**: PyTorch + dLLM library  
**Goal**: Training-time tree search with entropy-guided exploration for improved GRPO

---

## Mathematical Foundation

### 1. Entropy Computation (Exact for Discrete MDLM)

For a masked diffusion model at timestep $t$ with state $z_t \in \{0,1,\ldots,V\}^L$:

**Per-token Shannon entropy**:
$$H_i(z_t, t) = -\sum_{v=1}^{V} p_\theta(x_i = v | z_t, t) \log p_\theta(x_i = v | z_t, t)$$

where:
- $i$ indexes position in sequence ($1 \leq i \leq L$)
- $V$ is vocabulary size (≈50,000 for Qwen tokenizer)
- $p_\theta(x_i = v | z_t, t)$ is model's predicted probability for token $v$ at position $i$

**Aggregate entropy for node selection**:
$$H(z_t, t) = \frac{1}{L} \sum_{i=1}^{L} H_i(z_t, t)$$

**Properties**:
- $H_i(z_t, t) \in [0, \log V]$
- High entropy ($H \approx \log V \approx 10.8$): model is uncertain
- Low entropy ($H \approx 0$): model is confident
- Typically: $H(z_t=0) > H(z_t=128) > H(z_t=256)$ (decreases with denoising)

### 2. GRPO Loss (Base Formulation)

Standard Group Relative Policy Optimization for trajectory $\tau = (z_0, z_1, \ldots, z_T)$:

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}_{\tau \sim \pi_\theta} \left[ A(\tau) \cdot \log \pi_\theta(\tau) \right]$$

where:
- $A(\tau) = R(\tau) - \frac{1}{K}\sum_{k=1}^{K} R(\tau_k)$ is the advantage (group-normalized reward)
- $R(\tau)$ is terminal reward (e.g., code correctness, test pass rate)
- $K$ is group size (number of completions per prompt)

**Expanded per-transition**:
$$\mathcal{L}_{\text{GRPO}} = -\sum_{t=1}^{T} A(\tau) \cdot \log p_\theta(z_t | z_{t-1}, t)$$

### 3. Time-Weighted GRPO (TempFlow-GRPO)

Add temporal weighting to emphasize important timesteps:

$$\mathcal{L}_{\text{Time}} = -\sum_{t=1}^{T} w_{\text{time}}(t) \cdot A(\tau) \cdot \log p_\theta(z_t | z_{t-1}, t)$$

**Time weight definition**:
$$w_{\text{time}}(t) = \frac{(1 - t/T)^2}{\sum_{t'=1}^{T} (1 - t'/T)^2}$$

**Intuition**: Early timesteps (small $t$) have high weight, late timesteps (large $t$) have low weight.

**Numerical example** (T=256):
- $t=0$: $w_{\text{time}} \approx 0.0117$ (highest)
- $t=128$: $w_{\text{time}} \approx 0.0029$ (medium)
- $t=256$: $w_{\text{time}} \approx 0$ (lowest)

### 4. Entropy-Weighted GRPO (Our Contribution)

Add entropy weighting to emphasize uncertain regions:

$$\mathcal{L}_{\text{Entropy}} = -\sum_{t=1}^{T} w_{\text{ent}}(t) \cdot A(\tau) \cdot \log p_\theta(z_t | z_{t-1}, t)$$

**Entropy weight definition**:
$$w_{\text{ent}}(t) = \frac{H(z_t, t)}{\bar{H}(t)}$$

where $\bar{H}(t)$ is expected entropy at timestep $t$:
$$\bar{H}(t) = \left(1 - \frac{t}{T}\right) \cdot \log V$$

**Intuition**: Normalize entropy by what's expected at this timestep. If model is MORE uncertain than expected → higher weight.

**Numerical example** (T=256, V=50000):
- $t=0$: $\bar{H}(0) = 10.82$ (fully masked, maximum entropy)
- $t=128$: $\bar{H}(128) = 5.41$ (half denoised)
- $t=256$: $\bar{H}(256) = 0$ (fully denoised)

If measured $H(z_{128}, 128) = 7.5$ (higher than expected 5.41):
$$w_{\text{ent}}(128) = \frac{7.5}{5.41} = 1.39$$
→ This timestep gets 39% more weight because model is unusually uncertain

### 5. Tree-Based Advantages (MCTS Backpropagation)

Instead of trajectory-level advantage $A(\tau)$, compute per-node advantages via backpropagation:

**Leaf nodes** (final completions):
$$A_{\text{leaf}} = R(\text{leaf}) - \frac{1}{K}\sum_{k=1}^{K} R(\text{leaf}_k)$$

**Internal nodes** (partial completions):
$$A_{\text{internal}} = \frac{1}{|\text{children}|} \sum_{c \in \text{children}} A_c$$

**Root node**:
$$A_{\text{root}} = \frac{1}{|\text{leaves}|} \sum_{\ell \in \text{leaves}} A_\ell$$

**Intuition**: A node's value is the average value of its children. High-value nodes contributed to good completions; low-value nodes led to bad completions.

### 6. Combined Loss (Full Formulation)

$$\mathcal{L}_{\text{Full}} = -\sum_{n \in \mathcal{N}} \left[ \alpha_{\text{time}} \cdot w_{\text{time}}(t_n) + \alpha_{\text{ent}} \cdot w_{\text{ent}}(t_n) \right] \cdot A_n \cdot \log p_\theta(n | \text{parent}(n), t_n)$$

where:
- $\mathcal{N}$ is the set of all nodes in the tree (excluding root)
- $t_n$ is the timestep of node $n$
- $A_n$ is the backpropagated advantage of node $n$
- $\alpha_{\text{time}}, \alpha_{\text{ent}}$ are hyperparameters balancing the two weight types

**Normalization**: Both weight terms are normalized to sum to 1 across timesteps, so $\alpha$ values directly control relative importance.

**Default values**:
- $\alpha_{\text{time}} = 1.0$ (full time weighting)
- $\alpha_{\text{ent}} = 0.5$ (half entropy weighting)

**Interpretation**:
- $\alpha_{\text{ent}} = 0$: Pure time weighting (TempFlow-GRPO)
- $\alpha_{\text{time}} = 0$: Pure entropy weighting (our contribution)
- Both > 0: Hybrid that considers both temporal structure and uncertainty

### 7. Log Probability Computation (MDLM Specific)

For transition from $z_{t-1}$ to $z_t$:

$$\log p_\theta(z_t | z_{t-1}, t) = \sum_{i=1}^{L} \mathbb{1}[z_t^{(i)} \neq z_{t-1}^{(i)}] \cdot \log p_\theta(x_i = z_t^{(i)} | z_{t-1}, t)$$

where:
- $z_t^{(i)}$ is token at position $i$ in state $z_t$
- $\mathbb{1}[\cdot]$ is indicator function (1 if token changed, 0 otherwise)
- Only changed tokens contribute to log probability

**Implementation note**: For MDLM, only positions that were MASK in $z_{t-1}$ and became unmasked in $z_t$ contribute to the loss.

---

## Implementation Skeleton

### Core Data Structures

```python
from dataclasses import dataclass
from typing import List, Optional, Dict
import torch
import torch.nn.functional as F
import numpy as np

@dataclass
class MCTSNode:
    """
    Node in the MCTS tree representing a partial denoising trajectory
    """
    # Core state
    state: torch.Tensor              # [seq_len] - current token sequence with masks
    timestep: int                    # 0 to total_steps
    parent: Optional['MCTSNode']     # None for root
    children: List['MCTSNode']       # Empty for leaves
    
    # Computed values
    entropy: Optional[float] = None  # Aggregate entropy H(z_t, t)
    token_entropy: Optional[torch.Tensor] = None  # [seq_len] per-token entropy
    reward: Optional[float] = None   # Terminal reward (leaves only)
    advantage: Optional[float] = None  # Backpropagated advantage
    
    # Metadata
    is_leaf: bool = False
    is_completed: bool = False
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

@dataclass
class TreeTransition:
    """
    Represents a parent→child transition in the tree for loss computation
    """
    parent_state: torch.Tensor
    child_state: torch.Tensor
    timestep: int
    advantage: float
    entropy: float
    time_weight: float
    entropy_weight: float
```

### Configuration

```python
@dataclass
class MCTSConfig:
    """Hyperparameters for entropy-guided MCTS"""
    
    # Tree construction (DeepSearch style)
    max_tree_nodes: int = 30         # Total node budget
    branch_width: int = 3            # Branches created per expansion
    steps_per_expansion: int = 32    # Denoising steps between expansions
    
    # Sampling
    temperature: float = 0.8         # Stochastic sampling temperature
    
    # Loss weighting
    alpha_time: float = 1.0          # Time weight coefficient
    alpha_entropy: float = 0.5       # Entropy weight coefficient
    
    # Model specifics
    total_steps: int = 256           # Total denoising steps
    vocab_size: int = 50000          # Tokenizer vocabulary size
    
    # Training
    batch_size: int = 4              # Prompts per training batch
    learning_rate: float = 1e-5      # Conservative for RL
    max_grad_norm: float = 1.0       # Gradient clipping
```

### Entropy Computation Module

```python
class EntropyComputer:
    """
    Handles all entropy-related computations for MDLM
    """
    def __init__(self, config: MCTSConfig):
        self.config = config
    
    def compute_token_entropy(
        self, 
        model: torch.nn.Module,
        z_t: torch.Tensor,
        timestep: int
    ) -> torch.Tensor:
        """
        Compute exact Shannon entropy per token
        
        Args:
            model: MDLM model
            z_t: [batch, seq_len] current masked state
            timestep: int, current timestep
            
        Returns:
            entropy: [batch, seq_len] per-token entropy
            
        Math:
            H_i = -sum_v p(v) log p(v)
        """
        with torch.no_grad():
            # Get model predictions
            logits = model(z_t, timestep=timestep)  # [batch, seq_len, vocab_size]
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]
            
            # Compute Shannon entropy (vectorized)
            # Add small epsilon for numerical stability
            log_probs = torch.log(probs + 1e-10)  # [batch, seq_len, vocab_size]
            entropy = -(probs * log_probs).sum(dim=-1)  # [batch, seq_len]
            
        return entropy
    
    def aggregate_entropy(
        self,
        token_entropy: torch.Tensor,
        method: str = 'mean'
    ) -> float:
        """
        Aggregate per-token entropy to single score
        
        Args:
            token_entropy: [batch, seq_len] or [seq_len]
            method: 'mean', 'max', 'sum'
            
        Returns:
            float: aggregated entropy score
        """
        if method == 'mean':
            return token_entropy.mean().item()
        elif method == 'max':
            return token_entropy.max().item()
        elif method == 'sum':
            return token_entropy.sum().item()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def expected_entropy(self, timestep: int) -> float:
        """
        Expected entropy at given timestep for normalization
        
        Args:
            timestep: int, current timestep (0 to total_steps)
            
        Returns:
            float: expected entropy H_bar(t)
            
        Math:
            H_bar(t) = (1 - t/T) * log(V)
        """
        masking_ratio = 1.0 - (timestep / self.config.total_steps)
        max_entropy = np.log(self.config.vocab_size)
        return masking_ratio * max_entropy
    
    def compute_entropy_weight(
        self,
        measured_entropy: float,
        timestep: int
    ) -> float:
        """
        Compute normalized entropy weight
        
        Args:
            measured_entropy: float, actual measured H(z_t, t)
            timestep: int
            
        Returns:
            float: w_ent(t) = H(z_t, t) / H_bar(t)
            
        Math:
            w_ent(t) = H(z_t, t) / H_bar(t)
        """
        expected = self.expected_entropy(timestep)
        
        # Avoid division by zero at final timestep
        if expected < 1e-6:
            return 0.0
        
        return measured_entropy / expected
```

### Time Weighting Module

```python
class TimeWeighter:
    """
    Handles time-based weighting (TempFlow-GRPO style)
    """
    def __init__(self, config: MCTSConfig):
        self.config = config
        self._precompute_weights()
    
    def _precompute_weights(self):
        """
        Precompute and normalize time weights for all timesteps
        
        Math:
            w_time(t) = (1 - t/T)^2
            normalized so sum_t w_time(t) = 1
        """
        T = self.config.total_steps
        timesteps = np.arange(T)
        
        # Quadratic decay: (1 - t/T)^2
        weights = (1.0 - timesteps / T) ** 2
        
        # Normalize to sum to 1
        weights = weights / weights.sum()
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
    
    def get_weight(self, timestep: int) -> float:
        """
        Get precomputed time weight for timestep
        
        Args:
            timestep: int (0 to total_steps-1)
            
        Returns:
            float: normalized time weight
        """
        if timestep >= self.config.total_steps:
            return 0.0
        return self.weights[timestep].item()
```

### Tree Builder (DeepSearch Style)

```python
class EntropyGuidedTreeBuilder:
    """
    Constructs MCTS tree using global frontier selection
    """
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        config: MCTSConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.entropy_computer = EntropyComputer(config)
    
    def build_tree(
        self,
        prompt: str,
        device: str = 'cuda'
    ) -> tuple[MCTSNode, List[MCTSNode]]:
        """
        Build exploration tree using global frontier selection
        
        Args:
            prompt: str, input prompt
            device: str, torch device
            
        Returns:
            root: MCTSNode, tree root
            leaves: List[MCTSNode], final completions
            
        Algorithm:
            1. Initialize root with fully masked state
            2. While budget remains:
                a. Compute entropy for all leaf nodes
                b. Select top-k highest entropy globally (DeepSearch)
                c. Expand selected nodes with stochastic sampling
                d. Update leaf set
            3. Complete remaining leaves to final generation
        """
        # Initialize root
        root = self._initialize_root(prompt, device)
        
        all_nodes = [root]
        leaf_nodes = [root]
        nodes_used = 1
        
        # Main expansion loop
        while nodes_used < self.config.max_tree_nodes and leaf_nodes:
            
            # === GLOBAL FRONTIER SELECTION ===
            
            # 1. Compute entropy for all leaves
            for node in leaf_nodes:
                if node.entropy is None:
                    self._compute_node_entropy(node)
            
            # 2. Sort by entropy globally (DeepSearch key insight)
            leaf_nodes.sort(key=lambda n: n.entropy, reverse=True)
            
            # 3. Select top-k nodes to expand
            k = min(self.config.branch_width, len(leaf_nodes))
            nodes_to_expand = leaf_nodes[:k]
            
            # 4. Expand selected nodes
            newly_created = []
            for node in nodes_to_expand:
                # Check budget
                if nodes_used >= self.config.max_tree_nodes:
                    break
                
                # Create branches
                children = self._expand_node(node, device)
                
                for child in children:
                    if nodes_used >= self.config.max_tree_nodes:
                        break
                    
                    node.children.append(child)
                    newly_created.append(child)
                    all_nodes.append(child)
                    nodes_used += 1
                
                # Remove from leaf set
                leaf_nodes.remove(node)
            
            # Add new nodes to leaf set
            leaf_nodes.extend(newly_created)
        
        # Complete all leaves to final generation
        final_leaves = self._complete_leaves(leaf_nodes, device)
        
        return root, final_leaves
    
    def _initialize_root(self, prompt: str, device: str) -> MCTSNode:
        """
        Create root node with fully masked state
        
        Args:
            prompt: str
            device: str
            
        Returns:
            MCTSNode with t=0, fully masked
        """
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(device)
        
        # Create fully masked sequence
        # In MDLM, this means replacing all tokens with MASK token
        mask_token_id = self.tokenizer.mask_token_id
        fully_masked = torch.full_like(
            inputs.input_ids,
            fill_value=mask_token_id
        )
        
        root = MCTSNode(
            state=fully_masked[0],  # [seq_len]
            timestep=0,
            parent=None,
            children=[]
        )
        
        return root
    
    def _compute_node_entropy(self, node: MCTSNode):
        """
        Compute and store entropy for a node
        
        Args:
            node: MCTSNode to compute entropy for
            
        Side effects:
            Sets node.entropy and node.token_entropy
        """
        # Compute per-token entropy
        token_entropy = self.entropy_computer.compute_token_entropy(
            self.model,
            node.state.unsqueeze(0),  # Add batch dim
            node.timestep
        )
        
        # Store per-token entropy
        node.token_entropy = token_entropy[0]  # Remove batch dim
        
        # Aggregate to single score
        node.entropy = self.entropy_computer.aggregate_entropy(
            token_entropy,
            method='mean'
        )
    
    def _expand_node(
        self,
        node: MCTSNode,
        device: str
    ) -> List[MCTSNode]:
        """
        Create child nodes by stochastic denoising
        
        Args:
            node: MCTSNode to expand
            device: str
            
        Returns:
            List[MCTSNode]: child nodes
            
        Implementation:
            1. Run model for steps_per_expansion denoising steps
            2. Use stochastic sampling (temperature > 0) for diversity
            3. Create branch_width different samples
        """
        children = []
        
        for _ in range(self.config.branch_width):
            # Stochastic denoising for multiple steps
            child_state = self._denoise_steps(
                z_t=node.state,
                start_t=node.timestep,
                num_steps=self.config.steps_per_expansion,
                temperature=self.config.temperature,
                device=device
            )
            
            child = MCTSNode(
                state=child_state,
                timestep=node.timestep + self.config.steps_per_expansion,
                parent=node,
                children=[]
            )
            
            children.append(child)
        
        return children
    
    def _denoise_steps(
        self,
        z_t: torch.Tensor,
        start_t: int,
        num_steps: int,
        temperature: float,
        device: str
    ) -> torch.Tensor:
        """
        Perform multiple stochastic denoising steps
        
        Args:
            z_t: [seq_len] current state
            start_t: int, starting timestep
            num_steps: int, number of steps to denoise
            temperature: float, sampling temperature
            device: str
            
        Returns:
            [seq_len] denoised state
            
        Math:
            For each masked position:
                1. Get logits from model
                2. Sample from softmax(logits / temperature)
                3. Update masked positions with samples
        """
        current = z_t.clone()
        mask_token_id = self.tokenizer.mask_token_id
        
        with torch.no_grad():
            for step in range(num_steps):
                t = start_t + step
                
                # Get model predictions
                logits = self.model(
                    current.unsqueeze(0),  # Add batch dim
                    timestep=t
                )[0]  # Remove batch dim: [seq_len, vocab_size]
                
                # Apply temperature
                logits = logits / temperature
                
                # Get probabilities
                probs = F.softmax(logits, dim=-1)  # [seq_len, vocab_size]
                
                # Sample from distribution
                sampled = torch.multinomial(
                    probs,
                    num_samples=1
                ).squeeze(-1)  # [seq_len]
                
                # Update only masked positions
                mask_positions = (current == mask_token_id)
                current = torch.where(mask_positions, sampled, current)
        
        return current
    
    def _complete_leaves(
        self,
        leaf_nodes: List[MCTSNode],
        device: str
    ) -> List[MCTSNode]:
        """
        Complete all leaf nodes to final generation
        
        Args:
            leaf_nodes: List[MCTSNode] to complete
            device: str
            
        Returns:
            List[MCTSNode]: completed final leaves
        """
        final_leaves = []
        
        for node in leaf_nodes:
            if node.timestep < self.config.total_steps:
                # Denoise to completion
                final_state = self._denoise_steps(
                    z_t=node.state,
                    start_t=node.timestep,
                    num_steps=self.config.total_steps - node.timestep,
                    temperature=self.config.temperature,
                    device=device
                )
                
                # Create final leaf
                leaf = MCTSNode(
                    state=final_state,
                    timestep=self.config.total_steps,
                    parent=node,
                    children=[],
                    is_leaf=True,
                    is_completed=True
                )
                node.children.append(leaf)
                final_leaves.append(leaf)
            else:
                node.is_leaf = True
                node.is_completed = True
                final_leaves.append(node)
        
        return final_leaves
```

### GRPO Loss Computation

```python
class WeightedGRPOLoss:
    """
    Computes weighted GRPO loss with time + entropy weighting
    """
    def __init__(
        self,
        config: MCTSConfig,
        reward_function
    ):
        self.config = config
        self.reward_function = reward_function
        self.entropy_computer = EntropyComputer(config)
        self.time_weighter = TimeWeighter(config)
    
    def compute_loss(
        self,
        model: torch.nn.Module,
        root: MCTSNode,
        leaves: List[MCTSNode],
        prompts: List[str],
        tokenizer
    ) -> torch.Tensor:
        """
        Compute weighted GRPO loss from tree
        
        Args:
            model: MDLM model
            root: MCTSNode, tree root
            leaves: List[MCTSNode], final completions
            prompts: List[str], input prompts
            tokenizer: tokenizer
            
        Returns:
            loss: torch.Tensor, scalar loss
            
        Algorithm:
            1. Compute rewards for all leaves
            2. Compute advantages (group normalization)
            3. Backpropagate advantages through tree
            4. Collect all transitions
            5. Compute weighted loss for each transition
        """
        # Step 1: Compute rewards
        rewards = self._compute_rewards(leaves, prompts, tokenizer)
        
        # Step 2: Compute advantages
        advantages = rewards - rewards.mean()
        
        # Step 3: Assign advantages to leaves
        for leaf, adv in zip(leaves, advantages):
            leaf.advantage = adv.item()
        
        # Step 4: Backprop advantages
        self._backprop_advantages(root)
        
        # Step 5: Collect transitions
        transitions = self._collect_transitions(root)
        
        # Step 6: Compute weighted loss
        loss = self._compute_weighted_loss(model, transitions)
        
        return loss
    
    def _compute_rewards(
        self,
        leaves: List[MCTSNode],
        prompts: List[str],
        tokenizer
    ) -> torch.Tensor:
        """
        Compute terminal rewards for all leaves
        
        Args:
            leaves: List[MCTSNode]
            prompts: List[str]
            tokenizer: tokenizer
            
        Returns:
            rewards: torch.Tensor [num_leaves]
        """
        rewards = []
        
        for leaf, prompt in zip(leaves, prompts):
            # Decode completion
            completion = tokenizer.decode(
                leaf.state,
                skip_special_tokens=True
            )
            
            # Compute reward (external function)
            reward = self.reward_function(completion, prompt)
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def _backprop_advantages(self, node: MCTSNode) -> float:
        """
        Recursively backpropagate advantages from leaves to root
        
        Args:
            node: MCTSNode to backprop through
            
        Returns:
            float: node's advantage value
            
        Math:
            Leaf: A_leaf = reward - mean(rewards)
            Internal: A_internal = mean(child advantages)
            
        Side effects:
            Sets node.advantage for all nodes
        """
        if not node.children:  # Leaf
            # Advantage already set from rewards
            return node.advantage
        
        # Internal node: average children
        child_advantages = [
            self._backprop_advantages(child)
            for child in node.children
        ]
        node.advantage = sum(child_advantages) / len(child_advantages)
        
        return node.advantage
    
    def _collect_transitions(
        self,
        root: MCTSNode
    ) -> List[TreeTransition]:
        """
        Collect all parent→child transitions in tree
        
        Args:
            root: MCTSNode, tree root
            
        Returns:
            List[TreeTransition]: all transitions
        """
        transitions = []
        
        def traverse(node):
            for child in node.children:
                # Compute weights for this transition
                time_weight = self.time_weighter.get_weight(node.timestep)
                entropy_weight = self.entropy_computer.compute_entropy_weight(
                    node.entropy if node.entropy is not None else 0.0,
                    node.timestep
                )
                
                transition = TreeTransition(
                    parent_state=node.state,
                    child_state=child.state,
                    timestep=node.timestep,
                    advantage=child.advantage,
                    entropy=node.entropy if node.entropy is not None else 0.0,
                    time_weight=time_weight,
                    entropy_weight=entropy_weight
                )
                transitions.append(transition)
                
                # Recurse
                traverse(child)
        
        traverse(root)
        return transitions
    
    def _compute_weighted_loss(
        self,
        model: torch.nn.Module,
        transitions: List[TreeTransition]
    ) -> torch.Tensor:
        """
        Compute weighted GRPO loss from transitions
        
        Args:
            model: MDLM model
            transitions: List[TreeTransition]
            
        Returns:
            loss: torch.Tensor, scalar
            
        Math:
            L = -sum_n [ (α_time * w_time(t) + α_ent * w_ent(t)) 
                         * A_n * log p(child|parent, t) ]
        """
        total_loss = 0.0
        
        for trans in transitions:
            # Compute log probability
            log_prob = self._compute_log_prob(
                model,
                trans.parent_state,
                trans.child_state,
                trans.timestep
            )
            
            # Combined weight
            combined_weight = (
                self.config.alpha_time * trans.time_weight +
                self.config.alpha_entropy * trans.entropy_weight
            )
            
            # Weighted GRPO term
            # Note: negative sign because we maximize advantage
            loss_term = -combined_weight * trans.advantage * log_prob
            total_loss += loss_term
        
        # Average over transitions
        loss = total_loss / len(transitions)
        
        return loss
    
    def _compute_log_prob(
        self,
        model: torch.nn.Module,
        parent_state: torch.Tensor,
        child_state: torch.Tensor,
        timestep: int
    ) -> torch.Tensor:
        """
        Compute log p(child | parent, t) for MDLM
        
        Args:
            model: MDLM model
            parent_state: [seq_len] parent tokens
            child_state: [seq_len] child tokens
            timestep: int
            
        Returns:
            log_prob: torch.Tensor, scalar
            
        Math:
            log p(child | parent, t) = sum_i [ changed(i) * log p(child[i] | parent, t) ]
            where changed(i) = 1 if parent[i] != child[i], else 0
        """
        # Get model predictions for parent state
        logits = model(
            parent_state.unsqueeze(0),  # Add batch dim
            timestep=timestep
        )[0]  # Remove batch dim: [seq_len, vocab_size]
        
        # Convert to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # [seq_len, vocab_size]
        
        # Find changed positions
        changed = (parent_state != child_state)
        
        # Get log probs of actual child tokens
        child_log_probs = log_probs.gather(
            dim=-1,
            index=child_state.unsqueeze(-1)
        ).squeeze(-1)  # [seq_len]
        
        # Sum only over changed positions
        log_prob = (child_log_probs * changed.float()).sum()
        
        return log_prob
```

### Training Loop Scaffold

```python
class EntropyMCTSTrainer:
    """
    Main training loop for entropy-guided MCTS-GRPO
    """
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        config: MCTSConfig,
        reward_function,
        optimizer,
        device: str = 'cuda'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        self.tree_builder = EntropyGuidedTreeBuilder(
            model, tokenizer, config
        )
        self.loss_computer = WeightedGRPOLoss(
            config, reward_function
        )
        self.optimizer = optimizer
    
    def train_step(self, prompts: List[str]) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            prompts: List[str], batch of prompts
            
        Returns:
            dict with 'loss', 'avg_reward', 'tree_size'
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Build trees for all prompts
        trees = []
        all_leaves = []
        for prompt in prompts:
            root, leaves = self.tree_builder.build_tree(prompt, self.device)
            trees.append(root)
            all_leaves.extend(leaves)
        
        # Compute loss (batched across all trees)
        loss = self.loss_computer.compute_loss(
            self.model,
            trees[0],  # TODO: batch this properly
            all_leaves,
            prompts,
            self.tokenizer
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        
        # Optimizer step
        self.optimizer.step()
        
        # Collect metrics
        metrics = {
            'loss': loss.item(),
            'avg_reward': sum(leaf.advantage for leaf in all_leaves) / len(all_leaves),
            'tree_size': len(all_leaves)
        }
        
        return metrics
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            dataloader: DataLoader yielding batches of prompts
            
        Returns:
            dict with aggregated metrics
        """
        epoch_metrics = {'loss': [], 'avg_reward': [], 'tree_size': []}
        
        for batch_prompts in dataloader:
            metrics = self.train_step(batch_prompts)
            
            for key, value in metrics.items():
                epoch_metrics[key].append(value)
        
        # Average metrics
        avg_metrics = {
            key: sum(values) / len(values)
            for key, values in epoch_metrics.items()
        }
        
        return avg_metrics
```

---

## Usage Example

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# 1. Load model
model_name = "dllm-collection/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1"
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = model.to('cuda')

# 2. Define reward function
def code_correctness_reward(completion: str, prompt: str) -> float:
    """
    Example: check if code passes test cases
    """
    # TODO: implement actual test execution
    # For now, dummy reward
    return 1.0 if "def" in completion else 0.0

# 3. Configure
config = MCTSConfig(
    max_tree_nodes=30,
    branch_width=3,
    steps_per_expansion=32,
    temperature=0.8,
    alpha_time=1.0,
    alpha_entropy=0.5,
    total_steps=256,
    vocab_size=len(tokenizer),
    batch_size=4,
    learning_rate=1e-5
)

# 4. Setup optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate
)

# 5. Create trainer
trainer = EntropyMCTSTrainer(
    model=model,
    tokenizer=tokenizer,
    config=config,
    reward_function=code_correctness_reward,
    optimizer=optimizer,
    device='cuda'
)

# 6. Training loop
prompts = [
    "def fibonacci(n):",
    "def quicksort(arr):",
    "def binary_search(arr, target):",
    "def merge_sort(arr):"
]

for epoch in range(10):
    metrics = trainer.train_step(prompts)
    print(f"Epoch {epoch}: loss={metrics['loss']:.4f}, "
          f"reward={metrics['avg_reward']:.4f}, "
          f"tree_size={metrics['tree_size']}")
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Implement `MCTSNode` data structure
- [ ] Implement `EntropyComputer` with exact Shannon entropy
- [ ] Implement `TimeWeighter` with precomputed weights
- [ ] Unit test entropy computation against manual calculation
- [ ] Unit test time weights sum to 1

### Phase 2: Tree Construction
- [ ] Implement `EntropyGuidedTreeBuilder._initialize_root()`
- [ ] Implement `_compute_node_entropy()`
- [ ] Implement `_denoise_steps()` with stochastic sampling
- [ ] Implement global frontier selection in `build_tree()`
- [ ] Visualize tree structure (print tree depth, branching factor)

### Phase 3: Loss Computation
- [ ] Implement `WeightedGRPOLoss._compute_rewards()`
- [ ] Implement `_backprop_advantages()`
- [ ] Implement `_compute_log_prob()` for MDLM
- [ ] Implement `_compute_weighted_loss()`
- [ ] Test on synthetic tree with known advantages

### Phase 4: Integration
- [ ] Implement `EntropyMCTSTrainer.train_step()`
- [ ] Add gradient clipping and logging
- [ ] Test on single batch (overfit test)
- [ ] Scale to full dataset

### Phase 5: Evaluation
- [ ] Implement evaluation on HumanEval / MBPP
- [ ] Compare baseline GRPO vs entropy-guided MCTS
- [ ] Ablation studies (alpha_time, alpha_entropy, tree budget)
- [ ] Visualize which timesteps get explored most

---

## Debugging Tips

### 1. Entropy Sanity Checks
```python
# Entropy should decrease with timestep
for t in [0, 64, 128, 192, 256]:
    entropy = entropy_computer.compute_token_entropy(model, z_t, t)
    print(f"t={t}: H={entropy.mean():.2f}")
# Expected: ~10.8 → ~5.0 → ~2.0 → ~0.5 → ~0.0
```

### 2. Tree Structure Validation
```python
# Tree should have expected size
root, leaves = tree_builder.build_tree(prompt)
assert count_nodes(root) <= config.max_tree_nodes
assert len(leaves) > 1  # Should have multiple completions
```

### 3. Loss Magnitude Check
```python
# GRPO loss should be O(1) to O(10)
loss = loss_computer.compute_loss(...)
assert 0.1 < loss < 100, f"Suspicious loss magnitude: {loss}"
```

### 4. Gradient Flow Check
```python
# Verify gradients exist and are reasonable
loss.backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        assert grad_norm < 100, f"Exploding gradient in {name}: {grad_norm}"
```

---

## Performance Optimization Notes

### 1. Batch Tree Construction
Current implementation builds trees sequentially. For efficiency:
- Build multiple trees in parallel (batch the model forward passes)
- Use same entropy threshold across batch

### 2. Caching
- Cache model forward passes within tree (nodes at same timestep)
- Cache entropy computations for nodes

### 3. Memory Management
- Delete tree after loss computation
- Use gradient checkpointing for deep trees
- Consider mixed precision (fp16) training

### 4. Parallelization
- Multi-GPU: data parallelism across prompts
- Each GPU builds independent trees

