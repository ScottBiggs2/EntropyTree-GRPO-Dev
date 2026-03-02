"""Entropy-guided MCTS tree construction (DeepSearch-style). Phase 4."""

from typing import List, Tuple

import torch
import torch.nn.functional as F

from src.config import MCTSConfig
from src.tree_node import MCTSNode
from src.entropy import EntropyComputer
from src.utils import add_gumbel_noise, chat_template_to_token_ids


class EntropyGuidedTreeBuilder:
    """Build exploration tree with global frontier selection by entropy."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        config: MCTSConfig,
        entropy_computer: EntropyComputer,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.entropy_computer = entropy_computer
        self.device = config.device or next(model.parameters()).device
        self.mask_id = tokenizer.mask_token_id
        self.vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
        self._entropy_diagnostic_logged = False
        self.pad_id = (
            tokenizer.pad_token_id
            or tokenizer.eos_token_id
            or tokenizer.mask_token_id
        )

    def build_tree(self, prompt: str) -> Tuple[MCTSNode, List[MCTSNode]]:
        """Build tree; return (root, final_leaves)."""
        root = self._create_root(prompt)
        leaf_nodes: List[MCTSNode] = [root]
        nodes_used = 1

        while nodes_used < self.config.max_tree_nodes and leaf_nodes:
            for node in leaf_nodes:
                if node.entropy is None:
                    self._compute_node_entropy(node)

            leaf_nodes.sort(key=lambda n: n.entropy or -1.0, reverse=True)
            k = min(
                self.config.branch_width,
                len(leaf_nodes),
                self.config.max_tree_nodes - nodes_used,
            )
            if k <= 0:
                break
            to_expand = leaf_nodes[:k]
            newly_created: List[MCTSNode] = []

            for node in to_expand:
                if nodes_used >= self.config.max_tree_nodes:
                    break
                children = self._expand_node(node)
                for child in children:
                    if nodes_used >= self.config.max_tree_nodes:
                        break
                    node.children.append(child)
                    newly_created.append(child)
                    nodes_used += 1
                leaf_nodes.remove(node)
            leaf_nodes.extend(newly_created)

        final_leaves = self._complete_leaves(leaf_nodes)
        self._fill_missing_entropy(root)
        return root, final_leaves

    def _create_root(self, prompt: str) -> MCTSNode:
        messages = [{"role": "user", "content": prompt}]
        encoded = chat_template_to_token_ids(self.tokenizer, messages)
        prompt_len = len(encoded)
        max_new = self.config.max_new_tokens
        total_len = prompt_len + max_new

        state = torch.full(
            (total_len,), self.pad_id, dtype=torch.long, device=self.device
        )
        state[:prompt_len] = torch.tensor(encoded, device=self.device)
        state[prompt_len:prompt_len + max_new] = self.mask_id
        attn = torch.ones(total_len, device=self.device)

        root = MCTSNode(
            state=state,
            attention_mask=attn,
            prompt_len=prompt_len,
            step_index=0,
            parent=None,
            mask_id=self.mask_id,
            depth=0,
        )
        return root

    def _compute_node_entropy(self, node: MCTSNode) -> None:
        """Set node.entropy and node.token_entropy from model forward.
        Same-position: logits[t] = distribution for position t; mean entropy over masked positions only."""
        ids = node.state.unsqueeze(0)
        attn = node.attention_mask.unsqueeze(0)
        token_entropy = self.entropy_computer.compute_token_entropy(
            self.model, ids, attn
        )
        node.token_entropy = token_entropy[0]
        mask_pos = (node.state == self.mask_id) & (node.attention_mask.bool())
        if mask_pos.any():
            node.entropy = self.entropy_computer.aggregate_entropy(
                token_entropy, mask_pos.unsqueeze(0), method="mean"
            )
            # One-time diagnostic for first internal node (if tree still shows zeros)
            if node.depth == 1 and not self._entropy_diagnostic_logged:
                vals = token_entropy[0][mask_pos]
                n_masked = mask_pos.sum().item()
                self._entropy_diagnostic_logged = True
                print(
                    f"[entropy diagnostic] depth=1 node: n_masked={n_masked}, "
                    f"token_entropy at masked: min={vals.min().item():.6f} "
                    f"mean={vals.float().mean().item():.6f} max={vals.max().item():.6f}"
                )
        else:
            node.entropy = 0.0

    def _expand_node(self, node: MCTSNode) -> List[MCTSNode]:
        """Create branch_width children by stochastic _denoise_chunk."""
        children: List[MCTSNode] = []
        temp = self._node_temperature(node)
        for _ in range(self.config.branch_width):
            child = self._denoise_chunk(node, self.config.steps_per_expansion, temp)
            child.sampling_prob = 1.0 / self.config.branch_width
            child.depth = node.depth + 1
            children.append(child)
        return children

    def _node_temperature(self, node: MCTSNode) -> float:
        """
        Adaptive temperature: higher entropy → higher temperature → more stochastic sampling.
        Uses node.entropy vs. expected_entropy(masking_ratio, vocab_size) as an uncertainty ratio.
        """
        base_temp = self.config.temperature
        if node.entropy is None:
            return base_temp
        masking_ratio = node.masking_ratio()
        expected_h = self.entropy_computer.expected_entropy(
            masking_ratio, vocab_size=self.vocab_size
        )
        if expected_h < 1e-6:
            return base_temp
        uncertainty_ratio = float(node.entropy) / float(expected_h)
        temperature = base_temp * (0.7 + 0.6 * uncertainty_ratio)
        return max(0.5, min(1.3, temperature))

    def _denoise_chunk(self, node: MCTSNode, num_steps: int, temperature: float) -> MCTSNode:
        """Run num_steps denoising steps from node; return new child node."""
        state = node.state.clone()
        attn = node.attention_mask.clone()
        prompt_len = node.prompt_len
        max_new = self.config.max_new_tokens
        response_region = torch.zeros_like(state, dtype=torch.bool, device=state.device)
        response_region[prompt_len : prompt_len + max_new] = True

        with torch.no_grad():
            n_masked_initial = None
            for step in range(num_steps):
                mask_now = (state == self.mask_id) & response_region
                n_masked = mask_now.sum().item()
                if n_masked == 0:
                    break
                if n_masked_initial is None:
                    n_masked_initial = n_masked
                steps_left = num_steps - step
                # Cap k so we leave at least 1 masked token (so child entropy is computable)
                k_cap = max(1, (n_masked_initial - 1) // num_steps)
                k = min(k_cap, max(1, n_masked // steps_left))

                logits = self.model(
                    state.unsqueeze(0), attention_mask=attn.unsqueeze(0)
                ).logits[0]
                logits_n = add_gumbel_noise(logits, temperature)
                x0_pred = torch.argmax(logits_n, dim=-1)
                probs = F.softmax(logits, dim=-1)
                conf = torch.gather(
                    probs, -1, x0_pred.unsqueeze(-1)
                ).squeeze(-1)
                confidence = torch.where(
                    mask_now, conf, torch.full_like(conf, -1e9)
                )
                x0_pred = torch.where(mask_now, x0_pred, state)
                _, sel = torch.topk(confidence, k=min(k, n_masked))
                state[sel] = x0_pred[sel]

        child = MCTSNode(
            state=state,
            attention_mask=attn,
            prompt_len=prompt_len,
            step_index=node.step_index + num_steps,
            parent=node,
            mask_id=self.mask_id,
        )
        return child

    def _complete_leaves(self, leaf_nodes: List[MCTSNode]) -> List[MCTSNode]:
        """Denoise each leaf to full completion; return list of final completion nodes."""
        final_leaves: List[MCTSNode] = []
        for node in leaf_nodes:
            if node.num_masked_tokens() == 0:
                node.is_completed = True
                final_leaves.append(node)
                continue
            completed = self._denoise_to_completion(node)
            completed.is_completed = True
            completed.depth = node.depth + 1
            node.children.append(completed)
            final_leaves.append(completed)
        return final_leaves

    def _fill_missing_entropy(self, node: MCTSNode) -> None:
        """DFS: compute entropy for any node that has entropy is None (e.g. last-iteration children, completed leaves)."""
        if node.entropy is None:
            self._compute_node_entropy(node)
        for child in node.children:
            self._fill_missing_entropy(child)

    def _denoise_to_completion(self, node: MCTSNode) -> MCTSNode:
        """Run denoising until no masked tokens remain in response."""
        state = node.state.clone()
        attn = node.attention_mask.clone()
        prompt_len = node.prompt_len
        max_new = self.config.max_new_tokens
        response_region = torch.zeros_like(state, dtype=torch.bool, device=state.device)
        response_region[prompt_len : prompt_len + max_new] = True

        with torch.no_grad():
            temperature = self._node_temperature(node)
            while True:
                mask_now = (state == self.mask_id) & response_region
                n_masked = mask_now.sum().item()
                if n_masked == 0:
                    break
                k = min(n_masked, 64)  # unmask up to 64 per step for speed
                logits = self.model(
                    state.unsqueeze(0), attention_mask=attn.unsqueeze(0)
                ).logits[0]
                logits_n = add_gumbel_noise(logits, temperature)
                x0_pred = torch.argmax(logits_n, dim=-1)
                probs = F.softmax(logits, dim=-1)
                conf = torch.gather(
                    probs, -1, x0_pred.unsqueeze(-1)
                ).squeeze(-1)
                confidence = torch.where(
                    mask_now, conf, torch.full_like(conf, -1e9)
                )
                x0_pred = torch.where(mask_now, x0_pred, state)
                _, sel = torch.topk(confidence, k=k)
                state[sel] = x0_pred[sel]

        return MCTSNode(
            state=state,
            attention_mask=attn,
            prompt_len=prompt_len,
            step_index=node.step_index + 999,  # placeholder
            parent=node,
            mask_id=self.mask_id,
        )

    def generate_one_trajectory(
        self, prompt: str
    ) -> Tuple[str, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Generate one full completion from prompt, return (completion_str, list of (parent_state, child_state, attn)) for baseline GRPO."""
        root = self._create_root(prompt)
        state = root.state.clone()
        attn = root.attention_mask.clone()
        prompt_len = root.prompt_len
        max_new = self.config.max_new_tokens
        response_region = torch.zeros_like(state, dtype=torch.bool, device=state.device)
        response_region[prompt_len : prompt_len + max_new] = True
        transitions: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        with torch.no_grad():
            # Baseline GRPO path does not build a full tree, so we don't have node.entropy.
            # Use the global base temperature here.
            temperature = self.config.temperature
            while True:
                mask_now = (state == self.mask_id) & response_region
                n_masked = mask_now.sum().item()
                if n_masked == 0:
                    break
                k = min(n_masked, 64)
                parent_state = state.clone()
                parent_attn = attn.clone()
                logits = self.model(
                    state.unsqueeze(0), attention_mask=attn.unsqueeze(0)
                ).logits[0]
                logits_n = add_gumbel_noise(logits, temperature)
                x0_pred = torch.argmax(logits_n, dim=-1)
                probs = F.softmax(logits, dim=-1)
                conf = torch.gather(
                    probs, -1, x0_pred.unsqueeze(-1)
                ).squeeze(-1)
                confidence = torch.where(
                    mask_now, conf, torch.full_like(conf, -1e9)
                )
                x0_pred = torch.where(mask_now, x0_pred, state)
                _, sel = torch.topk(confidence, k=k)
                state[sel] = x0_pred[sel]
                transitions.append((parent_state, state.clone(), parent_attn))

        completion = self.tokenizer.decode(
            state[prompt_len : prompt_len + max_new].tolist(),
            skip_special_tokens=True,
        )
        return completion, transitions
