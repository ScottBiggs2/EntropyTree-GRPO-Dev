"""Entropy-guided MCTS tree construction for the Dream substack.

This mirrors the parent project's EntropyGuidedTreeBuilder but routes
all model interaction through ModelAdapter and supports (optional)
entropy-threshold adaptive branching.
"""

from typing import List, Tuple
import math

import torch
import torch.nn.functional as F

from dream.src.config import MCTSConfig
from dream.src.tree_node import MCTSNode
from dream.src.entropy import EntropyComputer
from dream.src.model_adapter import ModelAdapter


class EntropyGuidedTreeBuilder:
    """Build exploration tree with global frontier selection by entropy."""

    def __init__(
        self,
        adapter: ModelAdapter,
        tokenizer,
        config: MCTSConfig,
        entropy_computer: EntropyComputer,
    ):
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.config = config
        self.entropy_computer = entropy_computer
        self.device = config.device or adapter.device
        self.mask_id = tokenizer.mask_token_id
        self.vocab_size = adapter.vocab_size
        self._entropy_diagnostic_logged = False
        self.pad_id = (
            tokenizer.pad_token_id
            or tokenizer.eos_token_id
            or tokenizer.mask_token_id
        )
        self._gen_diag_count = 0

    # ---- Public API ----

    def build_tree(self, prompt: str) -> Tuple[MCTSNode, List[MCTSNode]]:
        """Build tree from a text prompt; return (root, final_leaves)."""
        root = self._create_root(prompt)
        leaf_nodes: List[MCTSNode] = [root]
        nodes_used = 1

        while nodes_used < self.config.max_tree_nodes and leaf_nodes:
            # Ensure entropy is computed for all current leaves.
            for node in leaf_nodes:
                if node.entropy is None:
                    self._compute_node_entropy(node)

            # Global frontier selection by entropy (highest first).
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

    # ---- Root / entropy helpers ----

    def _create_root(self, prompt: str) -> MCTSNode:
        """Create a root node with a masked response region."""
        # Use the tokenizer's chat template if available; for local tests
        # we allow prompt to be tokenized directly.
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors=None,
            )
            if hasattr(encoded, "input_ids"):
                ids = encoded.input_ids[0].tolist()
            elif isinstance(encoded, dict) and "input_ids" in encoded:
                ids = encoded["input_ids"][0]
            else:
                # Fall back to direct encode.
                ids = (
                    self.tokenizer.encode(prompt, add_special_tokens=True)
                    if isinstance(encoded, str)
                    else list(encoded)
                )
        else:
            ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        prompt_len = len(ids)
        max_new = self.config.max_new_tokens
        total_len = prompt_len + max_new

        state = torch.full(
            (total_len,), self.pad_id, dtype=torch.long, device=self.device
        )
        state[:prompt_len] = torch.tensor(ids, device=self.device)
        state[prompt_len : prompt_len + max_new] = self.mask_id
        attn = torch.ones(total_len, device=self.device)

        return MCTSNode(
            state=state,
            attention_mask=attn,
            prompt_len=prompt_len,
            step_index=0,
            parent=None,
            mask_id=self.mask_id,
            depth=0,
        )

    def _compute_node_entropy(self, node: MCTSNode) -> None:
        """Set node.entropy and node.token_entropy from adapter logits."""
        ids = node.state.unsqueeze(0)
        attn = node.attention_mask.unsqueeze(0)
        logits = self.adapter.forward_logits(ids, attn)
        token_entropy = self.entropy_computer.compute_token_entropy_from_logits(
            logits
        )
        node.token_entropy = token_entropy[0]
        mask_pos = (node.state == self.mask_id) & (node.attention_mask.bool())
        if mask_pos.any():
            node.entropy = self.entropy_computer.aggregate_entropy(
                token_entropy, mask_pos.unsqueeze(0), method="mean"
            )
            if node.depth == 1 and not self._entropy_diagnostic_logged:
                vals = token_entropy[0][mask_pos]
                self._entropy_diagnostic_logged = True
                print(
                    "[dream entropy diag] depth=1 "
                    f"n_masked={mask_pos.sum().item()} "
                    f"min={vals.min().item():.6f} "
                    f"mean={vals.float().mean().item():.6f} "
                    f"max={vals.max().item():.6f}"
                )
        else:
            node.entropy = 0.0

    # ---- Expansion / denoising ----

    def _expand_node(self, node: MCTSNode) -> List[MCTSNode]:
        """Create children by running a denoising chunk from this node.

        Uses ``train_sampling_temperature`` (if > 0) to promote diversity
        across sibling branches.
        """
        children: List[MCTSNode] = []
        base_temp = self._node_temperature(node)
        train_temp = getattr(self.config, "train_sampling_temperature", 0.0)
        temp = train_temp if train_temp > 0 else base_temp
        for _ in range(self.config.branch_width):
            if self.config.adaptive_stepping:
                child = self._denoise_chunk_adaptive(
                    node,
                    self.config.min_steps_per_expansion,
                    self.config.max_steps_per_expansion,
                    self.config.branch_threshold,
                    temp,
                )
            else:
                child = self._denoise_chunk(
                    node, self.config.steps_per_expansion, temp
                )
            child.sampling_prob = 1.0 / max(self.config.branch_width, 1)
            child.depth = node.depth + 1
            children.append(child)
        return children

    def _node_temperature(self, node: MCTSNode) -> float:
        """Adaptive temperature: higher entropy → slightly higher temperature."""
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
        ratio_clamped = min(uncertainty_ratio, 1.0)
        return base_temp * (0.8 + 0.3 * ratio_clamped)

    def _denoise_chunk(
        self, node: MCTSNode, num_steps: int, temperature: float
    ) -> MCTSNode:
        """Run fixed-step denoising from node; return child node.

        The per-step unmasking rate is derived from the *global* denoising
        schedule (total_denoising_steps), not the per-expansion budget.
        This ensures each expansion only partially denoises, leaving masked
        tokens for deeper tree levels.
        """
        state = node.state.clone()
        attn = node.attention_mask.clone()
        prompt_len = node.prompt_len
        max_new = self.config.max_new_tokens
        response_region = torch.zeros_like(state, dtype=torch.bool)
        response_region[prompt_len : prompt_len + max_new] = True

        total_T = self.config.total_denoising_steps
        steps_taken = 0
        with torch.no_grad():
            for step in range(num_steps):
                mask_now = (state == self.mask_id) & response_region
                n_masked = int(mask_now.sum().item())
                if n_masked == 0:
                    break
                global_step = node.step_index + step
                k = self.adapter.transfer_count(
                    n_masked, global_step, total_T
                )

                logits = self.adapter.forward_logits(
                    state.unsqueeze(0), attn.unsqueeze(0)
                )[0]
                x0_pred, conf = self.adapter.sample_and_confidence(
                    logits, mask_now, temperature, self.config.top_p
                )
                x0_pred = torch.where(mask_now, x0_pred, state)
                _, sel = torch.topk(conf, k=min(k, n_masked))
                state[sel] = x0_pred[sel]
                steps_taken += 1

        child = MCTSNode(
            state=state,
            attention_mask=attn,
            prompt_len=prompt_len,
            step_index=node.step_index + steps_taken,
            parent=node,
            mask_id=self.mask_id,
        )
        child.steps_in_edge = steps_taken
        return child

    def _denoise_chunk_adaptive(
        self,
        node: MCTSNode,
        min_steps: int,
        max_steps: int,
        branch_threshold: float,
        temperature: float,
    ) -> MCTSNode:
        """Entropy-threshold adaptive stepping.

        Per-step unmasking uses the *global* denoising schedule so that each
        expansion only partially denoises the response.
        """
        state = node.state.clone()
        attn = node.attention_mask.clone()
        prompt_len = node.prompt_len
        max_new = self.config.max_new_tokens
        response_region = torch.zeros_like(state, dtype=torch.bool)
        response_region[prompt_len : prompt_len + max_new] = True

        total_T = self.config.total_denoising_steps
        steps_taken = 0
        with torch.no_grad():
            for step in range(max_steps):
                mask_now = (state == self.mask_id) & response_region
                n_masked = int(mask_now.sum().item())
                if n_masked == 0:
                    break

                logits = self.adapter.forward_logits(
                    state.unsqueeze(0), attn.unsqueeze(0)
                )[0]
                probs = F.softmax(logits, dim=-1)
                token_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                masked_entropy = (
                    token_entropy[mask_now].mean().item() if mask_now.any() else 0.0
                )

                x0_pred, conf = self.adapter.sample_and_confidence(
                    logits, mask_now, temperature, self.config.top_p
                )
                global_step = node.step_index + step
                k = self.adapter.transfer_count(
                    n_masked, global_step, total_T
                )
                x0_pred = torch.where(mask_now, x0_pred, state)
                _, sel = torch.topk(conf, k=min(k, n_masked))
                state[sel] = x0_pred[sel]
                steps_taken += 1

                if steps_taken >= min_steps:
                    log_v = math.log(max(self.vocab_size, 1))
                    if log_v > 1e-6:
                        uncertainty_ratio = masked_entropy / log_v
                        if uncertainty_ratio > branch_threshold:
                            break

        child = MCTSNode(
            state=state,
            attention_mask=attn,
            prompt_len=prompt_len,
            step_index=node.step_index + steps_taken,
            parent=node,
            mask_id=self.mask_id,
        )
        child.steps_in_edge = steps_taken
        return child

    # ---- Completion helpers ----

    def _complete_leaves(self, leaf_nodes: List[MCTSNode]) -> List[MCTSNode]:
        """Denoise each leaf to full completion; return completion nodes."""
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
        """DFS: compute entropy for any node that lacks it."""
        if node.entropy is None:
            self._compute_node_entropy(node)
        for child in node.children:
            self._fill_missing_entropy(child)

    def generate_one_trajectory(
        self,
        prompt: str,
        temperature_override: float = 0.0,
    ) -> Tuple[str, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Generate one full completion without MCTS; record mask→token transitions for GRPO.

        Uses the same iterative unmasking loop as ``_denoise_to_completion`` (Dream
        adapter logits + ``sample_and_confidence``), not ``diffusion_generate``,
        so training-time log-probs stay consistent with the tree stack.

        Args:
            temperature_override: if > 0, use this instead of config.temperature
                to promote diversity across K baseline samples.
        """
        root = self._create_root(prompt)
        state = root.state.clone()
        attn = root.attention_mask.clone()
        prompt_len = root.prompt_len
        max_new = self.config.max_new_tokens
        response_region = torch.zeros_like(state, dtype=torch.bool, device=state.device)
        response_region[prompt_len : prompt_len + max_new] = True
        transitions: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        temperature = temperature_override if temperature_override > 0 else self.config.temperature
        # Quick sanity log: confirm we're sampling (temperature > 0).
        if self._gen_diag_count < 3:
            self._gen_diag_count += 1
            print(
                f"[dream-gen diag] mode=baseline_trajectory "
                f"temperature={float(temperature):.4f} top_p={float(self.config.top_p):.3f} "
                f"max_new_tokens={int(self.config.max_new_tokens)}"
            )

        with torch.no_grad():
            while True:
                mask_now = (state == self.mask_id) & response_region
                n_masked = int(mask_now.sum().item())
                if n_masked == 0:
                    break
                k = min(n_masked, 64)
                parent_state = state.clone()
                parent_attn = attn.clone()
                logits = self.adapter.forward_logits(
                    state.unsqueeze(0), attn.unsqueeze(0)
                )[0]
                x0_pred, conf = self.adapter.sample_and_confidence(
                    logits, mask_now, temperature, self.config.top_p
                )
                x0_pred = torch.where(mask_now, x0_pred, state)
                _, sel = torch.topk(conf, k=k)
                state[sel] = x0_pred[sel]
                transitions.append((parent_state, state.clone(), parent_attn))

        completion = self.tokenizer.decode(
            state[prompt_len : prompt_len + max_new].tolist(),
            skip_special_tokens=True,
        )
        return completion, transitions

    def _denoise_to_completion(self, node: MCTSNode) -> MCTSNode:
        """Run denoising until no masked tokens remain in response."""
        state = node.state.clone()
        attn = node.attention_mask.clone()
        prompt_len = node.prompt_len
        max_new = self.config.max_new_tokens
        response_region = torch.zeros_like(state, dtype=torch.bool)
        response_region[prompt_len : prompt_len + max_new] = True

        steps_taken = 0
        with torch.no_grad():
            temperature = self._node_temperature(node)
            while True:
                mask_now = (state == self.mask_id) & response_region
                n_masked = int(mask_now.sum().item())
                if n_masked == 0:
                    break
                k = min(n_masked, 64)
                logits = self.adapter.forward_logits(
                    state.unsqueeze(0), attn.unsqueeze(0)
                )[0]
                x0_pred, conf = self.adapter.sample_and_confidence(
                    logits, mask_now, temperature, self.config.top_p
                )
                x0_pred = torch.where(mask_now, x0_pred, state)
                _, sel = torch.topk(conf, k=k)
                state[sel] = x0_pred[sel]
                steps_taken += 1

        child = MCTSNode(
            state=state,
            attention_mask=attn,
            prompt_len=prompt_len,
            step_index=node.step_index + steps_taken,
            parent=node,
            mask_id=self.mask_id,
        )
        child.steps_in_edge = steps_taken
        return child

