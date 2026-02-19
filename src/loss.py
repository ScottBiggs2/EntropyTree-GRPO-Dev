"""Weighted GRPO loss (time + entropy weighting). Phase 6."""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from src.config import MCTSConfig
from src.tree_node import MCTSNode, TreeTransition
from src.entropy import EntropyComputer
from src.time_weight import TimeWeighter


class WeightedGRPOLoss:
    """Compute time- and entropy-weighted GRPO loss from tree transitions."""

    def __init__(
        self,
        config: MCTSConfig,
        entropy_computer: EntropyComputer,
        time_weighter: TimeWeighter,
        mask_id: int,
    ):
        self.config = config
        self.entropy_computer = entropy_computer
        self.time_weighter = time_weighter
        self.mask_id = mask_id

    def compute_loss(
        self,
        model: torch.nn.Module,
        root: MCTSNode,
        leaves: List[MCTSNode],
        prompt: str,
        vocab_size: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Returns (loss, metrics_dict)."""
        transitions = self._collect_transitions(root, vocab_size)
        if not transitions:
            return torch.tensor(0.0, device=next(model.parameters()).device), {}

        total_loss = 0.0
        for trans in transitions:
            log_prob = self._log_prob_transition(
                model, trans.parent_state, trans.child_state,
                trans.parent_attention_mask,
            )
            w = (
                self.config.alpha_time * trans.time_weight
                + self.config.alpha_entropy * trans.entropy_weight
            )
            total_loss = total_loss - w * trans.advantage * log_prob

        loss = total_loss / len(transitions)
        metrics = {"loss": loss.item(), "n_transitions": len(transitions)}
        return loss, metrics

    def _collect_transitions(self, root: MCTSNode, vocab_size: int) -> List[TreeTransition]:
        out: List[TreeTransition] = []

        def go(node: MCTSNode) -> None:
            for c in node.children:
                tw = self.time_weighter.get_weight(node.step_index)
                expected_h = self.entropy_computer.expected_entropy(
                    node.masking_ratio(), vocab_size=vocab_size
                )
                if expected_h < 1e-6:
                    ew = 0.0
                else:
                    ew = (node.entropy or 0.0) / expected_h
                trans = TreeTransition(
                    parent_state=node.state,
                    child_state=c.state,
                    parent_attention_mask=node.attention_mask,
                    child_attention_mask=c.attention_mask,
                    step_index=node.step_index,
                    advantage=c.advantage or 0.0,
                    entropy=node.entropy or 0.0,
                    time_weight=tw,
                    entropy_weight=ew,
                )
                out.append(trans)
                go(c)

        go(root)
        return out

    def _log_prob_transition(
        self,
        model: torch.nn.Module,
        parent_state: torch.Tensor,
        child_state: torch.Tensor,
        parent_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Log p(child | parent): only positions that changed (were mask -> unmasked) contribute."""
        changed = (parent_state != child_state) & (parent_state == self.mask_id)
        if not changed.any():
            return torch.tensor(0.0, device=parent_state.device)
        logits = model(
            parent_state.unsqueeze(0),
            attention_mask=parent_attention_mask.unsqueeze(0),
        ).logits[0]
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(
            -1, child_state.unsqueeze(-1)
        ).squeeze(-1)
        return (token_lp * changed.float()).sum()
