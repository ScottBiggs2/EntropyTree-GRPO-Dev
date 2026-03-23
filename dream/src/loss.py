"""Weighted GRPO loss with corrected time/entropy weighting for Dream stack."""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from dream.src.config import MCTSConfig
from dream.src.tree_node import MCTSNode, TreeTransition
from dream.src.entropy import EntropyComputer
from dream.src.time_weight import TimeWeighter
from dream.src.model_adapter import ModelAdapter


class WeightedGRPOLoss:
    """Compute time- and entropy-weighted GRPO loss from tree transitions."""

    def __init__(
        self,
        config: MCTSConfig,
        entropy_computer: EntropyComputer,
        time_weighter: TimeWeighter,
        mask_id: int,
        adapter: ModelAdapter,
    ):
        self.config = config
        self.entropy_computer = entropy_computer
        self.time_weighter = time_weighter
        self.mask_id = mask_id
        self.adapter = adapter

    @staticmethod
    def _parent_groups(transitions: List[TreeTransition]) -> List[List[TreeTransition]]:
        """Siblings share the same parent tensor — group for one forward per parent."""
        d: Dict[int, List[TreeTransition]] = defaultdict(list)
        for t in transitions:
            d[id(t.parent_state)].append(t)
        return list(d.values())

    def _transition_mask_change(self, trans: TreeTransition) -> torch.Tensor:
        return (trans.parent_state != trans.child_state) & (
            trans.parent_state == self.mask_id
        )

    def compute_loss(
        self,
        model: torch.nn.Module,
        root: MCTSNode,
        leaves: List[MCTSNode],
        prompt: str,
        vocab_size: int,
        backward_per_transition: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Returns (loss, metrics_dict).

        With ``loss_group_by_parent`` (default): one ``forward_logits`` per
        **parent node** for all child edges (siblings), then sum weighted
        log-prob terms — **identical gradients** to per-edge forwards, fewer
        transformer passes and a smaller autograd graph.

        With ``loss_backward_per_transition`` and grouping: one backward per
        **parent group** (sum of normalized contribs for that group's edges).
        """
        transitions = self._collect_transitions(root, vocab_size)
        if not transitions:
            dev = next(model.parameters()).device
            return torch.tensor(0.0, device=dev), {"n_transitions": 0}

        n_tr = len(transitions)
        mean_edge_delta = (
            sum(t.child_step_index - t.step_index for t in transitions) / n_tr
        )

        if backward_per_transition is None:
            backward_per_transition = getattr(
                self.config, "loss_backward_per_transition", True
            )
        group_by_parent = getattr(self.config, "loss_group_by_parent", True)

        n = len(transitions)
        device = next(model.parameters()).device

        sum_abs_adv = 0.0
        sum_w_time = 0.0
        sum_w_ent = 0.0
        sum_w_ent_raw = 0.0
        n_clamped_low = 0
        n_clamped_high = 0
        sum_weight = 0.0
        sum_weighted_term = 0.0

        def process_one_trans(
            trans: TreeTransition, raw_log_prob: torch.Tensor
        ) -> torch.Tensor:
            nonlocal sum_abs_adv, sum_w_time, sum_w_ent, sum_w_ent_raw, n_clamped_low, n_clamped_high, sum_weight, sum_weighted_term
            changed = self._transition_mask_change(trans)
            n_tok = max(int(changed.sum().item()), 1)
            log_prob = raw_log_prob / n_tok

            w_time = trans.time_weight
            w_ent = trans.entropy_weight
            ew_raw = float(getattr(trans, "entropy_weight_raw", w_ent))
            sum_w_ent_raw += ew_raw
            if ew_raw < self.config.entropy_weight_min - 1e-12:
                n_clamped_low += 1
            elif ew_raw > self.config.entropy_weight_max + 1e-12:
                n_clamped_high += 1
            w = self.config.alpha_time * w_time + self.config.alpha_entropy * w_ent
            contrib = -w * float(trans.advantage) * log_prob

            lp_val = float(log_prob.detach().item())
            adv_val = float(trans.advantage)
            sum_abs_adv += abs(adv_val)
            sum_w_time += float(w_time)
            sum_w_ent += float(w_ent)
            sum_weight += float(w)
            sum_weighted_term += float(-w * adv_val * lp_val)
            return contrib

        if backward_per_transition:
            total_logged = torch.tensor(0.0, device=device, dtype=torch.float32)
            if group_by_parent:
                groups = self._parent_groups(transitions)
                for group in groups:
                    chunk = torch.zeros((), device=device, dtype=torch.float32)
                    head = group[0]
                    any_change = any(
                        self._transition_mask_change(t).any().item() for t in group
                    )
                    logits = None
                    if any_change:
                        logits = self.adapter.forward_logits(
                            head.parent_state.unsqueeze(0),
                            head.parent_attention_mask.unsqueeze(0),
                        )[0]
                    for trans in group:
                        if logits is None:
                            raw_lp = torch.zeros(
                                (), device=head.parent_state.device, dtype=torch.float32
                            )
                        else:
                            raw_lp = self._log_prob_from_logits(
                                logits, trans.parent_state, trans.child_state
                            )
                        contrib = process_one_trans(trans, raw_lp)
                        chunk = chunk + contrib / n
                    if chunk.requires_grad:
                        chunk.backward()
                    total_logged = total_logged + chunk.detach()
            else:
                for trans in transitions:
                    raw_log_prob = self._log_prob_transition(
                        self.adapter,
                        trans.parent_state,
                        trans.child_state,
                        trans.parent_attention_mask,
                    )
                    contrib = process_one_trans(trans, raw_log_prob)
                    loss_i = contrib / n
                    if loss_i.requires_grad:
                        loss_i.backward()
                    total_logged = total_logged + loss_i.detach()

            metrics = {
                "loss": float(total_logged.item()),
                "n_transitions": n,
                "mean_abs_adv": sum_abs_adv / n,
                "mean_w_time": sum_w_time / n,
                "mean_w_ent": sum_w_ent / n,
                "mean_w_ent_raw": sum_w_ent_raw / n,
                "frac_entropy_clamped_low": float(n_clamped_low) / n,
                "frac_entropy_clamped_high": float(n_clamped_high) / n,
                "mean_edge_denoising_delta": mean_edge_delta,
                "mean_weight": sum_weight / n,
                "mean_weighted_adv_logp": sum_weighted_term / n,
                "n_loss_forwards": float(
                    len(self._parent_groups(transitions))
                    if group_by_parent
                    else n
                ),
            }
            return total_logged.detach(), metrics

        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        if group_by_parent:
            for group in self._parent_groups(transitions):
                head = group[0]
                any_change = any(
                    self._transition_mask_change(t).any().item() for t in group
                )
                logits = None
                if any_change:
                    logits = self.adapter.forward_logits(
                        head.parent_state.unsqueeze(0),
                        head.parent_attention_mask.unsqueeze(0),
                    )[0]
                for trans in group:
                    if logits is None:
                        raw_lp = torch.zeros(
                            (), device=head.parent_state.device, dtype=torch.float32
                        )
                    else:
                        raw_lp = self._log_prob_from_logits(
                            logits, trans.parent_state, trans.child_state
                        )
                    contrib = process_one_trans(trans, raw_lp)
                    total_loss = total_loss + contrib
        else:
            for trans in transitions:
                raw_log_prob = self._log_prob_transition(
                    self.adapter,
                    trans.parent_state,
                    trans.child_state,
                    trans.parent_attention_mask,
                )
                contrib = process_one_trans(trans, raw_log_prob)
                total_loss = total_loss + contrib

        loss = total_loss / n
        metrics = {
            "loss": float(loss.item()),
            "n_transitions": n,
            "mean_abs_adv": sum_abs_adv / n,
            "mean_w_time": sum_w_time / n,
            "mean_w_ent": sum_w_ent / n,
            "mean_w_ent_raw": sum_w_ent_raw / n,
            "frac_entropy_clamped_low": float(n_clamped_low) / n,
            "frac_entropy_clamped_high": float(n_clamped_high) / n,
            "mean_edge_denoising_delta": mean_edge_delta,
            "mean_weight": sum_weight / n,
            "mean_weighted_adv_logp": sum_weighted_term / n,
            "n_loss_forwards": float(
                len(self._parent_groups(transitions)) if group_by_parent else n
            ),
        }
        return loss, metrics

    def _collect_transitions(self, root: MCTSNode, vocab_size: int) -> List[TreeTransition]:
        out: List[TreeTransition] = []

        def go(node: MCTSNode) -> None:
            for c in node.children:
                parent_step = node.step_index
                child_step = c.step_index
                tw = self.time_weighter.get_interval_weight(parent_step, child_step)

                ew_raw = self.entropy_computer.compute_entropy_weight(
                    measured_masked_mean=node.entropy or 0.0,
                    vocab_size=vocab_size,
                    masking_ratio=node.masking_ratio(),
                    mode=self.config.entropy_norm_mode,
                )
                ew = max(
                    self.config.entropy_weight_min,
                    min(self.config.entropy_weight_max, ew_raw),
                )

                trans = TreeTransition(
                    parent_state=node.state,
                    child_state=c.state,
                    parent_attention_mask=node.attention_mask,
                    child_attention_mask=c.attention_mask,
                    step_index=parent_step,
                    child_step_index=child_step,
                    advantage=c.advantage or 0.0,
                    entropy=node.entropy or 0.0,
                    time_weight=tw,
                    entropy_weight=ew,
                    entropy_weight_raw=float(ew_raw),
                )
                out.append(trans)
                go(c)

        go(root)
        return out

    def _log_prob_from_logits(
        self,
        logits: torch.Tensor,
        parent_state: torch.Tensor,
        child_state: torch.Tensor,
    ) -> torch.Tensor:
        """Scalar sum of log p(child_tok | parent) over mask→token positions."""
        changed = (parent_state != child_state) & (parent_state == self.mask_id)
        if not changed.any():
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, child_state.unsqueeze(-1)).squeeze(-1)
        return (token_lp * changed.float()).sum()

    def _log_prob_transition(
        self,
        adapter: ModelAdapter,
        parent_state: torch.Tensor,
        child_state: torch.Tensor,
        parent_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Log p(child | parent): only mask->token changes contribute."""
        changed = (parent_state != child_state) & (parent_state == self.mask_id)
        if not changed.any():
            return torch.tensor(0.0, device=parent_state.device)
        logits = adapter.forward_logits(
            parent_state.unsqueeze(0),
            parent_attention_mask.unsqueeze(0),
        )[0]
        return self._log_prob_from_logits(logits, parent_state, child_state)

    def trajectory_log_prob_with_count(
        self,
        model: torch.nn.Module,
        transitions: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, int]:
        """Sum log p along a single trajectory; also count mask→token positions (for normalization).

        Used by ``BaselineGRPOTrainer`` (non-tree GRPO). ``model`` is unused;
        gradients flow through ``self.adapter`` (same as tree loss).
        """
        del model
        if not transitions:
            dev = next(self.adapter.model.parameters()).device
            return torch.tensor(0.0, device=dev), 0
        total_lp: Optional[torch.Tensor] = None
        total_tokens = 0
        for p, c, a in transitions:
            changed = (p != c) & (p == self.mask_id)
            total_tokens += int(changed.sum().item())
            lp = self._log_prob_transition(self.adapter, p, c, a)
            total_lp = lp if total_lp is None else (total_lp + lp)
        assert total_lp is not None
        return total_lp, total_tokens
