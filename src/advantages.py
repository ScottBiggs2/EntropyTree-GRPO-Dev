"""Advantage computation: simple averaging and BranchGRPO-style (Phase 5). D-007."""

from typing import Dict, List

import numpy as np
from src.tree_node import MCTSNode


class AdvantageComputer:
    """Compute advantages for tree nodes: simple or BranchGRPO (path-weighted + depth norm)."""

    @staticmethod
    def compute_advantages(
        root: MCTSNode,
        leaves: List[MCTSNode],
        rewards: List[float],
        mode: str = "branchgrpo",
        advantage_clip: float = 2.0,
    ) -> None:
        """Assign rewards to leaves, then run backprop. Mutates node.advantage (and fused_reward).
        advantage_clip: clamp each advantage to [-advantage_clip, +advantage_clip] (D-014)."""
        if len(leaves) != len(rewards):
            raise ValueError("leaves and rewards length mismatch")
        for leaf, r in zip(leaves, rewards):
            leaf.reward = r
        if mode == "simple":
            AdvantageComputer._backprop_simple(root, leaves, advantage_clip)
        elif mode == "branchgrpo":
            AdvantageComputer._backprop_branchgrpo(root, leaves, advantage_clip)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def _backprop_simple(
        root: MCTSNode, leaves: List[MCTSNode], advantage_clip: float = 2.0
    ) -> None:
        """Leaf advantage = reward - mean(rewards); internal = mean(children)."""
        mean_r = np.mean([leaf.reward for leaf in leaves])
        for leaf in leaves:
            a = (leaf.reward or 0.0) - mean_r
            leaf.advantage = float(np.clip(a, -advantage_clip, advantage_clip))

        def _backprop(node: MCTSNode) -> float:
            if node.is_leaf() and node.advantage is not None:
                return node.advantage
            if not node.children:
                return 0.0
            child_advs = [_backprop(c) for c in node.children]
            a = float(np.mean(child_advs))
            node.advantage = float(np.clip(a, -advantage_clip, advantage_clip))
            return node.advantage

        _backprop(root)

    @staticmethod
    def _backprop_branchgrpo(
        root: MCTSNode, leaves: List[MCTSNode], advantage_clip: float = 2.0
    ) -> None:
        """Path-weighted reward fusion, then depth-wise z-score normalization (BranchGRPO).
        Advantages are clipped to [-advantage_clip, +advantage_clip] for stability (D-014)."""
        AdvantageComputer._fuse_rewards_path_weighted(root)
        nodes_by_depth: Dict[int, List[MCTSNode]] = {}
        AdvantageComputer._collect_by_depth(root, nodes_by_depth, 0)
        for depth, nodes in nodes_by_depth.items():
            fused = [n.fused_reward for n in nodes if n.fused_reward is not None]
            if not fused:
                continue
            mean_d = float(np.mean(fused))
            std_d = float(np.std(fused)) + 1e-8
            for n in nodes:
                if n.fused_reward is not None:
                    a = (n.fused_reward - mean_d) / std_d
                    n.advantage = float(np.clip(a, -advantage_clip, advantage_clip))

    @staticmethod
    def _fuse_rewards_path_weighted(node: MCTSNode) -> float:
        """Bottom-up: leaf fused_reward = reward; internal = sum over children of sampling_prob * child.fused_reward."""
        if node.is_leaf():
            node.fused_reward = node.reward if node.reward is not None else 0.0
            return node.fused_reward
        for c in node.children:
            AdvantageComputer._fuse_rewards_path_weighted(c)
        fused = 0.0
        for c in node.children:
            p = getattr(c, "sampling_prob", 1.0)
            if c.fused_reward is not None:
                fused += p * c.fused_reward
        node.fused_reward = fused
        return fused

    @staticmethod
    def _collect_by_depth(
        node: MCTSNode,
        depth_dict: Dict[int, List[MCTSNode]],
        depth: int,
    ) -> None:
        if depth not in depth_dict:
            depth_dict[depth] = []
        depth_dict[depth].append(node)
        for c in node.children:
            AdvantageComputer._collect_by_depth(c, depth_dict, depth + 1)
