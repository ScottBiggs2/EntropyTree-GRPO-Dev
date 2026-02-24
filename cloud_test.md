Baseline: 

[baseline] epoch 0 loss=-0.22165163358052573 avg_reward=0.22499999999999998 max_reward=0.3 tree_nodes=0.0 tree_leaves=4.0 avg_entropy=0.0 epoch=0 method=bas
eline wall_sec=19.25180222839117
[baseline] epoch 1 loss=-0.022623737653096516 avg_reward=0.25 max_reward=0.3 tree_nodes=0.0 tree_leaves=4.0 avg_entropy=0.0 epoch=1 method=baseline wall_sec
=11.43008653447032
[baseline] Saved final.pt and config.json to checkpoints/baseline_grpo/dllm_grpo_baseline_test


EntropyTree Test: 

[entropy diagnostic] depth=1 node: n_masked=16, token_entropy at masked: min=0.088237 mean=0.589878 max=1.269172
[entropy_mcts] epoch 0 loss=-0.038151068543946266 avg_reward=0.3 max_reward=0.3 tree_nodes=10.0 tree_leaves=5.0 avg_entropy=0.49408312638600665 n_transitions
=9.0 epoch=0 method=entropy_mcts
[entropy_mcts] epoch 1 loss=-0.13779956164459387 avg_reward=0.10999999999999999 max_reward=0.15 tree_nodes=10.0 tree_leaves=5.0 avg_entropy=0.350354300604926
17 n_transitions=9.0 epoch=1 method=entropy_mcts
[entropy_mcts] Saved final.pt and config.json to checkpoints/entropy_mcts_grpo/2_epoch_retest

---

I notice that the loss is sometimes negative, or flips between positive and negative.

---

## How to interpret tree_nodes vs tree_leaves

- **tree_nodes** = total number of nodes in the search tree (root + every expanded child). So it’s the size of the whole tree.
- **tree_leaves** = number of *final* leaves: the completion nodes we assign rewards to and use for the GRPO loss (after expansion stops and `_complete_leaves` runs).

So **leaves ≤ nodes** always. With branching, many nodes are internal (they have children); only the “tip” nodes at the end of expansion become the leaves we complete and score.

Your EntropyTree run: **10 nodes, 5 leaves** is expected. It means the tree has 10 vertices total; 5 of them are internal (root + 4 others that were expanded), and 5 are final leaves (the completions used for reward and advantages). So you get 5 completion trajectories per prompt, coming from a small tree of 10 nodes. Baseline correctly reports **0 nodes, 4 leaves** because it doesn’t build a tree—it just samples 4 trajectories (num_baseline_samples=4); the “leaves” count is the number of completions.