"""
Phase 4: Visualize tree structure (depth, branching, entropy).
Run after model is available: python scripts/tree_viz.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MCTSConfig
from src.utils import load_model_and_tokenizer, get_device
from src.entropy import EntropyComputer
from src.tree_builder import EntropyGuidedTreeBuilder
from src.tree_node import MCTSNode


def count_nodes(root: MCTSNode) -> int:
    n = 1
    for c in root.children:
        n += count_nodes(c)
    return n


def tree_to_lines(node: MCTSNode, tokenizer, prefix: str = "", lines: list = None) -> list:
    """Build indented text lines for the tree (root, then children)."""
    if lines is None:
        lines = []
    ent = f", entropy={node.entropy:.6f}" if node.entropy is not None else ""
    lines.append(f"{prefix}step={node.step_index}, depth={node.depth}{ent}")
    for i, c in enumerate(node.children):
        lines.append(f"{prefix}  child[{i}]:")
        tree_to_lines(c, tokenizer, prefix + "    ", lines)
    return lines


def main():
    config = MCTSConfig(
        max_tree_nodes=10,
        branch_width=2,
        steps_per_expansion=16,
        max_new_tokens=64,
    )
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(config)
    model.eval()
    entropy_computer = EntropyComputer()
    builder = EntropyGuidedTreeBuilder(model, tokenizer, config, entropy_computer)

    prompt = "def fibonacci(n):"
    print(f"Building tree for prompt: {prompt!r}")
    root, leaves = builder.build_tree(prompt)

    n_nodes = count_nodes(root)
    print(f"Tree nodes: {n_nodes}, leaves: {len(leaves)}")
    print(f"Root: step_index={root.step_index}, masking_ratio={root.masking_ratio():.3f}, entropy={root.entropy}")
    for i, leaf in enumerate(leaves[:5]):
        preview = tokenizer.decode(leaf.state[root.prompt_len:root.prompt_len+32].tolist(), skip_special_tokens=True)
        print(f"  Leaf {i}: step_index={leaf.step_index}, preview={preview[:50]!r}...")
    if len(leaves) > 5:
        print(f"  ... and {len(leaves)-5} more leaves")

    # Write tree structure to a file so verification produces an artifact
    out_file = ROOT / "tree_structure.txt"
    with open(out_file, "w") as f:
        f.write(f"Prompt: {prompt!r}\n")
        f.write(f"Nodes: {n_nodes}, leaves: {len(leaves)}\n\n")
        for line in tree_to_lines(root, tokenizer):
            f.write(line + "\n")
    print(f"Tree structure written to {out_file}")

    print("Done.")


if __name__ == "__main__":
    main()
