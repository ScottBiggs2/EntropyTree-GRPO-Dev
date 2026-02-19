"""
Phase 7: Run one training step (overfit sanity check).
Usage: python scripts/single_step_train.py
Requires model download unless run with mock.
"""
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MCTSConfig
from src.utils import load_model_and_tokenizer
from src.entropy import EntropyComputer
from src.time_weight import TimeWeighter
from src.rewards import SyntaxReward
from src.advantages import AdvantageComputer
from src.loss import WeightedGRPOLoss
from src.trainer import EntropyMCTSTrainer


def main():
    config = MCTSConfig(
        max_tree_nodes=8,
        branch_width=2,
        steps_per_expansion=8,
        max_new_tokens=32,
        learning_rate=1e-5,
    )
    print("Loading model (may download)...")
    model, tokenizer = load_model_and_tokenizer(config)
    optimizer = __import__("torch").optim.AdamW(
        model.parameters(), lr=config.learning_rate
    )
    ec = EntropyComputer()
    tw = TimeWeighter(config.total_denoising_steps)
    loss_fn = WeightedGRPOLoss(config, ec, tw, tokenizer.mask_token_id)
    trainer = EntropyMCTSTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        reward_fn=SyntaxReward(),
        advantage_computer=AdvantageComputer(),
        loss_computer=loss_fn,
        optimizer=optimizer,
    )

    prompt = "def fibonacci(n):"
    print(f"One training step for prompt: {prompt!r}")
    t0 = time.perf_counter()
    metrics = trainer.train_step(prompt)
    t1 = time.perf_counter()
    print(f"Time: {t1 - t0:.2f}s")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("Done. Run again to see if loss changes (overfit check).")


if __name__ == "__main__":
    main()
