"""
Phase 7.5: Run all real-model verification steps in order and report pass/fail.
Exits with code 0 only if all four steps succeed.
Usage: from repo root, PYTHONPATH=. python scripts/verify_real_model.py
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ("validate_model", "scripts/validate_model.py", "Model load + forward + generate"),
    ("entropy_profile", "scripts/entropy_profile.py", "Entropy at step 0 and 32/64/128"),
    ("tree_viz", "scripts/tree_viz.py", "Tree build + structure print"),
    ("single_step_train", "scripts/single_step_train.py", "One training step"),
]


def main():
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    failed = []
    print("Real-model verification (Phase 7.5). Each step's output appears below its [RUN] line.\n")
    for name, script, desc in STEPS:
        path = ROOT / script
        if not path.exists():
            print(f"[FAIL] {name}: {path} not found")
            failed.append(name)
            continue
        print(f"[RUN] {name}: {desc} ...")
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(ROOT),
            env=env,
            timeout=600,
        )
        if result.returncode == 0:
            print(f"[PASS] {name}")
        else:
            print(f"[FAIL] {name} (exit code {result.returncode})")
            failed.append(name)
    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        sys.exit(1)
    print("\nAll real-model verification steps passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
