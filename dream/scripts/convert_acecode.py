#!/usr/bin/env python3
"""Convert TIGER-Lab/AceCode-89K (hard split) into Dream CodeTask JSONL files.

Follows DiffuCoder ``recipes/process_data.py`` filtering (bottom 20%% mean acc,
top 60%% std among qwen_coder_2.5 + llama3_instruct) and optional question
transform. See dream/PLAN_03_ENVIRONMENT_SCALEUP.md Step 2.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dream.src.formatting import build_code_task_prompt

CONSIDER_MODELS = ("qwen_coder_2.5", "llama3_instruct")

ENTRY_ASSERT_RE = re.compile(
    r"^\s*assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE
)
DEF_ENTRY_RE = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)


def _starter_line_from_completion(completion: str, entry_point: str) -> str:
    pattern = rf"(?m)^def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*:"
    m = re.search(pattern, completion)
    if not m:
        return ""
    line_end = completion.find("\n", m.start())
    if line_end == -1:
        return completion[m.start() :].strip()
    return completion[m.start() : line_end].strip()


def extract_entry_point(item: Dict[str, Any]) -> Optional[str]:
    tests = item.get("test_cases") or []
    for t in tests:
        if isinstance(t, str):
            m = ENTRY_ASSERT_RE.match(t.strip())
            if m:
                return m.group(1)
    inferences = item.get("inferences") or []
    if not inferences:
        return None
    best = max(inferences, key=lambda x: float(x.get("pass_rate") or 0.0))
    comp = best.get("completion") or ""
    m = DEF_ENTRY_RE.search(comp)
    if m:
        return m.group(1)
    return None


def validate_assertion_tests(test_cases: Sequence[Any]) -> bool:
    if not test_cases:
        return False
    for t in test_cases:
        if not isinstance(t, str):
            return False
        if not t.strip().lower().startswith("assert"):
            return False
    return True


def _best_completion(item: Dict[str, Any]) -> Optional[str]:
    inferences = item.get("inferences") or []
    if not inferences:
        return None
    best = max(inferences, key=lambda x: float(x.get("pass_rate") or 0.0))
    c = best.get("completion")
    return str(c) if c is not None else None


def transform_question_format(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    q = item.get("question") or ""
    if "you" in q.lower():
        return [item]
    inferences = item.get("inferences") or []
    if not inferences:
        return [item]
    best_completion = max(inferences, key=lambda x: float(x.get("pass_rate") or 0.0))[
        "completion"
    ]
    if not isinstance(best_completion, str):
        return [item]
    parts = best_completion.split(":\n", 1)
    func_def = parts[0].strip()
    if not func_def.startswith("def "):
        return [item]
    func_def = func_def[4:]
    new_item = dict(item)
    new_item["question"] = (
        f"""Please complete the following problem:
```
def {func_def}:
    \"\"\"
    {q}
    \"\"\"
```
"""
    )
    new_item["_transform_suffix"] = "qstruct"
    return [item, new_item]


def _attach_accs(item: Dict[str, Any]) -> Dict[str, Any]:
    inference = [
        x
        for x in (item.get("inferences") or [])
        if x.get("model_name") in CONSIDER_MODELS
    ]
    accs = [float(x.get("pass_rate") or 0.0) for x in inference]
    out = dict(item)
    out["accs"] = accs
    return out


def filter_difficulty(dataset: Any, difficulty: str) -> Any:
    import numpy as np

    avg_accs = [np.mean(x) for x in dataset["accs"]]
    std_accs = [np.std(x) for x in dataset["accs"]]
    if difficulty == "hard":
        split_acc = np.percentile(avg_accs, 20)
        dataset = dataset.filter(
            lambda x: float(np.mean(x["accs"])) <= float(split_acc),
            desc="Filter low mean accuracy",
        )
        split_std = np.percentile(std_accs, 40)
        dataset = dataset.filter(
            lambda x: float(np.std(x["accs"])) >= float(split_std),
            desc="Filter high std",
        )
    elif difficulty == "medium":
        lower_acc = np.percentile(avg_accs, 25)
        upper_acc = np.percentile(avg_accs, 75)
        dataset = dataset.filter(
            lambda x: lower_acc <= float(np.mean(x["accs"])) <= upper_acc,
            desc="Filter medium mean accuracy",
        )
    elif difficulty == "easy":
        split_acc = np.percentile(avg_accs, 75)
        dataset = dataset.filter(
            lambda x: float(np.mean(x["accs"])) >= float(split_acc),
            desc="Filter high mean accuracy",
        )
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    return dataset


def item_to_task_rows(
    item: Dict[str, Any],
    *,
    source_label: str,
    difficulty: str,
) -> List[Dict[str, Any]]:
    work = dict(item)
    suffix = work.pop("_transform_suffix", None)
    base_id = str(work.get("id") or work.get("task_id") or "unknown")
    task_id = f"{base_id}__{suffix}" if suffix else base_id

    test_cases = list(work.get("test_cases") or [])
    if not validate_assertion_tests(test_cases):
        return []

    entry_point = extract_entry_point(work)
    if not entry_point:
        return []

    ref = _best_completion(work)
    if not ref:
        return []

    starter_code = _starter_line_from_completion(ref, entry_point)
    question = str(work.get("question") or "").strip()
    instruction = question if question else f"Implement `{entry_point}` correctly."
    canonical_prompt = build_code_task_prompt(
        instruction=instruction,
        starter_code=starter_code,
        language="python",
    )

    meta: Dict[str, Any] = {
        "reference_completion": ref,
        "acecode_source": work.get("source"),
        "difficulty": difficulty,
    }

    row: Dict[str, Any] = {
        "task_id": task_id,
        "source": source_label,
        "split": "train",
        "instruction": instruction,
        "starter_code": starter_code,
        "entry_point": entry_point,
        "tests": test_cases,
        "test_format": "assertion",
        "canonical_prompt": canonical_prompt,
        "metadata": meta,
    }
    return [row]


def main() -> int:
    p = argparse.ArgumentParser(description="Convert AceCode-89K to Dream CodeTask JSONL")
    p.add_argument(
        "--dataset",
        type=str,
        default="TIGER-Lab/AceCode-89K",
        help="HuggingFace dataset id or local path (see --local-path)",
    )
    p.add_argument(
        "--local-path",
        type=str,
        default="",
        help="If set, load via datasets.load_from_disk(this) instead of the Hub",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(_repo_root / "dream" / "data"),
    )
    p.add_argument(
        "--difficulty",
        type=str,
        default="hard",
        choices=("easy", "medium", "hard"),
    )
    p.add_argument("--dev-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--skip-transform",
        action="store_true",
        help="Do not duplicate rows with structured question format (DiffuCoder transform)",
    )
    args = p.parse_args()

    try:
        import numpy as np  # noqa: F401
        from datasets import load_dataset, load_from_disk
    except ImportError as e:
        print(
            "ERROR: need `datasets` and `numpy`. Install with: pip install datasets numpy",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"acecode_{args.difficulty}_train.jsonl"
    dev_path = output_dir / f"acecode_{args.difficulty}_dev.jsonl"

    if args.local_path:
        lp = Path(args.local_path)
        if not lp.exists():
            print(f"ERROR: local path not found: {lp}", file=sys.stderr)
            return 1
        ds = load_from_disk(str(lp))
        if hasattr(ds, "keys") and "train" in ds:
            ds = ds["train"]
    else:
        ds = load_dataset(args.dataset, split="train")

    print(f"Loaded {len(ds)} examples from {args.dataset or args.local_path}")

    ds = ds.map(_attach_accs, desc="Attach accs", num_proc=4)
    ds = ds.filter(lambda x: len(x["accs"]) >= 1, desc="Drop rows without model accs")

    if args.difficulty in ("easy", "medium", "hard"):
        ds = filter_difficulty(ds, args.difficulty)

    print(f"After difficulty filter: {len(ds)} examples")

    transformed: List[Dict[str, Any]] = []
    for item in ds:
        row = dict(item)
        row.pop("accs", None)
        if args.skip_transform:
            transformed.append(row)
        else:
            transformed.extend(transform_question_format(row))

    print(f"After question transform: {len(transformed)} rows")

    source_label = "acecode_89k"
    all_rows: List[Dict[str, Any]] = []
    skipped = 0
    for raw in transformed:
        item = dict(raw)
        rows = item_to_task_rows(
            item, source_label=source_label, difficulty=args.difficulty
        )
        if not rows:
            skipped += 1
            continue
        all_rows.extend(rows)

    print(f"Converted tasks: {len(all_rows)}, skipped (bad tests/entry): {skipped}")

    rng = random.Random(args.seed)
    order = list(range(len(all_rows)))
    rng.shuffle(order)
    n_dev = int(round(len(all_rows) * args.dev_frac))
    n_dev = min(max(n_dev, 0), len(all_rows))
    dev_set = set(order[:n_dev])

    def _write(path: Path, indices: Sequence[int]) -> None:
        with path.open("w") as f:
            for i in indices:
                f.write(json.dumps(all_rows[i], ensure_ascii=False) + "\n")

    train_indices = [i for i in range(len(all_rows)) if i not in dev_set]
    dev_indices = [i for i in range(len(all_rows)) if i in dev_set]

    for i in train_indices:
        all_rows[i] = dict(all_rows[i])
        all_rows[i]["split"] = "train"
    for i in dev_indices:
        all_rows[i] = dict(all_rows[i])
        all_rows[i]["split"] = "dev"

    _write(train_path, train_indices)
    _write(dev_path, dev_indices)

    print(f"Wrote {train_path} ({len(train_indices)} lines)")
    print(f"Wrote {dev_path} ({len(dev_indices)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
