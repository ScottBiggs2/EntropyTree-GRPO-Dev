"""Shared helpers for HumanEval / MBPP → Dream CodeTask JSONL (Step 3).

HumanEval tests use ``candidate`` as the solution callable; Dream's assertion
runner executes ``exec(assertion, ns)`` after ``exec(code, ns)``, so assertions
must reference the real ``entry_point`` name. We rewrite ``candidate`` →
``entry_point`` during conversion.
"""

from __future__ import annotations

import ast
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

from dream.src.formatting import build_code_task_prompt

_CANDIDATE_RE = re.compile(r"\bcandidate\b")


def iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    """Read JSONL or .jsonl.gz (one JSON object per line)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".gz" or str(path).endswith(".jsonl.gz"):
        f = gzip.open(path, "rt", encoding="utf-8")
    else:
        f = path.open(encoding="utf-8")
    try:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}: each row must be a JSON object")
            yield row
    finally:
        f.close()


def replace_candidate_with_entry_point(assertions: List[str], entry_point: str) -> List[str]:
    return [_CANDIDATE_RE.sub(entry_point, a) for a in assertions]


def extract_assertions_from_humaneval_test(test_code: str) -> List[str]:
    """Parse HumanEval-style ``test`` field into individual assertion strings."""
    if not (test_code and test_code.strip()):
        return []

    line_asserts: List[str] = []
    for line in test_code.splitlines():
        s = line.strip()
        if not s.startswith("assert"):
            continue
        if len(s) == 6:
            continue
        if s[6] not in " (":
            continue
        line_asserts.append(s)

    if line_asserts:
        return line_asserts

    return _assertions_from_check_ast(test_code)


def _assertions_from_check_ast(test_code: str) -> List[str]:
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            out: List[str] = []
            for stmt in node.body:
                if isinstance(stmt, ast.Assert):
                    out.append(ast.unparse(stmt).strip())
            return out
    return []


def extract_entry_point_from_code(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None


def starter_code_from_solution(code: str, entry_point: str) -> str:
    """MBPP reference ``code`` is a full solution; derive a stub for prompting."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            kw: Dict[str, Any] = {
                "name": node.name,
                "args": node.args,
                "body": [ast.Pass()],
                "decorator_list": node.decorator_list,
                "returns": node.returns,
            }
            if sys.version_info < (3, 12):
                kw["type_comment"] = getattr(node, "type_comment", None)
            else:
                kw["type_params"] = getattr(node, "type_params", [])
            stub = ast.FunctionDef(**kw)
            ast.fix_missing_locations(stub)
            return ast.unparse(stub).strip() + "\n"
    return ""


def humaneval_to_row(
    item: Mapping[str, Any],
    *,
    source: str = "humaneval_plus",
) -> Dict[str, Any]:
    task_id = str(item["task_id"])
    prompt = str(item.get("prompt") or "").rstrip()
    entry_point = str(item.get("entry_point") or "").strip()
    canonical_solution = str(item.get("canonical_solution") or "")
    test_code = str(item.get("test") or "")

    if not entry_point or not prompt:
        raise ValueError(f"{task_id}: missing entry_point or prompt")

    assertions = extract_assertions_from_humaneval_test(test_code)
    assertions = replace_candidate_with_entry_point(assertions, entry_point)
    if not assertions:
        raise ValueError(f"{task_id}: could not parse assertions from test block")

    instruction = "Complete the following Python function."
    canonical_prompt = build_code_task_prompt(
        instruction=instruction,
        starter_code=prompt,
        language="python",
    )
    return {
        "task_id": task_id,
        "source": source,
        "split": "eval",
        "instruction": instruction,
        "starter_code": prompt,
        "entry_point": entry_point,
        "tests": assertions,
        "test_format": "assertion",
        "canonical_prompt": canonical_prompt,
        "metadata": {
            "canonical_solution": canonical_solution,
            "eval_dataset": "humaneval",
        },
    }


def mbpp_to_row(
    item: Mapping[str, Any],
    *,
    source: str = "mbpp_plus",
) -> Dict[str, Any]:
    task_id = str(item["task_id"])
    text = str(item.get("text") or item.get("prompt") or "").strip()
    code = str(item.get("code") or item.get("canonical_solution") or "")

    entry_point = str(item.get("entry_point") or "").strip()
    if not entry_point:
        entry_point = extract_entry_point_from_code(code) or ""
    if not entry_point:
        raise ValueError(f"{task_id}: could not determine entry_point")

    test_list = item.get("test_list") or []
    assertions: List[str] = []

    if isinstance(test_list, list) and test_list:
        for t in test_list:
            if not isinstance(t, str):
                raise ValueError(f"{task_id}: test_list entries must be strings")
            s = t.strip()
            if not s.lower().startswith("assert"):
                raise ValueError(f"{task_id}: test must start with assert: {s[:80]!r}")
            assertions.append(s)
    else:
        test_code = str(item.get("test") or "")
        if test_code:
            assertions = extract_assertions_from_humaneval_test(test_code)

    if not assertions:
        raise ValueError(f"{task_id}: no assertions found (tried test_list and test)")

    assertions = replace_candidate_with_entry_point(assertions, entry_point)

    starter_code = starter_code_from_solution(code, entry_point)
    if not starter_code.strip():
        for line in code.splitlines():
            ls = line.strip()
            if ls.startswith(f"def {entry_point}"):
                starter_code = ls.split(":")[0] + ":\n    pass\n"
                break

    instruction = text if text else f"Implement `{entry_point}` correctly."
    canonical_prompt = build_code_task_prompt(
        instruction=instruction,
        starter_code=starter_code,
        language="python",
    )

    meta: Dict[str, Any] = {
        "canonical_solution": code,
        "eval_dataset": "mbpp",
    }
    setup = item.get("test_setup_code")
    if setup:
        meta["test_setup_code"] = setup

    return {
        "task_id": task_id,
        "source": source,
        "split": "eval",
        "instruction": instruction,
        "starter_code": starter_code,
        "entry_point": entry_point,
        "tests": assertions,
        "test_format": "assertion",
        "canonical_prompt": canonical_prompt,
        "metadata": meta,
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
