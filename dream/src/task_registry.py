"""Task registry utilities for Dream code-GRPO experiments.

This layer sits above the current prompt-string based trainers and gives us a
stable task abstraction for:

- training/dev splits,
- canonical user-facing prompts,
- execution-time starter code / entry points,
- legacy execution-lite compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union


@dataclass(frozen=True)
class CodeTask:
    task_id: str
    source: str
    split: str
    instruction: str
    starter_code: str
    entry_point: str
    # args_expected: list of [arg..., expected]; assertion: list of assertion strings
    tests: Union[List[List[Any]], List[str]]
    canonical_prompt: str
    language: str = "python"
    prompt_type: str = "chat_code"
    test_format: str = "args_expected"
    metadata: Dict[str, Any] = field(default_factory=dict)


def task_field(task: Any, name: str, default: Any = None) -> Any:
    """Read a field from a :class:`CodeTask` or a legacy mapping row.

    Prefer this over ``getattr(...) or dict.get(...)``: :class:`CodeTask` has no
    ``.get``, and fields such as ``starter_code`` may be ``""`` where ``or`` would
    incorrectly fall through to dict-style access.
    """
    if isinstance(task, Mapping):
        return task.get(name, default)
    return getattr(task, name, default)


def _coerce_tests(raw_tests: Any) -> List[List[Any]]:
    if not isinstance(raw_tests, list):
        raise ValueError("tests must be a list")
    out: List[List[Any]] = []
    for item in raw_tests:
        if isinstance(item, list):
            row = item
        elif isinstance(item, tuple):
            row = list(item)
        else:
            raise ValueError("each test case must be a list/tuple")
        if len(row) < 2:
            raise ValueError("each test case must contain args and expected value")
        out.append(row)
    return out


def _coerce_assertion_tests(raw_tests: Any) -> List[str]:
    if not isinstance(raw_tests, list) or not raw_tests:
        raise ValueError("tests must be a non-empty list of assertion strings")
    out: List[str] = []
    for item in raw_tests:
        if not isinstance(item, str):
            raise ValueError("each assertion test must be a string")
        s = item.strip()
        if not s.lower().startswith("assert"):
            raise ValueError("each assertion test must start with 'assert'")
        out.append(s)
    return out


def _default_instruction(starter_code: str, entry_point: str) -> str:
    if starter_code.strip():
        return "Complete the following Python function."
    return f"Write a Python function named `{entry_point}`."


def _build_prompt(instruction: str, starter_code: str, language: str) -> str:
    from dream.src.formatting import build_code_task_prompt

    return build_code_task_prompt(
        instruction=instruction,
        starter_code=starter_code,
        language=language,
    )


def _task_from_standard_row(
    row: Mapping[str, Any],
    *,
    source_path: Path,
    index: int,
) -> CodeTask:
    task_id = str(row.get("task_id") or f"{source_path.stem}_{index:04d}")
    source = str(row.get("source") or source_path.stem)
    split = str(row.get("split") or "train")
    starter_code = str(row.get("starter_code") or "").rstrip()
    entry_point = str(row.get("entry_point") or "").strip()
    if not entry_point:
        raise ValueError(f"{task_id}: missing entry_point")
    instruction = str(
        row.get("instruction") or _default_instruction(starter_code, entry_point)
    ).strip()
    language = str(row.get("language") or "python")
    prompt_type = str(row.get("prompt_type") or "chat_code")
    test_format = str(row.get("test_format") or "args_expected").strip()
    if test_format == "assertion":
        tests = _coerce_assertion_tests(row.get("tests"))
    else:
        tests = _coerce_tests(row.get("tests"))
    canonical_prompt = str(
        row.get("canonical_prompt") or _build_prompt(instruction, starter_code, language)
    ).strip()
    metadata = dict(row.get("metadata") or {})
    return CodeTask(
        task_id=task_id,
        source=source,
        split=split,
        instruction=instruction,
        starter_code=starter_code,
        entry_point=entry_point,
        tests=tests,
        canonical_prompt=canonical_prompt,
        language=language,
        prompt_type=prompt_type,
        test_format=test_format,
        metadata=metadata,
    )


def _task_from_legacy_execution_lite(
    row: Mapping[str, Any],
    *,
    source_path: Path,
    index: int,
) -> CodeTask:
    starter_code = str(row.get("prompt") or "").rstrip()
    entry_point = str(row.get("function_name") or "").strip()
    if not starter_code or not entry_point:
        raise ValueError(
            f"{source_path}:{index}: legacy execution-lite row missing prompt/function_name"
        )
    split = "dev" if bool(row.get("eval", False)) else "train"
    source = str(row.get("source") or "execution_lite")
    instruction = str(
        row.get("instruction") or _default_instruction(starter_code, entry_point)
    ).strip()
    canonical_prompt = str(
        row.get("canonical_prompt")
        or _build_prompt(instruction, starter_code, "python")
    ).strip()
    metadata = dict(row.get("metadata") or {})
    metadata.setdefault("legacy_execution_lite", True)
    return CodeTask(
        task_id=str(row.get("task_id") or f"{source}_{index:04d}"),
        source=source,
        split=split,
        instruction=instruction,
        starter_code=starter_code,
        entry_point=entry_point,
        tests=_coerce_tests(row.get("tests")),
        canonical_prompt=canonical_prompt,
        language="python",
        prompt_type="chat_code",
        test_format="args_expected",
        metadata=metadata,
    )


def _row_to_task(row: Mapping[str, Any], *, source_path: Path, index: int) -> CodeTask:
    if {"prompt", "function_name", "tests"}.issubset(row.keys()):
        return _task_from_legacy_execution_lite(row, source_path=source_path, index=index)
    required = {"entry_point", "tests"}
    if not required.issubset(row.keys()):
        raise ValueError(
            f"{source_path}:{index}: task row missing required fields {sorted(required)}"
        )
    return _task_from_standard_row(row, source_path=source_path, index=index)


def _load_jsonl_rows(path: Path) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    with path.open() as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i}: invalid JSONL row: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{i}: each JSONL row must be an object")
            rows.append(row)
    return rows


def _load_json_rows(path: Path) -> List[Mapping[str, Any]]:
    with path.open() as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"{path}: JSON task file must contain a list of objects")
    rows: List[Mapping[str, Any]] = []
    for i, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{i}: each JSON row must be an object")
        rows.append(row)
    return rows


def load_code_tasks(path: str | Path) -> List[CodeTask]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        rows = _load_jsonl_rows(path)
    elif path.suffix == ".json":
        rows = _load_json_rows(path)
    else:
        raise ValueError(f"Unsupported task file type: {path.suffix}")
    tasks = [_row_to_task(row, source_path=path, index=i) for i, row in enumerate(rows)]
    if not tasks:
        raise ValueError(f"{path}: no tasks found")
    return tasks


def filter_code_tasks(tasks: Iterable[CodeTask], split: str) -> List[CodeTask]:
    split = split.strip().lower()
    if split in ("", "all", "*"):
        return list(tasks)
    return [task for task in tasks if task.split.lower() == split]


def load_code_tasks_for_split(path: str | Path, split: str) -> List[CodeTask]:
    return filter_code_tasks(load_code_tasks(path), split)


def build_prompt_lookup(tasks: Iterable[CodeTask]) -> Dict[str, CodeTask]:
    lookup: Dict[str, CodeTask] = {}
    for task in tasks:
        for alias in (task.canonical_prompt.strip(), task.starter_code.strip()):
            if alias:
                lookup[alias] = task
    return lookup


def infer_default_split(tasks: Iterable[CodeTask], phase: str) -> str:
    task_list = list(tasks)
    splits = {task.split.lower() for task in task_list}
    if phase == "initial_eval":
        if "dev" in splits:
            return "dev"
        if "eval" in splits:
            return "eval"
    if "train" in splits:
        return "train"
    if "dev" in splits:
        return "dev"
    return "all"


def export_execution_lite_rows(tasks: Iterable[CodeTask]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        rows.append(
            {
                "task_id": task.task_id,
                "source": task.source,
                "prompt": task.starter_code,
                "canonical_prompt": task.canonical_prompt,
                "instruction": task.instruction,
                "function_name": task.entry_point,
                "tests": task.tests,
                "test_format": task.test_format,
                "eval": task.split.lower() in {"dev", "eval", "test"},
                "metadata": task.metadata,
            }
        )
    return rows


def save_execution_lite_json(tasks: Iterable[CodeTask], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(export_execution_lite_rows(tasks), f, indent=2)
