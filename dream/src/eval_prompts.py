"""DiffuCoder-aligned prompt templates for EvalPlus (paper Tables 5–6).

Dream Instruct uses Qwen-style chat markers. DiffuCoder manually builds the full
string through the assistant prefill ending in `` ```python\\n`` so the model
continues inside a fenced block — we mirror that instead of ``apply_chat_template``
with ``add_generation_prompt=True`` (which omits the prefill).

See ``dream/docs/EVAL_PROTOCOL.md`` for the training vs eval distinction.
"""

from __future__ import annotations

import ast
import re
from typing import Final

from dream.src.formatting import strip_special_tokens

# Qwen / Dream turn markers (DiffuCoder Appendix B.2)
_IM_START: Final = "<|im_start|>"
_EOT: Final = "<|im_end|>"


def build_humaneval_prompt(prompt: str) -> str:
    """Full prompt string from system through the assistant `` ```python\\n`` prefill.

    HumanEval (Table 5): user message includes
    "Please complete the following problem:" and a fenced copy of ``prompt``.
    ``prompt`` is EvalPlus's HumanEval ``prompt`` field (function stub), which we
    take from ``CodeTask.starter_code`` in our JSONL.
    """
    body = (prompt or "").rstrip()
    return (
        f"{_IM_START}system\n"
        "You are a helpful assistant."
        f"{_EOT}\n"
        f"{_IM_START}user\n"
        "Please complete the following problem:\n"
        "```\n"
        f"{body}\n"
        "```\n"
        f"{_EOT}\n"
        f"{_IM_START}assistant\n"
        "Here is the code to solve this problem:\n"
        "```python\n"
    )


def build_mbpp_prompt(prompt: str) -> str:
    """Same as :func:`build_humaneval_prompt` but without the HumanEval-only line (Table 6)."""
    body = (prompt or "").strip()
    return (
        f"{_IM_START}system\n"
        "You are a helpful assistant."
        f"{_EOT}\n"
        f"{_IM_START}user\n"
        "```\n"
        f"{body}\n"
        "```\n"
        f"{_EOT}\n"
        f"{_IM_START}assistant\n"
        "Here is the code to solve this problem:\n"
        "```python\n"
    )


def extract_diffucoder_completion(generated_suffix: str) -> str:
    """Extract executable Python from new tokens after the `` ```python`` prefill.

    Aligns with DiffuCoder's ``extract_code`` idea: prefer a
    `` ```python ... ``` `` block. Handles:

    - clean continuation (body only, no closing fence),
    - closing `` ``` `` present,
    - stray ``<|im_end|>`` / ``<|dlm_pad|>`` (via :func:`dream.src.formatting.strip_special_tokens`).
    """
    text = strip_special_tokens(generated_suffix or "")
    if not text:
        return ""

    m = re.search(r"```python\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip("\n\r")

    # No closing fence: strip a duplicated opening ```python line if the model echoed it
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```python"):
        text = "\n".join(lines[1:])
        text = strip_special_tokens(text)

    text = text.rstrip()
    if text.endswith("```"):
        idx = text.rfind("```")
        if idx != -1:
            text = text[:idx].rstrip()

    return text.strip("\n\r")


def normalize_humaneval_evalplus_completion(completion: str, entry_point: str) -> str:
    """Make `completion` suitable for EvalPlus ``prompt + completion`` merge.

    EvalPlus evaluates ``solution = problem[\"prompt\"] + completion``. If the model
    emits a **full** ``def entry_point(...): ...`` inside the fence, concatenation
    duplicates the stub from ``prompt`` and every task fails. When we detect a
    single top-level function with the expected name, emit **body-only** lines
    (4-space indented) to follow the stub in the official HumanEval ``prompt``.
    """
    if not (completion or "").strip() or not (entry_point or "").strip():
        return completion
    text = completion.strip()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return completion
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return completion
    fn = tree.body[0]
    if fn.name != entry_point:
        return completion
    out_lines: list[str] = []
    for stmt in fn.body:
        block = ast.unparse(stmt)
        for line in block.splitlines():
            out_lines.append(f"    {line}" if line.strip() else line)
    return ("\n".join(out_lines) + "\n") if out_lines else ""
