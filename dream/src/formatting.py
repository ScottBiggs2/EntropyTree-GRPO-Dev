"""Prompt formatting and code extraction helpers for Dream code-GRPO."""

from __future__ import annotations

import re
from typing import Optional


SPECIAL_TOKENS = (
    "<|im_end|>",
    "<|im_start|>",
    "<|dlm_pad|>",
    "</s>",
)


def strip_special_tokens(text: str) -> str:
    cleaned = text or ""
    for token in SPECIAL_TOKENS:
        cleaned = cleaned.replace(token, "")
    return cleaned.strip("\n\r")


def build_code_task_prompt(
    instruction: str,
    starter_code: str = "",
    language: str = "python",
) -> str:
    instruction = (instruction or "").strip()
    starter_code = (starter_code or "").rstrip()
    if starter_code:
        return (
            f"{instruction}\n\n"
            f"Write only the code needed to solve it.\n\n"
            f"```{language}\n{starter_code}\n```"
        ).strip()
    return instruction


def extract_python_code(text: str) -> str:
    text = strip_special_tokens(text)
    if not text:
        return ""

    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        for block in fenced:
            block = block.strip("\n\r")
            if block.strip():
                return block

    # If the assistant echoed prose before the code, cut to the first likely code token.
    code_start = re.search(
        r"(?m)^(def |class |from |import |@|if __name__ == ['\"]__main__['\"]:)",
        text,
    )
    if code_start:
        return text[code_start.start() :].strip("\n\r")

    return text.strip("\n\r")


def normalize_completion_for_reward(
    completion: str,
    *,
    entry_point: Optional[str] = None,
) -> str:
    code = extract_python_code(completion)
    if not code:
        return ""

    if entry_point:
        pattern = rf"(?m)^def\s+{re.escape(entry_point)}\s*\("
        match = re.search(pattern, code)
        if match:
            return code[match.start() :].strip("\n\r")

    return code.strip("\n\r")
