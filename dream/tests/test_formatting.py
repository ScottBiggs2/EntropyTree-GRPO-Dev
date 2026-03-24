from dream.src.formatting import (
    build_code_task_prompt,
    extract_python_code,
    normalize_completion_for_reward,
)


def test_build_code_task_prompt_includes_starter_code_block():
    prompt = build_code_task_prompt(
        "Complete the function.",
        "def foo(x):",
    )
    assert "Complete the function." in prompt
    assert "```python" in prompt
    assert "def foo(x):" in prompt


def test_extract_python_code_from_fenced_block():
    text = "Here is the code:\n```python\ndef foo(x):\n    return x + 1\n```"
    code = extract_python_code(text)
    assert code.startswith("def foo(x):")
    assert "return x + 1" in code


def test_extract_python_code_from_assistant_style_output():
    text = (
        "Here is the code to solve this problem:\n"
        "def foo(x):\n"
        "    return x + 1\n"
        "<|im_end|>"
    )
    code = extract_python_code(text)
    assert code.startswith("def foo(x):")
    assert "<|im_end|>" not in code


def test_normalize_completion_prefers_entry_point_definition():
    text = "```python\ndef helper():\n    pass\n\ndef foo(x):\n    return x\n```"
    code = normalize_completion_for_reward(text, entry_point="foo")
    assert code.startswith("def foo(x):")


def test_normalize_completion_keeps_body_only_when_no_definition():
    text = "    if n <= 1:\n        return n\n    return n * factorial(n - 1)"
    code = normalize_completion_for_reward(text, entry_point="factorial")
    assert "return n * factorial" in code
