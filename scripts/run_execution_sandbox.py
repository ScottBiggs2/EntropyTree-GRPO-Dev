"""
Phase 8.5: Sandbox runner for execution-lite reward.
Reads raw completion from stdin, test config + prompt from JSON file (argv[1]).
Tries multiple strategies to extract the target function from the completion,
runs tests, prints fraction passed in [0,1].
No network/files; intended to be run with subprocess + timeout.
"""
import json
import re
import sys


def _try_exec(code, func_name):
    """Exec code and return the callable if func_name is defined, else None."""
    try:
        glbs = {}
        exec(code, glbs)
        fn = glbs.get(func_name)
        if fn is not None and callable(fn):
            return fn
    except Exception:
        pass
    return None


def _extract_function(prompt, completion, func_name):
    """Try multiple strategies to obtain the target function. Returns callable or None."""

    # Strategy 1: prompt + "\n" + completion (completion is just the function body)
    if prompt:
        fn = _try_exec((prompt + "\n" + completion).strip(), func_name)
        if fn is not None:
            return fn

    # Strategy 2: completion as-is (model included full function definition)
    fn = _try_exec(completion.strip(), func_name)
    if fn is not None:
        return fn

    # Strategy 3: strip markdown code fences then try both strategies again
    cleaned = re.sub(r'```python\s*\n?', '', completion)
    cleaned = re.sub(r'```\s*', '', cleaned).strip()
    if cleaned != completion.strip():
        if prompt:
            fn = _try_exec((prompt + "\n" + cleaned).strip(), func_name)
            if fn is not None:
                return fn
        fn = _try_exec(cleaned, func_name)
        if fn is not None:
            return fn

    # Strategy 4: extract individual markdown code blocks
    blocks = re.findall(r'```(?:python)?\s*\n?(.*?)```', completion, re.DOTALL)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        fn = _try_exec(block, func_name)
        if fn is not None:
            return fn
        if prompt:
            fn = _try_exec((prompt + "\n" + block).strip(), func_name)
            if fn is not None:
                return fn

    # Strategy 5: regex-extract the function definition from surrounding text
    pattern = rf'(def\s+{re.escape(func_name)}\s*\([^)]*\):\s*\n(?:[ \t]+[^\n]*\n?)+)'
    for text in [completion, cleaned]:
        matches = re.findall(pattern, text)
        for match in matches:
            fn = _try_exec(match.strip(), func_name)
            if fn is not None:
                return fn

    return None


def main():
    if len(sys.argv) != 2:
        print("0.0", file=sys.stderr)
        sys.exit(1)
    try:
        with open(sys.argv[1]) as f:
            config = json.load(f)
    except Exception:
        print("0.0", file=sys.stderr)
        sys.exit(1)

    completion = sys.stdin.read()
    prompt = config.get("prompt", "")
    func_name = config.get("function_name")
    tests = config.get("tests", [])

    if not func_name or not tests:
        print("0.0")
        return

    fn = _extract_function(prompt, completion, func_name)
    if fn is None:
        print("0.0")
        return

    passed = 0
    for item in tests:
        if len(item) < 2:
            continue
        args_list, expected = item[:-1], item[-1]
        try:
            result = fn(*args_list)
            if result == expected:
                passed += 1
        except Exception:
            pass
    frac = passed / len(tests) if tests else 0.0
    print(frac)


if __name__ == "__main__":
    main()
