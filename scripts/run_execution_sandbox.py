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


def _merge_executable_code(prompt, completion, func_name):
    """Return the first full code string that defines func_name, or None."""
    candidates = []
    if prompt:
        candidates.append((prompt + "\n" + completion).strip())
    candidates.append(completion.strip())
    cleaned = re.sub(r"```python\s*\n?", "", completion)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()
    if cleaned != completion.strip():
        if prompt:
            candidates.append((prompt + "\n" + cleaned).strip())
        candidates.append(cleaned)
    blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", completion, re.DOTALL)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        candidates.append(block)
        if prompt:
            candidates.append((prompt + "\n" + block).strip())
    pattern = rf"(def\s+{re.escape(func_name)}\s*\([^)]*\):\s*\n(?:[ \t]+[^\n]*\n?)+)"
    for text in [completion, cleaned]:
        matches = re.findall(pattern, text)
        for match in matches:
            candidates.append(match.strip())
    for code in candidates:
        if _try_exec(code, func_name) is not None:
            return code
    return None


def _extract_function(prompt, completion, func_name):
    """Try multiple strategies to obtain the target function. Returns callable or None."""
    code = _merge_executable_code(prompt, completion, func_name)
    if code is None:
        return None
    return _try_exec(code, func_name)


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
    test_format = config.get("test_format") or "args_expected"

    if not func_name or not tests:
        print("0.0")
        return

    if test_format == "assertion":
        code = _merge_executable_code(prompt, completion, func_name)
        if code is None:
            print("0.0")
            return
        ns = {}
        try:
            exec(code, ns)
        except Exception:
            print("0.0")
            return
        passed = 0
        for assertion in tests:
            if not isinstance(assertion, str):
                continue
            try:
                exec(assertion.strip(), ns)
                passed += 1
            except Exception:
                pass
        frac = passed / len(tests) if tests else 0.0
        print(frac)
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
