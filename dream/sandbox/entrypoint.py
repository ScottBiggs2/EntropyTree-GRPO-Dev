#!/usr/bin/env python3
"""Run untrusted Python in-container: read JSON config (stdin or file), print pass fraction [0,1]."""

from __future__ import annotations

import json
import os
import signal
import sys
from typing import Any


def _default_timeout() -> float:
    return float(os.environ.get("DREAM_SANDBOX_TEST_TIMEOUT", "5"))


def _run_with_timeout(fn, timeout_sec: float):
    """Run fn() in the main thread; raise TimeoutError on expiry (ITIMER_REAL)."""

    def _handler(_signum, _frame):
        raise TimeoutError("sandbox test timed out")

    old = signal.signal(signal.SIGALRM, _handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, max(0.001, float(timeout_sec)))
        try:
            return fn()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
    finally:
        signal.signal(signal.SIGALRM, old)


def _fraction(passed: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return passed / total


def _run_assertion_mode(code: str, test_cases: list[str], timeout_sec: float) -> float:
    ns: dict[str, Any] = {}
    try:
        exec(code, ns, ns)
    except Exception:
        return 0.0

    passed = 0
    for assertion in test_cases:
        try:

            def run_one():
                exec(assertion, ns, ns)

            _run_with_timeout(run_one, timeout_sec)
            passed += 1
        except Exception:
            pass
    return _fraction(passed, len(test_cases))


def _run_args_expected_mode(
    code: str, function_name: str, tests: list[list[Any]], timeout_sec: float
) -> float:
    ns: dict[str, Any] = {}
    try:
        exec(code, ns, ns)
    except Exception:
        return 0.0

    fn = ns.get(function_name)
    if fn is None or not callable(fn):
        return 0.0

    passed = 0
    for item in tests:
        if not item:
            continue
        if len(item) < 2:
            # Match scripts/run_execution_sandbox.py: skip malformed rows
            continue
        args_list, expected = item[:-1], item[-1]

        def call():
            result = fn(*args_list)
            if result != expected:
                raise AssertionError("result mismatch")

        try:
            _run_with_timeout(call, timeout_sec)
            passed += 1
        except Exception:
            pass

    # Align with run_execution_sandbox: fraction over all listed tests (including skipped)
    total = len(tests)
    if total == 0:
        return 0.0
    return _fraction(passed, total)


def _load_config() -> dict[str, Any]:
    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return json.load(sys.stdin)


def main() -> None:
    timeout_sec = _default_timeout()
    try:
        cfg = _load_config()
    except Exception:
        print("0.0")
        return

    code = (cfg.get("code") or "").strip()
    if not code:
        print("0.0")
        return

    fmt = (cfg.get("test_format") or "").strip().lower()
    if not fmt:
        if cfg.get("test_cases") is not None:
            fmt = "assertion"
        elif cfg.get("tests") is not None:
            fmt = "args_expected"
        else:
            print("0.0")
            return

    if fmt == "assertion":
        cases = cfg.get("test_cases") or []
        if not isinstance(cases, list) or not cases:
            print("0.0")
            return
        out = _run_assertion_mode(code, [str(c) for c in cases], timeout_sec)
        print(out)
        return

    if fmt == "args_expected":
        function_name = cfg.get("function_name")
        tests = cfg.get("tests") or []
        if not function_name or not isinstance(tests, list):
            print("0.0")
            return
        out = _run_args_expected_mode(str(code), str(function_name), tests, timeout_sec)
        print(out)
        return

    print("0.0")


if __name__ == "__main__":
    main()
