"""
Phase 8.5: Sandbox runner for execution-lite reward.
Reads code from stdin, tests config from JSON file (argv[1]).
Exec's code, runs tests (call function with args, compare to expected), prints fraction passed in [0,1].
No network/files; intended to be run with subprocess + timeout.
"""
import json
import sys


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
    code = sys.stdin.read()
    func_name = config.get("function_name")
    tests = config.get("tests", [])
    if not func_name or not tests:
        print("0.0")
        return
    try:
        glbs = {}
        exec(code, glbs)
        fn = glbs.get(func_name)
        if fn is None:
            print("0.0")
            return
    except Exception:
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
