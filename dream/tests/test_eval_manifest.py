import json
from pathlib import Path

from dream.src.eval_manifest import build_eval_run_manifest, write_eval_run_manifest


def test_write_eval_run_manifest_roundtrip(tmp_path: Path):
    p = tmp_path / "m.json"
    write_eval_run_manifest(p, model="m1", steps="128", extra_flag=True)
    data = json.loads(p.read_text())
    assert data["model"] == "m1"
    assert data["steps"] == "128"
    assert data["extra_flag"] is True


def test_build_includes_python_version():
    m = build_eval_run_manifest(foo="bar")
    assert "python" in m
    assert m["foo"] == "bar"
