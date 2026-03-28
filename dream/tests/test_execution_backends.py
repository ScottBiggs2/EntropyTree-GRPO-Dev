"""Tests for dream.src.execution_backends."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dream.src.execution_backends import (
    ContainerBackend,
    ExecutionBackend,
    SubprocessBackend,
    make_backend,
)


class TestSubprocessBackend:
    def test_is_execution_backend(self):
        b = SubprocessBackend()
        assert isinstance(b, ExecutionBackend)

    def test_args_expected_pass(self):
        b = SubprocessBackend(project_root=Path(__file__).resolve().parents[2])
        score = b.run_tests(
            code="def add(a, b): return a + b",
            function_name="add",
            tests=[[1, 2, 3], [0, 0, 0]],
            test_format="args_expected",
        )
        assert score == 1.0

    def test_args_expected_fail(self):
        b = SubprocessBackend(project_root=Path(__file__).resolve().parents[2])
        score = b.run_tests(
            code="def add(a, b): return a - b",
            function_name="add",
            tests=[[1, 2, 3]],
            test_format="args_expected",
        )
        assert score == 0.0

    def test_assertion_pass(self):
        b = SubprocessBackend(project_root=Path(__file__).resolve().parents[2])
        score = b.run_tests(
            code="def add(a, b): return a + b",
            function_name="add",
            tests=["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
            test_format="assertion",
        )
        assert score == 1.0

    def test_assertion_fail(self):
        b = SubprocessBackend(project_root=Path(__file__).resolve().parents[2])
        score = b.run_tests(
            code="def add(a, b): return a - b",
            function_name="add",
            tests=["assert add(1, 2) == 3"],
            test_format="assertion",
        )
        assert score == 0.0

    def test_empty_code(self):
        b = SubprocessBackend()
        assert b.run_tests(code="", function_name="f", tests=[[1, 2]]) == 0.0

    def test_empty_tests(self):
        b = SubprocessBackend()
        assert b.run_tests(code="def f(): pass", function_name="f", tests=[]) == 0.0


class TestContainerBackendDocker:
    """Tests that actually invoke Docker. Skipped if Docker is not available."""

    @pytest.fixture(autouse=True)
    def _require_docker(self):
        if shutil.which("docker") is None:
            pytest.skip("docker not on PATH")
        result = subprocess.run(
            ["docker", "ps"], capture_output=True, timeout=5
        )
        if result.returncode != 0:
            pytest.skip("docker daemon not running")

    def test_assertion_pass(self):
        b = ContainerBackend(image="dream-sandbox:latest", runtime="docker")
        score = b.run_tests(
            code="def add(a, b): return a + b",
            function_name="add",
            tests=["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
            test_format="assertion",
        )
        assert score == 1.0

    def test_assertion_fail(self):
        b = ContainerBackend(image="dream-sandbox:latest", runtime="docker")
        score = b.run_tests(
            code="def add(a, b): return a - b",
            function_name="add",
            tests=["assert add(1, 2) == 3"],
            test_format="assertion",
        )
        assert score == 0.0

    def test_args_expected_pass(self):
        b = ContainerBackend(image="dream-sandbox:latest", runtime="docker")
        score = b.run_tests(
            code="def add(a, b): return a + b",
            function_name="add",
            tests=[[1, 2, 3], [0, 0, 0]],
            test_format="args_expected",
        )
        assert score == 1.0

    def test_empty_code(self):
        b = ContainerBackend(image="dream-sandbox:latest", runtime="docker")
        assert b.run_tests(code="", function_name="f", tests=["assert True"]) == 0.0

    def test_partial_pass(self):
        b = ContainerBackend(image="dream-sandbox:latest", runtime="docker")
        score = b.run_tests(
            code="def add(a, b): return a + b",
            function_name="add",
            tests=["assert add(1, 2) == 3", "assert add(1, 2) == 99"],
            test_format="assertion",
        )
        assert score == pytest.approx(0.5)


class TestMakeBackend:
    def test_subprocess(self):
        b = make_backend("subprocess")
        assert isinstance(b, SubprocessBackend)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            make_backend("unknown_thing")

    def test_docker_without_binary(self):
        with mock.patch("shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="docker"):
                make_backend("docker")

    def test_container_autodetect_sif(self):
        with mock.patch("shutil.which", return_value="/usr/bin/apptainer"):
            b = make_backend("container", image="/path/to/sandbox.sif")
            assert isinstance(b, ContainerBackend)
            assert b.runtime == "apptainer"
