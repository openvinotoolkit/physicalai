from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from physicalai.isolate import Environment, PreparedEnvironment, SyncProfile, run, start


def _prepared(path: Path) -> PreparedEnvironment:
    python = path / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.write_bytes(b"interpreter")
    return PreparedEnvironment(path=path, key="a" * 64, profile=SyncProfile())


def test_run_passes_the_entrypoint_and_arguments_as_argv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path / "prepared")
    entrypoint = tmp_path / "worker with spaces.py"
    entrypoint.write_text("pass\n", encoding="utf-8")
    calls: list[tuple[list[str], dict[str, Any]]] = []
    completed = subprocess.CompletedProcess([], 0, stdout=b"done", stderr=b"")

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append((command, kwargs))
        return completed

    monkeypatch.setattr("physicalai.isolate._runner.subprocess.run", fake_run)

    result = run(
        prepared,
        entrypoint,
        args=("--label", "two words"),
        cwd=tmp_path,
        env={"WORKER_MODE": "test"},
        capture_output=True,
        timeout=5,
    )

    assert result is completed
    assert calls == [
        (
            [str(prepared.python), str(entrypoint), "--label", "two words"],
            {
                "check": True,
                "timeout": 5,
                "cwd": tmp_path,
                "env": {"WORKER_MODE": "test"},
                "input": None,
                "capture_output": True,
                "shell": False,
            },
        ),
    ]
    assert {field.name for field in fields(Environment)} == {"source", "profile"}


def test_run_and_start_execute_entrypoints_with_the_prepared_interpreter(tmp_path: Path) -> None:
    prepared_root = tmp_path / "prepared"
    python = prepared_root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    prepared = PreparedEnvironment(path=prepared_root, key="b" * 64, profile=SyncProfile())
    entrypoint = tmp_path / "worker.py"
    entrypoint.write_text("import sys\nprint(sys.argv[1])\n", encoding="utf-8")

    result = run(prepared, entrypoint, args=("run-entrypoint",), capture_output=True)
    session = start(prepared, entrypoint, args=("start-entrypoint",), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = session.communicate(timeout=10)

    assert result.stdout == b"run-entrypoint\n"
    assert session.returncode == 0
    assert stdout == b"start-entrypoint\n"
    assert stderr == b""


def test_start_returns_a_session_for_the_entrypoint_launched_now(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path / "prepared")
    entrypoint = tmp_path / "session.py"
    entrypoint.write_text("pass\n", encoding="utf-8")
    calls: list[tuple[list[str], dict[str, Any]]] = []

    class Child:
        args: list[str] = []
        pid = 123
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.returncode = 0
            return 0

        def communicate(
            self,
            input: bytes | None = None,
            timeout: float | None = None,
        ) -> tuple[bytes | None, bytes | None]:
            del input, timeout
            self.returncode = 0
            return b"out", b"err"

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        def send_signal(self, sig: int) -> None:
            self.returncode = -sig

    child = Child()

    def fake_popen(command: list[str], **kwargs: Any) -> Child:
        calls.append((command, kwargs))
        return child

    monkeypatch.setattr("physicalai.isolate._runner.subprocess.Popen", fake_popen)

    session = start(prepared, entrypoint, args=("--once",), env={"SESSION": "yes"})

    assert session.pid == 123
    assert session.wait() == 0
    assert calls == [
        (
            [str(prepared.python), str(entrypoint), "--once"],
            {
                "cwd": None,
                "env": {"SESSION": "yes"},
                "stdin": None,
                "stdout": None,
                "stderr": None,
                "shell": False,
            },
        ),
    ]
