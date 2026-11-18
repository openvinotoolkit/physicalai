from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from physicalai.isolate import Environment, PreparedEnvironment, SyncProfile, run, start
from physicalai.isolate._runner import _BOOTSTRAP


def _prepared(path: Path) -> PreparedEnvironment:
    python = path / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.write_bytes(b"interpreter")
    return PreparedEnvironment(path=path, key="a" * 64, profile=SyncProfile())


def _linked_prepared(path: Path) -> PreparedEnvironment:
    python = path / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    return PreparedEnvironment(path=path, key="b" * 64, profile=SyncProfile())


def _activated(prepared: PreparedEnvironment, search_path: str = os.defpath) -> dict[str, str]:
    return {
        "PATH": os.pathsep.join((str(prepared.python.parent), search_path)),
        "VIRTUAL_ENV": str(prepared.venv),
    }


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
            [str(prepared.python), "-c", _BOOTSTRAP, str(entrypoint), "--label", "two words"],
            {
                "check": True,
                "timeout": 5,
                "cwd": tmp_path,
                "env": {"WORKER_MODE": "test", **_activated(prepared)},
                "input": None,
                "capture_output": True,
                "shell": False,
            },
        ),
    ]
    assert {field.name for field in fields(Environment)} == {"source", "profile"}


def test_run_and_start_execute_entrypoints_with_the_prepared_interpreter(tmp_path: Path) -> None:
    prepared = _linked_prepared(tmp_path / "prepared")
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
            [str(prepared.python), "-c", _BOOTSTRAP, str(entrypoint), "--once"],
            {
                "cwd": None,
                "env": {"SESSION": "yes", **_activated(prepared)},
                "stdin": None,
                "stdout": None,
                "stderr": None,
                "shell": False,
            },
        ),
    ]


def test_run_removes_host_python_settings_from_an_explicit_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path / "prepared")
    entrypoint = tmp_path / "worker.py"
    entrypoint.write_text("pass\n", encoding="utf-8")
    environments: list[dict[str, str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        environments.append(kwargs["env"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("physicalai.isolate._runner.subprocess.run", fake_run)

    run(
        prepared,
        entrypoint,
        env={"PYTHONPATH": "/host", "VIRTUAL_ENV": "/host/venv", "PATH": "/usr/bin", "KEEP": "1"},
    )

    assert environments == [{"KEEP": "1", **_activated(prepared, "/usr/bin")}]


def test_children_do_not_import_from_the_host_python_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _linked_prepared(tmp_path / "prepared")
    host_packages = tmp_path / "host-packages"
    host_packages.mkdir()
    (host_packages / "host_only_module.py").write_text("", encoding="utf-8")
    entrypoint = tmp_path / "worker.py"
    entrypoint.write_text(
        "import importlib.util\n"
        "import os\n"
        "print(importlib.util.find_spec('host_only_module') is None)\n"
        "print(os.environ['VIRTUAL_ENV'])\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(host_packages))
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "host-venv"))

    result = run(prepared, entrypoint, capture_output=True)
    session = start(prepared, entrypoint, stdout=subprocess.PIPE)
    stdout, _ = session.communicate(timeout=10)

    expected = f"True\n{prepared.venv}\n".encode()
    assert result.stdout == expected
    assert stdout == expected


def test_children_do_not_import_files_next_to_a_script_entrypoint(tmp_path: Path) -> None:
    prepared = _linked_prepared(tmp_path / "prepared")
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "runtime.py").write_text("", encoding="utf-8")
    entrypoint = plugin / "_worker.py"
    entrypoint.write_text(
        "import importlib.util\n"
        "import sys\n"
        "print(importlib.util.find_spec('runtime') is None, __file__, sys.argv)\n",
        encoding="utf-8",
    )

    result = run(prepared, entrypoint, args=("a",), capture_output=True)
    session = start(prepared, entrypoint, args=("a",), stdout=subprocess.PIPE)
    stdout, _ = session.communicate(timeout=10)

    script = entrypoint.resolve()
    expected = f"True {script} {[str(script), 'a']}\n".encode()
    assert result.stdout == expected
    assert stdout == expected


def test_folder_entrypoints_import_only_their_own_modules(tmp_path: Path) -> None:
    prepared = _linked_prepared(tmp_path / "prepared")
    plugin = tmp_path / "plugin"
    worker = plugin / "_worker"
    worker.mkdir(parents=True)
    (plugin / "runtime.py").write_text("", encoding="utf-8")
    (worker / "helper.py").write_text("VALUE = 'helper'\n", encoding="utf-8")
    (worker / "__main__.py").write_text(
        "import importlib.util\n"
        "import helper\n"
        "print(helper.VALUE, importlib.util.find_spec('runtime') is None)\n",
        encoding="utf-8",
    )

    result = run(prepared, worker, capture_output=True)

    assert result.stdout == b"helper True\n"


def test_children_do_not_import_from_the_working_directory(tmp_path: Path) -> None:
    prepared = _linked_prepared(tmp_path / "prepared")
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "runtime.py").write_text("", encoding="utf-8")
    entrypoint = tmp_path / "worker.py"
    entrypoint.write_text(
        "import importlib.util\nprint(importlib.util.find_spec('runtime') is None)\n",
        encoding="utf-8",
    )

    result = run(prepared, entrypoint, cwd=workdir, capture_output=True)

    assert result.stdout == b"True\n"


def test_entrypoints_can_use_spawned_multiprocessing_workers(tmp_path: Path) -> None:
    prepared = _linked_prepared(tmp_path / "prepared")
    entrypoint = tmp_path / "worker.py"
    entrypoint.write_text(
        "import multiprocessing\n"
        "\n"
        "def double(value):\n"
        "    return value * 2\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    with multiprocessing.get_context('spawn').Pool(1) as pool:\n"
        "        print(pool.map(double, [1, 2]))\n",
        encoding="utf-8",
    )

    result = run(prepared, entrypoint, capture_output=True, timeout=60)

    assert result.stdout == b"[2, 4]\n"


def test_entrypoints_must_be_a_file_or_a_folder_with_a_main_module(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path / "prepared")
    folder = tmp_path / "worker"
    folder.mkdir()

    with pytest.raises(FileNotFoundError, match="__main__.py"):
        run(prepared, folder)
