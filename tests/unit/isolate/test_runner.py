from __future__ import annotations

import gc
import os
import signal
import subprocess
import sys
import time
import zipfile
from contextlib import suppress
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from physicalai.isolate import Environment, ExternalInterpreter, PreparedEnvironment, SyncProfile, run, start
from physicalai.isolate._runner import _BOOTSTRAP

posix_only = pytest.mark.skipif(os.name != "posix", reason="worker process groups are POSIX-only")


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


class _FakeChild:
    pid = 123
    stdin = stdout = stderr = None

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.communications: list[tuple[bytes | None, float | None]] = []

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.returncode = 0
        return 0

    def communicate(self, input: bytes | None = None, timeout: float | None = None) -> tuple[bytes, bytes]:
        self.communications.append((input, timeout))
        self.returncode = 0
        return b"out", b"err"


def _fake_popen(
    monkeypatch: pytest.MonkeyPatch,
    child: _FakeChild,
) -> list[tuple[list[str], dict[str, Any]]]:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_popen(command: list[str], **kwargs: Any) -> _FakeChild:
        calls.append((command, kwargs))
        return child

    monkeypatch.setattr("physicalai.isolate._runner.subprocess.Popen", fake_popen)
    # The fake pid must never reach a real process group; os.killpg is POSIX-only.
    monkeypatch.setattr("physicalai.isolate._runner.os.killpg", lambda pid, sig: None, raising=False)
    return calls


def _launch(prepared: PreparedEnvironment, entrypoint: Path, kwargs: dict[str, Any]) -> list[str]:
    lifeline = str(kwargs["pass_fds"][0]) if kwargs["pass_fds"] else "-"
    return [str(prepared.python), "-c", _BOOTSTRAP, lifeline, str(entrypoint)]


def _process_options() -> dict[str, Any]:
    return {"bufsize": -1, "shell": False, "process_group": 0 if os.name == "posix" else None}


def test_run_passes_the_entrypoint_and_arguments_as_argv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path / "prepared")
    entrypoint = tmp_path / "worker with spaces.py"
    entrypoint.write_text("pass\n", encoding="utf-8")
    child = _FakeChild()
    calls = _fake_popen(monkeypatch, child)

    result = run(
        prepared,
        entrypoint,
        args=("--label", "two words"),
        cwd=tmp_path,
        env={"WORKER_MODE": "test"},
        capture_output=True,
        timeout=5,
    )

    [(command, kwargs)] = calls
    assert command == [*_launch(prepared, entrypoint, kwargs), "--label", "two words"]
    assert len(kwargs.pop("pass_fds")) == (1 if os.name == "posix" else 0)
    assert kwargs == {
        "cwd": tmp_path,
        "env": {"WORKER_MODE": "test", **_activated(prepared)},
        "stdin": None,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        **_process_options(),
    }
    assert child.communications == [(None, 5)]
    assert (result.args, result.returncode, result.stdout, result.stderr) == (command, 0, b"out", b"err")
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
    calls = _fake_popen(monkeypatch, _FakeChild())

    session = start(prepared, entrypoint, args=("--once",), env={"SESSION": "yes"})

    assert session.pid == 123
    assert session.wait() == 0
    [(command, kwargs)] = calls
    assert command == [*_launch(prepared, entrypoint, kwargs), "--once"]
    assert len(kwargs.pop("pass_fds")) == (1 if os.name == "posix" else 0)
    assert kwargs == {
        "cwd": None,
        "env": {"SESSION": "yes", **_activated(prepared)},
        "stdin": None,
        "stdout": None,
        "stderr": None,
        **_process_options(),
    }


def test_run_removes_host_python_settings_from_an_explicit_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path / "prepared")
    entrypoint = tmp_path / "worker.py"
    entrypoint.write_text("pass\n", encoding="utf-8")
    calls = _fake_popen(monkeypatch, _FakeChild())

    run(
        prepared,
        entrypoint,
        env={"PYTHONPATH": "/host", "VIRTUAL_ENV": "/host/venv", "PATH": "/usr/bin", "KEEP": "1"},
    )

    assert [kwargs["env"] for _, kwargs in calls] == [{"KEEP": "1", **_activated(prepared, "/usr/bin")}]


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


_SPAWNING_WORKER = """\
import os
import subprocess
import sys
import time
from pathlib import Path
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
Path(sys.argv[1]).write_text(f"{os.getpid()} {child.pid}\\n", encoding="utf-8")
time.sleep(60)
"""

_HOST = """\
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from physicalai.isolate import PreparedEnvironment, SyncProfile, start
prepared = PreparedEnvironment(path=Path(sys.argv[1]), key="b" * 64, profile=SyncProfile())
pid_file = Path(sys.argv[3])
session = start(prepared, sys.argv[2], args=(str(pid_file),), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
while not pid_file.exists():
    time.sleep(0.05)
if sys.argv[4:] == ["after-worker-exit"]:
    # Reap the worker without telling the session, leaving only its subprocess behind.
    pid = int(pid_file.read_text(encoding="utf-8").split()[0])
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
os.kill(os.getpid(), signal.SIGKILL)
"""


def _read_pids(path: Path, timeout: float = 10.0) -> list[int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and (text := path.read_text(encoding="utf-8")).endswith("\n"):
            return [int(pid) for pid in text.split()]
        time.sleep(0.05)
    msg = f"The worker did not write {path}."
    raise AssertionError(msg)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with suppress(OSError):
        # Zombies waiting for an unrelated parent to reap them count as stopped.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    return True


def _survivors(pids: list[int], timeout: float = 5.0) -> list[int]:
    deadline = time.monotonic() + timeout
    while (alive := [pid for pid in pids if _alive(pid)]) and time.monotonic() < deadline:
        time.sleep(0.05)
    return alive


def _kill(pids: list[int]) -> None:
    for pid in pids:
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


@posix_only
def test_workers_and_their_subprocesses_exit_when_the_host_is_killed(tmp_path: Path) -> None:
    prepared = _linked_prepared(tmp_path / "prepared")
    worker = tmp_path / "worker.py"
    worker.write_text(_SPAWNING_WORKER, encoding="utf-8")
    pid_file = tmp_path / "pids"

    host = subprocess.run(
        [sys.executable, "-c", _HOST, str(prepared.path), str(worker), str(pid_file)],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert host.returncode == -signal.SIGKILL, host.stderr
    pids = _read_pids(pid_file)

    try:
        assert _survivors(pids) == []
    finally:
        _kill(pids)


@posix_only
def test_subprocesses_exit_when_the_host_is_killed_after_their_worker_exited(tmp_path: Path) -> None:
    prepared = _linked_prepared(tmp_path / "prepared")
    worker = tmp_path / "worker.py"
    worker.write_text(_SPAWNING_WORKER, encoding="utf-8")
    pid_file = tmp_path / "pids"

    host = subprocess.run(
        [sys.executable, "-c", _HOST, str(prepared.path), str(worker), str(pid_file), "after-worker-exit"],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert host.returncode == -signal.SIGKILL, host.stderr
    pids = _read_pids(pid_file)

    try:
        assert _survivors(pids) == []
    finally:
        _kill(pids)


_EXITING_WORKER = """\
import subprocess
import sys
from pathlib import Path
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
Path(sys.argv[1]).write_text(f"{child.pid}\\n", encoding="utf-8")
"""


@posix_only
def test_waiting_for_a_worker_stops_the_subprocesses_it_left_behind(tmp_path: Path) -> None:
    prepared = _linked_prepared(tmp_path / "prepared")
    worker = tmp_path / "worker.py"
    worker.write_text(_EXITING_WORKER, encoding="utf-8")
    pid_file = tmp_path / "pids"
    session = start(prepared, worker, args=(str(pid_file),))

    assert session.wait(timeout=10) == 0
    pids = _read_pids(pid_file)

    try:
        assert _survivors(pids) == []
    finally:
        _kill(pids)
        session.close()


@posix_only
def test_a_dropped_session_stops_its_worker(tmp_path: Path) -> None:
    prepared = _linked_prepared(tmp_path / "prepared")
    worker = tmp_path / "worker.py"
    worker.write_text(_SPAWNING_WORKER, encoding="utf-8")
    pid_file = tmp_path / "pids"
    session = start(prepared, worker, args=(str(pid_file),))
    pids = _read_pids(pid_file)
    process = session._process

    try:
        del session
        gc.collect()
        assert process.wait(timeout=10) == -signal.SIGKILL
        assert _survivors(pids) == []
    finally:
        _kill(pids)


def test_close_ignores_unflushed_input_for_a_worker_that_already_exited(tmp_path: Path) -> None:
    prepared = _linked_prepared(tmp_path / "prepared")
    worker = tmp_path / "worker.py"
    worker.write_text("", encoding="utf-8")
    session = start(prepared, worker, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    session.wait(timeout=10)
    assert session.stdin is not None
    session.stdin.write(b"x" * 16)

    session.close()

    assert session.stdout is not None
    assert session.stdout.closed


def test_start_rejects_bad_arguments_before_opening_a_lifeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path / "prepared")
    entrypoint = tmp_path / "worker.py"
    entrypoint.write_text("pass\n", encoding="utf-8")
    pipes: list[tuple[int, int]] = []
    real_pipe = os.pipe

    def tracked_pipe() -> tuple[int, int]:
        pipes.append(real_pipe())
        return pipes[-1]

    monkeypatch.setattr("physicalai.isolate._runner.os.pipe", tracked_pipe)

    with pytest.raises(TypeError, match="sequence of strings"):
        start(prepared, entrypoint, args="--flag")
    prepared.python.unlink()
    with pytest.raises(FileNotFoundError, match="no Python interpreter"):
        start(prepared, entrypoint)

    assert pipes == []


def test_start_rejects_pass_fds_on_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prepared = _prepared(tmp_path / "prepared")
    entrypoint = tmp_path / "worker.py"
    entrypoint.write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr("physicalai.isolate._runner._POSIX", False)

    def unexpected_popen(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("Popen must not run when pass_fds is unsupported.")

    monkeypatch.setattr("physicalai.isolate._runner.subprocess.Popen", unexpected_popen)

    with pytest.raises(ValueError, match="only supported on POSIX"):
        start(prepared, entrypoint, pass_fds=(3,))


def test_folder_entrypoints_can_come_from_a_zip_resource(tmp_path: Path) -> None:
    prepared = _linked_prepared(tmp_path / "prepared")
    archive = tmp_path / "package.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("worker/__main__.py", "import helper\nhelper.report()\n")
        bundle.writestr("worker/helper.py", "def report():\n    print('from zip')\n")

    result = run(prepared, zipfile.Path(archive, "worker/"), capture_output=True)

    assert result.stdout.strip() == b"from zip"


@posix_only
def test_close_stops_the_subprocesses_of_a_worker(tmp_path: Path) -> None:
    prepared = _linked_prepared(tmp_path / "prepared")
    worker = tmp_path / "worker.py"
    worker.write_text(_SPAWNING_WORKER, encoding="utf-8")
    pid_file = tmp_path / "pids"
    session = start(prepared, worker, args=(str(pid_file),))
    pids = _read_pids(pid_file)

    try:
        session.close(timeout=2)
        assert _survivors(pids) == []
    finally:
        _kill(pids)


@posix_only
def test_run_timeout_stops_the_subprocesses_of_a_worker(tmp_path: Path) -> None:
    prepared = _linked_prepared(tmp_path / "prepared")
    worker = tmp_path / "worker.py"
    worker.write_text(_SPAWNING_WORKER, encoding="utf-8")
    pid_file = tmp_path / "pids"

    with pytest.raises(subprocess.TimeoutExpired):
        run(prepared, worker, args=(str(pid_file),), timeout=2)
    pids = _read_pids(pid_file)

    try:
        assert _survivors(pids) == []
    finally:
        _kill(pids)


def test_close_lets_a_worker_exit_when_its_stdin_closes(tmp_path: Path) -> None:
    prepared = _linked_prepared(tmp_path / "prepared")
    worker = tmp_path / "worker.py"
    worker.write_text("import sys\nsys.stdin.read()\n", encoding="utf-8")
    session = start(prepared, worker, stdin=subprocess.PIPE)

    session.close(timeout=10)

    assert session.returncode == 0


@posix_only
def test_close_kills_a_worker_that_ignores_stdin_and_sigterm(tmp_path: Path) -> None:
    prepared = _linked_prepared(tmp_path / "prepared")
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import signal\nimport time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    session = start(prepared, worker, stdout=subprocess.PIPE)
    assert session.stdout is not None
    session.stdout.readline()

    session.close(timeout=0.5)

    assert session.returncode == -signal.SIGKILL


@posix_only
def test_ctrl_c_in_the_host_is_forwarded_to_the_worker(tmp_path: Path) -> None:
    prepared = _linked_prepared(tmp_path / "prepared")
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import signal\nimport sys\nimport time\n"
        "signal.signal(signal.SIGINT, lambda *_: sys.exit(0))\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    session = start(prepared, worker, stdout=subprocess.PIPE)
    assert session.stdout is not None
    session.stdout.readline()

    with pytest.raises(KeyboardInterrupt), session:
        raise KeyboardInterrupt

    assert session.returncode == 0


@posix_only
def test_released_session_does_not_signal_process_group_again(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path / "prepared")
    entrypoint = tmp_path / "worker.py"
    entrypoint.write_text("pass\n", encoding="utf-8")
    _fake_popen(monkeypatch, _FakeChild())
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr("physicalai.isolate._runner.os.killpg", lambda pid, sig: signals.append((pid, sig)))

    session = start(prepared, entrypoint)
    assert session.wait() == 0
    session.terminate()
    session.kill()
    session.close()

    assert signals == []


@posix_only
def test_worker_survives_when_host_has_closed_standard_descriptor(tmp_path: Path) -> None:
    prepared = _linked_prepared(tmp_path / "prepared")
    worker = tmp_path / "worker.py"
    worker.write_text("print('ok')\n", encoding="utf-8")
    host = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "from physicalai.isolate import PreparedEnvironment, SyncProfile, run\n"
        "os.close(0)\n"
        "prepared = PreparedEnvironment(path=Path(sys.argv[1]), key='b' * 64, profile=SyncProfile())\n"
        "result = run(prepared, sys.argv[2], capture_output=True, timeout=10)\n"
        "sys.stdout.buffer.write(result.stdout)\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", host, str(prepared.path), str(worker)],
        capture_output=True,
        timeout=30,
        check=True,
    )

    assert completed.stdout == b"ok\n"


@posix_only
def test_second_ctrl_c_during_exit_still_kills_worker_and_closes_streams(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _linked_prepared(tmp_path / "prepared")
    worker = tmp_path / "worker.py"
    worker.write_text(_SPAWNING_WORKER, encoding="utf-8")
    pid_file = tmp_path / "pids"
    session = start(prepared, worker, args=(str(pid_file),), stdout=subprocess.PIPE)
    pids = _read_pids(pid_file)
    original_exited = session._exited_within
    calls = 0

    def interrupt_grace_wait(timeout: float) -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt
        return original_exited(timeout)

    monkeypatch.setattr(session, "_exited_within", interrupt_grace_wait)

    try:
        with pytest.raises(KeyboardInterrupt), session:
            raise KeyboardInterrupt
        assert session.stdout is not None
        assert session.stdout.closed
        assert _survivors(pids) == []
    finally:
        _kill(pids)


@posix_only
def test_second_ctrl_c_during_close_still_kills_worker_and_closes_streams(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _linked_prepared(tmp_path / "prepared")
    worker = tmp_path / "worker.py"
    worker.write_text(_SPAWNING_WORKER, encoding="utf-8")
    pid_file = tmp_path / "pids"
    session = start(prepared, worker, args=(str(pid_file),), stdout=subprocess.PIPE)
    pids = _read_pids(pid_file)

    def interrupt_grace_wait(timeout: float) -> bool:
        del timeout
        raise KeyboardInterrupt

    monkeypatch.setattr(session, "_exited_within", interrupt_grace_wait)

    try:
        with pytest.raises(KeyboardInterrupt):
            session.close(timeout=2)
        assert session.stdout is not None
        assert session.stdout.closed
        assert _survivors(pids) == []
    finally:
        _kill(pids)


@posix_only
def test_start_forwards_pass_fds_without_supervisor_holding_them_open(tmp_path: Path) -> None:
    prepared = _linked_prepared(tmp_path / "prepared")
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import os, sys\n"
        "fd = int(sys.argv[1])\n"
        "os.write(fd, b'hello-ipc')\n"
        "os.close(fd)\n",
        encoding="utf-8",
    )
    read_fd, write_fd = os.pipe()
    try:
        session = start(prepared, worker, args=(str(write_fd),), pass_fds=(write_fd,), bufsize=0)
        os.close(write_fd)
        write_fd = -1
        # Reading until EOF before session.wait() verifies the supervisor did not keep write_fd open.
        chunks: list[bytes] = []
        while chunk := os.read(read_fd, 64):
            chunks.append(chunk)
        assert b"".join(chunks) == b"hello-ipc"
        assert session.wait(timeout=10) == 0
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


def test_external_interpreters_keep_the_host_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The caller manages an external interpreter, so isolate leaves its environment alone."""
    interpreter = ExternalInterpreter(Path(sys.executable))
    host_packages = tmp_path / "host-packages"
    host_packages.mkdir()
    (host_packages / "host_only_module.py").write_text("", encoding="utf-8")
    entrypoint = tmp_path / "worker.py"
    entrypoint.write_text(
        "import importlib.util\n"
        "import os\n"
        "print(importlib.util.find_spec('host_only_module') is not None)\n"
        "print(os.environ['VIRTUAL_ENV'])\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(host_packages))
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "host-venv"))

    result = run(interpreter, entrypoint, capture_output=True)
    session = start(interpreter, entrypoint, stdout=subprocess.PIPE)
    stdout, _ = session.communicate(timeout=10)
    explicit = run(
        interpreter,
        entrypoint,
        env={**os.environ, "VIRTUAL_ENV": str(tmp_path / "explicit-venv")},
        capture_output=True,
    )

    assert result.stdout == f"True\n{tmp_path / 'host-venv'}\n".encode()
    assert stdout == result.stdout
    assert explicit.stdout == f"True\n{tmp_path / 'explicit-venv'}\n".encode()
    assert session.args[0] == sys.executable


def test_external_interpreters_run_entrypoints_like_prepared_environments(tmp_path: Path) -> None:
    """Entrypoints still run through the bootstrap: files next to a script are not importable."""
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

    result = run(ExternalInterpreter(Path(sys.executable)), entrypoint, args=("a",), capture_output=True)

    script = entrypoint.resolve()
    assert result.stdout == f"True {script} {[str(script), 'a']}\n".encode()


def test_external_interpreters_must_exist(tmp_path: Path) -> None:
    entrypoint = tmp_path / "worker.py"
    entrypoint.write_text("", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="External interpreter has no Python interpreter"):
        start(ExternalInterpreter(tmp_path / "missing" / "python"), entrypoint)


def test_external_interpreter_paths_expand_the_home_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert ExternalInterpreter("~/venv/bin/python").python == tmp_path / "venv" / "bin" / "python"


def test_external_interpreters_accept_strings_and_reject_an_empty_path() -> None:
    interpreter = ExternalInterpreter(sys.executable)

    assert interpreter.python == Path(sys.executable).absolute()
    assert interpreter == ExternalInterpreter(Path(sys.executable))
    with pytest.raises(ValueError, match="cannot be empty"):
        ExternalInterpreter("")


@posix_only
def test_relative_external_interpreters_do_not_follow_the_session_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A relative interpreter is fixed against the host's directory, and its venv symlink is kept."""
    host = tmp_path / "host"
    (host / "venv" / "bin").mkdir(parents=True)
    (host / "venv" / "bin" / "python").symlink_to(sys.executable)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    entrypoint = tmp_path / "worker.py"
    entrypoint.write_text("import sys\nprint(sys.executable)\n", encoding="utf-8")
    monkeypatch.chdir(host)

    interpreter = ExternalInterpreter(Path("venv/bin/python"))
    result = run(interpreter, entrypoint, cwd=elsewhere, capture_output=True)

    assert interpreter.python == host / "venv" / "bin" / "python"
    assert result.stdout.decode().strip() == str(host / "venv" / "bin" / "python")


@posix_only
def test_close_stops_the_subprocesses_of_an_external_interpreter_worker(tmp_path: Path) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text(_SPAWNING_WORKER, encoding="utf-8")
    pid_file = tmp_path / "pids"
    session = start(ExternalInterpreter(Path(sys.executable)), worker, args=(str(pid_file),))
    pids = _read_pids(pid_file)

    try:
        session.close(timeout=2)
        assert _survivors(pids) == []
    finally:
        _kill(pids)
