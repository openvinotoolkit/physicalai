# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import select
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from physicalai.cli import robot as robot_module
from physicalai.robot.errors import RobotTransportError
from physicalai.robot.transport._lock import acquire_locks
from physicalai.robot.transport._owner_worker import OwnerExitReason, OwnerResult

from tests.unit.robot.transport.conftest import requires_zenoh


def _serve_cfg(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "name": "left-arm",
        "robot_class": "tests.unit.robot.transport.fake.FakeRobot",
        "robot_kwargs": {"port": "/dev/fake0"},
        "allow_remote": False,
        "rate_hz": 100.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_serve_runs_owner_in_foreground_with_persistent_timeout(capsys: object) -> None:
    captured: dict[str, object] = {}

    def _run_owner(config: object, shutdown: threading.Event, *, ready: object) -> OwnerResult:
        captured["config"] = config
        captured["shutdown"] = shutdown
        assert callable(ready)
        ready()
        shutdown.set()
        return OwnerResult(OwnerExitReason.SHUTDOWN, 0)

    with patch.object(robot_module, "run_owner", side_effect=_run_owner) as run:
        assert robot_module.serve(_serve_cfg()) == 0

    run.assert_called_once()
    config = captured["config"]
    assert config.idle_timeout is None  # type: ignore[attr-defined]
    assert config.name == "left-arm"  # type: ignore[attr-defined]


def test_signal_requests_runtime_shutdown() -> None:
    def _run_owner(_config: object, shutdown: threading.Event, *, ready: object) -> OwnerResult:  # noqa: ARG001
        signal.raise_signal(signal.SIGTERM)
        assert shutdown.is_set()
        return OwnerResult(OwnerExitReason.SHUTDOWN, 0)

    with patch.object(robot_module, "run_owner", side_effect=_run_owner):
        assert robot_module.serve(_serve_cfg()) == 0


def test_expected_startup_error_is_concise(capsys: object) -> None:
    error = RobotTransportError("name is already owned", phase="name_lock_contention")
    with patch.object(robot_module, "run_owner", side_effect=error):
        assert robot_module.serve(_serve_cfg()) == 1

    stderr = capsys.readouterr().err  # type: ignore[attr-defined]
    assert "name_lock_contention" in stderr
    assert "Traceback" not in stderr


def test_invalid_config_fails_before_runtime(capsys: object) -> None:
    with patch.object(robot_module, "run_owner") as run:
        assert robot_module.serve(_serve_cfg(name="left/arm")) == 1
    run.assert_not_called()
    assert "Invalid robot configuration" in capsys.readouterr().err  # type: ignore[attr-defined]


def test_discovery_json_is_sorted_and_clean(capsys: object) -> None:
    records = [
        {"name": "z-arm", "host": "b", "robot_class": "untrusted.Z", "num_joints": 7},
        {"name": "a-arm", "host": "a", "robot_class": "untrusted.A", "num_joints": 6},
    ]
    cfg = SimpleNamespace(timeout=1.0, allow_remote=False, json=True)
    with patch.object(robot_module, "discover_robots", return_value=records):
        assert robot_module.discover(cfg) == 0

    output = capsys.readouterr()  # type: ignore[attr-defined]
    assert output.err == ""
    assert [record["name"] for record in json.loads(output.out)] == ["a-arm", "z-arm"]


def test_empty_discovery_json_is_array(capsys: object) -> None:
    cfg = SimpleNamespace(timeout=1.0, allow_remote=False, json=True)
    with patch.object(robot_module, "discover_robots", return_value=[]):
        assert robot_module.discover(cfg) == 0
    assert capsys.readouterr().out == "[]\n"  # type: ignore[attr-defined]


def test_parser_does_not_expose_idle_timeout() -> None:
    parser = robot_module.build_parser()
    cfg = parser.parse_args(
        ["serve", "--name", "left-arm", "--robot_class", "pkg.mod.Robot"],
    )
    assert not hasattr(cfg.serve, "idle_timeout")


def test_discover_rejects_invalid_timeout(capsys: object) -> None:
    cfg = SimpleNamespace(timeout=float("nan"), allow_remote=False, json=False)
    with patch.object(robot_module, "discover_robots") as discover:
        assert robot_module.discover(cfg) == 1
    discover.assert_not_called()
    assert "timeout must be finite" in capsys.readouterr().err  # type: ignore[attr-defined]


@requires_zenoh
def test_serve_process_sigterm_disconnects_and_releases_locks(tmp_path: Path) -> None:
    name = f"cli-{uuid.uuid4().hex[:8]}"
    port = f"/dev/{name}"
    device_id = f"fake:{port}"
    marker = tmp_path / "disconnected"
    process = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "physicalai.cli.main",
            "robot",
            "serve",
            "--name",
            name,
            "--robot_class",
            "tests.unit.robot.transport.fake.FakeRobot",
            "--robot_kwargs.port",
            port,
            "--robot_kwargs.disconnect_marker",
            str(marker),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stderr is not None
    deadline = time.monotonic() + 20.0
    ready = False
    stderr_lines: list[str] = []
    try:
        while time.monotonic() < deadline:
            readable, _, _ = select.select([process.stderr], [], [], 0.5)
            if readable:
                line = process.stderr.readline()
                stderr_lines.append(line)
                if "Serving robot" in line:
                    ready = True
                    break
            if process.poll() is not None:
                break
        assert ready, "".join(stderr_lines)

        process.terminate()
        assert process.wait(timeout=10.0) == 0
        assert marker.exists()

        locks = acquire_locks(name, [device_id])
        locks.release_all()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)