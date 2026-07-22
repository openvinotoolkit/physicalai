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

from loguru import logger

from physicalai.cli import robot as robot_module
from physicalai.robot.errors import RobotTransportError
from physicalai.robot.transport import OwnerEvent, OwnerExitReason, OwnerResult
from physicalai.robot.transport._lock import acquire_locks

from tests.unit.robot.transport.conftest import requires_zenoh


def _serve_cfg(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "name": "left-arm",
        "robot_class": "tests.unit.robot.transport.fake.FakeRobot",
        "robot_kwargs": {"port": "/dev/fake0"},
        "allow_remote": False,
        "rate_hz": 100.0,
        "verbose": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_serve_runs_owner_in_foreground_with_persistent_timeout(capsys: object) -> None:
    captured: dict[str, object] = {}

    def _run_owner(config: object, shutdown: threading.Event, *, ready: object, on_event: object) -> OwnerResult:
        captured["config"] = config
        captured["shutdown"] = shutdown
        assert callable(ready)
        assert callable(on_event)
        ready()
        shutdown.set()
        return OwnerResult(OwnerExitReason.SHUTDOWN, 0)

    with patch.object(robot_module, "run_owner", side_effect=_run_owner) as run:
        assert robot_module.serve(_serve_cfg()) == 0

    run.assert_called_once()
    config = captured["config"]
    assert config.idle_timeout is None  # type: ignore[attr-defined]
    assert config.name == "left-arm"  # type: ignore[attr-defined]
    stderr = capsys.readouterr().err  # type: ignore[attr-defined]
    assert "unauthenticated" not in stderr
    assert "[local-only]" in stderr


def test_serve_allow_remote_warns_and_tags_mode(capsys: object) -> None:
    def _run_owner(  # noqa: ARG001
        _config: object,
        _shutdown: threading.Event,
        *,
        ready: object,
        on_event: object,
    ) -> OwnerResult:
        return OwnerResult(OwnerExitReason.SHUTDOWN, 0)

    with patch.object(robot_module, "run_owner", side_effect=_run_owner):
        assert robot_module.serve(_serve_cfg(allow_remote=True)) == 0

    stderr = capsys.readouterr().err  # type: ignore[attr-defined]
    assert robot_module._ALLOW_REMOTE_WARNING.strip() in stderr
    assert "[remote]" in stderr
    warning_at = stderr.index("WARNING: The action endpoint is unauthenticated")
    starting_at = stderr.index("Starting robot")
    assert warning_at < starting_at


def test_signal_requests_runtime_shutdown(capsys: object) -> None:
    def _run_owner(  # noqa: ARG001
        _config: object,
        shutdown: threading.Event,
        *,
        ready: object,
        on_event: object,
    ) -> OwnerResult:
        signal.raise_signal(signal.SIGTERM)
        assert shutdown.is_set()
        return OwnerResult(OwnerExitReason.SHUTDOWN, 0)

    with patch.object(robot_module, "run_owner", side_effect=_run_owner):
        assert robot_module.serve(_serve_cfg()) == 0
    assert "Shutdown requested by SIGTERM" in capsys.readouterr().err  # type: ignore[attr-defined]


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


def test_discovery_human_output_is_table(capsys: object) -> None:
    records = [
        {"name": "z-arm", "host": "b", "robot_class": "untrusted.Z", "num_joints": 7},
        {"name": "a-arm", "host": "a", "robot_class": "untrusted.A", "num_joints": 6},
    ]
    cfg = SimpleNamespace(timeout=1.0, allow_remote=False, json=False)
    with patch.object(robot_module, "discover_robots", return_value=records):
        assert robot_module.discover(cfg) == 0

    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "NAME" in output
    assert "ROBOT CLASS" in output
    assert output.index("a-arm") < output.index("z-arm")
    assert "2 robots found" in output


def test_owner_events_are_logged(capsys: object) -> None:
    def _run_owner(  # noqa: ARG001
        _config: object,
        _shutdown: threading.Event,
        *,
        ready: object,
        on_event: object,
    ) -> OwnerResult:
        assert callable(ready)
        assert callable(on_event)
        ready()
        on_event(OwnerEvent.SUBSCRIBERS_PRESENT)
        on_event(OwnerEvent.HEARTBEAT)
        on_event(OwnerEvent.NO_SUBSCRIBERS)
        return OwnerResult(OwnerExitReason.SHUTDOWN, 0)

    with patch.object(robot_module, "run_owner", side_effect=_run_owner):
        assert robot_module.serve(_serve_cfg()) == 0

    stderr = capsys.readouterr().err  # type: ignore[attr-defined]
    assert "Subscriber(s) connected" in stderr
    assert "Healthy" in stderr
    assert "Subscriber(s) connected" in stderr
    assert "No subscribers remain" in stderr


def test_verbose_controls_trace_details(capsys: object) -> None:
    def _run_owner(  # noqa: ARG001
        _config: object,
        _shutdown: threading.Event,
        *,
        ready: object,
        on_event: object,
    ) -> OwnerResult:
        logger.trace("startup phase detail")
        return OwnerResult(OwnerExitReason.SHUTDOWN, 0)

    with patch.object(robot_module, "run_owner", side_effect=_run_owner):
        assert robot_module.serve(_serve_cfg(verbose=False)) == 0
    assert "startup phase detail" not in capsys.readouterr().err  # type: ignore[attr-defined]

    with patch.object(robot_module, "run_owner", side_effect=_run_owner):
        assert robot_module.serve(_serve_cfg(verbose=True)) == 0
    assert "startup phase detail" in capsys.readouterr().err  # type: ignore[attr-defined]


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
