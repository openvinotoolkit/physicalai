# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from physicalai.robot.transport import _owner_worker
from physicalai.robot.transport._owner_config import RobotOwnerConfig
from physicalai.robot.transport._owner_worker import OwnerEvent, OwnerExitReason, _run_loop, main, run_owner

from .fake import FakeRobot


@dataclass
class _MatchingStatus:
    matching: bool = True


@dataclass
class _StatePublisher:
    matching: bool = True
    fail_put: bool = False
    puts: list[bytes] = field(default_factory=list)

    @property
    def matching_status(self) -> _MatchingStatus:
        return _MatchingStatus(self.matching)

    def put(self, payload: bytes) -> None:
        if self.fail_put:
            raise RuntimeError("fake publish failure")
        self.puts.append(payload)


class _ActionSubscriber:
    def try_recv(self) -> Any | None:
        return None


def _driver(**kwargs: object) -> FakeRobot:
    driver = FakeRobot(**kwargs)
    driver.connect()
    return driver


def test_shutdown_returns_shutdown() -> None:
    shutdown = threading.Event()
    shutdown.set()
    reason = _run_loop(
        _driver(),
        _StatePublisher(),
        _ActionSubscriber(),
        rate_hz=1000.0,
        idle_timeout=None,
        name="test",
        shutdown_event=shutdown,
    )
    assert reason is OwnerExitReason.SHUTDOWN


def test_idle_timeout_returns_idle_timeout() -> None:
    reason = _run_loop(
        _driver(),
        _StatePublisher(matching=False),
        _ActionSubscriber(),
        rate_hz=1000.0,
        idle_timeout=0.01,
        name="test",
        shutdown_event=threading.Event(),
    )
    assert reason is OwnerExitReason.IDLE_TIMEOUT


def test_repeated_reads_return_failure() -> None:
    reason = _run_loop(
        _driver(fail_observation=True),
        _StatePublisher(),
        _ActionSubscriber(),
        rate_hz=1000.0,
        idle_timeout=10.0,
        name="test",
        shutdown_event=threading.Event(),
    )
    assert reason is OwnerExitReason.CONSECUTIVE_READ_FAILURES


def test_subscriber_transitions_and_heartbeat_emit_events() -> None:
    events: list[OwnerEvent] = []
    publisher = _StatePublisher(matching=False)
    shutdown = threading.Event()

    def _change_matching() -> None:
        publisher.matching = True
        time.sleep(0.02)
        publisher.matching = False
        time.sleep(0.02)
        shutdown.set()

    thread = threading.Thread(target=_change_matching)
    thread.start()
    try:
        reason = _run_loop(
            _driver(),
            publisher,
            _ActionSubscriber(),
            rate_hz=1000.0,
            idle_timeout=None,
            name="test",
            shutdown_event=shutdown,
            on_event=events.append,
            heartbeat_interval_s=0.01,
        )
    finally:
        thread.join()

    assert reason is OwnerExitReason.SHUTDOWN
    assert OwnerEvent.SUBSCRIBERS_PRESENT in events
    assert OwnerEvent.NO_SUBSCRIBERS in events
    assert OwnerEvent.HEARTBEAT in events


def test_disconnect_failure_upgrades_clean_shutdown(monkeypatch: Any) -> None:  # noqa: ANN401
    driver = _driver(fail_disconnect=True)
    shutdown = threading.Event()
    shutdown.set()
    endpoints = SimpleNamespace(
        driver=driver,
        state_pub=_StatePublisher(),
        action_sub=_ActionSubscriber(),
        metadata_queryable=SimpleNamespace(undeclare=lambda: None),
        session=SimpleNamespace(close=lambda: None),
        locks=SimpleNamespace(release_all=lambda: None),
    )
    monkeypatch.setattr(_owner_worker, "_startup", lambda _config: endpoints)
    config = RobotOwnerConfig(
        name="left-arm",
        robot={"class_path": "tests.unit.robot.transport.fake.FakeRobot", "init_args": {}},
        idle_timeout=None,
    )

    result = run_owner(config, shutdown)

    assert result.reason is OwnerExitReason.SHUTDOWN
    assert result.exit_code == 1
    assert driver.disconnect_called


def test_ready_failure_still_disconnects(monkeypatch: Any) -> None:  # noqa: ANN401
    driver = _driver()
    endpoints = SimpleNamespace(
        driver=driver,
        state_pub=_StatePublisher(),
        action_sub=_ActionSubscriber(),
        metadata_queryable=SimpleNamespace(undeclare=lambda: None),
        session=SimpleNamespace(close=lambda: None),
        locks=SimpleNamespace(release_all=lambda: None),
    )
    monkeypatch.setattr(_owner_worker, "_startup", lambda _config: endpoints)
    config = RobotOwnerConfig(
        name="left-arm",
        robot={"class_path": "tests.unit.robot.transport.fake.FakeRobot", "init_args": {}},
        idle_timeout=None,
    )

    def _fail_ready() -> None:
        raise RuntimeError("readiness output failed")

    result = run_owner(config, threading.Event(), ready=_fail_ready)

    assert result.reason is OwnerExitReason.LOOP_FAILURE
    assert result.exit_code == 1
    assert driver.disconnect_called


def test_main_malformed_robot_stdin_signals_invalid_config(monkeypatch: Any) -> None:  # noqa: ANN401
    """Malformed new-shape robot: must ERROR with invalid_config, not crash."""
    payload = json.dumps(
        {
            "name": "left-arm",
            "robot": {"class_path": "tests.unit.robot.transport.fake.FakeRobot", "extra": 1},
        },
    )
    errors: list[tuple[str, str | None]] = []

    def _capture_error(msg: str, tb: str | None = None, *, phase: str | None = None, **_kwargs: object) -> None:
        errors.append((msg, phase))

    monkeypatch.setattr(_owner_worker, "signal_error", _capture_error)
    monkeypatch.setattr(_owner_worker.sys, "stdin", io.StringIO(payload))

    assert main() == 1
    assert len(errors) == 1
    assert errors[0][1] == "invalid_config"
    assert "invalid worker config" in errors[0][0]
