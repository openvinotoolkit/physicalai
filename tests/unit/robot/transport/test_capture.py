# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fresh captures must cross the driver-owning thread, not a subscriber cache."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest

from physicalai.robot import CompositeRobot, SharedRobot
from physicalai.robot.errors import RobotNotConnectedError, RobotTransportError
from physicalai.robot.transport import _owner_worker
from physicalai.robot.transport._codec import decode_state, encode_state
from physicalai.robot.transport._ids import state_key
from physicalai.robot.transport._owner_config import RobotOwnerConfig
from physicalai.robot.transport._owner_worker import OwnerExitReason, _run_loop

from .conftest import FAKE_ROBOT_CLASS, requires_zenoh
from .fake import FakeObservation, FakeRobot
from .test_owner_worker import _ActionSubscriber, _StatePublisher


def test_request_during_a_read_requires_the_next_capture() -> None:
    shutdown = threading.Event()
    requests: queue.Queue[Future[bytes]] = queue.Queue(maxsize=1)
    request: Future[bytes] = Future()
    driver = FakeRobot()
    reads = 0

    def read() -> FakeObservation:
        nonlocal reads
        reads += 1
        if reads == 1:
            requests.put_nowait(request)
            assert not request.running()
        else:
            assert request.running()
            shutdown.set()
        return FakeObservation(np.zeros(6), timestamp=float(reads))

    driver.get_observation = MagicMock(side_effect=read)
    reason = _run_loop(
        driver,
        _StatePublisher(),
        _ActionSubscriber(),
        rate_hz=1000,
        idle_timeout=None,
        name="capture",
        shutdown_event=shutdown,
        capture_requests=requests,
    )

    assert reason is OwnerExitReason.SHUTDOWN
    assert decode_state(request.result()).timestamp == 2.0


def test_capture_reports_driver_errors_instead_of_cached_state() -> None:
    requests: queue.Queue[Future[bytes]] = queue.Queue(maxsize=1)
    request: Future[bytes] = Future()
    requests.put_nowait(request)
    driver = FakeRobot(fail_observation=True)
    driver.connect()
    _run_loop(
        driver,
        _StatePublisher(),
        _ActionSubscriber(),
        rate_hz=1000,
        idle_timeout=None,
        name="capture",
        shutdown_event=threading.Event(),
        capture_requests=requests,
    )
    with pytest.raises(RuntimeError, match="observation failure"):
        request.result()


@pytest.mark.parametrize("payload", [None, b"invalid state"])
def test_failed_fresh_request_never_falls_back_to_cached_state(payload: bytes | None) -> None:
    robot = SharedRobot.attach("capture")
    robot._connected = True
    robot._latest = decode_state(
        encode_state(joint_positions=np.zeros(6), state=np.zeros(6), timestamp=1.0, sensor_data=None)
    )
    session = MagicMock()
    session.get.return_value = (
        [] if payload is None else [SimpleNamespace(ok=SimpleNamespace(payload=SimpleNamespace(to_bytes=lambda: payload)))]
    )
    robot._session = session
    with pytest.raises(RobotTransportError, match="fresh observation"):
        robot.get_fresh_observation()


def test_fresh_request_requires_connection() -> None:
    with pytest.raises(RobotNotConnectedError):
        SharedRobot.attach("capture").get_fresh_observation()


@pytest.mark.parametrize("invalid_payload", [True, False])
def test_capture_query_rejects_payloads_and_a_full_queue(monkeypatch: pytest.MonkeyPatch, invalid_payload: bool) -> None:
    callbacks: dict[str, Callable[[Any], None]] = {}
    session = MagicMock()
    session.declare_queryable.side_effect = lambda key, callback: callbacks.setdefault(key, callback)
    monkeypatch.setattr(_owner_worker, "open_session", lambda *args, **kwargs: session)
    requests: queue.Queue[Future[bytes]] = queue.Queue(maxsize=1)
    if not invalid_payload:
        requests.put_nowait(Future())
    config = RobotOwnerConfig(name="capture", robot={"class_path": FAKE_ROBOT_CLASS, "init_args": {}})
    _owner_worker._declare_zenoh_endpoints(config, b"", capture_requests=requests)
    query = MagicMock()
    query.payload = b"unexpected" if invalid_payload else None

    callbacks[state_key("capture")](query)

    query.reply.assert_not_called()
    query.reply_err.assert_called_once()
    assert requests.qsize() == (0 if invalid_payload else 1)


def test_composite_fresh_capture_recurses_into_buffered_components() -> None:
    left = FakeRobot()
    right = FakeRobot()
    left.get_observation = MagicMock(return_value=FakeObservation(np.zeros(6), timestamp=1.0))
    right.get_observation = MagicMock(return_value=FakeObservation(np.zeros(6), timestamp=1.0))
    left.get_fresh_observation = MagicMock(return_value=FakeObservation(np.full(6, 2.0), timestamp=2.0))
    right.get_fresh_observation = MagicMock(return_value=FakeObservation(np.full(6, 3.0), timestamp=3.0))
    robot = CompositeRobot({"left": left, "right": right})

    fresh = robot.get_fresh_observation()
    assert fresh.timestamp == 2.0
    left.get_fresh_observation.assert_called_once()
    right.get_fresh_observation.assert_called_once()
    left.get_observation.assert_not_called()
    right.get_observation.assert_not_called()
    fresh.joint_positions[:] = -1
    cached = robot.get_observation()
    assert cached.timestamp == 2.0
    np.testing.assert_array_equal(cached.joint_positions, [2.0] * 6 + [3.0] * 6)
    with pytest.raises(ConnectionError, match="did not advance"):
        robot.get_fresh_observation()


@requires_zenoh
def test_shared_capture_is_newer_than_the_request_not_its_stream_cache() -> None:
    name = f"capture-{uuid4().hex[:8]}"
    robot = SharedRobot(name, robot={"class_path": FAKE_ROBOT_CLASS, "init_args": {"device_ids": [f"fake:{name}"]}})
    owner = None
    try:
        robot.connect()
        owner = robot._owner
        requested_at = time.monotonic()
        observation = robot.get_fresh_observation()
        assert observation.timestamp >= requested_at
    finally:
        robot.disconnect()
        if owner is not None:
            owner.stop()
