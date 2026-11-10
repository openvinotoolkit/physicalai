# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import threading
from unittest.mock import Mock

import pytest

from physicalai.robot.errors import RobotCalibrationError, RobotDeviceAlreadyOwned, RobotTransportError
from physicalai.robot.transport import SharedRobot
from physicalai.robot.transport._ids import derive_endpoint_port
from physicalai.robot.transport._lock import NamedLock, acquire_locks
from physicalai.robot.transport._owner import RobotOwner
from physicalai.robot.transport._owner_config import RobotOwnerConfig
from physicalai.robot.transport._owner_worker import (
    _StartupError,
    _build_metadata,
    _declare_zenoh_endpoints,
    _startup,
    run_owner,
    signal_error,
)

from .conftest import FAKE_ROBOT_CLASS, requires_zenoh
from .fake import FakeRobot


def _fake_robot(**init_args: object) -> dict[str, object]:
    return {"class_path": FAKE_ROBOT_CLASS, "init_args": dict(init_args)}


def test_calibration_failure_preserves_diagnostics_and_releases_locks(
    unique_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = unique_id.replace("/", "-")
    device_id = f"fake:{unique_id}"
    driver = FakeRobot(device_ids=(device_id,))
    error = RobotCalibrationError("missing calibration", reason="missing", calibration_path="/calibration/leader.json")
    monkeypatch.setattr(driver, "connect", Mock(side_effect=error))
    monkeypatch.setattr(RobotOwnerConfig, "build", lambda _self: driver)
    config = RobotOwnerConfig(name=name, robot=_fake_robot())
    with pytest.raises(RobotCalibrationError) as caught:
        run_owner(config, threading.Event())
    assert caught.value.reason == error.reason
    assert caught.value.calibration_path == error.calibration_path
    assert caught.value.phase == "connection_failed"
    assert driver.disconnect_called
    acquire_locks(name, [device_id]).release_all()


@pytest.mark.parametrize("reason", ["missing", "invalid", "mismatch"])
def test_owner_reads_structured_calibration_error(reason: str) -> None:
    line = "ERROR:" + json.dumps({
        "msg": "driver failed",
        "phase": "connection_failed",
        "traceback": "diagnostic traceback",
        "calibration": {"reason": reason, "path": "/calibration/leader.json"},
    })
    with pytest.raises(RobotCalibrationError) as caught:
        RobotOwner._raise_from_error_line(line)
    assert caught.value.reason == reason
    assert caught.value.calibration_path == "/calibration/leader.json"
    assert "diagnostic traceback" in str(caught.value)


def test_stop_waits_for_a_slow_safe_disconnect_before_killing() -> None:
    # On SIGTERM the worker disconnects the driver, which may first park the arm for several seconds.
    process = Mock()
    process.poll.return_value = None
    process.returncode = 0
    owner = RobotOwner.__new__(RobotOwner)
    owner._process = process

    owner.stop()

    assert process.wait.call_args_list[0].kwargs["timeout"] >= 20.0
    process.kill.assert_not_called()


@pytest.mark.parametrize("calibration", [{"reason": "unknown"}, {"reason": "missing", "path": 123}, []])
def test_owner_does_not_trust_malformed_calibration_diagnostics(calibration: object) -> None:
    line = "ERROR:" + json.dumps({"msg": "failed", "phase": "connection_failed", "calibration": calibration})
    with pytest.raises(RobotTransportError) as caught:
        RobotOwner._raise_from_error_line(line)
    assert not isinstance(caught.value, RobotCalibrationError)


def test_calibration_diagnostics_round_trip_through_error_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    output = io.StringIO()
    monkeypatch.setattr(output, "close", lambda: None)
    monkeypatch.setattr("physicalai.robot.transport._owner_worker.sys.stdout", output)
    error = RobotCalibrationError("missing file", reason="missing", calibration_path="/calibration/leader.json")
    signal_error(str(error), phase=error.phase, calibration_error=error)
    with pytest.raises(RobotCalibrationError) as caught:
        RobotOwner._raise_from_error_line(output.getvalue().strip())
    assert caught.value.reason == error.reason
    assert caught.value.calibration_path == error.calibration_path


def test_shared_robot_does_not_erase_calibration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    error = RobotCalibrationError("missing file", reason="missing", calibration_path="/calibration/leader.json")
    monkeypatch.setattr(RobotOwner, "start", Mock(side_effect=error))
    robot = SharedRobot("calibration-test", robot=_fake_robot())
    with pytest.raises(RobotCalibrationError) as caught:
        robot._spawn_or_reprobe(timeout=1.0)
    assert caught.value is error


def _owner(unique_id: str, **robot_init_args: object) -> RobotOwner:
    config = RobotOwnerConfig(
        name=unique_id.replace("/", "-"),
        robot=_fake_robot(device_ids=[f"fake:{unique_id}"], **robot_init_args),
        idle_timeout=2.0,
    )
    return RobotOwner(config)


def test_startup_failure_after_connect_disconnects_and_releases_locks(
    unique_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = unique_id.replace("/", "-")
    device_id = f"fake:{unique_id}"
    driver = FakeRobot(device_ids=(device_id,), fail_observation=True)
    config = RobotOwnerConfig(name=name, robot=_fake_robot())
    monkeypatch.setattr(RobotOwnerConfig, "build", lambda _self: driver)

    with pytest.raises(_StartupError, match="fake observation failure") as exc_info:
        _startup(config)

    assert exc_info.value.phase == "unexpected_startup_failure"
    assert driver.disconnect_called
    locks = acquire_locks(name, [device_id])
    locks.release_all()


@pytest.mark.parametrize("allow_remote", [False, True])
def test_metadata_redacts_device_ids_for_remote_owner(allow_remote: bool) -> None:
    config = RobotOwnerConfig(name="left-arm", robot=_fake_robot(), allow_remote=allow_remote)
    metadata = _build_metadata(
        config,
        FakeRobot(device_ids=("serial:ttyUSB0",)),
        ("serial:ttyUSB0",),
        state_dim=6,
    )

    assert metadata["host"]
    assert metadata["robot_class"] == FAKE_ROBOT_CLASS
    if allow_remote:
        assert "device_ids" not in metadata
    else:
        assert metadata["device_ids"] == ["serial:ttyUSB0"]


@requires_zenoh
@pytest.mark.parametrize(("allow_remote", "bind_host"), [(False, "127.0.0.1"), (True, "0.0.0.0")])
def test_endpoint_collision_error_identifies_endpoint_and_remediation(
    unique_id: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    allow_remote: bool,
    bind_host: str,
) -> None:
    name = unique_id.replace("/", "-")
    config = RobotOwnerConfig(name=name, robot=_fake_robot(), allow_remote=allow_remote)

    def _fail_open_session(*_args: object, **_kwargs: object) -> None:
        raise OSError("address already in use")

    monkeypatch.setattr("physicalai.robot.transport._owner_worker.open_session", _fail_open_session)

    endpoint = f"tcp/{bind_host}:{derive_endpoint_port(name)}"
    with pytest.raises(_StartupError, match="address already in use") as exc_info:
        _declare_zenoh_endpoints(config, b"")

    assert exc_info.value.phase == "endpoint_collision"
    assert endpoint in str(exc_info.value)
    assert "different robot name" in str(exc_info.value)
    assert "local Zenoh router" in str(exc_info.value)


@requires_zenoh
class TestOwnerHandshake:
    def test_ready_path(self, unique_id: str) -> None:
        owner = _owner(unique_id)
        owner.start(timeout=20.0)
        assert owner.is_alive
        owner.stop()
        assert not owner.is_alive

    def test_context_manager(self, unique_id: str) -> None:
        with _owner(unique_id) as owner:
            assert owner.is_alive
        assert not owner.is_alive

    def test_error_path_hardware_failure(self, unique_id: str) -> None:
        owner = _owner(unique_id, fail_connect=True)
        try:
            owner.start(timeout=20.0)
        except RobotTransportError as exc:
            assert "fake hardware failure" in str(exc)
            assert exc.phase == "connection_failed"
        else:
            raise AssertionError("expected RobotTransportError")
        assert not owner.is_alive

    def test_error_path_device_lock_held(self, unique_id: str) -> None:
        device_id = f"fake:{unique_id}"
        lock = NamedLock("device", device_id)
        assert lock.acquire()
        try:
            owner = _owner(unique_id)
            try:
                owner.start(timeout=20.0)
            except RobotDeviceAlreadyOwned as exc:
                assert exc.phase == "device_lock_contention"
            else:
                raise AssertionError("expected RobotDeviceAlreadyOwned")
        finally:
            lock.release()

    def test_error_payload_includes_traceback(self, unique_id: str) -> None:
        owner = _owner(unique_id, fail_connect=True)
        try:
            owner.start(timeout=20.0)
        except RobotTransportError as exc:
            assert "worker traceback" in str(exc)
        else:
            raise AssertionError("expected RobotTransportError")
