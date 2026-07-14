# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from physicalai.robot.errors import RobotDeviceAlreadyOwned, RobotTransportError
from physicalai.robot.transport._lock import NamedLock
from physicalai.robot.transport._owner import RobotOwner
from physicalai.robot.transport._owner_config import RobotOwnerConfig

from .conftest import FAKE_ROBOT_CLASS, requires_zenoh


def _owner(unique_id: str, **robot_kwargs: object) -> RobotOwner:
    config = RobotOwnerConfig(
        name=unique_id.replace("/", "-"),
        robot_class=FAKE_ROBOT_CLASS,
        robot_kwargs={"device_ids": [f"fake:{unique_id}"], **robot_kwargs},
        idle_timeout=2.0,
    )
    return RobotOwner(config)


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
