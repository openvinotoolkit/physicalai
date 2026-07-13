# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from physicalai.robot.errors import RobotTransportError
from physicalai.robot.transport._lock import RobotLock
from physicalai.robot.transport._owner import RobotOwner
from physicalai.robot.transport._spec import RobotSpec

from .conftest import FAKE_FACTORY, requires_zenoh


def _owner(unique_id: str, **kwargs: object) -> RobotOwner:
    spec = RobotSpec("so101", {"port": f"/dev/{unique_id.replace('/', '-')}", **kwargs})
    return RobotOwner(
        spec,
        robot_id=f"physicalai/robot/{unique_id}",
        device_id=unique_id.replace("/", "-"),
        idle_timeout=2.0,
        _factory_override=FAKE_FACTORY,
    )


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
        with pytest.raises(RobotTransportError, match="fake hardware failure"):
            owner.start(timeout=20.0)
        assert not owner.is_alive

    def test_error_path_lock_held(self, unique_id: str) -> None:
        device_id = unique_id.replace("/", "-")
        lock = RobotLock(device_id)
        assert lock.acquire()
        try:
            owner = _owner(unique_id)
            with pytest.raises(RobotTransportError, match="lock already held"):
                owner.start(timeout=20.0)
        finally:
            lock.release()

    def test_error_payload_includes_traceback(self, unique_id: str) -> None:
        owner = _owner(unique_id, fail_connect=True)
        with pytest.raises(RobotTransportError, match="worker traceback"):
            owner.start(timeout=20.0)
