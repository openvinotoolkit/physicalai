# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from physicalai.robot.errors import RobotIdConflict, RobotNotConnectedError, RobotTransportError
from physicalai.robot.transport import SharedRobot, discover_robots

from .conftest import FAKE_FACTORY, requires_zenoh

if TYPE_CHECKING:
    from collections.abc import Generator

_NUM_JOINTS = 6
_STATE_DIM = 12  # fake ships positions + velocities


def _shared_robot(unique_id: str, **kwargs: object) -> SharedRobot:
    defaults: dict[str, object] = {
        "robot_id": unique_id,
        "port": f"/dev/{unique_id.replace('/', '-')}",
        "_factory_override": FAKE_FACTORY,
        "idle_timeout": 3.0,
    }
    defaults.update(kwargs)
    return SharedRobot("so101", **defaults)  # pyrefly: ignore


class TestConstruction:
    def test_requires_type_or_id(self) -> None:
        with pytest.raises(ValueError, match="robot_type or robot_id"):
            SharedRobot(None)

    def test_requires_derivable_id(self) -> None:
        with pytest.raises(ValueError, match="cannot derive robot_id"):
            SharedRobot("so101", role="follower")

    def test_derived_robot_id(self) -> None:
        robot = SharedRobot("so101", port="/dev/ttyUSB0")
        assert robot.robot_id.startswith("physicalai/robot/so101/")
        assert robot.robot_id.endswith("/ttyUSB0")

    def test_satisfies_robot_protocol(self) -> None:
        from physicalai.robot import Robot

        assert isinstance(SharedRobot("so101", port="/dev/ttyUSB0"), Robot)

    def test_not_connected_errors(self) -> None:
        robot = SharedRobot("so101", port="/dev/ttyUSB0")
        assert not robot.is_connected()
        with pytest.raises(RobotNotConnectedError):
            robot.get_observation()
        with pytest.raises(RobotNotConnectedError):
            robot.send_action(np.zeros(_NUM_JOINTS, dtype=np.float32))
        with pytest.raises(RobotNotConnectedError):
            _ = robot.joint_names

    def test_from_owner_is_attach_only(self) -> None:
        robot = SharedRobot.from_owner("physicalai/robot/nowhere/x")
        assert robot.robot_id == "physicalai/robot/nowhere/x"


@requires_zenoh
class TestSharedRobotLifecycle:
    @pytest.fixture
    def robot(self, unique_id: str) -> Generator[SharedRobot, None, None]:
        robot = _shared_robot(unique_id)
        yield robot
        owner = robot._owner
        robot.disconnect()
        if owner is not None:
            owner.stop()

    def test_spawn_connect_observe_act(self, robot: SharedRobot) -> None:
        robot.connect()
        assert robot.is_connected()
        assert robot.joint_names == [f"joint_{i}" for i in range(_NUM_JOINTS)]

        obs = robot.get_observation()
        assert obs.joint_positions.shape == (_NUM_JOINTS,)
        assert obs.joint_positions.dtype == np.float32
        # Owner-computed state shipped as-is (positions + velocities).
        assert obs.state.shape == (_STATE_DIM,)
        assert obs.state.dtype == np.float32
        assert obs.images is None

        target = np.arange(_NUM_JOINTS, dtype=np.float32)
        robot.send_action(target, goal_time=0.05)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if np.array_equal(robot.get_observation().joint_positions, target):
                break
            time.sleep(0.01)
        np.testing.assert_array_equal(robot.get_observation().joint_positions, target)

    def test_connect_idempotent(self, robot: SharedRobot) -> None:
        robot.connect()
        robot.connect()
        assert robot.is_connected()

    def test_second_instance_attaches(self, robot: SharedRobot, unique_id: str) -> None:
        robot.connect()
        second = _shared_robot(unique_id)
        second.connect()
        try:
            # Attached, not spawned: no owner subprocess handle of its own.
            assert second._owner is None
            assert second.get_observation().state.shape == (_STATE_DIM,)
        finally:
            second.disconnect()

    def test_disconnect_leaves_owner_running(self, robot: SharedRobot, unique_id: str) -> None:
        robot.connect()
        owner = robot._owner
        assert owner is not None and owner.is_alive
        robot.disconnect()
        assert not robot.is_connected()
        # Subscriber disconnect must not stop the owner (owner owns safe-state).
        assert owner.is_alive

    def test_observation_from_cache_when_no_new_sample(self, robot: SharedRobot) -> None:
        robot.connect()
        first = robot.get_observation()
        again = robot.get_observation()
        assert again.state.shape == first.state.shape

    def test_freshest_state_after_stall_not_backlog(self, robot: SharedRobot) -> None:
        """Ring(1) keeps only the newest sample while the subscriber stalls."""
        robot.connect()
        stale = robot.get_observation()
        # Stall for many owner periods (default 100 Hz -> ~50 ticks); the
        # native ring keeps buffering and evicting without the GIL.
        time.sleep(0.5)
        fresh = robot.get_observation()
        assert fresh.timestamp > stale.timestamp
        # Newest-or-nothing: a second immediate pull must not drain a backlog
        # of intermediate samples older than the one we just got.
        newest = robot.get_observation()
        assert newest.timestamp >= fresh.timestamp

    def test_meta_exposed(self, robot: SharedRobot) -> None:
        robot.connect()
        assert robot.meta is not None
        assert robot.meta["robot_type"] == "so101"
        assert robot.meta["state_dim"] == _STATE_DIM
        assert robot.meta["num_joints"] == _NUM_JOINTS


@requires_zenoh
class TestConflictsAndFailures:
    def test_id_conflict_on_different_hardware(self, unique_id: str) -> None:
        first = _shared_robot(unique_id)
        first.connect()
        try:
            impostor = _shared_robot(unique_id, port="/dev/other-port")
            with pytest.raises(RobotIdConflict, match="connection"):
                impostor.connect()
            assert not impostor.is_connected()
        finally:
            owner = first._owner
            first.disconnect()
            if owner is not None:
                owner.stop()

    def test_attach_only_no_owner_raises(self, unique_id: str) -> None:
        robot = SharedRobot.from_owner(f"physicalai/robot/{unique_id}")
        with pytest.raises(RobotTransportError, match="attach-only"):
            robot.connect()

    def test_spawn_failure_raises(self, unique_id: str) -> None:
        robot = _shared_robot(unique_id, fail_connect=True)
        with pytest.raises(RobotTransportError, match="failed to start robot owner"):
            robot.connect()
        assert not robot.is_connected()


@requires_zenoh
class TestOwnerIdleShutdown:
    def test_owner_exits_and_disconnects_driver_after_idle(self, unique_id: str) -> None:
        robot = _shared_robot(unique_id, idle_timeout=1.0)
        robot.connect()
        owner = robot._owner
        assert owner is not None
        proc = owner._process
        assert proc is not None
        robot.disconnect()

        # Owner detects zero subscribers via matching status and self-exits,
        # calling driver.disconnect() (exit code 0 = clean shutdown path ran).
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.2)
        assert proc.poll() == 0


@requires_zenoh
class TestDiscovery:
    def test_discover_robots_via_connected_session(self, unique_id: str) -> None:
        robot = _shared_robot(unique_id)
        robot.connect()
        try:
            # Query through a session connected to the owner's endpoint so
            # the test does not depend on multicast scouting availability.
            found = discover_robots(timeout=2.0, session=robot._session)
            ids = [m["robot_id"] for m in found]
            assert robot.robot_id in ids
            meta = next(m for m in found if m["robot_id"] == robot.robot_id)
            assert meta["robot_type"] == "so101"
        finally:
            owner = robot._owner
            robot.disconnect()
            if owner is not None:
                owner.stop()
