# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from physicalai.robot.errors import (
    RobotNameConflict,
    RobotNotConnectedError,
    RobotProtocolMismatch,
    RobotTransportError,
)
from physicalai.robot.transport import SharedRobot, discover_robots
from physicalai.robot.transport._codec import ROBOT_TRANSPORT_PROTOCOL_VERSION

from .conftest import FAKE_ROBOT_CLASS, requires_zenoh
from .fake import FakeRobot

if TYPE_CHECKING:
    from collections.abc import Generator

_NUM_JOINTS = 6
_STATE_DIM = 12  # fake ships positions + velocities


def _shared_robot(name: str, **robot_kwargs: object) -> SharedRobot:
    return SharedRobot(
        name,
        robot_class=FAKE_ROBOT_CLASS,
        robot_kwargs={"device_ids": [f"fake:{name}"], **robot_kwargs},
        idle_timeout=3.0,
    )


class TestConstruction:
    def test_invalid_name_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid robot name"):
            SharedRobot("bad/name")

    def test_attach_only_has_no_robot_class(self) -> None:
        robot = SharedRobot.attach("left-arm")
        assert robot.name == "left-arm"
        assert robot.device_ids == ()

    def test_class_object_normalized(self) -> None:
        robot = SharedRobot("left-arm", robot_class=FakeRobot, robot_kwargs={"port": "/dev/ttyUSB0"})
        assert robot._robot_class == "tests.unit.robot.transport.fake.FakeRobot"

    def test_satisfies_robot_protocol(self) -> None:
        from physicalai.robot import Robot

        assert isinstance(SharedRobot.attach("left-arm"), Robot)

    def test_not_connected_errors(self) -> None:
        robot = _shared_robot("x")
        assert not robot.is_connected()
        with pytest.raises(RobotNotConnectedError):
            robot.get_observation()
        with pytest.raises(RobotNotConnectedError):
            robot.send_action(np.zeros(_NUM_JOINTS, dtype=np.float32))
        with pytest.raises(RobotNotConnectedError):
            _ = robot.joint_names

    def test_device_ids_always_empty(self) -> None:
        assert _shared_robot("x").device_ids == ()


@requires_zenoh
class TestSharedRobotLifecycle:
    @pytest.fixture
    def robot(self, unique_id: str) -> Generator[SharedRobot, None, None]:
        name = unique_id.replace("/", "-")
        robot = _shared_robot(name)
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

    def test_second_instance_attaches(self, robot: SharedRobot) -> None:
        robot.connect()
        second = _shared_robot(robot.name)
        second.connect()
        try:
            # Attached, not spawned: no owner subprocess handle of its own.
            assert second._owner is None
            assert second.get_observation().state.shape == (_STATE_DIM,)
        finally:
            second.disconnect()

    def test_disconnect_leaves_owner_running(self, robot: SharedRobot) -> None:
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

    def test_metadata_exposed(self, robot: SharedRobot) -> None:
        robot.connect()
        assert robot.metadata is not None
        assert robot.metadata["protocol_version"] == ROBOT_TRANSPORT_PROTOCOL_VERSION
        assert robot.metadata["name"] == robot.name
        assert robot.metadata["robot_class"] == FAKE_ROBOT_CLASS
        assert robot.metadata["state_dim"] == _STATE_DIM
        assert robot.metadata["num_joints"] == _NUM_JOINTS
        assert robot.metadata["device_ids"] == [f"fake:{robot.name}"]


@requires_zenoh
class TestNameAndDeviceConflicts:
    def test_matching_device_race_attaches(self, unique_id: str) -> None:
        """Two instances for the same name and same devices must not conflict."""
        name = unique_id.replace("/", "-")
        first = _shared_robot(name)
        first.connect()
        try:
            second = _shared_robot(name)  # same derived device_ids
            second.connect()
            try:
                assert second.metadata is not None
            finally:
                second.disconnect()
        finally:
            owner = first._owner
            first.disconnect()
            if owner is not None:
                owner.stop()

    def test_differing_device_race_raises_name_conflict(self, unique_id: str) -> None:
        """A genuine concurrent race for the same name but different devices must conflict."""
        import threading

        name = unique_id.replace("/", "-")
        barrier = threading.Barrier(2)
        results: dict[str, object] = {}

        def _run(key: str, device_id: str) -> None:
            robot = SharedRobot(name, robot_class=FAKE_ROBOT_CLASS, robot_kwargs={"device_ids": [device_id]})
            barrier.wait()
            try:
                robot.connect()
            except Exception as exc:  # noqa: BLE001
                results[key] = exc
            else:
                results[key] = robot

        threads = [
            threading.Thread(target=_run, args=("a", f"fake:{unique_id}-A")),
            threading.Thread(target=_run, args=("b", f"fake:{unique_id}-B")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        outcomes = list(results.values())
        winners = [o for o in outcomes if isinstance(o, SharedRobot)]
        conflicts = [o for o in outcomes if isinstance(o, RobotNameConflict)]
        try:
            assert len(winners) == 1
            assert len(conflicts) == 1
        finally:
            for winner in winners:
                owner = winner._owner
                winner.disconnect()
                if owner is not None:
                    owner.stop()

    def test_class_mismatch_warns_but_attaches(self, unique_id: str, caplog: pytest.LogCaptureFixture) -> None:
        """robot_class mismatch on an existing owner is diagnostic, not fatal."""
        import logging

        name = unique_id.replace("/", "-")
        first = _shared_robot(name)
        first.connect()
        try:
            impostor = SharedRobot(
                name,
                robot_class="physicalai.robot.transport._session.open_session",
                robot_kwargs={"device_ids": [f"fake:{name}"]},
            )
            with caplog.at_level(logging.WARNING):
                impostor.connect()  # attaches to the same owner; must not raise
            try:
                assert impostor.is_connected()
            finally:
                impostor.disconnect()
        finally:
            owner = first._owner
            first.disconnect()
            if owner is not None:
                owner.stop()

    def test_attach_only_no_owner_raises(self, unique_id: str) -> None:
        robot = SharedRobot.attach(unique_id.replace("/", "-"))
        with pytest.raises(RobotTransportError, match="attach-only"):
            robot.connect()

    def test_spawn_failure_raises(self, unique_id: str) -> None:
        robot = _shared_robot(unique_id.replace("/", "-"), fail_connect=True)
        with pytest.raises(RobotTransportError, match="failed to start robot owner"):
            robot.connect()
        assert not robot.is_connected()

    def test_device_already_owned_under_another_name(self, unique_id: str) -> None:
        first_name = f"{unique_id.replace('/', '-')}-a"
        second_name = f"{unique_id.replace('/', '-')}-b"
        shared_device = f"fake:{unique_id}"

        first = SharedRobot(first_name, robot_class=FAKE_ROBOT_CLASS, robot_kwargs={"device_ids": [shared_device]})
        first.connect()
        try:
            from physicalai.robot.errors import RobotDeviceAlreadyOwned

            second = SharedRobot(second_name, robot_class=FAKE_ROBOT_CLASS, robot_kwargs={"device_ids": [shared_device]})
            with pytest.raises(RobotDeviceAlreadyOwned):
                second.connect()
        finally:
            owner = first._owner
            first.disconnect()
            if owner is not None:
                owner.stop()


@requires_zenoh
class TestProtocolAndMetadataValidation:
    def test_protocol_mismatch_rejected_before_action_publisher(self, unique_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
        import physicalai.robot.transport._shared_robot as shared_robot_module

        name = unique_id.replace("/", "-")
        owner_side = _shared_robot(name)
        owner_side.connect()
        try:
            monkeypatch.setattr(shared_robot_module, "ROBOT_TRANSPORT_PROTOCOL_VERSION", 999)
            attacher = SharedRobot.attach(name)
            with pytest.raises(RobotProtocolMismatch):
                attacher.connect()
            # Rejected before the action publisher was ever declared.
            assert attacher._action_pub is None
        finally:
            owner = owner_side._owner
            owner_side.disconnect()
            if owner is not None:
                owner.stop()

    def test_malformed_metadata_rejected(self, unique_id: str) -> None:
        name = unique_id.replace("/", "-")
        robot = SharedRobot.attach(name)
        with pytest.raises(RobotTransportError, match="malformed"):
            robot._validate_metadata(
                {
                    "protocol_version": ROBOT_TRANSPORT_PROTOCOL_VERSION,
                    "joint_names": ["a", "a"],
                    "num_joints": 2,
                    "state_dim": 2,
                },
            )


@requires_zenoh
class TestOwnerIdleShutdown:
    def test_owner_exits_and_disconnects_driver_after_idle(self, unique_id: str) -> None:
        robot = _shared_robot(unique_id.replace("/", "-"))
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
        robot = _shared_robot(unique_id.replace("/", "-"))
        robot.connect()
        try:
            # Query through a session connected to the owner's endpoint so
            # the test does not depend on multicast scouting availability.
            found = discover_robots(timeout=2.0, session=robot._session)
            names = [m["name"] for m in found]
            assert robot.name in names
            metadata = next(m for m in found if m["name"] == robot.name)
            assert metadata["robot_class"] == FAKE_ROBOT_CLASS
        finally:
            owner = robot._owner
            robot.disconnect()
            if owner is not None:
                owner.stop()

    def test_local_only_owner_still_reachable_on_same_host(self, unique_id: str) -> None:
        """allow_remote only gates off-host reachability, not same-host.

        The deterministic loopback rendezvous port is always used for
        same-host connect, regardless of ``allow_remote`` — that is what
        makes same-host spawn-or-attach work without depending on
        multicast scouting. Off-host unreachability (the actual security
        property) needs the owner's listen bind + scouting config, which
        is asserted directly in ``test_session.py``; verifying it
        end-to-end requires a real second host or network namespace
        (Phase 4 integration test), not a same-host unit test.
        """
        name = unique_id.replace("/", "-")
        local_only_owner = _shared_robot(name)
        local_only_owner.connect()
        try:
            same_host_attacher = SharedRobot.attach(name, allow_remote=False)
            same_host_attacher.connect()
            try:
                assert same_host_attacher.is_connected()
            finally:
                same_host_attacher.disconnect()
        finally:
            owner = local_only_owner._owner
            local_only_owner.disconnect()
            if owner is not None:
                owner.stop()
