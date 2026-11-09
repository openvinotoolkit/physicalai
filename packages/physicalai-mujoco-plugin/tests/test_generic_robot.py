# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration checks for generic profiles on the shared scenes (real MuJoCo, Menagerie download)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import pytest

from physicalai.config import Config
from physicalai.robot import SharedRobot
from physicalai.robot.errors import RobotTransportError
from physicalai.robot.interface import Robot
from physicalai_mujoco_plugin.robot import MuJoCoRobot
from physicalai_mujoco_plugin.sim import default_cameras

UR5E_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
# LeRobot's joint names: lerobot/aloha_sim_insertion_human (ALOHA); lerobot/svla_so100_pickplace and
# danaaubakirova/koch_test (SO-100, Koch; there with the arm prefix ``main_``).
ALOHA_JOINTS = [
    f"{side}_{joint}"
    for side in ("left", "right")
    for joint in ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper")
]
SO_ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


@pytest.mark.requires_download
def test_ur5e_runs_in_single_pick_place() -> None:
    """A dataset-tier profile runs in an SO-101 tabletop scene with its derived channels (PR 2 acceptance)."""
    robot = MuJoCoRobot(profile="ur5e", scene="single_pick_place", seed=7, cameras=[])
    assert isinstance(robot, Robot)
    assert robot.joint_names == UR5E_JOINTS  # before connect (DRV-6)
    assert Config.from_instance(robot).instantiate().profile.name == "ur5e"

    robot.connect()
    try:
        observation = robot.get_observation()
        assert observation.joint_positions.shape == (6,)
        assert observation.joint_positions.dtype == np.float32
        assert observation.sensor_data["velocities"].shape == (6,)
        # Home is Menagerie's `home` keyframe, held by the position actuators.
        np.testing.assert_allclose(observation.joint_positions, [-90, -90, 90, -90, -90, 0], atol=0.5)

        target = observation.joint_positions + np.array([10, 0, 0, 0, 0, 10], dtype=np.float32)
        robot.send_action(target)
        for _ in range(100):
            observation = robot.get_observation()
        np.testing.assert_allclose(observation.joint_positions, target, atol=1.0)
        # The UR5e has no camera of its own; it gets a generated wrist camera (CAM-3).
        assert [camera.name for camera in default_cameras(robot._sim)] == ["wrist", "overview"]  # noqa: SLF001
    finally:
        robot.disconnect()


@pytest.mark.parametrize(("profile", "joints"), [("aloha", ALOHA_JOINTS), ("so_arm100", SO_ARM_JOINTS), ("koch", SO_ARM_JOINTS)])
def test_dataset_arms_use_their_lerobot_joint_names(profile: str, joints: list[str]) -> None:
    """Spec issue 4: rename channels only where LeRobot datasets name the joints (no download before connect)."""
    assert MuJoCoRobot(profile=profile, cameras=[]).joint_names == joints


@pytest.mark.requires_download
def test_aloha_is_one_robot_with_both_arms_and_their_wrist_cameras() -> None:
    """Menagerie's ALOHA model keeps its arm spacing, so it attaches whole at single_pick_place's one mount."""
    robot = MuJoCoRobot(profile="aloha", scene="single_pick_place", cameras=[])
    robot.connect()
    try:
        observation = robot.get_observation()
        assert robot.joint_names == ALOHA_JOINTS
        assert observation.state.shape == (14,)
        assert robot._http_status()["units"] == ["degrees"] * 6 + ["metres"] + ["degrees"] * 6 + ["metres"]  # noqa: SLF001
        assert [camera.name for camera in default_cameras(robot._sim)] == [  # noqa: SLF001
            "wrist_cam_left",
            "overview",
            "wrist_cam_right",
        ]
    finally:
        robot.disconnect()


@pytest.mark.requires_download
def test_ur5e_cannot_switch_to_an_so101_only_scene() -> None:
    robot = MuJoCoRobot(profile="ur5e", scene="single_pick_place", cameras=[])
    robot.connect()
    try:
        assert robot._switch_to_scene("conveyor_sort") is False  # noqa: SLF001
        assert robot._http_status()["compatible_scenes"] == ["single_pick_place"]  # noqa: SLF001
    finally:
        robot.disconnect()


@contextmanager
def _shared_robot_for_test(robot: MuJoCoRobot) -> Iterator[SharedRobot]:
    """Retry test-name port collisions without retrying assertions or other startup failures."""
    config = Config.from_instance(robot)
    for attempt in range(3):
        # UUID names still hash into a finite port range. A bind-in-use failure needs a
        # fresh name, not a longer timeout; a port pre-check would race with other sockets.
        shared = SharedRobot.from_config(
            config,
            name=f"mujoco-ur5e-test-{uuid4().hex}",
            rate_hz=50.0,
            idle_timeout=10.0,
        )
        try:
            shared.connect()
        except RobotTransportError as exc:
            # SharedRobot wraps the structured worker error in its exception cause.
            error = exc.__cause__ if isinstance(exc.__cause__, RobotTransportError) else exc
            if attempt == 2 or error.phase != "endpoint_collision" or "Address already in use" not in str(error):
                raise
        else:
            break

    # disconnect() deliberately leaves the owner running and clears its handle.
    owner = shared._owner  # noqa: SLF001
    try:
        yield shared
    finally:
        try:
            shared.disconnect()
        finally:
            if owner is not None:
                owner.stop()


@pytest.mark.slow
@pytest.mark.requires_download
def test_generic_fixed_base_round_trips_through_shared_robot() -> None:
    """A generic profile works through the shared-owner transport (TST-4, fixed base)."""
    robot = MuJoCoRobot(profile="ur5e", scene="single_pick_place", cameras=[])
    with _shared_robot_for_test(robot) as shared:
        assert shared.joint_names == UR5E_JOINTS
        observation = shared.get_observation()
        assert observation.joint_positions.shape == (6,)
        assert observation.state.shape == (6,)
        assert set(observation.sensor_data) == {"velocities"}
        shared.send_action(observation.joint_positions)


@pytest.fixture
def shared_factory(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Keep retry/cleanup regression tests independent of downloads and network sockets."""
    factory = Mock()
    monkeypatch.setattr(SharedRobot, "from_config", factory)
    return factory


def _wrapped_startup_error(phase: str | None, message: str) -> RobotTransportError:
    error = RobotTransportError("failed to start robot owner")
    error.__cause__ = RobotTransportError(message, phase=phase)
    return error


@pytest.mark.parametrize("wrapped", [False, True])
def test_shared_robot_retries_bind_collision_with_a_new_name(shared_factory: Mock, wrapped: bool) -> None:
    collision = RobotTransportError("Address already in use", phase="endpoint_collision")
    if wrapped:
        collision = _wrapped_startup_error("endpoint_collision", "Address already in use")
    failed = Mock(connect=Mock(side_effect=collision))
    owner = Mock()
    connected = Mock(_owner=owner)
    # Model disconnect() clearing the owner handle so teardown must capture it beforehand.
    connected.disconnect.side_effect = lambda: setattr(connected, "_owner", None)
    shared_factory.side_effect = [failed, connected]

    with _shared_robot_for_test(MuJoCoRobot(profile="ur5e", cameras=[])) as shared:
        assert shared is connected
        owner.stop.assert_not_called()

    assert shared_factory.call_count == 2
    assert shared_factory.call_args_list[0].kwargs["name"] != shared_factory.call_args_list[1].kwargs["name"]
    connected.disconnect.assert_called_once_with()
    owner.stop.assert_called_once_with()


@pytest.mark.parametrize(
    ("phase", "message"),
    [
        ("connection_failed", "Address already in use"),
        ("construction_failed", "driver unavailable"),
        ("name_lock_contention", "Address already in use"),
        ("endpoint_collision", "failed to declare queryable"),
        (None, "Address already in use"),
    ],
)
def test_shared_robot_does_not_retry_other_startup_errors(
    shared_factory: Mock, phase: str | None, message: str
) -> None:
    error = _wrapped_startup_error(phase, message)
    shared_factory.return_value = Mock(connect=Mock(side_effect=error))

    with pytest.raises(RobotTransportError) as caught, _shared_robot_for_test(MuJoCoRobot(profile="ur5e", cameras=[])):
        pytest.fail("connection should have failed")

    assert caught.value is error
    shared_factory.assert_called_once()


@pytest.mark.parametrize("error", [RobotTransportError("startup timed out"), ValueError("invalid config")])
def test_shared_robot_does_not_retry_unclassified_connection_errors(shared_factory: Mock, error: Exception) -> None:
    shared_factory.return_value = Mock(connect=Mock(side_effect=error))

    with pytest.raises(type(error)) as caught, _shared_robot_for_test(MuJoCoRobot(profile="ur5e", cameras=[])):
        pytest.fail("connection should have failed")

    assert caught.value is error
    shared_factory.assert_called_once()


def test_shared_robot_collision_retry_is_bounded(shared_factory: Mock) -> None:
    errors = [_wrapped_startup_error("endpoint_collision", "Address already in use") for _ in range(3)]
    shared_factory.side_effect = [Mock(connect=Mock(side_effect=error)) for error in errors]

    with pytest.raises(RobotTransportError) as caught, _shared_robot_for_test(MuJoCoRobot(profile="ur5e", cameras=[])):
        pytest.fail("connection should have failed")

    assert caught.value is errors[-1]
    assert shared_factory.call_count == 3
    assert len({call.kwargs["name"] for call in shared_factory.call_args_list}) == 3


def test_shared_robot_stops_owner_without_retrying_body_failures(shared_factory: Mock) -> None:
    connected = Mock(_owner=Mock())
    shared_factory.return_value = connected

    with (
        pytest.raises(AssertionError, match="observation failed"),
        _shared_robot_for_test(MuJoCoRobot(profile="ur5e", cameras=[])),
    ):
        raise AssertionError("observation failed")

    shared_factory.assert_called_once()
    connected.disconnect.assert_called_once_with()
    connected._owner.stop.assert_called_once_with()


def test_shared_robot_stops_owner_even_if_disconnect_fails(shared_factory: Mock) -> None:
    connected = Mock(_owner=Mock(), disconnect=Mock(side_effect=RuntimeError("disconnect failed")))
    shared_factory.return_value = connected

    with (
        pytest.raises(RuntimeError, match="disconnect failed"),
        _shared_robot_for_test(MuJoCoRobot(profile="ur5e", cameras=[])),
    ):
        pass

    connected._owner.stop.assert_called_once_with()
