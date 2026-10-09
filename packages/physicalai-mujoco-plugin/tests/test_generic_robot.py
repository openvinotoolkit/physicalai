# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration checks for generic profiles on the shared scenes (real MuJoCo, Menagerie download)."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from physicalai.config import Config
from physicalai.robot import SharedRobot
from physicalai.robot.interface import Robot
from physicalai_mujoco_plugin.robot import MuJoCoRobot
from physicalai_mujoco_plugin.sim import default_cameras

UR5E_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]


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
        assert [camera.name for camera in default_cameras(robot._sim)] == ["overview"]  # noqa: SLF001 - UR5e has none
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


@pytest.mark.slow
@pytest.mark.requires_download
def test_generic_fixed_base_round_trips_through_shared_robot() -> None:
    """A generic profile works through the shared-owner transport (TST-4, fixed base)."""
    robot = MuJoCoRobot(profile="ur5e", scene="single_pick_place", cameras=[])
    # The subscriber retries its localhost connection with a backoff, so the first metadata
    # reply can take ~3 s; a shorter idle timeout lets the owner exit before it arrives.
    shared = SharedRobot.from_config(
        Config.from_instance(robot),
        name=f"mujoco-ur5e-test-{uuid4().hex}",
        rate_hz=50.0,
        idle_timeout=10.0,
    )
    try:
        shared.connect()
        assert shared.joint_names == UR5E_JOINTS
        observation = shared.get_observation()
        assert observation.joint_positions.shape == (6,)
        assert observation.state.shape == (6,)
        assert set(observation.sensor_data) == {"velocities"}
        shared.send_action(observation.joint_positions)
    finally:
        shared.disconnect()
