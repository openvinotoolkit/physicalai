# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Teleoperation on the OpenArm twin: the viewer's gripper handles and a headset's controllers."""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import math
import queue
import time
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import pytest
from fastapi.testclient import TestClient

from physicalai_mujoco_plugin.http_server import HomeCommand, SetTeleopCommand, SetToolTargetCommand, build_app
from physicalai_mujoco_plugin.robot import MuJoCoRobot
from physicalai_mujoco_plugin.teleop import HEADSET_JOINT_SPEED, MOUSE_JOINT_SPEED
from physicalai_mujoco_plugin.viser_controls import _teleop_markdown
from physicalai_mujoco_plugin.xr import XRController, XRPose, XRSample

if TYPE_CHECKING:
    from collections.abc import Iterator

    from physicalai_mujoco_plugin.teleop import Teleop, TeleopArm


class _FakeHeadset:
    """Stands in for ``XRServer``: the test sets the sample."""

    def __init__(self) -> None:
        self.sample: XRSample | None = None

    def latest(self, max_age: float = 0.5) -> XRSample | None:  # noqa: ARG002
        return self.sample

    def status(self) -> dict:
        return {"urls": [], "connected": True, "session": self.sample is not None, "controllers": []}

    def stop(self) -> None:
        pass


@pytest.fixture
def robot() -> Iterator[MuJoCoRobot]:
    robot = MuJoCoRobot("openarm", cameras=[])
    robot.connect()
    yield robot
    robot.disconnect()


def _ticks(robot: MuJoCoRobot, count: int) -> None:
    for _ in range(count):
        robot.get_observation()


def _teleop(robot: MuJoCoRobot) -> Teleop:
    teleop = robot._teleop  # noqa: SLF001
    assert teleop is not None
    return teleop


def _joints(arm: TeleopArm) -> np.ndarray:
    return arm.channels.model_targets()[list(arm.joint_channels)]


def _tool(robot: MuJoCoRobot, arm: TeleopArm) -> np.ndarray:
    data = robot._sim.data  # noqa: SLF001
    rotation = data.xmat[arm.body].reshape(3, 3)
    return data.xpos[arm.body] + rotation @ np.asarray(arm.effector.pos)


def _wxyz(rotation: np.ndarray) -> tuple[float, ...]:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation.reshape(9))
    return tuple(quat)


def _control_dt(robot: MuJoCoRobot) -> float:
    return robot._substeps * float(robot._sim.model.opt.timestep)  # noqa: SLF001


@pytest.mark.requires_download
class TestHandles:
    def test_both_arms_are_teleoperable_with_their_grippers(self, robot: MuJoCoRobot) -> None:
        arms = _teleop(robot).arms
        assert [(sorted(arm.sides), len(arm.joints), arm.gripper is not None) for arm in arms] == [
            (["left"], 7, True),
            (["right"], 7, True),
        ]
        assert [arm["name"] for arm in robot._http_status()["teleop"]["arms"]] == [  # noqa: SLF001
            "openarm_left_ee_base_link",
            "openarm_right_ee_base_link",
        ]

    def test_dragging_a_handle_moves_only_its_arm(self, robot: MuJoCoRobot) -> None:
        robot._submit_command(SetTeleopCommand(mouse=True))  # noqa: SLF001
        _ticks(robot, 2)
        left, right = _teleop(robot).arms
        right_start = _joints(right)
        goal = left.target_pos + (0.05, 0.05, -0.04)
        robot._submit_command(SetToolTargetCommand(left.name, tuple(goal), _wxyz(left.target_rot), dragging=True))  # noqa: SLF001
        _ticks(robot, 150)
        assert left.reached
        # The position actuators let the arm sag a little under gravity below the IK's pose.
        assert np.linalg.norm(_tool(robot, left) - goal) < 0.03
        np.testing.assert_allclose(_joints(right), right_start, atol=1e-6)

    def test_letting_go_of_an_unreachable_target_puts_it_back_at_the_gripper(self, robot: MuJoCoRobot) -> None:
        robot._submit_command(SetTeleopCommand(mouse=True))  # noqa: SLF001
        _ticks(robot, 2)
        left = _teleop(robot).arms[0]
        far = tuple(left.target_pos + (1.5, 0.0, 0.0))
        robot._submit_command(SetToolTargetCommand(left.name, far, _wxyz(left.target_rot), dragging=True))  # noqa: SLF001
        _ticks(robot, 3)
        assert not left.reached
        np.testing.assert_allclose(left.target_pos, far)  # the handle stays under the pointer
        robot._submit_command(SetToolTargetCommand(left.name, far, _wxyz(left.target_rot), dragging=False))  # noqa: SLF001
        _ticks(robot, 3)
        assert left.target_pos[0] < far[0] - 0.5

    def test_teleoperation_ignores_actions_until_it_is_switched_off(self, robot: MuJoCoRobot) -> None:
        robot._submit_command(SetTeleopCommand(mouse=True))  # noqa: SLF001
        _ticks(robot, 2)
        held = robot.get_observation().joint_positions.copy()
        moved = held.copy()
        index = robot.joint_names.index("left_joint_1")
        moved[index] += 20.0
        robot.send_action(moved)
        _ticks(robot, 60)
        assert robot.get_observation().joint_positions[index] == pytest.approx(held[index], abs=1.0)
        robot._submit_command(SetTeleopCommand(mouse=False))  # noqa: SLF001
        _ticks(robot, 1)
        robot.send_action(moved)
        _ticks(robot, 100)
        assert robot.get_observation().joint_positions[index] == pytest.approx(moved[index], abs=1.0)

    def test_joint_targets_move_no_faster_than_the_speed_limit(self, robot: MuJoCoRobot) -> None:
        robot._submit_command(SetTeleopCommand(mouse=True))  # noqa: SLF001
        _ticks(robot, 2)
        left = _teleop(robot).arms[0]
        before = _joints(left)
        goal = tuple(left.target_pos + (0.0, 0.25, 0.1))
        robot._submit_command(SetToolTargetCommand(left.name, goal, _wxyz(left.target_rot), dragging=True))  # noqa: SLF001
        _ticks(robot, 1)
        assert np.max(np.abs(_joints(left) - before)) <= MOUSE_JOINT_SPEED * _control_dt(robot) + 1e-9

    def test_going_home_moves_the_targets_with_the_arms(self, robot: MuJoCoRobot) -> None:
        robot._submit_command(SetTeleopCommand(mouse=True))  # noqa: SLF001
        _ticks(robot, 2)
        left = _teleop(robot).arms[0]
        home = left.target_pos.copy()
        moved = tuple(home + (0.0, 0.1, 0.0))
        robot._submit_command(SetToolTargetCommand(left.name, moved, _wxyz(left.target_rot), dragging=False))  # noqa: SLF001
        _ticks(robot, 100)
        robot._submit_command(HomeCommand())  # noqa: SLF001
        _ticks(robot, 2)
        np.testing.assert_allclose(left.target_pos, home, atol=1e-6)


# --- The headset ------------------------------------------------------------------------------------


@pytest.fixture
def headset_robot(robot: MuJoCoRobot) -> MuJoCoRobot:
    robot._xr = _FakeHeadset()  # type: ignore[assignment]  # noqa: SLF001
    return robot


def _yaw(degrees: float) -> np.ndarray:
    """A turn about WebXR's vertical (+y): positive turns -z (ahead) toward -x (left)."""
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _send(
    robot: MuJoCoRobot,
    *,
    position: tuple[float, ...],
    squeeze: float,
    trigger: float = 0.0,
    side: str = "left",
    rotation: np.ndarray | None = None,
    turn: float = 0.0,
) -> None:
    """Send one frame: the robot shown with the eye point at the origin, its ahead along WebXR's turned -z."""
    pose = XRPose(np.asarray(position, dtype=float), np.eye(3) if rotation is None else rotation)
    frame = XRPose(np.zeros(3), _yaw(turn))
    sample = XRSample(time.monotonic(), None, {side: XRController(pose, trigger, squeeze)}, frame)
    robot._xr.sample = sample  # type: ignore[union-attr]  # noqa: SLF001


@pytest.mark.requires_download
class TestHeadset:
    def test_the_arm_moves_only_while_its_grip_is_held(self, headset_robot: MuJoCoRobot) -> None:
        robot = headset_robot
        _send(robot, position=(-0.2, 1.0, -0.3), squeeze=0.0)
        _ticks(robot, 5)
        left, right = _teleop(robot).arms
        start, held_right = left.target_pos.copy(), _joints(right).copy()
        _send(robot, position=(-0.2, 1.0, -0.4), squeeze=0.0)  # moved, not gripped
        _ticks(robot, 5)
        np.testing.assert_allclose(left.target_pos, start)
        _send(robot, position=(-0.2, 1.0, -0.4), squeeze=1.0)  # grip: the arm is held from here
        _ticks(robot, 2)
        _send(robot, position=(-0.2, 1.05, -0.45), squeeze=1.0)  # 5 cm ahead and 5 cm up
        _ticks(robot, 150)
        np.testing.assert_allclose(left.target_pos, start + (0.05, 0.0, 0.05), atol=1e-9)
        assert left.reached
        _send(robot, position=(-0.4, 1.3, -0.1), squeeze=0.0)  # let go, then move away
        _ticks(robot, 5)
        np.testing.assert_allclose(left.target_pos, start + (0.05, 0.0, 0.05), atol=1e-9)
        np.testing.assert_allclose(_joints(right), held_right)  # the other controller never gripped

    def test_ahead_is_the_robots_ahead_as_the_page_shows_it(self, headset_robot: MuJoCoRobot) -> None:
        robot = headset_robot
        _send(robot, position=(0.0, 1.0, -0.3), squeeze=1.0, turn=90.0)  # the robot faces WebXR's -x
        _ticks(robot, 2)
        left = _teleop(robot).arms[0]
        start = left.target_pos.copy()
        _send(robot, position=(-0.05, 1.0, -0.3), squeeze=1.0, turn=90.0)  # 5 cm toward where it faces
        _ticks(robot, 2)
        np.testing.assert_allclose(left.target_pos, start + (0.05, 0.0, 0.0), atol=1e-9)

    def test_the_trigger_closes_the_gripper(self, headset_robot: MuJoCoRobot) -> None:
        robot = headset_robot
        _send(robot, position=(-0.2, 1.0, -0.3), squeeze=0.0, trigger=1.0)
        _ticks(robot, 100)
        data = robot._sim.data  # noqa: SLF001
        assert data.joint("openarm_left_finger_joint1").qpos[0] == pytest.approx(0.0, abs=0.02)
        _send(robot, position=(-0.2, 1.0, -0.3), squeeze=0.0, trigger=0.0)
        _ticks(robot, 100)
        assert data.joint("openarm_left_finger_joint1").qpos[0] == pytest.approx(math.radians(45.0), abs=0.03)

    def test_joint_targets_move_no_faster_than_the_headset_speed(self, headset_robot: MuJoCoRobot) -> None:
        robot = headset_robot
        _send(robot, position=(0.0, 1.0, 0.0), squeeze=1.0)
        _ticks(robot, 2)
        left = _teleop(robot).arms[0]
        before = _joints(left)
        _send(robot, position=(0.0, 1.2, -0.2), squeeze=1.0)
        _ticks(robot, 1)
        assert np.max(np.abs(_joints(left) - before)) <= HEADSET_JOINT_SPEED * _control_dt(robot) + 1e-9

    def test_actions_are_ignored_during_a_session_and_count_again_after_it(self, headset_robot: MuJoCoRobot) -> None:
        robot = headset_robot
        _send(robot, position=(-0.2, 1.0, -0.3), squeeze=0.0)
        _ticks(robot, 2)
        moved = robot.get_observation().joint_positions.copy()
        index = robot.joint_names.index("left_joint_1")
        moved[index] += 20.0
        robot.send_action(moved)
        _ticks(robot, 50)
        assert robot.get_observation().joint_positions[index] != pytest.approx(moved[index], abs=1.0)
        robot._xr.sample = None  # type: ignore[union-attr]  # noqa: SLF001 - the session ended
        _ticks(robot, 1)
        robot.send_action(moved)
        _ticks(robot, 100)
        assert robot.get_observation().joint_positions[index] == pytest.approx(moved[index], abs=1.0)

    def test_the_elbow_comes_back_after_a_wandering_round_trip(self, headset_robot: MuJoCoRobot) -> None:
        """Without the posture in the IK's null space, the elbow ends up about 8 degrees elsewhere."""
        robot = headset_robot
        _send(robot, position=(-0.2, 1.0, -0.3), squeeze=1.0)
        _ticks(robot, 2)
        left = _teleop(robot).arms[0]
        before = _joints(left).copy()
        path = np.cumsum(np.random.default_rng(0).normal(0.0, 0.01, (150, 3)), axis=0)
        for index, offset in enumerate([*path, *path[::-1]]):
            angle = 0.6 * math.sin(index / 20)
            roll = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(angle), -math.sin(angle)], [0.0, math.sin(angle), math.cos(angle)]])
            _send(robot, position=tuple(np.array([-0.2, 1.0, -0.3]) + offset), squeeze=1.0, rotation=roll)
            _ticks(robot, 2)
        _send(robot, position=(-0.2, 1.0, -0.3), squeeze=1.0)
        _ticks(robot, 200)
        assert left.reached
        assert np.max(np.abs(np.degrees(_joints(left) - before))) < 0.5

    def test_the_operator_looks_out_from_the_shoulders_along_the_arms(self, robot: MuJoCoRobot) -> None:
        view = _teleop(robot).view
        np.testing.assert_allclose(view.eye, (0.0, 0.0, 0.335 + 0.27), atol=0.01)  # the shoulders at 0.335 m
        assert view.yaw == pytest.approx(0.0, abs=0.05)


def test_post_teleop_switches_the_handles_and_refuses_robots_without_grippers() -> None:
    commands: queue.Queue = queue.Queue()
    status: dict = {"teleop": {"arms": [{"name": "gripper"}]}}
    client = TestClient(build_app(service_name="t", buffers={}, commands=commands, get_status=lambda: status))
    response = client.post("/teleop", json={"mouse": True})
    assert response.status_code == 200
    assert commands.get_nowait() == SetTeleopCommand(mouse=True)
    assert client.post("/teleop", json={"mouse": True, "hands": True}).status_code == 422
    status["teleop"] = {"arms": []}
    assert client.post("/teleop", json={"mouse": True}).status_code == 409


def test_the_panel_says_what_drives_the_arms_and_where_to_open_the_headset_page() -> None:
    arms = [{"name": "left_hand", "held": True, "reached": True}, {"name": "right_hand", "held": False, "reached": False}]
    xr = {"urls": ["https://mac.local:8443/"], "connected": False, "session": False, "controllers": []}
    off = _teleop_markdown({"mouse": False, "headset": False, "arms": arms, "xr": xr})
    assert off.startswith("Off:")
    assert "`https://mac.local:8443/`" in off
    xr = {**xr, "connected": True, "session": True, "controllers": ["left"]}
    on = _teleop_markdown({"mouse": False, "headset": True, "arms": arms, "xr": xr})
    assert "in a session (left)" in on
    assert "**left_hand**: held" in on
    assert "**right_hand**: holding still, out of reach" in on
