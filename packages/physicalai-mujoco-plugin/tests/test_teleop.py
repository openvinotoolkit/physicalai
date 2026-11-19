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
from physicalai_mujoco_plugin.xr import XRController, XRPose, XRSample, xr_to_robot

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
    return data.xpos[arm.body] + rotation @ arm.tool_offset


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
        assert left.reached, "on its way, the arm is not out of reach yet"
        _ticks(robot, 300)
        assert not left.moving
        assert not left.reached, "stopped short of the target"
        np.testing.assert_allclose(left.target_pos, far)  # the handle stays under the pointer
        robot._submit_command(SetToolTargetCommand(left.name, far, _wxyz(left.target_rot), dragging=False))  # noqa: SLF001
        _ticks(robot, 3)
        assert left.target_pos[0] < far[0] - 0.5
        _ticks(robot, 300)
        assert left.reached, "the target moved to where the gripper stopped: reached"
        assert "out of reach" not in _teleop_markdown(robot._http_status()["teleop"])  # noqa: SLF001

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


# --- Grab points ------------------------------------------------------------------------------------


def _send_world(robot: MuJoCoRobot, controllers: dict[str, tuple[np.ndarray, float, float]]) -> None:
    """Send controllers at robot-world positions: ``{side: (position, squeeze, trigger)}``."""
    frame = XRPose(np.zeros(3), np.eye(3))
    rotation, translation = xr_to_robot(frame, _teleop(robot).view)
    poses = {
        side: XRController(XRPose(rotation.T @ (np.asarray(position) - translation), rotation.T), trigger, squeeze)
        for side, (position, squeeze, trigger) in controllers.items()
    }
    robot._xr.sample = XRSample(time.monotonic(), None, poses, frame)  # type: ignore[union-attr]  # noqa: SLF001


def _point(robot: MuJoCoRobot, index: int) -> np.ndarray:
    point = _teleop(robot).grab_points[index]
    data = robot._sim.data  # noqa: SLF001
    return data.xpos[point.body] + data.xmat[point.body].reshape(3, 3) @ np.asarray(point.offset)


def _index(robot: MuJoCoRobot, arm: TeleopArm, joint: str | None) -> int:
    points = _teleop(robot).grab_points
    return next(
        i for i, p in enumerate(points) if p.arm == arm.name and (p.joint is None if joint is None else arm.joints[p.joint or 0] == joint)
    )


@pytest.mark.requires_download
class TestGrabPoints:
    def test_each_arm_offers_its_tool_point_and_the_joints_worth_grabbing(self, robot: MuJoCoRobot) -> None:
        left = _teleop(robot).arms[0]
        # The shoulder joints stand still or close to it, and joint 7 shares joint 6's place.
        assert [None if p.joint is None else left.joints[p.joint] for p in left.points] == [
            None,
            "openarm_left_joint4",
            "openarm_left_joint5",
            "openarm_left_joint6",
        ]
        assert len(_teleop(robot).grab_points) == 8

    def test_a_controller_near_a_point_hovers_it(self, headset_robot: MuJoCoRobot) -> None:
        robot = headset_robot
        left = _teleop(robot).arms[0]
        elbow = _index(robot, left, "openarm_left_joint4")
        _send_world(robot, {"left": (_point(robot, elbow) + (0.0, 0.0, 0.03), 0.0, 0.0)})
        _ticks(robot, 2)
        assert _teleop(robot).grab_state() == {"hover": {"left": elbow}, "held": {}}
        _send_world(robot, {"left": (_point(robot, elbow) + (0.0, 0.0, 0.2), 0.0, 0.0)})
        _ticks(robot, 2)
        assert _teleop(robot).grab_state() == {"hover": {}, "held": {}}

    def test_holding_a_joint_moves_it_and_the_arm_beyond_rigidly(self, headset_robot: MuJoCoRobot) -> None:
        robot = headset_robot
        left = _teleop(robot).arms[0]
        elbow = _index(robot, left, "openarm_left_joint4")
        start = _point(robot, elbow)
        _send_world(robot, {"left": (start, 0.0, 0.0)})
        _ticks(robot, 2)
        before = _joints(left).copy()
        _send_world(robot, {"left": (start, 1.0, 0.0)})
        _ticks(robot, 2)
        assert _teleop(robot).grab_state()["held"] == {"left": elbow}
        _send_world(robot, {"left": (start + (0.05, 0.04, 0.0), 1.0, 0.0)})  # the elbow out and forward
        _ticks(robot, 150)
        assert np.linalg.norm(_point(robot, elbow) - (start + (0.05, 0.04, 0.0))) < 0.015
        np.testing.assert_allclose(_joints(left)[3:], before[3:], atol=1e-9)  # the elbow and beyond keep their angles
        np.testing.assert_allclose(left.target_pos, _tool(robot, left), atol=0.03)  # the gripper target came along

    def test_holding_the_gripper_and_the_elbow_places_both(self, headset_robot: MuJoCoRobot) -> None:
        robot = headset_robot
        left = _teleop(robot).arms[0]
        elbow = _index(robot, left, "openarm_left_joint4")
        tool, joint = _point(robot, _index(robot, left, None)), _point(robot, elbow)
        _send_world(robot, {"left": (joint, 1.0, 0.0), "right": (tool, 1.0, 0.0)})
        _ticks(robot, 2)
        assert _teleop(robot).grab_state()["held"] == {"left": elbow, "right": _index(robot, left, None)}
        target_pos = left.target_pos.copy()
        _send_world(robot, {"left": (joint + (0.0, 0.08, 0.0), 1.0, 0.0), "right": (tool, 1.0, 0.0)})  # the elbow out
        _ticks(robot, 200)
        np.testing.assert_allclose(left.target_pos, target_pos)  # the gripper's target stays
        assert np.linalg.norm(_tool(robot, left) - target_pos) < 0.03
        assert _point(robot, elbow)[1] > joint[1] + 0.03  # the elbow swung out as far as the arm allows

    def test_a_placed_elbow_stays_placed_when_the_gripper_moves_after(self, headset_robot: MuJoCoRobot) -> None:
        robot = headset_robot
        left = _teleop(robot).arms[0]
        elbow, tool = _index(robot, left, "openarm_left_joint4"), _index(robot, left, None)
        joint, hand = _point(robot, elbow), _point(robot, tool)
        _send_world(robot, {"left": (joint, 1.0, 0.0), "right": (hand, 1.0, 0.0)})
        _ticks(robot, 2)
        _send_world(robot, {"left": (joint + (0.0, 0.08, 0.0), 1.0, 0.0), "right": (hand, 1.0, 0.0)})
        _ticks(robot, 200)
        assert _point(robot, elbow)[1] > joint[1] + 0.05
        _send_world(robot, {"left": (joint, 0.0, 0.0), "right": (hand, 1.0, 0.0)})  # let go of the elbow
        _ticks(robot, 2)
        _send_world(robot, {"left": (joint, 0.0, 0.0), "right": (hand + (0.03, 0.0, 0.0), 1.0, 0.0)})
        _ticks(robot, 200)
        assert _point(robot, elbow)[1] > joint[1] + 0.03  # still swung out: not pulled back toward home

    def test_an_elbow_held_where_the_arm_cannot_follow_never_pulls_the_gripper_off(self, headset_robot: MuJoCoRobot) -> None:
        robot = headset_robot
        teleop = _teleop(robot)
        left = teleop.arms[0]
        elbow, tool = _index(robot, left, "openarm_left_joint4"), _index(robot, left, None)
        joint, hand = _point(robot, elbow), _point(robot, tool)
        _send_world(robot, {"left": (joint, 1.0, 0.0), "right": (hand, 1.0, 0.0)})
        _ticks(robot, 2)
        target = left.target_pos.copy()
        _send_world(robot, {"left": (joint + (-0.3, 0.0, 0.0), 1.0, 0.0), "right": (hand, 1.0, 0.0)})  # 30 cm back
        _ticks(robot, 200)
        solved, _ = teleop._tool(left, left.goal)  # noqa: SLF001
        assert np.linalg.norm(solved - target) < 0.01  # the gripper's pose first
        assert left.reached

    def test_the_held_gripper_stays_on_its_pose_all_the_way_while_the_elbow_moves(self, headset_robot: MuJoCoRobot) -> None:
        """Every tick's joint targets keep the gripper's pose (what the virtual leader publishes), not only the end."""
        robot = headset_robot
        teleop = _teleop(robot)
        left = teleop.arms[0]
        elbow, tool = _index(robot, left, "openarm_left_joint4"), _index(robot, left, None)
        joint, hand = _point(robot, elbow), _point(robot, tool)
        _send_world(robot, {"left": (joint, 1.0, 0.0), "right": (hand, 1.0, 0.0)})
        _ticks(robot, 2)
        target_pos, target_rot = left.target_pos.copy(), left.target_rot.copy()
        _send_world(robot, {"left": (joint + (0.0, 0.2, 0.0), 1.0, 0.0), "right": (hand, 1.0, 0.0)})
        worst_position = worst_angle = 0.0
        for _ in range(100):
            _ticks(robot, 1)
            position, rotation = teleop._tool(left, _joints(left))  # noqa: SLF001
            worst_position = max(worst_position, float(np.linalg.norm(position - target_pos)))
            cosine = (np.trace(rotation.T @ target_rot) - 1) / 2
            worst_angle = max(worst_angle, math.degrees(math.acos(min(max(cosine, -1.0), 1.0))))
        assert worst_position < 0.01
        assert worst_angle < 5.0
        assert _point(robot, elbow)[1] > joint[1] + 0.03  # and the elbow did swing out

    def test_an_elbow_let_go_early_stays_where_it_was_not_where_it_was_headed(self, headset_robot: MuJoCoRobot) -> None:
        robot = headset_robot
        teleop = _teleop(robot)
        left = teleop.arms[0]
        elbow, tool = _index(robot, left, "openarm_left_joint4"), _index(robot, left, None)
        joint, hand = _point(robot, elbow), _point(robot, tool)
        _send_world(robot, {"left": (joint, 1.0, 0.0), "right": (hand, 1.0, 0.0)})
        _ticks(robot, 2)
        _send_world(robot, {"left": (joint + (0.0, 0.3, 0.0), 1.0, 0.0), "right": (hand, 1.0, 0.0)})
        _ticks(robot, 1)  # the joint targets have only begun to move
        _send_world(robot, {"left": (joint, 0.0, 0.0), "right": (hand, 1.0, 0.0)})
        _ticks(robot, 1)
        np.testing.assert_allclose(left.posture, _joints(left), atol=HEADSET_JOINT_SPEED * _control_dt(robot) + 1e-9)
        released = _joints(left).copy()
        _ticks(robot, 100)
        assert np.max(np.abs(np.degrees(_joints(left) - released))) < 3.0  # it does not carry on toward the far goal

    def test_a_far_grip_takes_the_controllers_own_gripper_and_the_trigger_closes_it(self, headset_robot: MuJoCoRobot) -> None:
        robot = headset_robot
        right = _teleop(robot).arms[1]
        far = np.array([0.0, -0.6, 0.9])
        _send_world(robot, {"right": (far, 1.0, 1.0)})
        _ticks(robot, 2)
        assert _teleop(robot).grab_state()["held"] == {"right": _index(robot, right, None)}
        assert right.opening == 0.0

    def test_the_leader_publishes_the_teleoperated_arms_for_studio_to_record(self, headset_robot: MuJoCoRobot) -> None:
        """Studio's virtual leader (``GET /leader``) reads the joint targets the headset sets."""
        robot = headset_robot
        _send(robot, position=(0.0, 1.0, 0.0), squeeze=1.0)
        _ticks(robot, 2)
        _send(robot, position=(0.0, 1.0, -0.08), squeeze=1.0)
        _ticks(robot, 100)
        leader = robot._automation.leader_snapshot()  # type: ignore[union-attr]  # noqa: SLF001
        assert leader["joint_names"] == robot.joint_names
        targets = robot._targets_to_units(robot._arm_targets())  # noqa: SLF001
        np.testing.assert_allclose(leader["joint_positions"], targets, atol=1e-6)
        index = robot.joint_names.index("left_joint_4")
        assert abs(leader["joint_positions"][index] - 90.0) > 1.0  # the arm moved off its home

