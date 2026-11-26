# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Headset teleoperation on every arm profile: the viewpoint, a far grip, the grippers, and a still grip that moves nothing."""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import pytest

from physicalai_mujoco_plugin.profiles.registry import PROFILES
from physicalai_mujoco_plugin.robot import MuJoCoRobot
from physicalai_mujoco_plugin.xr import XRController, XRPose, XRSample, xr_to_robot

if TYPE_CHECKING:
    from collections.abc import Iterator

    from physicalai_mujoco_plugin.teleop import Teleop, TeleopArm

ARM_PROFILES = [name for name, profile in PROFILES.items() if profile.end_effectors]
GRIPPER_PROFILES = [name for name in ARM_PROFILES if any(e.gripper for e in PROFILES[name].end_effectors)]


class _FakeHeadset:
    def __init__(self) -> None:
        self.sample: XRSample | None = None

    def latest(self, max_age: float = 0.5) -> XRSample | None:  # noqa: ARG002
        return self.sample

    def status(self) -> dict:
        return {"urls": [], "code": None, "connected": True, "session": self.sample is not None, "controllers": []}

    def set_grab_state(self, state: object) -> None:
        pass

    def stop(self) -> None:
        pass


def _robot(name: str, *, bimanual: bool = False) -> Iterator[MuJoCoRobot]:
    robot = MuJoCoRobot(name, cameras=[], bimanual=bimanual)
    robot.connect()
    try:
        robot._xr = _FakeHeadset()  # type: ignore[assignment]  # noqa: SLF001
        yield robot
    finally:
        robot.disconnect()


def _teleop(robot: MuJoCoRobot) -> Teleop:
    teleop = robot._teleop  # noqa: SLF001
    assert teleop is not None
    return teleop


def _send(robot: MuJoCoRobot, controllers: dict[str, tuple[np.ndarray, float, float]]) -> None:
    """Send controllers at robot-world positions: ``{side: (position, squeeze, trigger)}``."""
    frame = XRPose(np.zeros(3), np.eye(3))
    rotation, translation = xr_to_robot(frame, _teleop(robot).view)
    poses = {
        side: XRController(XRPose(rotation.T @ (np.asarray(position) - translation), rotation.T), trigger, squeeze)
        for side, (position, squeeze, trigger) in controllers.items()
    }
    robot._xr.sample = XRSample(time.monotonic(), None, poses, frame)  # type: ignore[union-attr]  # noqa: SLF001


def _ticks(robot: MuJoCoRobot, count: int) -> None:
    for _ in range(count):
        robot.get_observation()


def _far(robot: MuJoCoRobot) -> np.ndarray:
    return np.asarray(_teleop(robot).view.eye) + (0.0, 0.0, 0.5)  # above the head: away from the arm


def _finger_spread(robot: MuJoCoRobot, arm: TeleopArm) -> float:
    """Mean distance from the tool point of the geoms that move with the gripper (its fingers or jaw)."""
    model, data = robot._sim.model, robot._sim.data  # noqa: SLF001
    chain_bodies = {int(model.joint(joint).bodyid[0]) for joint in arm.joints}
    teleop = _teleop(robot)
    tool, _ = teleop._tool(arm, teleop._targets(arm))  # noqa: SLF001

    hand = int(model.body_parentid[arm.body])  # the fingers hang from the tool's body or its parent

    def moves(body: int) -> bool:  # under the hand, below a joint that is not the arm's
        moving = False
        while body > 0:
            if body == hand:
                return moving
            moving = moving or (model.body_jntnum[body] > 0 and body not in chain_bodies)
            body = int(model.body_parentid[body])
        return False

    geoms = [geom for geom in range(model.ngeom) if moves(int(model.geom_bodyid[geom]))]
    return float(np.mean([np.linalg.norm(data.geom_xpos[geom] - tool) for geom in geoms]))


@pytest.mark.mujoco_smoke
@pytest.mark.parametrize("name", GRIPPER_PROFILES)
def test_the_trigger_closes_and_opens_every_gripper(name: str) -> None:
    for robot in _robot(name):
        for arm in _teleop(robot).arms:
            assert arm.gripper is not None, f"{arm.name} has its gripper"
            side = sorted(arm.sides)[0]
            spreads = []
            for trigger, target in ((1.0, arm.gripper.closed), (0.0, arm.gripper.opened)):
                _send(robot, {side: (_far(robot), 0.0, trigger)})
                _ticks(robot, 300)
                assert arm.channels.model_targets()[arm.gripper.channel] == pytest.approx(target)
                spreads.append(_finger_spread(robot, arm))
            assert spreads[0] < spreads[1] - 0.005, f"{arm.name}: the trigger closes it, letting go opens it"


@pytest.mark.mujoco_smoke
def test_a_still_grip_leaves_a_continuous_joint_where_it_is() -> None:
    """Kinova's first joint has no limits: its winding (here a full turn) is kept, not folded back."""
    for robot in _robot("kinova_gen3"):
        arm = _teleop(robot).arms[0]
        robot._sim.channels[0].place({arm.joints[0]: 2 * math.pi})  # noqa: SLF001
        _ticks(robot, 2)
        before = _teleop(robot)._targets(arm)  # noqa: SLF001
        _send(robot, {"right": (_far(robot), 1.0, 0.0)})
        _ticks(robot, 50)
        moved = np.degrees(np.abs(_teleop(robot)._targets(arm) - before))  # noqa: SLF001
        assert moved.max() < 0.5, f"a still grip moved the joints {moved.round(1)} degrees"


@pytest.mark.mujoco_smoke
@pytest.mark.parametrize("name", ["so101", "trossen_wxai", "franka_fr3"])
def test_each_trigger_closes_its_own_robots_gripper_in_a_pair(name: str) -> None:
    """Two robots name their channels after their anchors (``left_gripper``); each controller drives its own."""
    for robot in _robot(name, bimanual=True):
        left, right = _teleop(robot).arms
        assert left.gripper is not None
        assert right.gripper is not None
        _send(robot, {"left": (_far(robot), 0.0, 1.0), "right": (_far(robot), 0.0, 0.0)})
        _ticks(robot, 5)
        assert left.channels.model_targets()[left.gripper.channel] == pytest.approx(left.gripper.closed)
        assert right.channels.model_targets()[right.gripper.channel] == pytest.approx(right.gripper.opened)


FORWARD_M = 0.03
PAIRS = ["so101", "trossen_wxai"]
"""Profiles also run as two robots (``--bimanual``): one arm each, the left controller the left robot."""


@pytest.mark.mujoco_smoke
@pytest.mark.parametrize(
    ("name", "bimanual"), [*((name, False) for name in ARM_PROFILES), *((name, True) for name in PAIRS)]
)
def test_a_far_grip_moves_each_gripper_ahead(name: str, bimanual: bool) -> None:  # noqa: FBT001
    for robot in _robot(name, bimanual=bimanual):
        teleop = _teleop(robot)
        assert teleop.arms, "an arm profile gives the headset arms to drive"
        if bimanual:
            assert [sorted(arm.sides) for arm in teleop.arms] == [["left"], ["right"]]
        view = teleop.view
        ahead = np.array([math.cos(view.yaw), math.sin(view.yaw), 0.0])
        for arm in teleop.arms:
            teleop._forward(arm, arm.home)  # noqa: SLF001
            top = max(float(teleop._data.xanchor[joint][2]) for joint in arm.joint_ids)  # noqa: SLF001
            assert view.eye[2] > top, "the eyes stand above the arm"
            assert arm.points[0].joint is None
            if len(arm.joints) >= 5:  # noqa: PLR2004
                assert len(arm.points) >= 2, "a longer arm offers a joint to grab"
        for arm in teleop.arms:
            side = sorted(arm.sides)[0]
            far = _far(robot)  # no grab point near: the controller grabs its own arm's gripper
            _send(robot, {side: (far, 0.0, 0.0)})
            _ticks(robot, 1)
            _send(robot, {side: (far, 1.0, 0.0)})
            _ticks(robot, 1)
            start = arm.target_pos.copy()
            _send(robot, {side: (far + FORWARD_M * ahead, 1.0, 0.0)})
            _ticks(robot, 100)
            np.testing.assert_allclose(arm.target_pos, start + FORWARD_M * ahead, atol=1e-9)
            reached, _ = teleop._tool(arm, arm.goal)  # noqa: SLF001
            assert np.linalg.norm(reached - arm.target_pos) < 0.015, f"{arm.name} follows the controller"
            _send(robot, {side: (far, 0.0, 0.0)})
            _ticks(robot, 1)


@pytest.mark.mujoco_smoke
def test_letting_go_of_a_reachable_target_on_the_way_keeps_it() -> None:
    """Let go of a handle before the arm gets there: the target stays, and the arm still reaches it.

    The FR3's last joint turned 120 degrees takes the arm longer than one solve, and longer than the
    drag; the target is out of reach only if the arm stops short of it.
    """
    for robot in _robot("franka_fr3"):
        teleop = _teleop(robot)
        teleop.set_inputs(mouse=True)
        _ticks(robot, 2)
        arm = teleop.arms[0]
        turned = teleop._targets(arm)  # noqa: SLF001
        turned[-1] += math.radians(120.0)
        position, rotation = teleop._tool(arm, turned)  # noqa: SLF001
        wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(wxyz, rotation.reshape(9))
        teleop.set_target(arm.name, tuple(position), tuple(wxyz), dragging=True)
        _ticks(robot, 5)
        teleop.set_target(arm.name, tuple(position), tuple(wxyz), dragging=False)
        _ticks(robot, 300)
        np.testing.assert_allclose(arm.target_rot, rotation, atol=1e-6)
        _, reached = teleop._tool(arm, teleop._targets(arm))  # noqa: SLF001
        angle = math.degrees(math.acos(min(max((np.trace(reached.T @ rotation) - 1) / 2, -1.0), 1.0)))
        assert angle < 1.0

