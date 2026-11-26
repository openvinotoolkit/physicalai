# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Arm teleoperation inside the simulation: drag the grippers in the 3D viewer, or move the arms with VR controllers.

Every arm of the robot (one per attached robot and ``RobotProfile.end_effectors`` entry) has a
gripper target: a pose of its tool point. Damped least-squares inverse kinematics turns the targets
into joint targets, which each control tick writes to the arm's channels like an action would,
moving each joint at most ``MOUSE_JOINT_SPEED`` or ``HEADSET_JOINT_SPEED``. While teleoperation
drives the arms, actions from Studio or a policy are ignored. Two inputs move the targets:

* ``mouse``: the viewer shows a handle at each gripper (:meth:`Teleop.set_target`). Letting go of a
  target the arm cannot reach puts it back where the gripper got to.
* the headset: VR controllers over WebXR (:mod:`~physicalai_mujoco_plugin.xr`), whose poses arrive in
  the robot's world frame. Each arm offers grab points (:class:`GrabPoint`): its tool point, and
  the joints along it that the joints before them move. A controller within ``HOVER_RADIUS_M`` of
  one hovers it (the headset shows a ring there); holding the grip button then grabs it, and the
  point moves as the controller does from where it was gripped. A grip far from every point grabs
  the tool point of the controller's own arm (the left controller the left arm), so the operator
  need not reach for the gripper. Letting go leaves the arm holding.

  Holding the tool point moves and turns the gripper; the trigger closes it. Holding a joint moves
  that joint with the joints before it, and the joints after it keep their angles, so the arm
  beyond moves along rigidly. Holding the tool point with one controller and a joint (an elbow)
  with the other places both: the gripper's pose first, the joint as close as the arm allows.

The IK keeps each arm near a reference posture in the null space of the gripper pose, so the elbow of
a seven-joint arm stays where it was rather than drifting along the ways the arm could reach the same
pose. The reference starts at the home pose, and becomes the arm's pose whenever a joint is let go, so
an elbow the operator placed stays placed.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    import mujoco

    from physicalai_mujoco_plugin.channels import ArmChannels
    from physicalai_mujoco_plugin.compose import RobotBinding
    from physicalai_mujoco_plugin.profiles import EndEffector, RobotProfile

Side = Literal["left", "right"]

MOUSE_JOINT_SPEED = 3.0
"""Fastest a joint target moves after a viewer handle, rad/s (m/s for slides)."""
HEADSET_JOINT_SPEED = 1.5
"""Fastest a joint target moves after a VR controller, rad/s: slower, since a hand moves fast and far."""
LIMIT_MARGIN = math.radians(2.0)
"""How far inside its range teleoperation keeps each joint."""
MIN_TOOL_HEIGHT = 0.005
"""Lowest a gripper target goes: just above the table top, which scenes put at z = 0."""
ORIENTATION_WEIGHT = 0.3
"""Metres of position error that weigh as much as one radian of orientation error."""
IK_ITERATIONS = 12
IK_DAMPING = 0.02
IK_MAX_STEP = 0.1
"""Largest change of one joint in one IK iteration, rad."""
POSTURE_GAIN = 0.1
"""Share of the way back to the reference posture that each IK iteration takes, in the gripper pose's null space."""
GRIP_ON, GRIP_OFF = 0.6, 0.4
"""A controller's grip button grabs above the first value and lets go below the second."""
HOVER_RADIUS_M = 0.05
"""A controller this close to a grab point hovers it."""
GRAB_POINT_SPACING_M = 0.08
"""Least distance between an arm's grab points, and from the points no joint of the arm moves (its base)."""
COST_TOLERANCE = 1e-6
"""Squared task errors closer than this (a millimetre) count as equal, so the second task decides."""
MOVING_RAD = 1e-5
"""A step of every joint smaller than this ends the arm's motion."""
REACHED_POSITION_M = 0.01
REACHED_ANGLE_RAD = math.radians(5.0)
EYE_ABOVE_SHOULDERS = 0.27
"""How far above the arms' shoulders the headset puts the operator's eyes, about as on a person."""
EYE_ABOVE_ARM = 0.15
"""How far above the arm's highest joint at home the eyes stay at least (an arm whose elbow stands up)."""


def _vector(values: np.ndarray) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def _quat(matrix: np.ndarray) -> np.ndarray:
    import mujoco  # noqa: PLC0415

    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=np.float64).reshape(9))
    return quat


def _matrix(wxyz: tuple[float, ...] | np.ndarray) -> np.ndarray:
    import mujoco  # noqa: PLC0415

    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(wxyz, dtype=np.float64))
    return matrix.reshape(3, 3)


@dataclass(frozen=True)
class Controller:
    """A VR controller in the robot's world frame: its grip pose, trigger and grip button (0 to 1)."""

    position: np.ndarray
    rotation: np.ndarray
    """3x3 rotation matrix."""
    trigger: float
    squeeze: float


@dataclass(frozen=True)
class OperatorView:
    """Where the headset puts the operator: the eye point and the way they face, in the robot's world frame."""

    eye: tuple[float, float, float]
    yaw: float
    """Heading of the operator's ahead about the world's z axis, rad (0: along +x)."""


@dataclass(frozen=True)
class GrabPoint:
    """A point of an arm that a VR controller can grab: its tool point, or one of its joints."""

    arm: str
    joint: int | None
    """Index of the joint in the arm's chain; ``None`` for the tool point."""
    body: int
    offset: tuple[float, float, float]
    """The point in the body's frame."""


@dataclass(frozen=True)
class Gripper:
    """An arm's gripper channel and its targets, in model units, at closed and fully open."""

    channel: int
    """Index of the channel in its ``ArmChannels``."""
    closed: float
    opened: float


@dataclass
class TeleopArm:
    """One teleoperated arm: an end effector of one attached robot, its joint chain and its targets."""

    name: str
    """The end effector's site or body name, with the robot's prefix."""
    sides: frozenset[Side]
    """The controllers that grab its tool point from afar: its own side, or both for a robot with one arm."""
    channels: ArmChannels
    body: int
    """The body that carries the tool point."""
    tool_offset: np.ndarray
    """The tool point in ``body``'s frame."""
    tool_rotation: np.ndarray
    """The tool frame's orientation in ``body``'s frame."""
    joints: tuple[str, ...]
    """The arm's hinge and slide joints from the base to the end effector, each driven by a channel."""
    joint_ids: tuple[int, ...]
    joint_channels: tuple[int, ...]
    qpos_adr: np.ndarray
    dof_adr: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    home: np.ndarray
    gripper: Gripper | None
    points: tuple[GrabPoint, ...] = ()
    """The tool point, then the joints a controller can grab, base first."""
    posture: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """The IK's reference posture: the home pose, or the pose a held joint was let go at."""
    target_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_rot: np.ndarray = field(default_factory=lambda: np.eye(3))
    joint_targets: dict[int, np.ndarray] = field(default_factory=dict)
    """Targets of the held joints' positions, by index in the chain."""
    tool_held: bool = False
    """A controller holds the tool point."""
    opening: float | None = None
    """Gripper target from 0 (closed) to 1 (open); ``None`` leaves the gripper alone."""
    reached: bool = True
    """The IK meets the targets within ``REACHED_POSITION_M`` and ``REACHED_ANGLE_RAD``."""
    goal: np.ndarray | None = None
    """This tick's joint targets: the IK's step toward the targets, within the joints' speed."""
    dirty: bool = False
    """A target moved since the IK last solved them."""
    moving: bool = False
    """The joints are still on their way: the last solve took a full step."""
    released: bool = False
    """The tool point was just let go."""
    snap: bool = False
    """The tool point was let go: if the arm stops short of the target, the target goes to where it stopped."""
    written: np.ndarray | None = None
    """Joint targets this arm wrote last tick; anything else means something else moved the arm."""


@dataclass(frozen=True)
class _Grab:
    """A controller holding a grab point: the controller's pose and the point's target when the grip closed."""

    arm: TeleopArm
    point: int
    """Index of the point in ``Teleop.grab_points``."""
    joint: int | None
    position: np.ndarray
    rotation: np.ndarray
    target_pos: np.ndarray
    target_rot: np.ndarray


def _side(prefix: str, effector: EndEffector) -> Side | None:
    for text in (prefix, effector.site or effector.body or ""):
        if "left" in text.lower():
            return "left"
        if "right" in text.lower():
            return "right"
    return None


def _chain(model: mujoco.MjModel, body: int) -> list[int]:
    """Return the hinge and slide joints from the root to *body*, root first."""
    import mujoco  # noqa: PLC0415

    joints: list[int] = []
    moving = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
    while body > 0:
        start, count = model.body_jntadr[body], model.body_jntnum[body]
        joints[:0] = [joint for joint in range(start, start + count) if int(model.jnt_type[joint]) in moving]
        body = int(model.body_parentid[body])
    return joints


def _gripper(model: mujoco.MjModel, channels: ArmChannels, effector: EndEffector, prefix: str) -> Gripper | None:
    """Return the end effector's gripper channel (``EndEffector.gripper``) and its closed and open targets.

    The targets are the ends of the channel's joint range, or, for a tendon, of its actuator's
    control range in model units.

    Returns:
        The gripper; ``None`` without one, or when the channel has no range to close and open over.
    """
    if effector.gripper is None:
        return None
    # Two attached robots name their channels after their anchors (``left_gripper``).
    name = next((name for name in (effector.gripper, f"{prefix}{effector.gripper}") if name in channels.names), None)
    if name is None:
        return None
    index = channels.names.index(name)
    member = channels.channels[index].first
    if member.range is not None:
        low, high = member.range
    elif model.actuator_ctrllimited[member.actuator_id] and member.ctrl_per_unit > 0:
        low, high = (float(v) / member.ctrl_per_unit for v in model.actuator_ctrlrange[member.actuator_id])
    else:
        return None
    return Gripper(index, low, high) if effector.gripper_opens_up else Gripper(index, high, low)


def _scaled_step(q: np.ndarray, step: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Return *step* shortened as a whole to stay within the bounds and ``IK_MAX_STEP``.

    Shortening the whole step keeps its direction, which clipping each joint would bend, taking
    the gripper off its pose.
    """
    moving = np.abs(step) > 1e-12  # noqa: PLR2004
    if not moving.any():
        return step
    largest = float(np.max(np.abs(step)))
    room = np.where(step > 0, upper - q, lower - q)
    ratios = np.maximum(room[moving] / step[moving], 0.0)
    return step * min(1.0, IK_MAX_STEP / largest, float(np.min(ratios)))


def _better(cost: tuple[float, float], best: tuple[float, float]) -> bool:
    """Return whether *cost* (first task, second task) beats *best*: on the first, else on the second."""
    if cost[0] < best[0] - COST_TOLERANCE:
        return True
    return cost[0] <= best[0] + COST_TOLERANCE and cost[1] < best[1]


def _damped_inverse(jacobian: np.ndarray) -> np.ndarray:
    """Return the damped least-squares inverse of *jacobian* (``IK_DAMPING``)."""
    rows = jacobian.shape[0]
    return jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + IK_DAMPING**2 * np.eye(rows))


def _posture_error(arm: TeleopArm, q: np.ndarray) -> np.ndarray:
    """Return the way from *q* to the arm's reference posture; an unlimited joint goes the shorter way round."""
    error = arm.posture - q
    unlimited = np.isinf(arm.lower)
    error[unlimited] = (error[unlimited] + math.pi) % (2 * math.pi) - math.pi
    return error


def _tool_frame(model: mujoco.MjModel, prefix: str, effector: EndEffector) -> tuple[int, np.ndarray, np.ndarray] | None:
    """Return the body carrying the end effector's tool point, the point and the tool's orientation in it."""
    import mujoco  # noqa: PLC0415

    if effector.site is not None:
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}{effector.site}")
        if site < 0:
            return None
        return int(model.site_bodyid[site]), model.site_pos[site].copy(), _matrix(model.site_quat[site])
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{effector.body}")
    return (body, np.asarray(effector.pos, dtype=np.float64), np.eye(3)) if body >= 0 else None


def build_arms(
    model: mujoco.MjModel,
    bindings: tuple[RobotBinding, ...],
    channels: tuple[ArmChannels, ...],
    profile: RobotProfile,
) -> tuple[TeleopArm, ...]:
    """Return one teleoperated arm per robot and end effector whose joints all have channels.

    An arm's side comes from its robot's prefix or its end effector's name; failing that, the first
    of two arms is the left one, and a robot with one arm answers both controllers.

    Returns:
        The arms, in anchor order and then end-effector order, without grab points yet.
    """
    found = []
    for binding, arm_channels in zip(bindings, channels, strict=True):
        by_joint = {channel.first.joint for channel in arm_channels.channels}
        for effector in profile.end_effectors:
            tool = _tool_frame(model, binding.prefix, effector)
            if tool is None:
                continue  # a custom model without the profile's end effector
            chain = [joint for joint in _chain(model, tool[0]) if model.joint(joint).name in by_joint]
            if chain:
                found.append((binding, arm_channels, effector, tool, chain))
    arms: list[TeleopArm] = []
    for index, (binding, arm_channels, effector, (body, offset, rotation), chain) in enumerate(found):
        side = _side(binding.prefix, effector) or ("left" if index == 0 else "right")
        sides: frozenset[Side] = frozenset(("left", "right")) if len(found) == 1 else frozenset((side,))
        names = tuple(model.joint(joint).name for joint in chain)
        # A joint without limits (a continuous wrist) keeps its winding: no bounds, and the posture's
        # pull on it takes the way round that is shorter (``_posture_error``).
        limited = model.jnt_limited[chain].astype(bool)
        lower = np.where(limited, model.jnt_range[chain, 0] + LIMIT_MARGIN, -np.inf)
        upper = np.where(limited, model.jnt_range[chain, 1] - LIMIT_MARGIN, np.inf)
        home_qpos = binding.layout.home_qpos
        home = np.clip(
            [home_qpos[name][0] if len(home_qpos.get(name, ())) == 1 else 0.0 for name in names], lower, upper
        )
        by_joint = {channel.first.joint: position for position, channel in enumerate(arm_channels.channels)}
        arms.append(
            TeleopArm(
                name=f"{binding.prefix}{effector.site or effector.body}",
                sides=sides,
                channels=arm_channels,
                body=body,
                tool_offset=offset,
                tool_rotation=rotation,
                joints=names,
                joint_ids=tuple(chain),
                joint_channels=tuple(by_joint[name] for name in names),
                qpos_adr=model.jnt_qposadr[chain].copy(),
                dof_adr=model.jnt_dofadr[chain].copy(),
                lower=lower,
                upper=upper,
                home=home,
                gripper=_gripper(model, arm_channels, effector, binding.prefix),
                posture=home.copy(),
            )
        )
    return tuple(arms)


class Teleop:
    """Targets of every arm, and the IK that drives the arms toward them (simulation thread only)."""

    def __init__(
        self,
        model: mujoco.MjModel,
        bindings: tuple[RobotBinding, ...],
        channels: tuple[ArmChannels, ...],
        profile: RobotProfile,
        workspace: tuple[float, float] | None = None,
    ) -> None:
        """Find the arms of the attached robots and their grab points; nothing drives them until an input does.

        Args:
            model: The composed model.
            bindings: The attached robots.
            channels: Each robot's channels.
            profile: The robots' profile, with their end effectors.
            workspace: Where on the table the arms work, in world metres (the scene's spawn centre),
                for the way the operator faces; ``None`` takes the way the arms reach at home.
        """
        import mujoco  # noqa: PLC0415

        self.model = model
        self.arms = build_arms(model, bindings, channels, profile)
        self.mouse = False
        """The viewer's gripper handles drive the arms."""
        self.headset = False
        """A headset session runs (set by the simulation every tick)."""
        self._data = mujoco.MjData(model)
        for arm in self.arms:
            arm.points = self._grab_points(arm)
        self.grab_points: tuple[GrabPoint, ...] = tuple(point for arm in self.arms for point in arm.points)
        """Every arm's grab points; the headset's page and :meth:`grab_state` number them in this order."""
        self._grabs: dict[str, _Grab] = {}
        """What each controller holds, by its side."""
        self._hover: dict[str, int] = {}
        """The grab point each free controller hovers, by its side."""
        self._held: frozenset[str] = frozenset()
        """Names of the arms a controller holds, for :meth:`status` on other threads."""
        self._synced = False
        self.view = self._operator_view(workspace)

    @property
    def drives(self) -> bool:
        """Whether teleoperation drives the arms (and actions are ignored)."""
        return bool(self.arms) and (self.mouse or self.headset)

    def set_inputs(self, *, mouse: bool | None = None, headset: bool | None = None) -> None:
        """Switch the inputs given; teleoperation that starts takes the targets from where the arms are."""
        was_driving = self.drives
        if mouse is not None:
            self.mouse = mouse
        if headset is not None:
            self.headset = headset
            if not headset:
                for side in list(self._grabs):
                    self._release(side)
                self._hover.clear()
        if self.drives and not was_driving:
            self._synced = False

    def arm(self, name: str) -> TeleopArm | None:
        """Return the arm called *name*, or ``None``."""
        return next((arm for arm in self.arms if arm.name == name), None)

    def set_target(self, name: str, position: tuple[float, ...], wxyz: tuple[float, ...], *, dragging: bool) -> None:
        """Move an arm's gripper target (a viewer drag); letting go of an unreachable target puts it back."""
        arm = self.arm(name)
        if arm is None or not self.mouse:
            return
        self._move_tool(arm, np.asarray(position, dtype=np.float64), _matrix(wxyz))
        arm.released = not dragging
        arm.snap = arm.snap and not dragging  # a new drag keeps its target

    def tick(self, data: mujoco.MjData, dt: float, controllers: Mapping[str, Controller] | None = None) -> None:
        """Move the targets after the controllers and write every arm's joint targets.

        Args:
            data: The simulation's data, its kinematics computed by the last step.
            dt: Control period, seconds.
            controllers: The headset's controllers in the robot's world frame, by side, while its
                session runs.
        """
        if not self.drives:
            return
        self._data.qpos[:] = data.qpos
        if not self._synced:
            self.sync()
            self._synced = True
        for arm in self.arms:
            if arm.written is not None and not np.allclose(self._targets(arm), arm.written, atol=1e-6):
                self._sync_arm(arm)  # a reset, home or scene command moved the arm
        if self.headset:
            self._follow_controllers(data, controllers or {})
        speed = HEADSET_JOINT_SPEED if self.headset else MOUSE_JOINT_SPEED
        for arm in self.arms:
            current = self._targets(arm)
            if arm.dirty or arm.released or arm.moving:
                self._solve_targets(arm, current, speed * dt)
            if arm.goal is not None:
                arm.channels.write_model_targets(dict(zip(arm.joints, arm.goal, strict=True)))
            if arm.gripper is not None and arm.opening is not None:
                gripper = arm.gripper
                value = gripper.closed + arm.opening * (gripper.opened - gripper.closed)
                arm.channels.write_model_target(gripper.channel, value)
            arm.written = self._targets(arm)

    def sync(self) -> None:
        """Put every target where its arm's joint targets put the gripper, and the posture back home."""
        for arm in self.arms:
            self._sync_arm(arm)

    def grab_state(self) -> dict[str, dict[str, int]]:
        """Return the grab point each controller hovers and holds, by side, as indices into :attr:`grab_points`.

        Returns:
            ``{"hover": {side: index}, "held": {side: index}}``.
        """
        return {"hover": dict(self._hover), "held": {side: grab.point for side, grab in self._grabs.items()}}

    def status(self) -> dict[str, Any]:
        """Return the inputs and each arm's target, for the viewer panel and ``GET /health``.

        Returns:
            A JSON-friendly snapshot.
        """
        held = self._held
        return {
            "mouse": self.mouse,
            "headset": self.headset,
            "arms": [
                {
                    "name": arm.name,
                    "sides": sorted(arm.sides),
                    "position": [float(v) for v in arm.target_pos],
                    "wxyz": [float(v) for v in _quat(arm.target_rot)],
                    "reached": arm.reached,
                    "held": arm.name in held,
                    "gripper": arm.gripper is not None,
                }
                for arm in self.arms
            ],
        }

    # ------------------------------------------------------------------
    # Controllers
    # ------------------------------------------------------------------

    def _follow_controllers(self, data: mujoco.MjData, controllers: Mapping[str, Controller]) -> None:
        """Hover, grab, move and let go of grab points after the controllers; the triggers close the grippers."""
        for side in list(self._grabs):
            controller = controllers.get(side)
            if controller is None or controller.squeeze < GRIP_OFF:
                self._release(side)
        self._hover = {}
        openings: dict[str, float] = {}
        for side, controller in sorted(controllers.items()):
            grab = self._grabs.get(side)
            if grab is None:
                hovered = self._nearest_point(data, controller.position)
                if hovered is not None:
                    self._hover[side] = hovered
                if controller.squeeze > GRIP_ON:
                    grab = self._grab(data, side, controller, hovered)
            if grab is None:
                arm = self._free_arm(side)
                if arm is not None and not arm.tool_held:
                    # An arm both controllers drive and neither holds closes as far as either trigger asks.
                    openings[arm.name] = min(openings.get(arm.name, 1.0), 1.0 - controller.trigger)
                continue
            position = grab.target_pos + controller.position - grab.position
            if grab.joint is None:
                self._move_tool(grab.arm, position, controller.rotation @ grab.rotation.T @ grab.target_rot)
                openings[grab.arm.name] = 1.0 - controller.trigger
            else:
                grab.arm.joint_targets[grab.joint] = position
                grab.arm.dirty = True
        for arm in self.arms:
            if arm.name in openings:
                arm.opening = openings[arm.name]

    def _nearest_point(self, data: mujoco.MjData, position: np.ndarray) -> int | None:
        """Return the free grab point nearest *position* within ``HOVER_RADIUS_M``, or ``None``."""
        held = {grab.point for grab in self._grabs.values()}
        best, best_distance = None, HOVER_RADIUS_M
        for index, point in enumerate(self.grab_points):
            if index in held:
                continue
            distance = float(np.linalg.norm(self._point_position(data, point) - position))
            if distance <= best_distance:
                best, best_distance = index, distance
        return best

    def _grab(self, data: mujoco.MjData, side: str, controller: Controller, hovered: int | None) -> _Grab | None:
        """Grab the hovered point, else the tool point of the controller's own arm while it is free.

        Returns:
            The grab, or ``None`` when there is nothing to grab.
        """
        if hovered is not None:
            point = self.grab_points[hovered]
            arm = self.arm(point.arm)
        else:
            arm = self._free_arm(side)
            if arm is None or arm.tool_held:
                return None
            hovered = self.grab_points.index(arm.points[0])
            point = arm.points[0]
        assert arm is not None  # noqa: S101 - every grab point names one of the arms
        if point.joint is None:
            target_pos, target_rot = arm.target_pos.copy(), arm.target_rot.copy()
            arm.tool_held = True
            arm.snap = False
        else:
            target_pos, target_rot = self._point_position(data, point), np.eye(3)
            arm.joint_targets[point.joint] = target_pos
        grab = _Grab(arm, hovered, point.joint, controller.position, controller.rotation, target_pos, target_rot)
        self._grabs[side] = grab
        self._hover.pop(side, None)
        self._held = frozenset(grab.arm.name for grab in self._grabs.values())
        return grab

    def _free_arm(self, side: str) -> TeleopArm | None:
        """Return the arm whose tool point the controller on *side* grabs from afar."""
        return next((arm for arm in self.arms if side in arm.sides), None)

    def _release(self, side: str) -> None:
        grab = self._grabs.pop(side, None)
        if grab is None:
            return
        arm = grab.arm
        if grab.joint is None:
            arm.tool_held = False
            arm.released = True
        else:
            arm.joint_targets.pop(grab.joint, None)
            # The joint stays where it is now, which the joint targets lag behind the IK's goal to.
            arm.posture = self._targets(arm)
            if arm.tool_held or arm.joint_targets:
                arm.dirty = True  # solve again toward the new posture, from where the arm is
            else:
                arm.target_pos, arm.target_rot = self._tool(arm, arm.posture)
                arm.goal = None  # nothing holds the arm: it stops here
        self._held = frozenset(grab.arm.name for grab in self._grabs.values())

    # ------------------------------------------------------------------
    # Targets and IK
    # ------------------------------------------------------------------

    @staticmethod
    def _move_tool(arm: TeleopArm, position: np.ndarray, rotation: np.ndarray) -> None:
        arm.target_pos = np.array([position[0], position[1], max(position[2], MIN_TOOL_HEIGHT)])
        arm.target_rot = rotation
        arm.dirty = True

    def _solve_targets(self, arm: TeleopArm, current: np.ndarray, limit: float) -> None:
        """Take this tick's step toward the arm's targets: the IK within *limit* of each joint's target.

        Solving within the step, rather than clipping a step toward a far solution, keeps the task
        order on the way too: a joint pulled away never takes the held gripper off its pose.
        With joints held and the tool point free, the gripper goes where the joints take it. A
        released tool target stays while the arm still makes its way toward it; only if the arm
        stops short does the target go to where the gripper got to.
        """
        tool = arm.tool_held or not arm.joint_targets
        arm.snap = (arm.snap or arm.released) and tool
        arm.goal, met = self._solve(arm, current, tool=tool, limit=limit)
        arm.moving = float(np.max(np.abs(arm.goal - current))) > MOVING_RAD
        arm.reached = met or arm.moving  # out of reach only once the arm stops short
        if not tool or (arm.snap and not arm.moving and not met):
            arm.target_pos, arm.target_rot = self._tool(arm, arm.goal)
            arm.reached = arm.reached or tool  # the gripper is at its snapped target
        if not arm.moving:
            arm.snap = False
        arm.dirty = arm.released = False

    def _sync_arm(self, arm: TeleopArm) -> None:
        arm.target_pos, arm.target_rot = self._tool(arm, self._targets(arm))
        arm.posture = arm.home.copy()
        arm.joint_targets.clear()
        arm.tool_held = False
        arm.opening = None
        arm.goal = None
        arm.reached = True
        arm.dirty = arm.released = arm.moving = arm.snap = False
        arm.written = None
        for side in [side for side, grab in self._grabs.items() if grab.arm is arm]:
            del self._grabs[side]
        self._held = frozenset(grab.arm.name for grab in self._grabs.values())

    @staticmethod
    def _targets(arm: TeleopArm) -> np.ndarray:
        targets = arm.channels.model_targets()
        return np.array([targets[index] for index in arm.joint_channels])

    def _forward(self, arm: TeleopArm, q: np.ndarray) -> None:
        import mujoco  # noqa: PLC0415

        self._data.qpos[arm.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self._data)
        mujoco.mj_comPos(self.model, self._data)

    def _tool(self, arm: TeleopArm, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the tool point and the tool frame's orientation at joint positions *q*."""
        self._forward(arm, q)
        return self._tool_now(arm, self._data)

    @staticmethod
    def _tool_now(arm: TeleopArm, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        rotation = data.xmat[arm.body].reshape(3, 3)
        return data.xpos[arm.body] + rotation @ arm.tool_offset, rotation @ arm.tool_rotation

    @staticmethod
    def _point_position(data: mujoco.MjData, point: GrabPoint) -> np.ndarray:
        return data.xpos[point.body] + data.xmat[point.body].reshape(3, 3) @ np.asarray(point.offset)

    def _solve(
        self, arm: TeleopArm, q0: np.ndarray, *, tool: bool, limit: float | None = None
    ) -> tuple[np.ndarray, bool]:
        """Damped least squares from *q0* toward the arm's targets.

        The tool point's pose comes first. Held joints come second, in its null space, so a joint
        held where the arm cannot follow never pulls the gripper off its pose; without a held
        joint, the reference posture pulls there instead.

        Args:
            arm: The arm.
            q0: The joint positions to start from.
            tool: Whether the tool point's pose is a target. Without it, only the joints before the
                held joints move.
            limit: How far each joint may move from *q0* (back inside its range at most this far);
                ``None``: as far as its range.

        Returns:
            The joint positions, and whether they meet the first targets (the tool point's pose, or
            the held joints without it): the last iterate when it does, else the best one.
        """
        free = np.ones(len(q0), dtype=bool)
        if not tool:
            free = np.arange(len(q0)) < max(arm.joint_targets)  # the joints before the farthest held joint
        lower, upper = arm.lower, arm.upper
        if limit is not None:
            lower = np.minimum(np.maximum(lower, q0 - limit), q0 + limit)
            upper = np.maximum(np.minimum(upper, q0 + limit), q0 - limit)
        q = np.clip(q0, lower, upper)
        best, best_cost, best_reached = q.copy(), (math.inf, math.inf), False
        reached = False
        for iteration in range(IK_ITERATIONS + 1):
            self._forward(arm, q)
            (first, first_jacobian, reached), second = self._tasks(arm, tool=tool)
            cost = (float(first @ first), float(second[0] @ second[0]) if second is not None else 0.0)
            if _better(cost, best_cost):
                best, best_cost, best_reached = q.copy(), cost, reached
            if iteration == IK_ITERATIONS:
                break
            # Joints held at a bound drop out of both tasks, so the others make up for them.
            active = free.copy()
            for _ in range(len(q)):
                step = self._step(arm, q, (first, first_jacobian), second, active, posture=tool)
                pushing = active & (((q <= lower + 1e-9) & (step < 0)) | ((q >= upper - 1e-9) & (step > 0)))
                if not pushing.any():
                    break
                active &= ~pushing
            if float(np.max(np.abs(step))) < 1e-5:  # noqa: PLR2004
                break
            q = np.clip(q + _scaled_step(q, step, lower, upper), lower, upper)
        return (q, True) if reached else (best, best_reached)

    @staticmethod
    def _step(
        arm: TeleopArm,
        q: np.ndarray,
        first: tuple[np.ndarray, np.ndarray],
        second: tuple[np.ndarray, np.ndarray] | None,
        active: np.ndarray,
        *,
        posture: bool,
    ) -> np.ndarray:
        """Return one step over the *active* joints: toward the first task, then the second in its null space.

        Without a second task (and with *posture*), the reference posture pulls in the null space.
        """
        rows, jacobian = first[0], first[1] * active
        inverse = _damped_inverse(jacobian)
        step = inverse @ rows
        null = (np.eye(len(q)) - inverse @ jacobian) * active
        if second is not None:
            rows, jacobian = second[0], second[1] * active
            step += null @ _damped_inverse(jacobian @ null) @ (rows - jacobian @ step)
        elif posture:
            step += null @ (POSTURE_GAIN * _posture_error(arm, q) * active)
        return step

    def _tasks(
        self, arm: TeleopArm, *, tool: bool
    ) -> tuple[tuple[np.ndarray, np.ndarray, bool], tuple[np.ndarray, np.ndarray] | None]:
        """Return the first task (errors, Jacobian over the arm's joints, whether met) and the second, if any.

        The first is the tool point's pose and the second the held joints' positions; without the
        tool point, the held joints are the first.

        Returns:
            ``((errors, jacobian, met), (errors, jacobian) or None)``.
        """
        joints = self._joint_task(arm)
        if not tool:
            assert joints is not None  # noqa: S101 - without the tool point, a joint is held
            return joints, None
        return self._tool_task(arm), joints[:2] if joints is not None else None

    def _joint_task(self, arm: TeleopArm) -> tuple[np.ndarray, np.ndarray, bool] | None:
        """Return the held joints' position errors, their Jacobian and whether they are met; ``None`` without any."""
        import mujoco  # noqa: PLC0415

        if not arm.joint_targets:
            return None
        jacp = np.zeros((3, self.model.nv))
        rows, jacobians, met = [], [], True
        for index, target in arm.joint_targets.items():
            joint = arm.joint_ids[index]
            anchor = self._data.xanchor[joint].copy()
            rows.append(target - anchor)
            mujoco.mj_jac(self.model, self._data, jacp, None, anchor, int(self.model.jnt_bodyid[joint]))
            jacobians.append(jacp[:, arm.dof_adr].copy())
            met = met and bool(np.linalg.norm(rows[-1]) <= REACHED_POSITION_M)
        return np.concatenate(rows), np.vstack(jacobians), met

    def _tool_task(self, arm: TeleopArm) -> tuple[np.ndarray, np.ndarray, bool]:
        """Return the tool point's weighted pose error, its Jacobian and whether it is met."""
        import mujoco  # noqa: PLC0415

        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        point, current = self._tool_now(arm, self._data)
        local = np.zeros(3)
        mujoco.mju_subQuat(local, _quat(arm.target_rot), _quat(current))
        position_error, angle_error = arm.target_pos - point, current @ local
        mujoco.mj_jac(self.model, self._data, jacp, jacr, point, arm.body)
        errors = np.concatenate([position_error, ORIENTATION_WEIGHT * angle_error])
        jacobian = np.vstack([jacp[:, arm.dof_adr], ORIENTATION_WEIGHT * jacr[:, arm.dof_adr]])
        met = bool(np.linalg.norm(position_error) <= REACHED_POSITION_M)
        return errors, jacobian, met and float(np.linalg.norm(angle_error)) <= REACHED_ANGLE_RAD

    # ------------------------------------------------------------------
    # Grab points and the headset's viewpoint
    # ------------------------------------------------------------------

    def _grab_points(self, arm: TeleopArm) -> tuple[GrabPoint, ...]:
        """Return the arm's tool point and the joints worth grabbing, measured at home.

        A joint is worth grabbing when the joints before it move it, and it stands at least
        ``GRAB_POINT_SPACING_M`` from the joints no joint moves, from the joints already taken and
        from the tool point: of joints closer together (a wrist's), the one nearest the base.

        Returns:
            The grab points, the tool point first.
        """
        import mujoco  # noqa: PLC0415

        self._forward(arm, arm.home)
        data = self._data
        tool = GrabPoint(arm.name, None, arm.body, _vector(arm.tool_offset))
        tool_position = self._point_position(data, tool)
        taken: list[np.ndarray] = []
        points = [tool]
        jacp = np.zeros((3, self.model.nv))
        for index, joint in enumerate(arm.joint_ids):
            anchor = data.xanchor[joint].copy()
            body = int(self.model.jnt_bodyid[joint])
            mujoco.mj_jac(self.model, data, jacp, None, anchor, body)
            if index == 0 or np.linalg.norm(jacp[:, arm.dof_adr[:index]]) < 1e-3:  # noqa: PLR2004 - fixed
                taken.append(anchor)
                continue
            nearest = min(float(np.linalg.norm(anchor - other)) for other in [*taken, tool_position])
            if nearest < GRAB_POINT_SPACING_M:
                continue
            taken.append(anchor)
            rotation = data.xmat[body].reshape(3, 3)
            offset = rotation.T @ (anchor - data.xpos[body])
            points.append(GrabPoint(arm.name, index, body, _vector(offset)))
        return tuple(points)

    def _operator_view(self, workspace: tuple[float, float] | None) -> OperatorView:
        """Return where the operator stands: above the arms' shoulders at home, facing their workspace.

        A shoulder is the higher of an arm's first two joints. Ahead is the level direction from the
        shoulders to *workspace*, else to the tool points at home (an arm folded at home can point
        them backwards); +x when that is under 5 cm (arms that face each other). The
        eyes stand ``EYE_ABOVE_SHOULDERS`` above the shoulders, and ``EYE_ABOVE_ARM`` above every
        joint of the arms at home.

        Returns:
            The view; for a robot without arms, 1 m behind and 1.2 m above the world origin, along +x.
        """
        if not self.arms:
            return OperatorView((-1.0, 0.0, 1.2), 0.0)
        shoulders, tools, tops = [], [], []
        for arm in self.arms:
            self._forward(arm, arm.home)
            anchors = [self._data.xanchor[joint].copy() for joint in arm.joint_ids]
            shoulders.append(max(anchors[:2], key=operator.itemgetter(2)))
            tops.append(max(float(anchor[2]) for anchor in anchors))
            tools.append(self._tool_now(arm, self._data)[0])
        shoulder = np.mean(shoulders, axis=0)
        reach = (np.asarray(workspace) if workspace is not None else np.mean(tools, axis=0)[:2]) - shoulder[:2]
        yaw = math.atan2(reach[1], reach[0]) if np.linalg.norm(reach) >= 0.05 else 0.0  # noqa: PLR2004
        height = max(max(float(point[2]) for point in shoulders) + EYE_ABOVE_SHOULDERS, max(tops) + EYE_ABOVE_ARM)
        return OperatorView((float(shoulder[0]), float(shoulder[1]), height), yaw)


__all__ = [
    "HEADSET_JOINT_SPEED",
    "HOVER_RADIUS_M",
    "MOUSE_JOINT_SPEED",
    "Controller",
    "GrabPoint",
    "Gripper",
    "OperatorView",
    "Teleop",
    "TeleopArm",
    "build_arms",
]
