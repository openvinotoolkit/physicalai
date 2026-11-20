# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Arm teleoperation inside the simulation: drag the grippers in the 3D viewer, or move them with VR controllers.

Every arm of the robot (one per attached robot and ``RobotProfile.end_effectors`` entry) has a
gripper target: a pose of its tool point. Damped least-squares inverse kinematics turns the target
into joint targets, which each control tick writes to the arm's channels like an action would,
moving each joint at most ``MOUSE_JOINT_SPEED`` or ``HEADSET_JOINT_SPEED``. While teleoperation
drives the arms, actions from Studio or a policy are ignored. Two inputs move the targets:

* ``mouse``: the viewer shows a handle at each gripper (:meth:`Teleop.set_target`). Letting go of a
  target the arm cannot reach puts it back where the gripper got to.
* the headset: VR controllers over WebXR (:mod:`~physicalai_mujoco_plugin.xr`), whose poses arrive in
  the robot's world frame. Holding a controller's grip button grabs its arm (the left controller
  the left arm): the target then moves and turns as the controller does from where it was gripped,
  so the operator can let go, move the hand back and grip again. The trigger closes the gripper.

The IK keeps each arm near a reference posture (its home pose) in the null space of the gripper
pose, so the elbow of a seven-joint arm stays where it was rather than drifting along the ways the
arm could reach the same pose.
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
REACHED_POSITION_M = 0.01
REACHED_ANGLE_RAD = math.radians(5.0)
EYE_ABOVE_SHOULDERS = 0.27
"""How far above the arms' shoulders the headset puts the operator's eyes, about as on a person."""


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
class Gripper:
    """An arm's gripper channel and its targets, in model units, at closed and fully open."""

    channel: int
    """Index of the channel in its ``ArmChannels``."""
    closed: float
    opened: float


@dataclass
class TeleopArm:
    """One teleoperated arm: an end effector of one attached robot, its joint chain and its target."""

    name: str
    """The end effector's site or body name, with the robot's prefix."""
    sides: frozenset[Side]
    """The controllers that drive it: its own side, or both for a robot with one arm."""
    channels: ArmChannels
    effector: EndEffector
    site: int | None
    body: int
    joints: tuple[str, ...]
    """The arm's hinge and slide joints from the base to the end effector, each driven by a channel."""
    joint_channels: tuple[int, ...]
    qpos_adr: np.ndarray
    dof_adr: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    home: np.ndarray
    """The joints' home positions: the IK's reference posture."""
    gripper: Gripper | None
    target_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_rot: np.ndarray = field(default_factory=lambda: np.eye(3))
    opening: float | None = None
    """Gripper target from 0 (closed) to 1 (open); ``None`` leaves the gripper alone."""
    reached: bool = True
    """The IK meets the target within ``REACHED_POSITION_M`` and ``REACHED_ANGLE_RAD``."""
    goal: np.ndarray | None = None
    """Joint positions the IK found for the target; the joint targets move toward them."""
    dirty: bool = False
    """The target moved since the IK last solved it."""
    released: bool = False
    """The input just let go: an unreachable target goes back to where the gripper got to."""
    written: np.ndarray | None = None
    """Joint targets this arm wrote last tick; anything else means something else moved the arm."""


@dataclass(frozen=True)
class _Grab:
    """A controller holding an arm: the controller's pose and the arm's target when the grip closed."""

    arm: TeleopArm
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


def _posture_error(arm: TeleopArm, q: np.ndarray) -> np.ndarray:
    """Return the way from *q* to the arm's reference posture; an unlimited joint goes the shorter way round."""
    error = arm.home - q
    unlimited = np.isinf(arm.lower)
    error[unlimited] = (error[unlimited] + math.pi) % (2 * math.pi) - math.pi
    return error


def _effector_ids(model: mujoco.MjModel, prefix: str, effector: EndEffector) -> tuple[int | None, int]:
    """Return the end effector's site (``None`` for a body point) and its body; the body is -1 when missing."""
    import mujoco  # noqa: PLC0415

    if effector.site is not None:
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}{effector.site}")
        return site, int(model.site_bodyid[site]) if site >= 0 else -1
    return None, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{effector.body}")


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
        The arms, in anchor order and then end-effector order.
    """
    found: list[tuple[RobotBinding, ArmChannels, EndEffector, int | None, int, list[int]]] = []
    for binding, arm_channels in zip(bindings, channels, strict=True):
        by_joint = {channel.first.joint for channel in arm_channels.channels}
        for effector in profile.end_effectors:
            site, body = _effector_ids(model, binding.prefix, effector)
            if body < 0:
                continue  # a custom model without the profile's end effector
            chain = [joint for joint in _chain(model, body) if model.joint(joint).name in by_joint]
            if chain:
                found.append((binding, arm_channels, effector, site, body, chain))
    arms: list[TeleopArm] = []
    for index, (binding, arm_channels, effector, site, body, chain) in enumerate(found):
        side = _side(binding.prefix, effector) or ("left" if index == 0 else "right")
        sides: frozenset[Side] = frozenset(("left", "right")) if len(found) == 1 else frozenset((side,))
        names = tuple(model.joint(joint).name for joint in chain)
        # A joint without limits (a continuous wrist) keeps its winding: no bounds, and the posture's
        # pull on it takes the way round that is shorter (``_posture_error``).
        limited = model.jnt_limited[chain].astype(bool)
        lower = np.where(limited, model.jnt_range[chain, 0] + LIMIT_MARGIN, -np.inf)
        upper = np.where(limited, model.jnt_range[chain, 1] - LIMIT_MARGIN, np.inf)
        home_qpos = binding.layout.home_qpos
        home = np.array([home_qpos[name][0] if len(home_qpos.get(name, ())) == 1 else 0.0 for name in names])
        by_joint = {channel.first.joint: position for position, channel in enumerate(arm_channels.channels)}
        arms.append(
            TeleopArm(
                name=f"{binding.prefix}{effector.site or effector.body}",
                sides=sides,
                channels=arm_channels,
                effector=effector,
                site=site,
                body=body,
                joints=names,
                joint_channels=tuple(by_joint[name] for name in names),
                qpos_adr=model.jnt_qposadr[chain].copy(),
                dof_adr=model.jnt_dofadr[chain].copy(),
                lower=lower,
                upper=upper,
                home=np.clip(home, lower, upper),
                gripper=_gripper(model, arm_channels, effector, binding.prefix),
            )
        )
    return tuple(arms)


class Teleop:
    """Gripper targets of every arm, and the IK that drives the arms toward them (simulation thread only)."""

    def __init__(
        self,
        model: mujoco.MjModel,
        bindings: tuple[RobotBinding, ...],
        channels: tuple[ArmChannels, ...],
        profile: RobotProfile,
    ) -> None:
        """Find the arms of the attached robots; nothing drives them until an input does."""
        import mujoco  # noqa: PLC0415

        self.model = model
        self.arms = build_arms(model, bindings, channels, profile)
        self.mouse = False
        """The viewer's gripper handles drive the arms."""
        self.headset = False
        """A headset session runs (set by the simulation every tick)."""
        self._data = mujoco.MjData(model)
        self._grabs: dict[str, _Grab] = {}
        """What each controller holds, by its side."""
        self._held: frozenset[str] = frozenset()
        """Names of the arms a controller holds, for :meth:`status` on other threads."""
        self._synced = False
        self.view = self._operator_view()

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
                self._release_all()
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
        self._move_target(arm, np.asarray(position, dtype=np.float64), _matrix(wxyz))
        arm.released = not dragging

    def tick(self, data: mujoco.MjData, dt: float, controllers: Mapping[str, Controller] | None = None) -> None:
        """Move the targets after the controllers and write every arm's joint targets.

        Args:
            data: The simulation's data.
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
            self._follow_controllers(controllers or {})
        speed = HEADSET_JOINT_SPEED if self.headset else MOUSE_JOINT_SPEED
        for arm in self.arms:
            current = self._targets(arm)
            if arm.dirty or arm.released:
                self._solve_target(arm, current)
            if arm.goal is not None:
                step = np.clip(arm.goal - current, -speed * dt, speed * dt)
                arm.channels.write_model_targets(dict(zip(arm.joints, current + step, strict=True)))
            if arm.gripper is not None and arm.opening is not None:
                gripper = arm.gripper
                value = gripper.closed + arm.opening * (gripper.opened - gripper.closed)
                arm.channels.write_model_target(gripper.channel, value)
            arm.written = self._targets(arm)

    def sync(self) -> None:
        """Put every target where its arm's joint targets put the gripper."""
        for arm in self.arms:
            self._sync_arm(arm)

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

    def _follow_controllers(self, controllers: Mapping[str, Controller]) -> None:
        """Grab, move and let go of arms after the controllers' grip buttons; the triggers close the grippers."""
        for side in list(self._grabs):
            controller = controllers.get(side)
            if controller is None or controller.squeeze < GRIP_OFF:
                self._release(side)
        openings: dict[str, float] = {}
        for side, controller in sorted(controllers.items()):
            grab = self._grabs.get(side)
            if grab is None and controller.squeeze > GRIP_ON:
                grab = self._grab(side, controller)
            if grab is not None:
                turn = controller.rotation @ grab.rotation.T
                position = grab.target_pos + controller.position - grab.position
                self._move_target(grab.arm, position, turn @ grab.target_rot)
                openings[grab.arm.name] = 1.0 - controller.trigger
                continue
            arm = self._free_arm(side)
            if arm is not None and arm.name not in self._held:
                # An arm both controllers drive and neither holds closes as far as either trigger asks.
                openings[arm.name] = min(openings.get(arm.name, 1.0), 1.0 - controller.trigger)
        for arm in self.arms:
            if arm.name in openings:
                arm.opening = openings[arm.name]

    def _grab(self, side: str, controller: Controller) -> _Grab | None:
        arm = self._free_arm(side)
        if arm is None:
            return None
        grab = _Grab(arm, controller.position, controller.rotation, arm.target_pos.copy(), arm.target_rot.copy())
        self._grabs[side] = grab
        self._held = frozenset(grab.arm.name for grab in self._grabs.values())
        return grab

    def _free_arm(self, side: str) -> TeleopArm | None:
        """Return the arm the controller on *side* drives that no other controller holds."""
        held = {grab.arm.name for other, grab in self._grabs.items() if other != side}
        return next((arm for arm in self.arms if side in arm.sides and arm.name not in held), None)

    def _release(self, side: str) -> None:
        grab = self._grabs.pop(side, None)
        if grab is not None:
            grab.arm.released = True
            self._held = frozenset(grab.arm.name for grab in self._grabs.values())

    def _release_all(self) -> None:
        for side in list(self._grabs):
            self._release(side)

    # ------------------------------------------------------------------
    # Targets and IK
    # ------------------------------------------------------------------

    @staticmethod
    def _move_target(arm: TeleopArm, position: np.ndarray, rotation: np.ndarray) -> None:
        arm.target_pos = np.array([position[0], position[1], max(position[2], MIN_TOOL_HEIGHT)])
        arm.target_rot = rotation
        arm.dirty = True

    def _solve_target(self, arm: TeleopArm, current: np.ndarray) -> None:
        """Solve the IK for the arm's target; a released target the arm cannot reach goes back to the gripper."""
        arm.goal, error, angle = self._solve(arm, current, arm.target_pos, arm.target_rot)
        arm.reached = error <= REACHED_POSITION_M and angle <= REACHED_ANGLE_RAD
        if arm.released and not arm.reached:
            arm.target_pos, arm.target_rot = self._tool(arm, arm.goal)
        arm.dirty = arm.released = False

    def _sync_arm(self, arm: TeleopArm) -> None:
        arm.target_pos, arm.target_rot = self._tool(arm, self._targets(arm))
        arm.opening = None
        arm.goal = None
        arm.reached = True
        arm.dirty = arm.released = False
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
        """Return the tool point and the end effector's orientation at joint positions *q*."""
        self._forward(arm, q)
        return self._tool_now(arm)

    def _tool_now(self, arm: TeleopArm) -> tuple[np.ndarray, np.ndarray]:
        data = self._data
        if arm.site is not None:
            return data.site_xpos[arm.site].copy(), data.site_xmat[arm.site].reshape(3, 3).copy()
        rotation = data.xmat[arm.body].reshape(3, 3).copy()
        return data.xpos[arm.body] + rotation @ np.asarray(arm.effector.pos), rotation

    def _solve(
        self,
        arm: TeleopArm,
        q0: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Damped least squares from *q0* toward the gripper pose, and toward home in its null space.

        Returns:
            The joint positions, their position error (m) and orientation error (rad): the last
            iterate when it meets the pose (it carries the posture's motion), else the best one.
        """
        q = np.clip(q0, arm.lower, arm.upper)
        best, best_cost, best_errors = q.copy(), math.inf, (math.inf, math.inf)
        errors = (math.inf, math.inf)
        for iteration in range(IK_ITERATIONS + 1):
            self._forward(arm, q)
            point, error = self._pose_error(arm, position, rotation)
            errors = (float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:])) / ORIENTATION_WEIGHT)
            cost = float(error @ error)
            if cost < best_cost:
                best, best_cost, best_errors = q.copy(), cost, errors
            if iteration == IK_ITERATIONS:
                break
            step, posture = self._step(arm, q, point, error)
            if errors[0] < 2e-4 and errors[1] < 2e-3 and float(np.max(np.abs(posture))) < 1e-4:  # noqa: PLR2004
                break
            q = np.clip(q + np.clip(step + posture, -IK_MAX_STEP, IK_MAX_STEP), arm.lower, arm.upper)
        if errors[0] <= REACHED_POSITION_M and errors[1] <= REACHED_ANGLE_RAD:
            return q, errors[0], errors[1]
        return best, best_errors[0], best_errors[1]

    def _pose_error(self, arm: TeleopArm, position: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the tool point and the weighted error toward the target pose: 3 position, 3 rotation rows."""
        import mujoco  # noqa: PLC0415

        point, current = self._tool_now(arm)
        local = np.zeros(3)
        mujoco.mju_subQuat(local, _quat(rotation), _quat(current))
        return point, np.concatenate([position - point, ORIENTATION_WEIGHT * (current @ local)])

    def _step(
        self, arm: TeleopArm, q: np.ndarray, point: np.ndarray, error: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return one damped least-squares step toward the gripper pose, and one toward home in its null space."""
        import mujoco  # noqa: PLC0415

        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self._data, jacp, jacr, point, arm.body)
        jacobian = np.vstack([jacp[:, arm.dof_adr], ORIENTATION_WEIGHT * jacr[:, arm.dof_adr]])
        inverse = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + IK_DAMPING**2 * np.eye(6))
        posture = (np.eye(len(q)) - inverse @ jacobian) @ (POSTURE_GAIN * _posture_error(arm, q))
        return inverse @ error, posture

    # ------------------------------------------------------------------
    # The headset's viewpoint
    # ------------------------------------------------------------------

    def _operator_view(self) -> OperatorView:
        """Return where the operator stands: above the arms' shoulders at home, facing the way they reach.

        A shoulder is the higher of an arm's first two joints. Ahead is the level direction from the
        shoulders to the tool points; +x when that is under 5 cm (arms that face each other).

        Returns:
            The view; for a robot without arms, 1 m behind and 1.2 m above the world origin, along +x.
        """
        if not self.arms:
            return OperatorView((-1.0, 0.0, 1.2), 0.0)
        shoulders, tools = [], []
        for arm in self.arms:
            self._forward(arm, arm.home)
            anchors = [self._data.xanchor[self.model.joint(name).id] for name in arm.joints[:2]]
            shoulders.append(max(anchors, key=operator.itemgetter(2)).copy())
            tools.append(self._tool_now(arm)[0])
        shoulder, tool = np.mean(shoulders, axis=0), np.mean(tools, axis=0)
        reach = (tool - shoulder)[:2]
        yaw = math.atan2(reach[1], reach[0]) if np.linalg.norm(reach) >= 0.05 else 0.0  # noqa: PLR2004
        top = max(float(point[2]) for point in shoulders)
        return OperatorView((float(shoulder[0]), float(shoulder[1]), top + EYE_ABOVE_SHOULDERS), yaw)


__all__ = [
    "HEADSET_JOINT_SPEED",
    "MOUSE_JOINT_SPEED",
    "Controller",
    "Gripper",
    "OperatorView",
    "Teleop",
    "TeleopArm",
    "build_arms",
]
