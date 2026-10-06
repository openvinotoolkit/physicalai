# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Scripted demonstrator for the ``conveyor_sort`` scene.

The demonstrator reads privileged simulation state (item poses, belt speed,
the sorting rule) and writes arm joint targets, so it can produce consistent
demonstrations faster than a human can teleoperate moving items. Each cycle it:

1. picks the most downstream item in the pick window,
2. hovers over it, tracking the belt, with the jaw lined up with the item's faces,
3. descends while tracking, closes, and lifts,
4. carries the item over the bin the rule assigns and drops it.

Grasps close along the belt so the jaw's push can never shove an item into a
rail. Inverse kinematics is damped least squares with the grasp point as the
primary task and "fingers down, jaw at yaw" in the remaining freedom; the SO-101
has five arm joints, so the gripper tilts a little where it cannot be vertical.

The demonstrator writes ``data.ctrl`` for the arm and gripper position
actuators (radians) and does not step the simulation.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from physicalai_mujoco_so101_plugin.conveyor import ConveyorItem, ConveyorSort

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER = "gripper"
TCP_LOCAL = np.array([0.015, 0.0, -0.09])
"""Grasp point in the gripper frame: between the jaw pads, ~10 mm off the fixed jaw."""
MAX_IK_STEP = 0.25
"""Largest joint change (rad) per IK iteration."""
SYMMETRY = {"cube": np.pi / 2, "hex": np.pi / 3, "cylinder": None}
"""Yaw period of each item shape's faces (``None``: any yaw works)."""

Phase = Literal["idle", "approach", "descend", "close", "lift", "carry", "lower", "release", "rise", "return"]


@dataclass
class DemoConfig:
    """Motion parameters for the demonstrator (metres, seconds, radians)."""

    grip_open: float = 0.5
    """Gripper angle while approaching: a 5.2 cm gap for 3 cm items."""
    grip_closed: float = -0.15
    hover_z: float = 0.14
    carry_z: float = 0.15
    drop_z: float = 0.10
    grasp_dz: float = 0.005
    """Grasp this far above the item centre, for fingertip clearance over the belt."""
    pick_y: tuple[float, float] = (-0.15, 0.10)
    """Belt window (world y) where the IK is exact and the gripper within ~3 deg of vertical.

    Grasping further upstream tilts the gripper 4-5 deg, which squeezes cubes out sideways
    into the far rail.
    """
    wait_y: float = 0.08
    """Hover over the belt here while the next item approaches the window."""
    belt_x: float = 0.25
    cart_speed: float = 0.35
    """Speed limit of the Cartesian setpoint."""
    descend_speed: float = 0.2
    lift_speed: float = 0.25
    close_s: float = 0.3
    release_s: float = 0.3
    approach_timeout_s: float = 4.0
    lead_s: float = 0.12
    """Look-ahead on moving items, to cover the position servos' lag."""
    reach_lookahead_s: float = 1.5
    """An item is only chosen if it will still be in the window this long from now."""


@dataclass
class DemoStats:
    """Counters for one demonstrator run."""

    attempts: int = 0
    grasp_failed: int = 0
    approach_timeouts: int = 0


@dataclass
class _Plan:
    phase: Phase = "idle"
    item: ConveyorItem | None = None
    since: float = 0.0
    yaw: float = 0.0
    flip: float = 0.0
    setpoint: np.ndarray = field(default_factory=lambda: np.zeros(3))


class ArmIK:
    """Damped least-squares IK for the grasp point, fingers down, jaw at a given yaw."""

    def __init__(self, model: object) -> None:
        """Resolve the arm joints; IK runs on a private `MjData` so the simulation is untouched."""
        import mujoco  # noqa: PLC0415

        self._mujoco = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS]
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in joints])
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in joints])
        self.lower = np.array([model.jnt_range[j][0] for j in joints])
        self.upper = np.array([model.jnt_range[j][1] for j in joints])
        self.body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper"))
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def tcp(self, data: object) -> np.ndarray:
        """World position of the grasp point (needs up-to-date kinematics in `data`).

        Returns:
            The grasp point, in metres.
        """
        return data.xpos[self.body] + data.xmat[self.body].reshape(3, 3) @ TCP_LOCAL

    def gripper_yaw(self, data: object) -> float:
        """World yaw of the jaw's closing axis (the gripper x axis).

        Returns:
            The yaw in radians.
        """
        x = data.xmat[self.body].reshape(3, 3)[:, 0]
        return float(np.arctan2(x[1], x[0]))

    def tilt(self, data: object) -> float:
        """Angle between the fingers and straight down.

        Returns:
            The tilt in radians.
        """
        return float(np.arccos(np.clip(data.xmat[self.body].reshape(3, 3)[2, 2], -1.0, 1.0)))

    def solve(  # noqa: PLR0914 - one numerical routine; splitting it hides the algorithm
        self, q0: np.ndarray, target: np.ndarray, yaw: float, iters: int = 8
    ) -> tuple[np.ndarray, float]:
        """Joint angles that put the grasp point at `target` with the jaw closing along `yaw`.

        Position is the primary task. Orientation uses only the null-space freedom
        left over, and joints pinned at a limit drop out of both (an active set), so
        the solution never trades position for orientation.

        Returns:
            The arm joint angles and the remaining position error in metres.
        """
        mujoco, model, data = self._mujoco, self.model, self.data
        q = q0.copy()
        x_des = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        z_des = np.array([0.0, 0.0, 1.0])
        r_des = np.column_stack([x_des, np.cross(z_des, x_des), z_des])
        quat, e_rot = np.zeros(4), np.zeros(3)
        e_pos = np.zeros(3)
        for _ in range(iters):
            data.qpos[self.qpos_adr] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            p = self.tcp(data)
            e_pos = target - p
            mujoco.mju_mat2Quat(quat, (r_des @ data.xmat[self.body].reshape(3, 3).T).ravel())
            mujoco.mju_quat2Vel(e_rot, quat, 1.0)
            mujoco.mj_jac(model, data, self._jacp, self._jacr, p, self.body)
            free = np.ones(len(q), dtype=bool)
            dq = np.zeros(len(q))
            for _ in range(len(q)):
                jp = self._jacp[:, self.dof_adr] * free
                jr = self._jacr[:, self.dof_adr] * free
                jp_pinv = jp.T @ np.linalg.inv(jp @ jp.T + 1e-5 * np.eye(3))
                dq = jp_pinv @ e_pos
                null = np.eye(len(q)) - jp_pinv @ jp
                jr_n = jr @ null
                dq += null @ (jr_n.T @ np.linalg.solve(jr_n @ jr_n.T + 1e-3 * np.eye(3), e_rot - jr @ dq))
                dq *= free
                # Active set: a joint already at a limit and pushed further drops out of both tasks.
                at_upper, at_lower = q >= self.upper - 1e-9, q <= self.lower + 1e-9
                blocked = free & ((at_upper & (dq > 0)) | (at_lower & (dq < 0)))
                if not blocked.any():
                    break
                free &= ~blocked
            # Small steps: a big jump can overshoot into another IK branch and diverge.
            biggest = float(np.max(np.abs(dq)))
            if biggest > MAX_IK_STEP:
                dq *= MAX_IK_STEP / biggest
            q = np.clip(q + dq, self.lower, self.upper)
        return q, float(np.linalg.norm(e_pos))


def _yaw_distance(a: float, b: float) -> float:
    """Distance between two jaw yaws, modulo the jaw's 180 deg symmetry.

    Returns:
        The distance in radians, in ``[0, pi/2]``.
    """
    return float(abs((a - b + np.pi / 2) % np.pi - np.pi / 2))


class ConveyorDemonstrator:
    """Sort items off the belt with privileged state, one control tick at a time."""

    def __init__(self, model: object, conveyor: ConveyorSort, config: DemoConfig | None = None) -> None:
        """Bind to a model and its conveyor controller."""
        import mujoco  # noqa: PLC0415

        self._mujoco = mujoco
        self.model = model
        self.conveyor = conveyor
        self.config = config or DemoConfig()
        self.ik = ArmIK(model)
        self.stats = DemoStats()
        self._plan = _Plan()
        self._q: np.ndarray | None = None
        self._grip = self.config.grip_open
        self._actuators = [
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)) for name in (*ARM_JOINTS, GRIPPER)
        ]

    @property
    def phase(self) -> Phase:
        """Current phase of the pick-and-place cycle."""
        return self._plan.phase

    @property
    def target(self) -> ConveyorItem | None:
        """Item the current cycle is working on, if any."""
        return self._plan.item

    def reset(self) -> None:
        """Forget the current cycle; the next `step` starts from the arm's current pose."""
        self._plan = _Plan()
        self._q = None

    # ------------------------------------------------------------------
    # Grasp geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _grasp_yaw(data: object, item: ConveyorItem, flip: float) -> float:
        """Closing yaw along the belt that lines the jaw up with the item's faces.

        Returns:
            The yaw in radians, turned by `flip` (``0`` or ``pi``).
        """
        along = np.pi / 2
        period = SYMMETRY.get(item.shape)
        if period is not None:
            rot = data.xmat[item.body_id].reshape(3, 3)
            base = float(np.arctan2(rot[1, 0], rot[0, 0]))
            along = base + float(np.round((along - base) / period)) * period
        return along + flip

    def _pick_flip(self, data: object, item: ConveyorItem) -> float:
        """Choose between the two 180 deg jaw variants for this grasp.

        A tilted gripper dips the open moving jaw's tip into the belt, and a yaw
        beyond the wrist's range never lines up. Prefer a variant that is upright
        enough and reachable, then the smaller wrist turn, then the smaller tilt.

        Returns:
            ``0`` or ``pi``.
        """
        cfg = self.config
        q_now = self._q if self._q is not None else data.qpos[self.ik.qpos_adr].copy()
        target = data.xpos[item.body_id] + np.array([0.0, 0.0, cfg.grasp_dz])
        options = []
        for flip in (0.0, np.pi):
            yaw = self._grasp_yaw(data, item, flip)
            q, err = self.ik.solve(q_now, target, yaw, iters=40)
            self.ik.data.qpos[self.ik.qpos_adr] = q
            self._mujoco.mj_kinematics(self.model, self.ik.data)
            tilt = self.ik.tilt(self.ik.data)
            yaw_err = _yaw_distance(self.ik.gripper_yaw(self.ik.data), yaw)
            usable = err < 0.003 and tilt < np.radians(7) and yaw_err < np.radians(3)  # noqa: PLR2004
            options.append((usable, -abs(float(q[4] - q_now[4])), -tilt, flip))
        return max(options)[3]

    def _choose(self, data: object) -> ConveyorItem | None:
        """Most downstream item that is on the belt and will still be in the window.

        Returns:
            The item to pick next, or ``None`` if none is pickable yet.
        """
        cfg = self.config
        speed = self.conveyor.belt_speed
        best, best_y = None, np.inf
        for item in self.conveyor.items_in_play:
            p = data.xpos[item.body_id]
            if p[2] < 0.06 or abs(p[0] - cfg.belt_x) > 0.05:  # noqa: PLR2004 - not riding the belt
                continue
            if p[1] >= cfg.pick_y[1] or p[1] - speed * cfg.reach_lookahead_s <= cfg.pick_y[0]:
                continue
            if p[1] < best_y:
                best, best_y = item, float(p[1])
        return best

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def _move_toward(self, goal: np.ndarray, dt: float, speed: float | None = None) -> None:
        plan = self._plan
        step = goal - plan.setpoint
        dist = float(np.linalg.norm(step))
        limit = (speed or self.config.cart_speed) * dt
        plan.setpoint = goal.copy() if dist <= limit else plan.setpoint + step * (limit / dist)

    def _raise_to(self, z: float, dt: float, speed: float | None = None) -> bool:
        goal = self._plan.setpoint.copy()
        goal[2] = z
        self._move_toward(goal, dt, speed)
        return abs(self._plan.setpoint[2] - z) < 0.003  # noqa: PLR2004

    def step(self, data: object, dt: float, *, apply: bool = True) -> np.ndarray:  # noqa: C901, PLR0912, PLR0915
        """Advance the cycle by one control tick of `dt` seconds.

        With `apply`, the targets are written to the arm's actuators. Without it
        they are only returned, for a caller that routes them elsewhere (a
        virtual leader arm, say).

        Returns:
            Joint targets in radians, ordered like ``SO101_JOINT_ORDER`` (arm joints, then gripper).
        """
        cfg, plan = self.config, self._plan
        now = float(data.time)
        if self._q is None:
            self._mujoco.mj_kinematics(self.model, data)
            self._q = data.qpos[self.ik.qpos_adr].copy()
            plan.setpoint = self.ik.tcp(data).copy()
            plan.yaw = self.ik.gripper_yaw(data)
        belt_v = np.array([0.0, -self.conveyor.belt_speed, 0.0])

        if plan.phase == "idle":
            self._grip = cfg.grip_open
            self._move_toward(np.array([cfg.belt_x, cfg.wait_y, cfg.hover_z]), dt)
            item = self._choose(data)
            if item is not None:
                plan.item, plan.phase, plan.since = item, "approach", now
                plan.flip = self._pick_flip(data, item)
                plan.yaw = self._grasp_yaw(data, item, plan.flip)
                self.stats.attempts += 1
        elif plan.phase in {"approach", "descend", "close"}:
            item = plan.item
            if item is None or not self.conveyor.is_in_play(item):
                plan.phase = "return"
            else:
                grasp = data.xpos[item.body_id] + belt_v * cfg.lead_s + np.array([0.0, 0.0, cfg.grasp_dz])
                plan.yaw = self._grasp_yaw(data, item, plan.flip)
                if plan.phase == "approach" and now - plan.since > cfg.approach_timeout_s:
                    # Could not line up (yaw out of the wrist's range, say): let the next item go first.
                    self.stats.approach_timeouts += 1
                    plan.phase = "return"
                elif plan.phase == "approach":
                    goal = np.array([grasp[0], grasp[1], cfg.hover_z])
                    plan.setpoint += belt_v * dt  # ride along with the belt
                    self._move_toward(goal, dt)
                    lined_up = _yaw_distance(self.ik.gripper_yaw(data), plan.yaw) < np.radians(5)
                    if np.linalg.norm(plan.setpoint - goal) < 0.004 and lined_up:  # noqa: PLR2004
                        plan.phase, plan.since = "descend", now
                elif plan.phase == "descend":
                    plan.setpoint += belt_v * dt
                    self._move_toward(grasp, dt, cfg.descend_speed)
                    if np.linalg.norm(plan.setpoint - grasp) < 0.003:  # noqa: PLR2004
                        plan.phase, plan.since = "close", now
                else:
                    plan.setpoint = grasp
                    self._grip = cfg.grip_closed
                    if now - plan.since > cfg.close_s:
                        plan.phase, plan.since = "lift", now
        elif plan.phase in {"lift", "carry", "lower"} and plan.item is None:
            plan.phase = "return"
        elif plan.phase == "lift":
            if self._raise_to(cfg.carry_z, dt, cfg.lift_speed):
                if data.xpos[plan.item.body_id][2] > 0.1:  # noqa: PLR2004 - came up with the gripper
                    plan.phase, plan.since = "carry", now
                else:
                    self.stats.grasp_failed += 1
                    plan.phase = "return"
        elif plan.phase == "carry":
            # Put the item, not the grasp point, over the bin centre: a grasped item sits up to
            # ~1 cm off the grasp point (pushed against the fixed jaw).
            self._mujoco.mj_kinematics(self.model, data)
            offset = (self.ik.tcp(data) - data.xpos[plan.item.body_id])[:2]
            bin_pos = data.xpos[self.conveyor.bin_body_ids[plan.item.target_bin]]
            goal = np.array([bin_pos[0] + offset[0], bin_pos[1] + offset[1], cfg.carry_z])
            self._move_toward(goal, dt)
            if np.linalg.norm(plan.setpoint - goal) < 0.003:  # noqa: PLR2004
                plan.phase, plan.since = "lower", now
        elif plan.phase == "lower":
            if self._raise_to(cfg.drop_z, dt, cfg.descend_speed):
                plan.phase, plan.since = "release", now  # time the release from when the jaws open
                self._grip = cfg.grip_open
        elif plan.phase == "release":
            self._grip = cfg.grip_open
            if now - plan.since > cfg.release_s:
                plan.phase = "rise"
        elif plan.phase in {"rise", "return"}:
            self._grip = cfg.grip_open
            if self._raise_to(cfg.hover_z, dt):
                plan.phase, plan.item = "idle", None

        self._q, _ = self.ik.solve(self._q, plan.setpoint, plan.yaw)
        targets = np.array([*self._q, self._grip], dtype=np.float64)
        if apply:
            for actuator, value in zip(self._actuators, targets, strict=True):
                data.ctrl[actuator] = value
        return targets
