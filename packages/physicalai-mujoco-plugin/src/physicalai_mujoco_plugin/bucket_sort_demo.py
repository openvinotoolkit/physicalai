# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Scripted demonstrator for the ``bucket_sort`` scene: a Franka FR3, a Panda or the FR3 Duo, with the Franka Hand.

What differs per robot (joint and actuator names, the gripper's command, the home pose, the joint
speed limits) comes from its arms in :mod:`.bucket_sort_arms`; on the Duo, each attempt goes to one
arm while the other holds its home.

It reads privileged simulation state (the instruction, the bucket and object poses) and writes the
arm's joint targets one control tick at a time, so the autopilot (``autopilot.py``) can drive the arm
or publish the targets as a virtual leader. Each episode it:

1. waits a moment for the objects to settle, then plans one attempt: the hand turns its fingers
   across the named object's short horizontal side (of the two symmetric yaws, the one whose plan
   keeps the wrist further from its limits), stops above the object, descends straight down and
   closes;
2. lifts the object, carries it above the rim of the named bucket, lowers it into the opening and
   opens the hand a little;
3. rises and returns home, then waits for the episode's success and the scene's next episode.

Motions are minimum-jerk profiles of the hand's position, yaw and opening, overlapped where one move
blends into the next (as human reaching does), turned into joint targets by damped least-squares
inverse kinematics seeded from the previous tick, so the arm never jumps to another IK branch. A
plan is checked before it runs: IK residual, joint limits with a margin, joint speed, and the
clearance of the swept hand (and of the held object) from the buckets, the other objects and the
table. Every episode draws its heights, speeds, grasp offset, yaw choice and pauses from its own
generator, drawn from the demonstrator's: the same seed repeats the same episodes.

A grasp that does not lift the object, an object that slips out on the way, or one that lands
outside the bucket gets one retry: the hand lets go and returns home, and once the object rests (on
the table, in reach, its long side level) a new attempt with fresh parameters starts from there.
After that the episode is marked ``failed`` and the demonstrator asks for a scene reset
(:meth:`BucketSortDemonstrator.take_reset_request`), so a recorder can discard the episode and the
next one starts. A target object that moves or turns before the grasp re-plans the approach; a target bucket
that moves during the carry or the descent into it, up to the release, re-plans the place. The
buckets are free bodies: their poses are read from the live data on every check and copied, with
the objects', from ``qpos`` into every plan.

In ``leader`` mode nothing but a client sends the targets back. When the arm does not reach a
plan's last target within :data:`CATCH_UP_S`, the demonstrator stops (``stalled``) instead of
failing and resetting episodes that nobody runs.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import starmap
from typing import TYPE_CHECKING, Literal

import numpy as np
from loguru import logger

from physicalai_mujoco_plugin.bucket_sort import BUCKET_COLORS, BUCKET_INTERIOR, OBJECT_Y, OBJECTS, bucket_containing
from physicalai_mujoco_plugin.bucket_sort_arms import FINGER_JOINTS, FR3_ARM, FRANKA_GRIP_OPEN, FrankaArm

if TYPE_CHECKING:
    from collections.abc import Callable

    from physicalai_mujoco_plugin.bucket_sort import BucketSortTask, Instruction

ARM_JOINTS = FR3_ARM.joints
ACTUATORS = (*FR3_ARM.actuators, FR3_ARM.gripper)
"""The ``franka_fr3`` channels, in action order: seven arm joints (radians in the model) and the hand's width (m)."""
HAND_BODY = "hand"
TCP_LOCAL = np.array([0.0, 0.0, 0.1034])
"""Franka's TCP in the hand frame: between the finger tips, 103.4 mm from the hand's base."""
HOME_Q = np.array(FR3_ARM.home)
"""Franka's ready pose, the scene's home: the hand 0.49 m above the table, fingers down."""
HOME_YAW = -np.pi / 2
"""World yaw of the closing axis (the hand's y axis) at :data:`HOME_Q`."""
GRIP_OPEN = FRANKA_GRIP_OPEN
"""The hand's full opening, in metres."""

FR3_DQ_MAX = np.array(FR3_ARM.dq_max)
"""FR3 joint velocity limits, rad/s (Franka's FR3 datasheet)."""
FR3_DDQ_MAX = np.array(FR3_ARM.ddq_max)
"""FR3 joint acceleration limits, rad/s^2."""
FR3_DDDQ_MAX = np.full(7, 5000.0)
"""FR3 joint jerk limits, rad/s^3."""
JOINT_MARGIN = float(np.radians(7.0))
"""The IK keeps every joint this far inside the FR3's hardware limits."""
SPEED_FRACTION = 0.5
"""A plan may use at most this fraction of the FR3's joint velocity limits."""
ACCEL_FRACTION = 0.5
"""A plan may use at most this fraction of the FR3's joint acceleration limits."""
START_BLEND_S = 0.5
"""A plan fades in any first-tick IK correction over this long."""
MAX_STRETCH = 3.0
"""A move that is too fast for the joints is stretched to at most this many times its duration."""
MAX_TICK_SPEED = 0.8
"""Joint targets never change faster than this fraction of the limits, even where the IK would jump."""

MIN_TCP_Z = 0.013
"""Lowest grasp: the finger pads reach 9 mm below the TCP, so they stay 4 mm above the table."""
CLEARANCE = {
    "bucket": 0.012,
    "objects": 0.008,
    "table": 0.002,
    "held_bucket": 0.010,
    "held_objects": 0.008,
    "arm": 0.03,
}
"""Smallest planned distance, in metres, from the robot (or the object it holds) to each kind of obstacle.

``arm`` is the working arm's distance from another arm that holds still (the Duo's)."""
IK_POS_TOL = 0.003
IK_ROT_TOL = float(np.radians(2.0))
SHOULDER_SINGULAR_WIDTH = 0.3
SHOULDER_SINGULAR_WEIGHT = 1.0
IK_DAMPING = 0.03
"""Damping of the least-squares step (m, rad): small, the targets stay well inside the workspace."""
MAX_IK_STEP = 0.1
"""Largest joint change, in radians, of one IK iteration."""
CHECK_EVERY = 2
"""Clearance is checked every this many ticks of a plan."""
WRIST_PROBE_Z = 0.15
"""TCP height at which a tilted arm's joint 7 is estimated for a yaw over a point (:meth:`Planner.predicted_q7`)."""

LIFT_CHECK_DZ = 0.04
"""The grasp is checked once the hand has risen this far above the grasp height."""
LIFT_CHECK_WAIT_S = 1.0
"""The lift is judged at most this long after its planned tick, however far the arm lags."""
LIFTED_DZ = 0.02
"""A held object has risen at least this far above where it lay."""
HELD_DISTANCE = 0.045
"""A held object's centre stays within this distance of the TCP."""
SETTLE_TIMEOUT_S = 10.0
"""Give up on an object that does not come to rest within this long."""
SETTLE_CHECK_S = 1.2
"""After returning home, wait this long for the placed object to come to rest before judging it."""
CATCH_UP_S = 1.5
"""At the end of a plan, wait at most this long for the arm to reach its last target."""
STALL_ERROR = 0.3
"""A joint further than this (rad) from its target for :data:`CATCH_UP_S` means nobody follows the targets.

The position servos lag up to 0.2 rad at the plan's top speed; a client easing the targets adds
its own delay."""
BRAKE_S = 0.35
"""An abandoned motion slows to a stop over this long."""
FAILURE_HOLD_S = 1.0
"""A failed episode is reported this long, at home, before the reset is requested."""
REPLAN_SHIFT = 0.012
"""Re-plan the approach when the target object moved this far (m) before the grasp."""
REPLAN_TURN = float(np.radians(5.0))
"""Re-plan the approach when the target object's grasp axis turned this far before the grasp."""
RETRY_X = (0.24, 0.52)
RETRY_Y = (OBJECT_Y[0] - 0.02, OBJECT_Y[1] + 0.02)
"""Where a retry picks the object up again: around the spawn area (``bucket_sort.OBJECT_X`` and
``OBJECT_Y``), with room for the nudge a missed grasp gives it."""
MAX_REPLANS = 3
DEFAULT_MAX_ATTEMPTS = 2
"""Attempts per episode, the retry included, unless a recorder asks for others (``RecordingOptions``)."""
STUCK_PHASES = frozenset({"carry", "lower", "release"})
"""Where a stuck start puts the hand: on the way to the bucket or over it (``stuck_start``)."""
STUCK_GRIP = (0.0, 0.03)
"""The hand's opening at a stuck start, metres: closed or almost, as if it held the object."""
STUCK_MOVE_S = (1.5, 2.5)
"""How long the move to a stuck start takes, seconds."""
STUCK_TRIES = 8
"""Poses of the planned path tried before a stuck start gives up."""
STILL_SPEED = 0.01
"""With several arms, an arm whose joints are all slower than this (rad/s) is still (``_arms_still``)."""
STILL_S = 0.1
"""How long the arms must be still on their targets before an attempt is planned."""
TARGET_TOLERANCE = 1e-4
"""How close (rad) an arm's actuator targets must be to the ones it holds; a client's degrees round a little."""
OUT_OF_REACH = 0.02
"""With several arms, a best-effort plan whose IK misses by more than this (m), or that comes too close
to an arm holding still, does not run: the episode fails instead (no arm reaches it)."""

Phase = Literal[
    "idle",
    "settle",
    "approach",
    "descend",
    "close",
    "lift",
    "carry",
    "lower",
    "release",
    "retreat",
    "return",
    "done",
    "failed",
    "stalled",
]
Outcome = Literal["running", "success", "failed", "stalled"]
PlanKind = Literal["attempt", "place", "home"]
"""A whole attempt (pick, place, home), the rest of one with the object in the hand, or the way home."""


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class DemoParams:
    """Motion parameters of one attempt (metres, seconds, radians); :meth:`sample` randomizes them."""

    start_pause_s: float = 0.6
    """Wait after the episode starts, for the objects to settle."""
    transit_speed: float = 0.42
    """Peak hand speed of the long moves (approach, carry, return), m/s."""
    fine_speed: float = 0.12
    """Peak hand speed of the descent onto the object and into the bucket, m/s."""
    lift_speed: float = 0.18
    yaw_speed: float = 1.5
    """Peak turn rate of the hand about the vertical, rad/s."""
    held_accel: float = 1.0
    """Peak hand acceleration with the object in the hand, m/s^2."""
    pregrasp_height: float = 0.10
    """Height of the stop above the grasp."""
    lift_height: float = 0.10
    """Height above the grasp of the vertical lift, before the carry blends in."""
    carry_z: float = 0.23
    """TCP height of the carry: the held object passes 6 cm or more above the 0.12 m rims."""
    release_z: float = 0.14
    """TCP height where the hand opens over the bucket."""
    grasp_offset: float = 0.0
    """Grasp this far (m) from the object's centre along its long side."""
    grasp_dz: float = 0.002
    """Grasp this far above the object's centre (or :data:`MIN_TCP_Z`)."""
    yaw_noise: float = 0.0
    """Turn of the closing axis away from square to the object's short side."""
    open_margin: float = 0.025
    """Opening beyond the object's width on the way down."""
    release_margin: float = 0.02
    """Opening beyond the object's width to let it go."""
    close_s: float = 0.6
    settle_s: float = 0.3
    """Pause with the hand closed before lifting."""
    release_pause_s: float = 0.2
    """Pause over the bucket before opening."""
    after_release_s: float = 0.4
    blend: float = 0.25
    """How much of a move overlaps the next one, as a fraction of its duration (0: stop between moves)."""
    place_offset: tuple[float, float] = (0.0, -0.01)
    """Release point in the bucket frame, from the interior's centre: across, along (+ toward the open end)."""
    reorient: bool = True
    """Turn the object during the carry so its long side runs along the bucket."""
    release_yaw_noise: float = 0.0
    yaw_choice: float = 0.5
    """Uniform draw that picks between two grasp yaws whose plans are about as good."""

    @classmethod
    def sample(cls, rng: np.random.Generator) -> DemoParams:
        """Draw a randomized set of parameters, every one inside what the scene allows.

        Returns:
            The parameters.
        """
        u = rng.uniform
        return cls(
            start_pause_s=float(u(0.4, 1.2)),
            transit_speed=float(u(0.30, 0.55)),
            fine_speed=float(u(0.08, 0.16)),
            lift_speed=float(u(0.12, 0.24)),
            yaw_speed=float(u(1.0, 2.0)),
            held_accel=float(u(0.7, 1.2)),
            pregrasp_height=float(u(0.06, 0.13)),
            lift_height=float(u(0.07, 0.13)),
            carry_z=float(u(0.20, 0.27)),
            release_z=float(u(0.12, 0.18)),
            grasp_offset=float(u(-0.006, 0.006)),
            grasp_dz=float(u(0.0, 0.004)),
            yaw_noise=float(np.radians(u(-7.0, 7.0))),
            open_margin=float(u(0.018, 0.035)),
            release_margin=float(u(0.012, 0.03)),
            close_s=float(u(0.45, 0.8)),
            settle_s=float(u(0.15, 0.5)),
            release_pause_s=float(u(0.0, 0.4)),
            after_release_s=float(u(0.25, 0.6)),
            blend=float(u(0.1, 0.35)),
            place_offset=(float(u(-0.008, 0.008)), float(u(-0.03, 0.01))),
            reorient=bool(rng.random() < 0.75),  # noqa: PLR2004
            release_yaw_noise=float(np.radians(u(-10.0, 10.0))),
            yaw_choice=float(rng.random()),
        )


# ----------------------------------------------------------------------
# Minimum-jerk motion script
# ----------------------------------------------------------------------


def min_jerk(s: np.ndarray | float) -> np.ndarray:
    """Return the minimum-jerk profile ``10 s^3 - 15 s^4 + 6 s^5`` of a move's progress, clipped to [0, 1]."""
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    return s**3 * (10.0 - 15.0 * s + 6.0 * s * s)


def move_duration(distance: float, peak_speed: float, minimum: float = 0.3, peak_accel: float | None = None) -> float:
    """Return how long a minimum-jerk move of *distance* takes at a peak speed of *peak_speed*.

    Its peak speed is 1.875 times its mean speed, and its peak acceleration ``5.77 distance / T^2``.

    Returns:
        The duration in seconds, at least *minimum*, and long enough to stay under *peak_accel*.
    """
    duration = max(minimum, 1.875 * abs(distance) / peak_speed)
    if peak_accel is not None:
        duration = max(duration, float(np.sqrt(5.77 * abs(distance) / peak_accel)))
    return duration


@dataclass
class _Move:
    phase: Phase
    t0: float
    duration: float
    delta: np.ndarray
    target: np.ndarray
    """The state ``(x, y, z, yaw, grip)`` the script reaches once this move (and those before it) end."""
    q_delta: np.ndarray | None = None
    """Change of the joint-space reference posture over this move (see :meth:`Planner.solve`)."""


class MotionScript:
    """A sum of minimum-jerk moves of the state ``(x, y, z, yaw, grip)``.

    Each move adds its displacement with a minimum-jerk profile; a move may start before the
    previous one ends (``blend``), which rounds the corner between them while keeping the speed,
    acceleration and jerk continuous.
    """

    def __init__(self, start: np.ndarray, slow: dict[str, float] | None = None) -> None:
        """Start at *start*, ``(x, y, z, yaw, grip)``; moves of a phase in *slow* take that many times longer."""
        self.slow = dict(slow or {})
        self.start = np.asarray(start, dtype=np.float64).copy()
        self.end = self.start.copy()
        self.moves: list[_Move] = []
        self.t_end = 0.0

    def add(
        self,
        phase: Phase,
        *,
        pos: np.ndarray | None = None,
        yaw: float | None = None,
        grip: float | None = None,
        duration: float,
        blend: float = 0.0,
    ) -> None:
        """Move to the given position, yaw and opening (the others keep their value) in *duration* seconds.

        Args:
            phase: The phase it reports while it runs.
            pos: TCP position.
            yaw: Closing-axis yaw, unwrapped: the hand turns the way the numbers say.
            grip: Commanded opening.
            duration: Seconds.
            blend: Fraction of the previous move it overlaps.
        """
        duration = float(duration) * self.slow.get(phase, 1.0)
        target = self.end.copy()
        if pos is not None:
            target[:3] = pos
        if yaw is not None:
            target[3] = yaw
        if grip is not None:
            target[4] = grip
        last = self.moves[-1] if self.moves else None
        t0 = self.t_end
        if last is not None and blend > 0.0 and np.any(last.delta):
            t0 = max(last.t0, self.t_end - blend * last.duration)
        self.moves.append(_Move(phase, t0, float(duration), target - self.end, target.copy()))
        self.end = target
        self.t_end = max(self.t_end, t0 + float(duration))

    def pause(self, phase: Phase, seconds: float) -> None:
        """Hold still for *seconds*, reporting *phase*."""
        if seconds > 0.0:
            self.moves.append(_Move(phase, self.t_end, float(seconds), np.zeros(5), self.end.copy()))
            self.t_end += float(seconds)

    def at(self, t: np.ndarray) -> np.ndarray:
        """Return the state at times *t*, shape ``(len(t), 5)``."""
        t = np.asarray(t, dtype=np.float64)
        state = np.repeat(self.start[None, :], len(t), axis=0)
        for move in self.moves:
            if np.any(move.delta):
                state += min_jerk((t - move.t0) / move.duration)[:, None] * move.delta[None, :]
        return state

    def reference(self, t: np.ndarray, q_start: np.ndarray) -> np.ndarray:
        """Return the joint-space reference posture at times *t*, shape ``(len(t), 7)``.

        It moves like the state, a minimum-jerk profile per move, by each move's ``q_delta``.
        """
        t = np.asarray(t, dtype=np.float64)
        ref = np.repeat(np.asarray(q_start, dtype=np.float64)[None, :], len(t), axis=0)
        for move in self.moves:
            if move.q_delta is not None and np.any(move.q_delta):
                ref += min_jerk((t - move.t0) / move.duration)[:, None] * move.q_delta[None, :]
        return ref

    def phases(self, t: np.ndarray) -> list[Phase]:
        """Return the phase at each time: that of the last move started by then."""
        starts = np.array([move.t0 for move in self.moves])
        out: list[Phase] = []
        for value in np.asarray(t):
            index = int(np.searchsorted(starts, value + 1e-9, side="right")) - 1
            out.append(self.moves[max(index, 0)].phase)
        return out

    def first_time(self, phase: Phase) -> float:
        """Return when the first move of *phase* starts, or the end of the script.

        Returns:
            Seconds from the start.
        """
        return next((move.t0 for move in self.moves if move.phase == phase), self.t_end)


# ----------------------------------------------------------------------
# Kinematics
# ----------------------------------------------------------------------


def hand_rotation(yaw: float) -> np.ndarray:
    """Return the hand's world orientation with the fingers down and the closing axis at *yaw*.

    Returns:
        The 3x3 rotation; its columns are the hand's x, y (closing) and z (finger) axes.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[-s, c, 0.0], [c, s, 0.0], [0.0, 0.0, -1.0]])


def wrap_near(angle: float, reference: float, period: float = 2.0 * np.pi) -> float:
    """Return *angle* plus a whole number of periods, as close as possible to *reference*.

    Returns:
        The shifted angle.
    """
    return float(angle + period * np.round((reference - angle) / period))


class FrankaIK:
    """Damped least-squares IK of a Franka arm's TCP: a position, fingers down, the closing axis at a yaw.

    The FR3 has one joint more than the task needs. That redundancy (where the wrist sits on a
    small circle about the hand's vertical axis, with the elbow swinging along) is resolved in the
    task's null space: :meth:`solve` pulls toward a reference posture, and :meth:`optimize` moves
    the arm away from its joint limits, which matters most for the elbow (folded over the near
    objects, nearly straight over the far end of a bucket). Joints stay :data:`JOINT_MARGIN` inside
    the hardware limits; a joint pinned there drops out of the solve (an active set). It runs on a
    private ``MjData``, so the simulation is untouched.

    The arm (:class:`~.bucket_sort_arms.FrankaArm`) names the joints, the hand and the home pose; it
    defaults to the ``franka_fr3`` profile's FR3.
    """

    def __init__(self, model: object, arm: FrankaArm = FR3_ARM) -> None:
        """Resolve *arm*'s joints and hand on *model*."""
        import mujoco  # noqa: PLC0415

        self._mujoco = mujoco
        self.model = model
        self.arm = arm
        self.data = mujoco.MjData(model)
        joints = [model.joint(arm.name(name)).id for name in arm.joints]
        self.qpos_adr = np.array([int(model.jnt_qposadr[j]) for j in joints])
        self.dof_adr = np.array([int(model.jnt_dofadr[j]) for j in joints])
        self.hard_lower = np.array([float(model.jnt_range[j][0]) for j in joints])
        self.hard_upper = np.array([float(model.jnt_range[j][1]) for j in joints])
        self.lower = self.hard_lower + JOINT_MARGIN
        self.upper = self.hard_upper - JOINT_MARGIN
        self._centre = (self.hard_lower + self.hard_upper) / 2.0
        self._half = (self.hard_upper - self.hard_lower) / 2.0
        self.finger_adr = np.array([int(model.jnt_qposadr[model.joint(arm.name(name)).id]) for name in FINGER_JOINTS])
        self.hand = int(model.body(arm.name(HAND_BODY)).id)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self.damping = IK_DAMPING
        self.home = np.array(arm.home)
        """The arm's home pose: plans start and end there."""
        self.home_yaw = float(arm.home_yaw) if arm.home_yaw is not None else self.fk(self.home)[1]
        """World yaw of the closing axis at :attr:`home`."""

    def set_arm(self, q: np.ndarray, grip: float | None = None) -> None:
        """Put the private model's arm at *q* (and the fingers at half *grip* each) and update its kinematics."""
        self.data.qpos[self.qpos_adr] = q
        if grip is not None:
            self.data.qpos[self.finger_adr] = min(max(grip, 0.0), GRIP_OPEN) / 2.0
        self._mujoco.mj_kinematics(self.model, self.data)

    def tcp(self, data: object | None = None) -> np.ndarray:
        """Return the TCP's world position (of *data*, by default the private one)."""
        data = self.data if data is None else data
        return data.xpos[self.hand] + data.xmat[self.hand].reshape(3, 3) @ TCP_LOCAL

    def yaw(self, data: object | None = None) -> float:
        """Return the closing axis' world yaw, in ``(-pi, pi]``."""
        data = self.data if data is None else data
        axis = data.xmat[self.hand].reshape(3, 3)[:, 1]
        return float(np.arctan2(axis[1], axis[0]))

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, float]:
        """Return the TCP position and closing-axis yaw at *q*.

        Returns:
            ``(position, yaw)``; the yaw is wrapped to ``(-pi, pi]``.
        """
        self.set_arm(q)
        return self.tcp(), self.yaw()

    def continuous_yaw(self, q: np.ndarray) -> float:
        """Return the closing-axis yaw at *q* on the branch the planner uses: home is :attr:`home_yaw`.

        Turning the hand about the vertical turns joint 7 the other way, and the base joint turns it
        along, so ``home_yaw + (q1 - q1_home) - (q7 - q7_home)`` tells which turn of the wrapped yaw
        is meant. On a tilted base (the Duo's) the base joint turns the hand less than that, but far
        less than the half turn that would pick the wrong branch.

        Returns:
            The yaw in radians.
        """
        _, yaw = self.fk(q)
        return wrap_near(yaw, self.home_yaw + float(q[0] - self.home[0]) - float(q[6] - self.home[6]))

    def limit_gradient(self, q: np.ndarray) -> np.ndarray:
        """Return the descent direction of ``sum(((q - centre) / half range) ** 6)``.

        The cost is flat in the middle of each joint's range and steep near its ends.

        Returns:
            Per joint, unscaled.
        """
        x = (q - self._centre) / self._half
        gradient = -6.0 * x**5 / self._half
        # Joint 2 near zero lines joints 1 and 3 up (a shoulder singularity): push it off zero.
        width = SHOULDER_SINGULAR_WIDTH
        gradient[1] += SHOULDER_SINGULAR_WEIGHT * 2.0 * q[1] / width**2 * np.exp(-((q[1] / width) ** 2))
        return gradient

    def margin(self, q: np.ndarray) -> float:
        """Return the smallest distance (rad) of any joint to its hardware limit."""
        return float(min((q - self.hard_lower).min(), (self.hard_upper - q).min()))

    def solve(  # noqa: PLR0914 - one numerical routine; splitting it hides the algorithm
        self,
        q0: np.ndarray,
        pos: np.ndarray,
        yaw: float,
        *,
        iters: int = 6,
        posture: np.ndarray | None = None,
        posture_gain: float = 0.5,
        limit_gain: float = 0.0,
        null_step: float = 0.05,
        tol: float = 2e-5,
    ) -> tuple[np.ndarray, float, float]:
        """Return joint angles near *q0* that put the TCP at *pos*, fingers down, closing along *yaw*.

        Args:
            q0: Seed.
            pos: TCP target.
            yaw: Closing-axis yaw.
            iters: Gauss-Newton iterations at most.
            posture: Where the null space pulls (each iteration, by *posture_gain* of the gap).
            posture_gain: Fraction of the gap to *posture* closed per iteration.
            limit_gain: Step along :meth:`limit_gradient` per iteration.
            null_step: Largest null-space change per iteration (rad).
            tol: Position tolerance (m) to stop early; rotation uses ten times it in radians.

        Returns:
            ``(q, position error in m, orientation error in rad)``.
        """
        mujoco, model, data = self._mujoco, self.model, self.data
        q = np.clip(np.asarray(q0, dtype=np.float64).copy(), self.lower, self.upper)
        r_des = hand_rotation(yaw)
        quat, e_rot = np.zeros(4), np.zeros(3)
        n = len(q)
        eye6, eye_n = np.eye(6), np.eye(n)
        pos_err, rot_err = np.inf, np.inf
        for it in range(iters + 1):
            data.qpos[self.qpos_adr] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            p = self.tcp()
            e_pos = pos - p
            mujoco.mju_mat2Quat(quat, (r_des @ data.xmat[self.hand].reshape(3, 3).T).ravel())
            mujoco.mju_quat2Vel(e_rot, quat, 1.0)
            pos_err, rot_err = float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot))
            pull = np.zeros(n)
            if posture is not None:
                pull += posture_gain * (posture - q)
            if limit_gain > 0.0:
                pull += limit_gain * self.limit_gradient(q)
            size = float(np.abs(pull).max())
            if size > null_step:
                pull *= null_step / size
            converged = pos_err < tol and rot_err < tol * 10.0 and size < 1e-4  # noqa: PLR2004
            if converged or it == iters:
                break
            mujoco.mj_jac(model, data, self._jacp, self._jacr, p, self.hand)
            jac = np.vstack([self._jacp[:, self.dof_adr], self._jacr[:, self.dof_adr]])
            error = np.concatenate([e_pos, e_rot])
            free = np.ones(n, dtype=bool)
            dq = np.zeros(n)
            for _pass in range(n):
                jf = jac * free
                pinv = jf.T @ np.linalg.inv(jf @ jf.T + self.damping**2 * eye6)
                dq = pinv @ error + (eye_n - pinv @ jf) @ (pull * free)
                dq *= free
                blocked = free & (((q >= self.upper - 1e-9) & (dq > 0)) | ((q <= self.lower + 1e-9) & (dq < 0)))
                if not blocked.any():
                    break
                free &= ~blocked
            biggest = float(np.max(np.abs(dq)))
            if biggest > MAX_IK_STEP:
                dq *= MAX_IK_STEP / biggest
            q = np.clip(q + dq, self.lower, self.upper)
        return q, pos_err, rot_err

    def optimize(self, q0: np.ndarray, pos: np.ndarray, yaw: float, iters: int = 80) -> tuple[np.ndarray, float, float]:
        """Solve for *pos* and *yaw*, then move along the null space away from the joint limits.

        The result stays on the seed's side of the null space, so consecutive key poses solved from
        each other connect without a jump.

        Returns:
            ``(q, position error in m, orientation error in rad)``.
        """
        q, e_pos, e_rot = self.solve(q0, pos, yaw, iters=30)
        return (
            self.solve(q, pos, yaw, iters=iters, limit_gain=0.02, null_step=0.03, tol=1e-6)
            if iters
            else (q, e_pos, e_rot)
        )


# ----------------------------------------------------------------------
# Scene geometry
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectShape:
    """An object's collision box in its body frame; the hand closes across its long side."""

    half: np.ndarray
    """Half extents along the body's x, y and z."""

    @property
    def long_axis(self) -> int:
        """The body axis of the longest side."""
        return int(np.argmax(self.half))

    @property
    def half_length(self) -> float:
        """Half the longest side."""
        return float(self.half[self.long_axis])

    def grasp_axes(self, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Return how to grasp the object lying with *rotation*: across its long side, horizontally.

        This holds lying flat and also rolled about the long side (the resistor rolls), as long as
        the long side lies roughly level.

        Returns:
            ``(closing axis, long axis, width)``: horizontal unit vectors in the world frame and the
            box's width along the closing axis; ``None`` when the long side is not level.
        """
        along = rotation[:, self.long_axis].copy()
        if abs(float(along[2])) > 0.35:  # noqa: PLR2004 - standing on end, or nearly
            return None
        along[2] = 0.0
        along /= np.linalg.norm(along)
        closing = np.array([-along[1], along[0], 0.0])
        width = float(2.0 * np.sum(self.half * np.abs(rotation.T @ closing)))
        return closing, along, width


def object_shape(model: object, name: str) -> ObjectShape:
    """Measure object *name*'s collision hull in its body frame.

    Returns:
        Its box and grasp axis.
    """
    import mujoco  # noqa: PLC0415

    body = int(model.body(name).id)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    rotation = np.zeros(9)
    for geom in np.flatnonzero(np.asarray(model.geom_bodyid) == body):
        if int(model.geom_contype[geom]) == 0 and int(model.geom_conaffinity[geom]) == 0:
            continue
        mesh = int(model.geom_dataid[geom])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
        points = model.mesh_vert[start : start + count] @ rotation.reshape(3, 3).T + model.geom_pos[geom]
        lo, hi = np.minimum(lo, points.min(axis=0)), np.maximum(hi, points.max(axis=0))
    return ObjectShape(half=(hi - lo) / 2.0)


class _Clearance:
    """Distances from the hand (and a held object) to the buckets, the other objects and the table.

    With *others*, also from the whole working arm to those arms, which hold still.
    """

    def __init__(
        self, model: object, data: object, arm: FrankaArm = FR3_ARM, others: tuple[FrankaArm, ...] = ()
    ) -> None:
        import mujoco  # noqa: PLC0415

        self._mujoco = mujoco
        self.model = model
        self.data = data

        def collision_geoms(body_names: tuple[str, ...]) -> np.ndarray:
            bodies = [int(model.body(name).id) for name in body_names]
            return np.array([
                g
                for g in range(model.ngeom)
                if int(model.geom_bodyid[g]) in bodies and (model.geom_contype[g] or model.geom_conaffinity[g])
            ])

        hand = tuple(arm.name(body) for body in (HAND_BODY, "left_finger", "right_finger"))
        self.robot = collision_geoms((*hand, *(arm.name(link) for link in reversed(arm.links))))
        self.fingers = collision_geoms(hand[1:])
        # The base (link 0) never moves: its distance from another arm's base is the mount's.
        moving = tuple(arm.name(link) for link in arm.arm_links[1:])
        self.arm = collision_geoms((*hand, *moving)) if others else None
        self.other_arms = (
            collision_geoms(
                tuple(o.name(b) for o in others for b in (*o.arm_links, HAND_BODY, "left_finger", "right_finger"))
            )
            if others
            else None
        )
        self.buckets = collision_geoms(tuple(f"bucket_{color}" for color in BUCKET_COLORS))
        self.objects = {name: collision_geoms((name,)) for name in OBJECTS}
        self.table = np.array([int(model.geom("table_top").id)])
        self._fromto = np.zeros(6)

    def _min_distance(self, group_a: np.ndarray, group_b: np.ndarray, cap: float = 0.05) -> float:
        data, model = self.data, self.model
        pa, pb = data.geom_xpos[group_a], data.geom_xpos[group_b]
        ra, rb = model.geom_rbound[group_a], model.geom_rbound[group_b]
        centre = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
        near = centre - ra[:, None] - rb[None, :] < cap
        best = cap
        for i, j in zip(*np.nonzero(near), strict=True):
            distance = self._mujoco.mj_geomDistance(model, data, int(group_a[i]), int(group_b[j]), cap, self._fromto)
            best = min(best, float(distance))
        return best

    def measure(self, *, target: str, held: bool) -> dict[str, float]:
        """Return the smallest distance to each kind of obstacle at the private data's current pose.

        Args:
            target: The object being picked: the fingers may touch it, and it is the held object.
            held: Whether the hand carries *target* (its pose in the private data must follow the hand).

        Returns:
            Distances in metres by :data:`CLEARANCE` key, capped at 5 cm.
        """
        others = np.concatenate([geoms for name, geoms in self.objects.items() if name != target])
        out = {
            "bucket": self._min_distance(self.robot, self.buckets),
            "objects": self._min_distance(self.robot, others),
            "table": self._min_distance(self.fingers, self.table),
        }
        if held:
            out["held_bucket"] = self._min_distance(self.objects[target], self.buckets)
            out["held_objects"] = self._min_distance(self.objects[target], others)
        if self.arm is not None and self.other_arms is not None:
            out["arm"] = self._min_distance(self.arm, self.other_arms)
        return out


# ----------------------------------------------------------------------
# Plans
# ----------------------------------------------------------------------


@dataclass
class JointPlan:
    """Joint targets for every control tick of one plan, with what was checked about them."""

    kind: PlanKind
    dt: float
    q: np.ndarray
    """Arm targets, ``(ticks, 7)``, radians."""
    grip: np.ndarray
    """Hand opening targets, ``(ticks,)``, metres."""
    phases: list[Phase]
    cartesian: np.ndarray
    """The scripted ``(x, y, z, yaw, grip)`` per tick."""
    lift_check: int | None = None
    """Tick at which the grasp is checked."""
    hold_until: int | None = None
    """Last tick (exclusive) the object must still be in the hand: the start of the release."""
    carry_start: int | None = None
    """First tick of the carry: from there to the release the bucket's pose is checked every tick."""
    bucket_pose: tuple[np.ndarray, float] | None = None
    """The target bucket's position and yaw the plan was made for."""
    yaw_grasp: float = 0.0
    yaw_release: float = 0.0
    pos_error: float = 0.0
    rot_error: float = 0.0
    clearance: dict[str, float] = field(default_factory=dict)
    limit_margin: float = np.inf
    """Smallest distance (rad) of any joint to its hardware limit."""
    peak_speed_ratio: float = 0.0
    """Largest joint speed as a fraction of the FR3's limit."""
    peak_accel_ratio: float = 0.0
    """Largest joint acceleration as a fraction of the FR3's limit."""
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether every check passed."""
        return not self.problems

    def __len__(self) -> int:
        """Return the number of ticks.

        Returns:
            The plan's length.
        """
        return len(self.q)

    def targets(self, tick: int) -> np.ndarray:
        """Return the targets of *tick*, clamped to the plan's ends.

        Returns:
            Eight values: seven joints (rad) and the opening (m).
        """
        tick = min(max(tick, 0), len(self.q) - 1)
        return np.array([*self.q[tick], self.grip[tick]])


def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth each column of *values* with a Gaussian of *sigma* samples, holding the end values.

    Returns:
        The smoothed array; the first and last rows are unchanged when the ends are at rest.
    """
    radius = max(1, int(np.ceil(3.0 * sigma)))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate([np.repeat(values[:1], radius, axis=0), values, np.repeat(values[-1:], radius, axis=0)])
    return np.stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(values.shape[1])], axis=1)


def joint_smoothness(q: np.ndarray, dt: float) -> dict[str, np.ndarray]:
    """Peak joint speed, acceleration and jerk of a target trajectory, by finite differences.

    Returns:
        ``velocity``, ``acceleration`` and ``jerk``: per-joint peaks (rad/s, rad/s^2, rad/s^3).
    """
    q = np.asarray(q, dtype=np.float64)
    zeros = np.zeros(q.shape[1])
    v = np.diff(q, axis=0) / dt if len(q) > 1 else np.zeros((1, q.shape[1]))
    a = np.diff(v, axis=0) / dt if len(v) > 1 else np.zeros((1, q.shape[1]))
    j = np.diff(a, axis=0) / dt if len(a) > 1 else np.zeros((1, q.shape[1]))
    return {
        "velocity": np.abs(v).max(axis=0) if len(v) else zeros,
        "acceleration": np.abs(a).max(axis=0) if len(a) else zeros,
        "jerk": np.abs(j).max(axis=0) if len(j) else zeros,
    }


@dataclass
class SceneSnapshot:
    """What a plan needs from the simulation: the objects and buckets (free joints, in ``qpos``) as they are now."""

    qpos: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray

    @classmethod
    def take(cls, data: object) -> SceneSnapshot:
        """Copy the poses out of *data*.

        Returns:
            The snapshot.
        """
        return cls(np.array(data.qpos), np.array(data.mocap_pos), np.array(data.mocap_quat))


class Planner:
    """Build and check joint plans of one arm for the bucket_sort scene."""

    def __init__(self, model: object, arm: FrankaArm = FR3_ARM, others: tuple[FrankaArm, ...] = ()) -> None:
        """Bind to the scene model and the *arm* that works; planning runs on private data.

        Args:
            model: The bucket_sort model.
            arm: The arm the plans move.
            others: Arms that hold still meanwhile; the plans keep clear of them.
        """
        import mujoco  # noqa: PLC0415

        self._mujoco = mujoco
        self.model = model
        self.arm = arm
        self.ik = FrankaIK(model, arm)
        self.dq_max = np.array(arm.dq_max)
        self.ddq_max = np.array(arm.ddq_max)
        self.clearance = _Clearance(model, self.ik.data, arm, others)
        self._wrist_probe: tuple[tuple[float, float], float, float] | None = None
        """The last point :meth:`predicted_q7` solved a tilted arm for: the point, joint 7 and the yaw there."""
        self.shapes = {name: object_shape(model, name) for name in OBJECTS}
        self._object_qpos = {name: int(model.jnt_qposadr[model.joint(f"{name}:joint").id]) for name in OBJECTS}
        self._object_body = {name: int(model.body(name).id) for name in OBJECTS}
        self._bucket_body = {color: int(model.body(f"bucket_{color}").id) for color in BUCKET_COLORS}
        self.held_width = 0.04
        """The width the last attempt script grasps; the place re-plan and the clearance use it."""
        self.tick_null_step = 0.01
        """Largest null-space change per IK iteration while following the reference posture."""
        self.tick_limit_gain = 0.0
        self.tick_iters = 3
        self.smooth_ticks = 3.0
        """Standard deviation, in ticks, of the Gaussian that smooths the joint targets after the IK."""
        """Step along the joint-limit gradient per tick, so a joint slows before its soft limit."""

    # -- scene readout ----------------------------------------------------

    def load(self, snapshot: SceneSnapshot) -> None:
        """Put the snapshot's objects and buckets into the private data."""
        data = self.ik.data
        data.qpos[:] = snapshot.qpos
        data.mocap_pos[:] = snapshot.mocap_pos
        data.mocap_quat[:] = snapshot.mocap_quat
        self._mujoco.mj_kinematics(self.model, data)

    def object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the object's position and rotation matrix in the private data (after :meth:`load`)."""
        body = self._object_body[name]
        return self.ik.data.xpos[body].copy(), self.ik.data.xmat[body].reshape(3, 3).copy()

    def bucket_pose(self, color: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the bucket's position and rotation matrix in the private data (after :meth:`load`)."""
        body = self._bucket_body[color]
        return self.ik.data.xpos[body].copy(), self.ik.data.xmat[body].reshape(3, 3).copy()

    def carry_held(self, name: str, live: object, q: np.ndarray) -> None:
        """Move object *name* in the private data from the hand in *live* to the hand at *q* (after :meth:`load`).

        The object keeps its pose relative to the hand, so a plan that starts at *q* (where braking
        stops) sees it where the hand will hold it then, not where it is now.
        """
        body, adr = self._object_body[name], self._object_qpos[name]
        hand_rot = live.xmat[self.ik.hand].reshape(3, 3)
        relative_pos = hand_rot.T @ (live.xpos[body] - self.ik.tcp(live))
        relative_rot = hand_rot.T @ live.xmat[body].reshape(3, 3)
        data = self.ik.data
        self.ik.set_arm(q)
        hand_rot = data.xmat[self.ik.hand].reshape(3, 3)
        data.qpos[adr : adr + 3] = self.ik.tcp() + hand_rot @ relative_pos
        quat = np.zeros(4)
        self._mujoco.mju_mat2Quat(quat, (hand_rot @ relative_rot).ravel())
        data.qpos[adr + 3 : adr + 7] = quat
        self._mujoco.mj_kinematics(self.model, data)

    # -- grasp geometry ---------------------------------------------------

    def grasp_axes(self, name: str, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Return the closing axis, the long axis and the width to grasp object *name* across.

        An object whose long side is not level gets its flat-lying axes (and is unlikely to be held).

        Returns:
            Horizontal world unit vectors and the width in metres.
        """
        shape = self.shapes[name]
        axes = shape.grasp_axes(rotation)
        if axes is None:
            axes = shape.grasp_axes(np.eye(3))
        return axes  # type: ignore[return-value]

    def grasp_axis_yaw(self, name: str, rotation: np.ndarray) -> float:
        """Return the world yaw of the closing axis across the object (one of two, pi apart).

        Returns:
            The yaw in radians.
        """
        closing, _, _ = self.grasp_axes(name, rotation)
        return float(np.arctan2(closing[1], closing[0]))

    def grasp_width(self, name: str, rotation: np.ndarray) -> float:
        """Return the width the hand closes on, at most the hand's opening."""
        return min(GRIP_OPEN, self.grasp_axes(name, rotation)[2])

    def grasp_point(self, name: str, position: np.ndarray, rotation: np.ndarray, params: DemoParams) -> np.ndarray:
        """Return the TCP's grasp position: the centre, shifted along the long side, at a safe height.

        Returns:
            The position.
        """
        _, along, _ = self.grasp_axes(name, rotation)
        # Keep the fingers (2 cm wide) on the object.
        limit = self.shapes[name].half_length - 0.012
        point = position + along * float(np.clip(params.grasp_offset, -limit, limit))
        point[2] = max(float(position[2]), MIN_TCP_Z) + params.grasp_dz
        return point

    def predicted_q7(self, yaw: float, xy: np.ndarray) -> float:
        """Estimate joint 7 for the hand at *xy* with closing-axis yaw *yaw* (see :meth:`FrankaIK.continuous_yaw`).

        An upright arm at the origin turns its base toward *xy*. A tilted one (the Duo's) is solved
        for: the hand over *xy* at its home yaw, from home, then joint 7 turns against the rest of the yaw.

        Returns:
            Radians.
        """
        ik = self.ik
        if self.arm.upright:
            q1 = float(np.arctan2(xy[1], xy[0]))
            return float(ik.home[6] + (q1 - ik.home[0]) - (yaw - ik.home_yaw))
        key = (float(xy[0]), float(xy[1]))
        if self._wrist_probe is None or self._wrist_probe[0] != key:
            q, _, _ = ik.solve(ik.home, np.array([*key, WRIST_PROBE_Z]), ik.home_yaw, iters=30)
            self._wrist_probe = (key, float(q[6]), ik.continuous_yaw(q))
        _, q7, solved_yaw = self._wrist_probe
        return float(q7 - (yaw - solved_yaw))

    def yaw_candidates(self, axis_yaw: float, current: float, xy: np.ndarray, *, period: float = np.pi) -> list[float]:
        """Return the closing yaws that line up with *axis_yaw*, nearest to *current* first.

        The hand is symmetric under half a turn, so each axis has a yaw every pi; the candidates are
        the two nearest to *current* whose joint 7 stays inside its range.

        Returns:
            One or two yaws, unwrapped near *current*.
        """
        base = wrap_near(axis_yaw, current, period)
        options = [base + k * period for k in (-2, -1, 0, 1, 2)]
        limit_lo, limit_hi = self.ik.lower[6] + 0.1, self.ik.upper[6] - 0.1
        feasible = [y for y in options if limit_lo <= self.predicted_q7(y, xy) <= limit_hi]
        feasible.sort(key=lambda y: abs(y - current))
        return feasible[:2] if feasible else [base]

    def key_margin(self, q: np.ndarray, instruction: Instruction, params: DemoParams, yaw: float) -> float:
        """Return the joint-limit margin of an attempt's key postures with grasp yaw *yaw* (after :meth:`load`).

        Returns:
            The smallest margin (rad) at the grasp and over the bucket, each solved from the one before.
        """
        name = instruction.object
        position, rotation = self.object_pose(name)
        grasp = self.grasp_point(name, position, rotation, params)
        key, _, _ = self.ik.optimize(q, grasp, yaw, iters=40)
        margin = self.ik.margin(key)
        bucket_pos, bucket_rot = self.bucket_pose(instruction.bucket)
        across = float(np.arctan2(bucket_rot[1, 0], bucket_rot[0, 0]))
        release_yaw = self.yaw_candidates(across, yaw, bucket_pos[:2])[0] if params.reorient else yaw
        over = np.array([bucket_pos[0], bucket_pos[1], params.release_z])
        key, e_pos, _ = self.ik.optimize(key, over, release_yaw, iters=40)
        return min(margin, self.ik.margin(key)) - (1.0 if e_pos > IK_POS_TOL else 0.0)

    # -- building ---------------------------------------------------------

    def attempt_script(
        self,
        start: np.ndarray,
        instruction: Instruction,
        params: DemoParams,
        *,
        grasp_yaw: float,
        slow: dict[str, float] | None = None,
    ) -> tuple[MotionScript, float]:
        """Script one pick-and-place-and-home attempt from *start* (after :meth:`load`).

        Args:
            start: ``(x, y, z, yaw, grip)`` now.
            instruction: What to move where.
            params: Heights, speeds and pauses.
            grasp_yaw: The closing yaw to grasp with (unwrapped).
            slow: Stretch the moves of these phases by these factors.

        Returns:
            The script and the release yaw.
        """
        p = params
        name = instruction.object
        position, rotation = self.object_pose(name)
        grasp = self.grasp_point(name, position, rotation, p)
        self.held_width = self.grasp_width(name, rotation)
        script = MotionScript(start, slow)
        x0 = start[:3]
        open_width = min(GRIP_OPEN, self.held_width + p.open_margin)
        pre = grasp + np.array([0.0, 0.0, p.pregrasp_height])
        # Get up and clear first when starting low and away from the object (a retry).
        if x0[2] < pre[2] - 0.02 and np.linalg.norm(x0[:2] - grasp[:2]) > 0.03:  # noqa: PLR2004
            up = np.array([x0[0], x0[1], max(pre[2], x0[2] + 0.05)])
            script.add(
                "approach",
                pos=up,
                grip=max(start[4], open_width),
                duration=move_duration(up[2] - x0[2], p.lift_speed, 0.5),
            )
        here = script.end[:3]
        script.add(
            "approach",
            pos=pre,
            yaw=grasp_yaw,
            grip=open_width,
            duration=max(
                move_duration(float(np.linalg.norm(pre - here)), p.transit_speed, 0.8),
                move_duration(grasp_yaw - script.end[3], p.yaw_speed, 0.8),
            ),
            blend=p.blend,
        )
        script.add("descend", pos=grasp, duration=move_duration(p.pregrasp_height, p.fine_speed, 0.6), blend=p.blend)
        script.add("close", grip=0.0, duration=p.close_s)
        script.pause("close", p.settle_s)
        lift = grasp + np.array([0.0, 0.0, p.lift_height])
        script.add("lift", pos=lift, duration=move_duration(p.lift_height, p.lift_speed, 0.6, p.held_accel))

        yaw_release = self._add_place(script, instruction, params, grasp_yaw, (position - grasp)[:2])
        self.add_return(script, p)
        return script, yaw_release

    def place_script(
        self,
        start: np.ndarray,
        instruction: Instruction,
        params: DemoParams,
        offset: np.ndarray,
        slow: dict[str, float] | None = None,
    ) -> tuple[MotionScript, float]:
        """Script the rest of an attempt from *start* with the object in the hand (after :meth:`load`).

        Args:
            start: ``(x, y, z, yaw, grip)`` now.
            instruction: What to move where.
            params: Heights, speeds and pauses.
            offset: The held object's centre from the TCP, horizontally.
            slow: Stretch the moves of these phases by these factors.

        Returns:
            The script and the release yaw.
        """
        script = MotionScript(start, slow)
        if start[2] < params.carry_z - REPLAN_SHIFT:  # re-planned on the way down into the bucket: rise first
            rise = params.carry_z - float(start[2])
            script.add(
                "carry",
                pos=np.array([start[0], start[1], params.carry_z]),
                duration=move_duration(rise, params.lift_speed, 0.5),
            )
        yaw_release = self._add_place(script, instruction, params, float(start[3]), np.asarray(offset)[:2])
        self.add_return(script, params)
        return script, yaw_release

    def _add_place(  # noqa: PLR0914
        self, script: MotionScript, instruction: Instruction, p: DemoParams, yaw: float, offset: np.ndarray
    ) -> float:
        """Add the carry over the bucket, the descent into its opening and the release.

        The object's centre, not the TCP, goes over the release point; *offset* (the centre from the
        TCP) turns with the hand.

        Returns:
            The release yaw.
        """
        bucket_pos, bucket_rot = self.bucket_pose(instruction.bucket)
        (x_lo, x_hi), (y_lo, y_hi), _ = BUCKET_INTERIOR
        local = np.array([(x_lo + x_hi) / 2 + p.place_offset[0], (y_lo + y_hi) / 2 + p.place_offset[1], 0.0])
        place = bucket_pos + bucket_rot @ local
        across = float(np.arctan2(bucket_rot[1, 0], bucket_rot[0, 0]))
        yaw_release = self.yaw_candidates(across + p.release_yaw_noise, yaw, place[:2])[0] if p.reorient else yaw
        turn = yaw_release - yaw
        c, s = np.cos(turn), np.sin(turn)
        offset = np.array([c * offset[0] - s * offset[1], s * offset[0] + c * offset[1]])
        over = np.array([place[0] - offset[0], place[1] - offset[1], p.carry_z])
        script.add(
            "carry",
            pos=over,
            yaw=yaw_release,
            duration=max(
                move_duration(float(np.linalg.norm(over - script.end[:3])), p.transit_speed, 0.8, p.held_accel),
                move_duration(turn, p.yaw_speed, 0.8),
            ),
            blend=p.blend,
        )
        release = np.array([over[0], over[1], p.release_z])
        script.add(
            "lower",
            pos=release,
            duration=move_duration(p.carry_z - p.release_z, p.fine_speed, 0.5, p.held_accel),
            blend=p.blend,
        )
        script.pause("release", p.release_pause_s)
        script.add("release", grip=min(GRIP_OPEN, self.held_width + p.release_margin), duration=0.45)
        script.pause("release", p.after_release_s)
        return yaw_release

    @staticmethod
    def add_return(script: MotionScript, p: DemoParams) -> None:
        """Add the retreat straight up; :meth:`solve` adds the joint-space way home after it."""
        here = script.end[:3]
        retreat_z = max(here[2] + 0.06, 0.22)
        script.add(
            "retreat",
            pos=np.array([here[0], here[1], retreat_z]),
            duration=move_duration(retreat_z - here[2], p.lift_speed, 0.5),
        )

    def checked(
        self,
        build: Callable[[dict[str, float]], tuple[MotionScript, float]],
        q: np.ndarray,
        dt: float,
        *,
        kind: PlanKind,
        target: str,
        bucket: str,
        held_from: float | None = None,
    ) -> JointPlan:
        """Build a script, solve it, and stretch the moves that are too fast until the joints keep to the limits.

        Args:
            build: Returns the script and its release yaw for a ``slow`` map (phase to stretch factor).
            q: Joint angles at the start.
            dt: Control period.
            kind: What the plan is for.
            target: The object being picked or held.
            bucket: The bucket it goes to; the plan records its pose.
            held_from: When the hand holds the object; ``None`` from the first ``lift`` move.

        Returns:
            The last plan solved (``problems`` says whether it passed).
        """
        slow: dict[str, float] = {}
        plan: JointPlan | None = None
        bucket_pos, bucket_rot = self.bucket_pose(bucket)
        for _ in range(4):
            script, yaw_release = build(slow)
            start = script.first_time("lift") + 0.1 if held_from is None else held_from
            release = next((m.t0 for m in script.moves if m.phase == "release"), script.t_end)
            plan = self.solve(script, q, dt, kind=kind, target=target, held_from=start, held_until=release)
            plan.yaw_release = yaw_release
            velocity = np.diff(plan.q, axis=0) / dt
            speed = (np.abs(velocity) / self.dq_max).max(axis=1)
            accel = (np.abs(np.diff(velocity, axis=0)) / (dt * self.ddq_max)).max(axis=1)
            factors: dict[str, float] = {}
            for k in np.flatnonzero(speed > 0.9 * SPEED_FRACTION):
                ratio = float(speed[k] / (0.85 * SPEED_FRACTION))
                factors[plan.phases[k + 1]] = max(factors.get(plan.phases[k + 1], 1.0), ratio)
            for k in np.flatnonzero(accel > 0.9 * ACCEL_FRACTION):
                ratio = float(np.sqrt(accel[k] / (0.85 * ACCEL_FRACTION)))
                factors[plan.phases[k + 1]] = max(factors.get(plan.phases[k + 1], 1.0), ratio)
            factors.pop("return", None)  # the way home is timed from its own speed limit
            too_fast = plan.peak_speed_ratio > SPEED_FRACTION or plan.peak_accel_ratio > ACCEL_FRACTION
            if not too_fast or not factors:
                break
            for phase, factor in factors.items():
                slow[phase] = min(MAX_STRETCH, slow.get(phase, 1.0) * factor)
        assert plan is not None  # noqa: S101 - the loop runs at least once
        plan.bucket_pose = (bucket_pos, float(np.arctan2(bucket_rot[1, 0], bucket_rot[0, 0])))
        self._annotate(plan)
        return plan

    @staticmethod
    def _annotate(plan: JointPlan) -> None:
        """Mark the ticks where the demonstrator checks the grasp, the hold and the bucket."""
        lift_ticks = [k for k, phase in enumerate(plan.phases) if phase == "lift"]
        if lift_ticks:
            z0 = plan.cartesian[lift_ticks[0], 2]
            above = [k for k in lift_ticks if plan.cartesian[k, 2] >= z0 + LIFT_CHECK_DZ]
            plan.lift_check = above[0] if above else lift_ticks[-1]
        release_ticks = [k for k, phase in enumerate(plan.phases) if phase == "release"]
        plan.hold_until = release_ticks[0] if release_ticks else None
        carry_ticks = [k for k, phase in enumerate(plan.phases) if phase == "carry"]
        plan.carry_start = carry_ticks[0] if carry_ticks else None
        if plan.kind == "place":
            plan.lift_check = None

    def solve(  # noqa: PLR0914
        self,
        script: MotionScript,
        q_start: np.ndarray,
        dt: float,
        *,
        kind: PlanKind,
        target: str,
        held_from: float | None = None,
        held_until: float | None = None,
        home_speed: float = 0.3,
        home_blend: float = 0.3,
    ) -> JointPlan:
        """Turn *script* into joint targets every *dt*, then the way home, and check them (after :meth:`load`).

        Args:
            script: The motion.
            q_start: Joint angles at the start (the IK's first seed).
            dt: Control period.
            kind: What the plan is for.
            target: The object being picked (and held from *held_from* to *held_until*, in seconds).
            held_from: When the hand starts carrying the object.
            held_until: When it lets go.
            home_speed: After the script, the arm returns home (:attr:`FrankaIK.home`) in joint space, each
                joint at most this fraction of its speed and acceleration limits.
            home_blend: Fraction of the script's last move that the way home overlaps.

        Returns:
            The plan; ``problems`` lists every check that failed.
        """
        q_start = np.asarray(q_start, dtype=np.float64).copy()
        q_script, pos_error, rot_error = self._follow_script(script, q_start, dt)
        n_script = len(q_script)
        cartesian = script.at(np.arange(n_script) * dt)
        phases = script.phases(np.arange(n_script) * dt)
        # The way home, in joint space: a minimum-jerk move of every joint onto the ready pose,
        # starting while the last move of the script ends.
        delta = self.ik.home - q_script[-1]
        home_s = max(
            1.2,
            1.875 * float(np.max(np.abs(delta) / (home_speed * self.dq_max))),
            float(np.sqrt(5.77 * np.max(np.abs(delta) / (home_speed * self.ddq_max)))),
        )
        last = script.moves[-1] if script.moves else None
        start = script.t_end - (home_blend * last.duration if last is not None and np.any(last.delta) else 0.0)
        ticks = max(n_script, int(np.ceil((start + home_s) / dt)) + 1)
        ramp = min_jerk((np.arange(ticks) * dt - start) / home_s)
        extra = ticks - n_script
        q = np.vstack([q_script, np.repeat(q_script[-1:], extra, axis=0)]) + ramp[:, None] * delta
        cartesian = np.vstack([cartesian, np.repeat(cartesian[-1:], extra, axis=0)])
        grip = cartesian[:, 4] + ramp * (GRIP_OPEN - cartesian[-1, 4])
        home_tick = int(np.ceil(start / dt))
        all_phases: list[Phase] = [*phases, *([phases[-1]] * extra)]
        returning: list[Phase] = ["return"] * (ticks - home_tick)
        all_phases = [*all_phases[:home_tick], *returning]
        plan = JointPlan(
            kind=kind,
            dt=dt,
            q=q,
            grip=grip,
            phases=all_phases,
            cartesian=cartesian,
            pos_error=pos_error,
            rot_error=rot_error,
        )
        self._check_joints(plan)
        held = None
        if held_from is not None and held_until is not None:
            held = (int(np.ceil(held_from / dt)), int(np.floor(held_until / dt)))
        self._check_clearance(plan, target, held)
        return plan

    def _follow_script(  # noqa: PLR0914 - one numerical routine; splitting it hides the algorithm
        self, script: MotionScript, q_start: np.ndarray, dt: float
    ) -> tuple[np.ndarray, float, float]:
        """Solve the IK every tick along *script*, following its reference posture in the null space.

        Key postures come first: at the end of each move, the IK solution furthest from the joint
        limits on the previous key's side of the null space. The reference posture moves between
        them like the hand does. The targets are then smoothed, and any first-tick correction fades in.

        Returns:
            ``(targets (ticks, 7), largest position error, largest orientation error)``.
        """
        ik, data = self.ik, self.ik.data
        key = q_start
        for move in script.moves:
            if not np.any(move.delta[:4]):
                move.q_delta = None
                continue
            new_key, _, _ = ik.optimize(key, move.target[:3], float(move.target[3]))
            move.q_delta = new_key - key
            key = new_key
        n_script = max(2, int(np.ceil(script.t_end / dt)) + 1)
        times = np.arange(n_script) * dt
        cartesian = script.at(times)
        reference = script.reference(times, q_start)
        q_script = np.zeros((n_script, 7))
        current = q_start.copy()
        pos_error = rot_error = 0.0
        step_limit = MAX_TICK_SPEED * self.dq_max * dt
        for k in range(n_script):
            solved, e_pos, e_rot = ik.solve(
                current,
                cartesian[k, :3],
                float(cartesian[k, 3]),
                iters=self.tick_iters if k else 30,
                posture=reference[k] if self.tick_null_step > 0.0 else None,
                null_step=self.tick_null_step,
                limit_gain=self.tick_limit_gain,
            )
            # Never faster than MAX_TICK_SPEED of the limits, even where the IK would jump.
            step = np.clip(solved - current, -step_limit, step_limit)
            if np.any(step != solved - current):
                ik.set_arm(current + step)
                e_pos = float(np.linalg.norm(cartesian[k, :3] - ik.tcp()))
            current += step
            q_script[k] = current
            pos_error, rot_error = max(pos_error, e_pos), max(rot_error, e_rot)
        if self.smooth_ticks > 0.0:
            # Round off the kinks a joint limit or the null space leaves.
            q_script = gaussian_smooth(q_script, self.smooth_ticks)
        # The first IK tick snaps a hand that is not quite level (after braking, say) onto the task;
        # fade that correction in over START_BLEND_S instead, so the targets start where the arm is.
        fade = 1.0 - min_jerk(times / START_BLEND_S)
        q_script -= fade[:, None] * (q_script[0] - q_start)[None, :]
        # Measure the task again on the targets as they will run (after the fade-in).
        for k in range(int(np.ceil(START_BLEND_S / dt)), n_script, CHECK_EVERY):
            ik.set_arm(q_script[k])
            rot = data.xmat[ik.hand].reshape(3, 3) @ hand_rotation(float(cartesian[k, 3])).T
            pos_error = max(pos_error, float(np.linalg.norm(cartesian[k, :3] - ik.tcp())))
            rot_error = max(rot_error, float(np.arccos(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0))))
        return q_script, pos_error, rot_error

    def _check_joints(self, plan: JointPlan) -> None:
        """Record the IK residual, the joint-limit margin and the peak joint speed and acceleration."""
        ik, q = self.ik, plan.q
        if plan.pos_error > IK_POS_TOL or plan.rot_error > IK_ROT_TOL:
            plan.problems.append(f"IK residual {plan.pos_error * 1000:.1f} mm, {np.degrees(plan.rot_error):.1f} deg")
        plan.limit_margin = float(min((q - ik.hard_lower).min(), (ik.hard_upper - q).min()))
        smooth = joint_smoothness(q, plan.dt)
        plan.peak_speed_ratio = float((smooth["velocity"] / self.dq_max).max())
        plan.peak_accel_ratio = float((smooth["acceleration"] / self.ddq_max).max())
        if plan.peak_speed_ratio > SPEED_FRACTION:
            plan.problems.append(f"joint speed {plan.peak_speed_ratio:.0%} of the {self.arm.label} limit")
        if plan.peak_accel_ratio > ACCEL_FRACTION:
            plan.problems.append(f"joint acceleration {plan.peak_accel_ratio:.0%} of the {self.arm.label} limit")

    def arm_clearance(self, q: np.ndarray, snapshot: SceneSnapshot) -> float:
        """Return how close the working arm comes to the arms holding still, at each row of *q* (capped at 5 cm).

        Every row is checked: braking is short. The arms holding still stand as in *snapshot*.

        Returns:
            Metres; infinite without other arms.
        """
        clearance = self.clearance
        if clearance.arm is None or clearance.other_arms is None:
            return float(np.inf)
        self.load(snapshot)
        worst = float(np.inf)
        for row in np.atleast_2d(q):
            self.ik.set_arm(row)
            worst = min(worst, clearance._min_distance(clearance.arm, clearance.other_arms))  # noqa: SLF001
        return worst

    def _check_clearance(self, plan: JointPlan, target: str, held: tuple[int, int] | None) -> None:
        """Sweep the hand (and, between the *held* ticks, the object in it) and record the clearances."""
        ik, data = self.ik, self.ik.data
        adr = self._object_qpos[target]
        position, rotation = self.object_pose(target)
        relative: tuple[np.ndarray, np.ndarray] | None = None
        rest_qpos = np.array(data.qpos[adr : adr + 7])
        worst: dict[str, float] = dict.fromkeys(CLEARANCE, float(np.inf))
        quat = np.zeros(4)
        for k in range(0, len(plan.q), CHECK_EVERY):
            holding = held is not None and held[0] <= k < held[1]
            ik.set_arm(plan.q[k], self.held_width if holding else float(plan.grip[k]))
            if holding:
                hand_pos, hand_rot = ik.tcp(), data.xmat[ik.hand].reshape(3, 3).copy()
                if relative is None:
                    # As grasped: the object sits where it lies now, relative to the closed hand.
                    relative = (hand_rot.T @ (position - hand_pos), hand_rot.T @ rotation)
                data.qpos[adr : adr + 3] = hand_pos + hand_rot @ relative[0]
                self._mujoco.mju_mat2Quat(quat, (hand_rot @ relative[1]).ravel())
                data.qpos[adr + 3 : adr + 7] = quat
            else:
                data.qpos[adr : adr + 7] = rest_qpos
            self._mujoco.mj_kinematics(self.model, data)
            for key, value in self.clearance.measure(target=target, held=holding).items():
                worst[key] = min(worst[key], value)
        data.qpos[adr : adr + 7] = rest_qpos
        plan.clearance = {key: value for key, value in worst.items() if np.isfinite(value)}
        plan.problems += [
            f"{key} clearance {value * 1000:.1f} mm" for key, value in plan.clearance.items() if value < CLEARANCE[key]
        ]


# ----------------------------------------------------------------------
# The demonstrator
# ----------------------------------------------------------------------


def _severity(plan: JointPlan) -> float:
    """Rank failing plans by how far past the budgets they go; IK error and clearance weigh most.

    Returns:
        A score; lower is better.
    """
    return (
        max(0.0, plan.pos_error - IK_POS_TOL) * 1000.0
        + max(0.0, plan.peak_speed_ratio - SPEED_FRACTION)
        + max(0.0, plan.peak_accel_ratio - ACCEL_FRACTION)
        + sum(max(0.0, CLEARANCE[key] - value) * 1000.0 for key, value in plan.clearance.items())
    )


@dataclass(frozen=True)
class StuckStart:
    """The move to a stuck start: targets for every control tick, model units in the channels' order."""

    targets: np.ndarray
    """``(ticks, channels)``: the working arm goes there, the others hold their targets."""
    arm: int
    """The arm that moves, and works the episode's first attempt."""


def smoothstep(s: np.ndarray) -> np.ndarray:
    """Return ``3 s^2 - 2 s^3`` of *s* clipped to [0, 1]: from rest to rest.

    Returns:
        The profile.
    """
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


@dataclass
class DemoStats:
    """Counters over the demonstrator's lifetime."""

    episodes: int = 0
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    retries: int = 0
    replans: int = 0
    plan_problems: int = 0
    """Plans that ran although no candidate passed every check."""


@dataclass
class _Episode:
    serial: int
    instruction: Instruction | None
    rng: np.random.Generator
    params: DemoParams
    start_time: float
    attempt: int = 0
    outcome: Outcome = "running"
    failure: str | None = None
    replans: int = 0
    reasons: list[str] = field(default_factory=list)
    """Why each attempt ended early, in order."""
    retry_pending: bool = False
    """Back home after a failed attempt, the next one starts once the object rests."""
    reset_requested: bool = False


class BucketSortDemonstrator:
    """Put the instructed object in the instructed bucket, one control tick at a time.

    With several arms (the Duo's), one works an attempt while the others hold their targets: each
    attempt goes to the arm that reaches both the object and the bucket, the nearer one first, and
    its plans keep clear of the arms that hold still.
    """

    def __init__(
        self,
        model: object,
        task: BucketSortTask,
        *,
        arms: tuple[FrankaArm, ...] = (FR3_ARM,),
        rng: np.random.Generator | None = None,
        randomize: bool = True,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        reset_on_failure: bool = True,
    ) -> None:
        """Bind to the scene model and its task.

        Args:
            model: The bucket_sort model.
            task: The scene's task (``Sim.on_reset``): the instruction and the success test.
            arms: The robot's arms in channel order (:data:`~.bucket_sort_arms.BUCKET_SORT_ARMS`);
                the default is the ``franka_fr3`` profile's FR3.
            rng: Draws each episode's own generator; the same seed repeats the same episodes.
            randomize: Draw each attempt's parameters (:meth:`DemoParams.sample`); otherwise the defaults.
            max_attempts: Attempts per episode, the retry included.
            reset_on_failure: Ask for a scene reset after a failed episode (:meth:`take_reset_request`).
        """
        from physicalai_mujoco_plugin.profiles.derive import control_kind  # noqa: PLC0415

        self.model = model
        self.task = task
        self.arms = tuple(arms)
        self.planners = tuple(Planner(model, arm, tuple(o for o in self.arms if o is not arm)) for arm in self.arms)
        self._active = 0
        self.stats = DemoStats()
        self.randomize = randomize
        self.max_attempts = int(max_attempts)
        self.reset_on_failure = reset_on_failure
        self._rng = rng if rng is not None else np.random.default_rng()
        self._actuators = np.array([
            int(model.actuator(arm.name(name)).id) for arm in self.arms for name in (*arm.actuators, arm.gripper)
        ])
        """Every arm's actuators, in output order: seven joints, then the hand, arm by arm."""
        # ctrl per model-unit target: 1 for the FR3, 6375 for the Panda's finger tendon (255 per 0.04 m).
        self._ctrl_per_unit = np.array([
            control_kind(model, int(a))[1] * float(model.actuator_gear[a, 0]) for a in self._actuators
        ])
        self._grip_per_width = np.array([arm.grip_per_width for arm in self.arms])
        self._hold: list[np.ndarray | None] = [None] * len(self.arms)
        self._still_since: float | None = None
        self._pinned_arm: int | None = None
        """The arm a stuck start moved, which works the episode's first attempt (several arms)."""
        """Since when every arm stands still on its targets (several arms)."""
        """The targets each arm that does not work holds, model units (seven joints and the opening in metres)."""
        self._object_body = {name: int(model.body(name).id) for name in OBJECTS}
        self._bucket_body = {color: int(model.body(f"bucket_{color}").id) for color in BUCKET_COLORS}
        self._q: np.ndarray | None = None
        self._q_prev: np.ndarray | None = None
        self._braking_from = self.ik.home.copy()
        self._grip = GRIP_OPEN
        self._generation: int | None = None
        self._episode: _Episode | None = None
        self._plan: JointPlan | None = None
        self._tick = 0
        self._phase: Phase = "idle"
        self._since = 0.0
        self._plan_end: float | None = None
        self._lag_since: float | None = None
        self._lift_check_since: float | None = None
        self._rest_z = 0.0
        self._planned_object: np.ndarray | None = None
        self._planned_axis = 0.0
        self._reset_requested = False
        self.plans: list[JointPlan] = []
        """Every plan of the current episode, for diagnostics."""

    # -- readouts ---------------------------------------------------------

    @property
    def planner(self) -> Planner:
        """The planner of the arm that works now."""
        return self.planners[self._active]

    @property
    def ik(self) -> FrankaIK:
        """The IK of the arm that works now."""
        return self.planners[self._active].ik

    @property
    def active_arm(self) -> FrankaArm:
        """The arm that works now; the others hold still."""
        return self.arms[self._active]

    @property
    def phase(self) -> Phase:
        """What the demonstrator is doing."""
        return self._phase

    @property
    def target(self) -> str | None:
        """The object of the current instruction, if any."""
        episode = self._episode
        return episode.instruction.object if episode is not None and episode.instruction is not None else None

    @property
    def last_plan(self) -> JointPlan | None:
        """The plan that ran last."""
        return self.plans[-1] if self.plans else None

    def status(self) -> dict[str, object]:
        """Return the episode status for ``/health``: instruction, attempt, outcome and why it failed.

        ``outcome`` is ``running``, ``success``, ``failed`` (a recorder discards the episode) or
        ``stalled`` (the arm does not follow the published targets). ``clean`` is false once an
        attempt went wrong (a miss, a slip, a drop), even when the episode still succeeds, for
        example by a dropped object landing in the bucket; a recorder that keeps only smooth
        demonstrations keeps clean successes.
        """
        episode = self._episode
        return {
            "clean": not episode.reasons if episode is not None else None,
            "instruction": episode.instruction.text if episode is not None and episode.instruction else None,
            "episode": episode.serial if episode is not None else None,
            "attempt": episode.attempt if episode is not None else 0,
            "max_attempts": self.max_attempts,
            "outcome": episode.outcome if episode is not None else None,
            "failure": episode.failure if episode is not None else None,
        }

    def take_reset_request(self) -> bool:
        """Return whether a failed episode wants a scene reset now, once.

        Returns:
            ``True`` once per failed episode.
        """
        requested, self._reset_requested = self._reset_requested, False
        return requested

    def reset(self) -> None:
        """Forget the episode in progress; the next :meth:`step` starts a new one from the arm's pose."""
        self._q = None
        self._q_prev = None
        self._plan = None
        self._episode = None
        self._generation = None
        self._phase = "idle"
        self._reset_requested = False
        self._pinned_arm = None

    # -- control ----------------------------------------------------------

    def step(self, data: object, dt: float, *, apply: bool = True) -> np.ndarray:
        """Advance by one control tick of *dt* seconds.

        With *apply*, the targets are written to the actuators; otherwise they are only returned.

        Returns:
            Eight targets per arm in model units, the channels' order (``ACTUATORS`` for the FR3): the
            seven arm joints (radians) and the hand's actuator (the FR3's opening in metres).
        """
        new_episode = self._episode is None or self._generation != self.task.reset_count
        if self._q is None or new_episode:
            self._take_targets(data)
        if new_episode:
            self._start_episode(data)
        self._q_prev = self._q.copy()
        self._advance(data, dt)
        targets = self._targets(self._q, self._grip)  # type: ignore[arg-type]
        if apply:
            self.write_targets(data, targets)
        return targets

    def _take_targets(self, data: object) -> None:
        """Start from the arms as they are: the working arm's joints, every other arm's targets."""
        # A scene reset puts the arm home: start from there, not from the last episode's targets.
        self._q = np.array(data.qpos[self.ik.qpos_adr])
        self._grip = self._ctrl_targets(data, self._active)[7]
        self._hold = [None if i == self._active else self._ctrl_targets(data, i) for i in range(len(self.arms))]

    def _targets(self, q: np.ndarray, grip: float) -> np.ndarray:
        """Return every channel's target, model units: the working arm at *q* and *grip* (m), the others holding.

        Returns:
            Eight values per arm.
        """
        targets = np.concatenate([
            [*q, grip] if i == self._active else self._hold[i]  # type: ignore[list-item]
            for i in range(len(self.arms))
        ])
        targets[7::8] *= self._grip_per_width
        return targets

    def write_targets(self, data: object, targets: np.ndarray) -> None:
        """Write targets in model units (:meth:`step`'s) to the actuators."""
        data.ctrl[self._actuators] = targets * self._ctrl_per_unit

    def stuck_start(self, data: object, dt: float, rng: np.random.Generator) -> StuckStart | None:
        """Plan the move to a pose a policy gets stuck in, for the episode about to start (after a reset).

        The pose comes from the attempt the demonstrator plans for this episode's instruction and
        layout: a tick of its carry, its descent into the bucket or its release, where the hand would
        hold the object. The hand gets there closed or almost (:data:`STUCK_GRIP`) and empty, by a
        smoothstep move in joint space of :data:`STUCK_MOVE_S`. The planned attempt passed every check
        (reach, joint limits and speed, clearance, and on the Duo the other arm's); the move is checked
        for the hand's clearance from the buckets, the other objects, the table and the other arm.
        With several arms, the planned arm moves and works the first attempt; the others hold.

        Returns:
            The move, or ``None`` when no instruction, no plan that passes every check, or no clear
            move was found; the demonstrator then starts as usual.
        """
        instruction = self.task.instruction
        if instruction is None:
            return None
        self.reset()
        self._take_targets(data)
        snapshot = SceneSnapshot.take(data)
        params = DemoParams.sample(rng) if self.randomize else DemoParams()
        if len(self.arms) > 1:
            plan = self._plan_with_an_arm(snapshot, instruction, params, dt, rng)
        else:
            plan = self.plan_attempt(snapshot, self._q, self._start_state(), instruction, params, dt, rng, report=False)  # type: ignore[arg-type]
        ticks = (
            [] if plan is None or not plan.ok else [k for k, phase in enumerate(plan.phases) if phase in STUCK_PHASES]
        )
        move = self._stuck_move(snapshot, plan, ticks, dt, rng) if plan is not None and ticks else None
        self.reset()  # the episode starts from the stuck pose, at the next step
        if move is None:
            return None
        self._pinned_arm = self._active if len(self.arms) > 1 else None
        logger.info(
            "Demonstrator: a stuck start for the {}, {:.1f} s away", self.active_arm.label, len(move.targets) * dt
        )
        return move

    def _stuck_move(
        self, snapshot: SceneSnapshot, plan: JointPlan, ticks: list[int], dt: float, rng: np.random.Generator
    ) -> StuckStart | None:
        """Return a clear smoothstep move from the current targets to a stuck pose of *plan*, if one is found.

        Returns:
            The move, or ``None`` after :data:`STUCK_TRIES` poses whose move is not clear.
        """
        planner, instruction = self.planner, self.task.instruction
        start, grip0 = np.asarray(self._q, dtype=np.float64), float(self._grip)
        for _ in range(STUCK_TRIES):
            goal = plan.q[int(rng.choice(ticks))]
            grip = float(rng.uniform(*STUCK_GRIP))
            count = max(2, round(float(rng.uniform(*STUCK_MOVE_S)) / dt))
            s = smoothstep(np.arange(1, count + 1) / count)
            q = start[None, :] + s[:, None] * (goal - start)[None, :]
            opening = grip0 + s * (grip - grip0)
            move = JointPlan(
                kind="home", dt=dt, q=q, grip=opening, phases=["return"] * count, cartesian=np.zeros((count, 5))
            )
            planner.load(snapshot)
            planner._check_clearance(move, instruction.object, None)  # noqa: SLF001  # type: ignore[union-attr]
            if move.problems:
                continue
            targets = np.stack(list(starmap(self._targets, zip(q, opening, strict=True))))
            return StuckStart(targets=targets, arm=self._active)
        return None

    def _ctrl_targets(self, data: object, arm: int) -> np.ndarray:
        """Return arm *arm*'s current targets from ``ctrl``: seven joints and the opening in metres.

        Returns:
            Eight values.
        """
        rows = slice(8 * arm, 8 * arm + 8)
        targets = np.array(data.ctrl[self._actuators[rows]]) / self._ctrl_per_unit[rows]
        targets[7] = float(np.clip(targets[7] / self._grip_per_width[arm], 0.0, GRIP_OPEN))
        return targets

    def _switch_arm(self, arm: int) -> None:
        """Let arm *arm* work from its held targets; the one that worked holds its own."""
        hold = self._hold[arm]
        if arm == self._active or hold is None or self._q is None:  # only the working arm has no hold
            return
        self._hold[self._active] = np.array([*self._q, self._grip])
        self._hold[arm] = None
        self._active = arm
        self._q, self._grip = hold[:7].copy(), float(hold[7])
        self._q_prev = self._q.copy()
        self._braking_from = self._q.copy()

    def _start_episode(self, data: object) -> None:
        self._generation = self.task.reset_count
        seed = int(self._rng.integers(2**63))
        rng = np.random.default_rng(seed)
        params = DemoParams.sample(rng) if self.randomize else DemoParams()
        self.stats.episodes += 1
        self._episode = _Episode(self.stats.episodes, self.task.instruction, rng, params, float(data.time))
        self._plan = None
        self.plans = []
        self._set_phase("settle" if self.task.instruction is not None else "idle", data)

    def _set_phase(self, phase: Phase, data: object) -> None:
        self._phase = phase
        self._since = float(data.time)
        if phase == "settle":
            self._still_since = None  # the arms must stand still anew before this attempt

    def _advance(self, data: object, dt: float) -> None:
        episode = self._episode
        now = float(data.time)
        if episode is None or episode.instruction is None or self._phase in {"idle", "stalled"}:
            return
        if self._phase == "settle":
            self._settle(data, dt, now)
            return
        settling = episode.outcome == "running" and self._phase in {"retreat", "return", "done"}
        if settling and self.task.evaluate(self.model, data).success:
            episode.outcome = "success"
            self.stats.successes += 1
        plan = self._plan
        if plan is not None:
            if self._tick < len(plan):
                self._follow(data, plan, dt)
            elif self._arrived(data, plan, now):
                self._plan = None
                if episode.retry_pending and episode.outcome == "running":
                    episode.retry_pending = False
                    self._set_phase("settle", data)
                else:
                    self._set_phase("failed" if episode.outcome == "failed" else "done", data)
            return
        if self._phase == "done" and episode.outcome == "running" and now - self._since >= SETTLE_CHECK_S:
            self._judge(data, dt)
        elif (
            self._phase == "failed"
            and self.reset_on_failure
            and not episode.reset_requested
            and now - self._since >= FAILURE_HOLD_S
        ):
            episode.reset_requested = self._reset_requested = True

    def _settle(self, data: object, dt: float, now: float) -> None:
        """Start the attempt once the pause is over and the object rests; a retry also needs it graspable."""
        episode = self._episode
        still = self._arms_still(data, now)
        if now - self._since >= episode.params.start_pause_s and self._resting(data) and still:  # type: ignore[union-attr]
            if episode.attempt == 0:  # type: ignore[union-attr]
                self._begin_attempt(data, dt)
            elif self._graspable(data):
                self._begin_attempt(data, dt, new_attempt=True)
            else:
                self._fail(data, f"{episode.reasons[-1]}, and it cannot be picked up where it lies", dt)  # type: ignore[union-attr]
        elif now - self._since > SETTLE_TIMEOUT_S and not still:
            # No way home either: it would be checked against an arm that is not where it will be.
            self._stop_and_fail(data, dt, None, "the arms do not come to rest", freeze_at=self._q)
        elif now - self._since > SETTLE_TIMEOUT_S:
            self._fail(data, "the object does not come to rest", dt)

    def _arms_still(self, data: object, now: float) -> bool:
        """Return whether every arm has stood still on its targets for :data:`STILL_S` (several arms).

        A plan is checked against the arms that hold still where they stand when it is planned. An arm
        still moving (toward a target a client sent before the autopilot took over, say) would stand
        elsewhere: so every arm's actuator targets must be the ones the demonstrator gives it (the
        working arm's, the holds), and every joint slower than :data:`STILL_SPEED`. A position servo
        with a fixed target that has come to rest stays there, so the snapshot holds for the plan.

        Returns:
            ``True`` with one arm, which has no other arm to keep clear of.
        """
        if len(self.arms) == 1:
            return True
        for index, planner in enumerate(self.planners):
            target = self._q if index == self._active else self._hold[index][:7]  # type: ignore[index]
            targets = (
                np.array(data.ctrl[self._actuators[8 * index : 8 * index + 7]])
                / self._ctrl_per_unit[8 * index : 8 * index + 7]
            )
            moving = np.abs(np.array(data.qvel[planner.ik.dof_adr])).max() >= STILL_SPEED
            if moving or np.abs(targets - target).max() > TARGET_TOLERANCE:
                self._still_since = None
                return False
        if self._still_since is None:
            self._still_since = now
        return now - self._still_since >= STILL_S

    def _follow(self, data: object, plan: JointPlan, dt: float) -> None:
        """Output the plan's next targets, and check that the attempt still goes as planned."""
        episode = self._episode
        tick = self._tick
        if self._lagging(data, plan, tick):
            return
        targets = plan.targets(tick)
        self._q, self._grip = targets[:7], float(targets[7])
        self._phase = plan.phases[tick]
        self._tick += 1
        if plan.kind == "home" or episode is None or episode.instruction is None:
            return
        name = episode.instruction.object
        if self._phase in {"approach", "descend"} and self._planned_object is not None:
            body = self._object_body[name]
            moved = np.linalg.norm(data.xpos[body][:2] - self._planned_object[:2])
            axis = self.planner.grasp_axis_yaw(name, data.xmat[body].reshape(3, 3))
            # The hand closes the same way turned by half a turn.
            turned = abs(wrap_near(axis, self._planned_axis, np.pi) - self._planned_axis)
            if moved > REPLAN_SHIFT or turned > REPLAN_TURN:
                self._replan(data, dt, f"the {name} moved", attempt=True)
            return
        if plan.lift_check is not None and tick == plan.lift_check:
            self._lift_check_since = float(data.time)
        if self._lift_check_since is not None:
            # The arm lags its targets (the servos, a client's easing): judge the lift once the
            # hand itself is up, or after LIFT_CHECK_WAIT_S at the latest.
            up = float(self.ik.tcp(data)[2]) >= plan.cartesian[plan.lift_check or 0, 2] - 0.005
            if not up and float(data.time) - self._lift_check_since < LIFT_CHECK_WAIT_S:
                return
            self._lift_check_since = None
            if not self._holding(data, name, lifted=True):
                self._retry_or_fail(data, "the grasp did not lift the object", dt)
            return
        if plan.hold_until is not None and (plan.lift_check or 0) < tick < plan.hold_until:
            if not self._holding(data, name, lifted=False):
                self._retry_or_fail(data, "the object slipped out of the hand", dt)
                return
            if plan.carry_start is not None and tick >= plan.carry_start and self._bucket_moved(data, plan):
                self._replan(data, dt, f"the {episode.instruction.bucket} bucket moved", attempt=False)

    def _lagging(self, data: object, plan: JointPlan, tick: int) -> bool:
        """Stall when the arm stays far from the targets: in ``leader`` mode, nobody sends them back.

        Returns:
            Whether the demonstrator just stalled.
        """
        error = float(np.abs(np.array(data.qpos[self.ik.qpos_adr]) - plan.q[max(tick - 1, 0)]).max())
        now = float(data.time)
        if error <= STALL_ERROR:
            self._lag_since = None
            return False
        if self._lag_since is None:
            self._lag_since = now
        if now - self._lag_since < CATCH_UP_S:
            return False
        self._stall(error)
        return True

    def _stall(self, error: float) -> None:
        episode = self._episode
        if episode is not None:
            episode.outcome, episode.failure = "stalled", "the arm does not follow the targets"
        logger.warning("Demonstrator: the arm does not follow its targets (error {:.2f} rad); stopping", error)
        self._plan, self._plan_end, self._lag_since = None, None, None
        self._phase = "stalled"

    def _bucket_moved(self, data: object, plan: JointPlan) -> bool:
        if plan.bucket_pose is None:
            return False
        color = self._episode.instruction.bucket  # type: ignore[union-attr]
        body = self._bucket_body[color]
        rotation = data.xmat[body].reshape(3, 3)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        position, planned_yaw = plan.bucket_pose
        turned = abs(wrap_near(yaw - planned_yaw, 0.0))
        return bool(np.linalg.norm(data.xpos[body][:2] - position[:2]) > REPLAN_SHIFT or turned > np.radians(5.0))

    def _holding(self, data: object, name: str, *, lifted: bool) -> bool:
        position = data.xpos[self._object_body[name]]
        near = float(np.linalg.norm(position - self.ik.tcp(data))) < HELD_DISTANCE
        return near and (not lifted or float(position[2]) > self._rest_z + LIFTED_DZ)

    def _arrived(self, data: object, plan: JointPlan, now: float) -> bool:
        """Return whether the arm reached the plan's last target; one that stays far away stalls the demonstrator.

        In ``leader`` mode a client sends the targets back; when none does, the arm never gets there.
        """
        if self._plan_end is None:
            self._plan_end = now
        error = float(np.abs(np.array(data.qpos[self.ik.qpos_adr]) - plan.q[-1]).max())
        waited = now - self._plan_end
        if (error < 0.02 and waited >= 0.1) or (waited >= CATCH_UP_S and error <= STALL_ERROR):  # noqa: PLR2004
            self._plan_end = None
            return True
        if waited >= CATCH_UP_S:
            self._stall(error)
        return False

    def _judge(self, data: object, dt: float) -> None:
        """Back home after an attempt: did the object land in the bucket, and nothing else?"""
        episode = self._episode
        instruction = episode.instruction  # type: ignore[union-attr]
        state = self.task.evaluate(self.model, data)
        if state.success:
            episode.outcome = "success"  # type: ignore[union-attr]
            self.stats.successes += 1
            return
        others = [name for name, color in state.in_bucket.items() if color is not None and name != instruction.object]
        if others:
            self._fail(data, f"the {others[0]} ended up in a bucket", dt)
            return
        where = state.in_bucket[instruction.object]
        if where == instruction.bucket:
            return  # in place but still moving (or touched): judge again next tick
        reason = f"the object landed in the {where} bucket" if where else "the object landed outside the bucket"
        self._retry_or_fail(data, reason, dt)

    # -- attempts ---------------------------------------------------------

    def _start_state(self) -> np.ndarray:
        q = self._q if self._q is not None else self.ik.home
        tcp, _ = self.ik.fk(q)
        return np.array([*tcp, self.ik.continuous_yaw(q), self._grip])

    def _resting(self, data: object) -> bool:
        name = self._episode.instruction.object  # type: ignore[union-attr]
        return bool(self.task.evaluate(self.model, data).resting[name])

    def _graspable(self, data: object) -> bool:
        """Return whether the instructed object lies where an attempt can pick it up.

        Returns:
            ``True`` on the table, in reach, out of the buckets and with its long side level.
        """
        name = self._episode.instruction.object  # type: ignore[union-attr]
        body = self._object_body[name]
        position = data.xpos[body]
        rotation = data.xmat[body].reshape(3, 3)
        return bool(
            RETRY_X[0] <= position[0] <= RETRY_X[1]
            and RETRY_Y[0] <= position[1] <= RETRY_Y[1]
            and position[2] < 0.05  # noqa: PLR2004
            and self.planner.shapes[name].grasp_axes(rotation) is not None
            and bucket_containing(self.model, data, position) is None
        )

    def _retry_or_fail(self, data: object, reason: str, dt: float) -> None:
        """After a failed attempt: let go, go home, and retry once the object rests; or fail the episode."""
        episode = self._episode
        if episode is None:
            return
        episode.reasons.append(reason)
        if episode.attempt >= self.max_attempts:
            self._fail(data, reason, dt)
            return
        logger.info("Demonstrator: {}; retrying", reason)
        self.stats.retries += 1
        if self.randomize:
            episode.params = DemoParams.sample(episode.rng)
        episode.retry_pending = True
        self._go_home(data, dt)

    def _fail(self, data: object, reason: str, dt: float) -> None:
        """Mark the episode failed, let go of anything held and go home; the reset follows."""
        episode = self._episode
        if episode is None:
            return
        if episode.outcome != "failed":
            self.stats.failures += 1
        episode.outcome, episode.failure = "failed", reason
        logger.info("Demonstrator: episode {} failed: {}", episode.serial, reason)
        self._go_home(data, dt)

    def _go_home(self, data: object, dt: float) -> None:
        """Brake, open the hand, rise and return home.

        The arm may be moving when an attempt is abandoned: the targets first slow to a stop at a
        constant deceleration over :data:`BRAKE_S`, so no joint jumps from speed to rest.
        """
        episode = self._episode
        brake = self._brake(dt)
        self.planner.load(SceneSnapshot.take(data))
        script = MotionScript(self._start_state())
        script.add("release", grip=GRIP_OPEN, duration=0.4)
        self.planner.add_return(script, episode.params)  # type: ignore[union-attr]
        plan = self.planner.solve(script, self._q, dt, kind="home", target=episode.instruction.object)  # type: ignore[union-attr]
        self._run(plan, data, brake)

    def _brake(self, dt: float) -> np.ndarray:
        """Return targets that slow the arm from its current target speed to a stop.

        The next plan is planned from where they stop (``self._q`` is the last of them until
        :meth:`_run` puts the braking in front of that plan).

        Returns:
            ``(ticks, 7)`` joint targets.
        """
        q = np.asarray(self._q, dtype=np.float64)
        velocity = (q - self._q_prev) / dt if self._q_prev is not None else np.zeros(7)
        t = np.arange(1, round(BRAKE_S / dt) + 1) * dt
        brake = q[None, :] + velocity[None, :] * (t - t**2 / (2.0 * BRAKE_S))[:, None]
        brake = np.clip(brake, self.ik.lower, self.ik.upper)
        self._braking_from = q.copy()
        self._q = brake[-1].copy()
        return brake

    def _replan(self, data: object, dt: float, reason: str, *, attempt: bool) -> None:
        """Plan again from here: the whole attempt (before the grasp) or the place (with the object held)."""
        episode = self._episode
        if episode is None:
            return
        if episode.replans >= MAX_REPLANS:
            self._fail(data, f"{reason} too often", dt)
            return
        episode.replans += 1
        self.stats.replans += 1
        logger.info("Demonstrator: {}; planning again", reason)
        brake = self._brake(dt)
        if attempt:
            self._begin_attempt(data, dt, new_attempt=False, brake=brake)
        else:
            self._begin_place(data, dt, brake=brake)

    def _begin_attempt(
        self, data: object, dt: float, *, new_attempt: bool = True, brake: np.ndarray | None = None
    ) -> None:
        """Plan an attempt from the current targets and start it."""
        episode = self._episode
        if episode is None or episode.instruction is None:
            return
        if new_attempt or episode.attempt == 0:
            episode.attempt += 1
            self.stats.attempts += 1
        snapshot = SceneSnapshot.take(data)
        if brake is None and len(self.arms) > 1:
            plan = self._plan_with_an_arm(snapshot, episode.instruction, episode.params, dt, episode.rng)
            if plan is None:
                instruction = episode.instruction
                self._fail(data, f"no arm reaches the {instruction.object} and the {instruction.bucket} bucket", dt)
                return
        else:
            plan = self.plan_attempt(
                snapshot, self._q, self._start_state(), episode.instruction, episode.params, dt, episode.rng
            )
        self.planner.load(snapshot)
        position, rotation = self.planner.object_pose(episode.instruction.object)
        self._rest_z = float(position[2])
        self._planned_object = position.copy()
        self._planned_axis = self.planner.grasp_axis_yaw(episode.instruction.object, rotation)
        self._run(plan, data, brake)

    def _plan_with_an_arm(
        self, snapshot: SceneSnapshot, instruction: Instruction, params: DemoParams, dt: float, rng: np.random.Generator
    ) -> JointPlan | None:
        """Plan an attempt with the arm that suits it, and let that arm work (several arms, at rest).

        Every arm plans, in the order of :meth:`_arm_preference`: the first whose plan passes every
        check works, else the best of the plans that may run (:meth:`_may_run`: within
        :data:`OUT_OF_REACH` of their path and clear of the other arm), however well a plan that may
        not run scores. The ranking only orders the arms:
        its probe does not try the other grasp yaw or the turn for the release, which a plan may need.

        Returns:
            The plan, for the arm that now works; ``None`` when no arm can do the attempt.
        """
        preference = {i: self._arm_preference(i, snapshot, instruction, params) for i in range(len(self.arms))}
        order = sorted(preference, key=lambda i: preference[i])
        episode = self._episode
        if self._pinned_arm is not None:
            if episode is None or episode.attempt <= 1:
                order = [self._pinned_arm]  # a stuck start's arm recovers on its own: it works the first attempt
            self._pinned_arm = None
        best: tuple[int, JointPlan] | None = None
        """The best plan that may run (:meth:`_may_run`), and its arm."""
        closest: JointPlan | None = None
        """The best of the plans that may not, for the log."""
        for arm in order:
            self._switch_arm(arm)
            plan = self.plan_attempt(snapshot, self._q, self._start_state(), instruction, params, dt, rng, report=False)  # type: ignore[arg-type]
            if plan.ok:
                logger.info("Demonstrator: the {} works this attempt", self.active_arm.label)
                return plan
            if self._may_run(plan) and (best is None or _severity(plan) < _severity(best[1])):
                best = (arm, plan)
            elif best is None and (closest is None or _severity(plan) < _severity(closest)):
                closest = plan
        if best is None:
            if closest is not None:
                logger.info("Demonstrator: no arm can do this attempt ({})", "; ".join(closest.problems))
            return None
        self._switch_arm(best[0])
        logger.info("Demonstrator: the {} works this attempt", self.active_arm.label)
        self._report_problems(best[1])
        return best[1]

    def _arm_preference(
        self, arm: int, snapshot: SceneSnapshot, instruction: Instruction, params: DemoParams
    ) -> tuple[bool, float]:
        """Rank arm *arm* for an attempt: whether it reaches the grasp and the bucket from home, and how far it goes.

        The probe keeps the object's grasp yaw over the bucket; an arm it misses may still plan the
        attempt with the other yaw or a turn for the release, so it ranks such an arm last.

        Returns:
            A sort key; lower is better.
        """
        planner = self.planners[arm]
        ik = planner.ik
        planner.load(snapshot)
        position, rotation = planner.object_pose(instruction.object)
        grasp = planner.grasp_point(instruction.object, position, rotation, params)
        bucket, _ = planner.bucket_pose(instruction.bucket)
        over = np.array([bucket[0], bucket[1], params.release_z])
        q_grasp, e_grasp, _ = ik.optimize(ik.home, grasp, planner.grasp_axis_yaw(instruction.object, rotation), 0)
        _, e_over, _ = ik.optimize(q_grasp, over, ik.continuous_yaw(q_grasp), 0)
        home, _ = ik.fk(ik.home)
        reaches = e_grasp <= IK_POS_TOL and e_over <= IK_POS_TOL
        distance = float(np.linalg.norm((grasp - home)[:2]) + np.linalg.norm((bucket - home)[:2]))
        return (not reaches, distance)

    def _begin_place(self, data: object, dt: float, brake: np.ndarray | None = None) -> None:
        """Plan the carry and the release again, with the object in the hand, and start it."""
        episode = self._episode
        instruction = episode.instruction  # type: ignore[union-attr]
        self.planner.load(SceneSnapshot.take(data))
        # Plan from where braking stops, with the object where the hand will hold it then.
        self.planner.carry_held(instruction.object, data, self._q)
        start = self._start_state()
        position, _ = self.planner.object_pose(instruction.object)
        offset = (position - start[:3])[:2]
        plan = self.planner.checked(
            lambda slow: self.planner.place_script(start, instruction, episode.params, offset, slow),  # type: ignore[union-attr]
            self._q,
            dt,
            kind="place",
            target=instruction.object,
            bucket=instruction.bucket,
            held_from=0.0,
        )
        self._run(plan, data, brake)

    def plan_attempt(
        self,
        snapshot: SceneSnapshot,
        q: np.ndarray,
        start: np.ndarray,
        instruction: Instruction,
        params: DemoParams,
        dt: float,
        rng: np.random.Generator | None = None,
        *,
        report: bool = True,
    ) -> JointPlan:
        """Plan one attempt of the working arm: the better grasp yaw first, then the other, then safer variants.

        The two yaws are ranked by the joint-limit margin of their key postures (grasp, then
        release, solved from each other). When the margins are within 10 deg, ``params.yaw_choice``
        ranks them instead, so both yaws appear in the demonstrations.

        With *rng*, two fresh parameter draws come last: a path near the shoulder singularity
        (objects close in front of the base) often clears with other heights and speeds.

        Without *report*, a plan that fails a check is neither counted nor logged.

        Returns:
            The first plan that passes every check, else the one with the fewest problems.
        """
        planner = self.planner
        planner.load(snapshot)
        position, rotation = planner.object_pose(instruction.object)
        axis = planner.grasp_axis_yaw(instruction.object, rotation) + params.yaw_noise
        candidates = planner.yaw_candidates(axis, float(start[3]), position[:2])
        if len(candidates) == 2:  # noqa: PLR2004
            margins = [planner.key_margin(q, instruction, params, yaw) for yaw in candidates]
            if abs(margins[0] - margins[1]) < np.radians(10.0):
                candidates = candidates if params.yaw_choice < 0.5 else candidates[::-1]  # noqa: PLR2004
            elif margins[1] > margins[0]:
                candidates = candidates[::-1]
        variants = [
            params,
            replace(params, release_z=max(params.release_z, 0.15), carry_z=max(params.carry_z, 0.25)),
            replace(params, release_z=0.16, carry_z=0.26, blend=0.0, place_offset=(0.0, -0.01), grasp_offset=0.0),
        ]
        if rng is not None:
            variants += [DemoParams.sample(rng) for _ in range(2)]
        best: JointPlan | None = None
        eligible: JointPlan | None = None
        """With several arms, the best plan that may run at all (:meth:`_may_run`)."""
        for variant in variants:
            for yaw in candidates:
                planner.load(snapshot)

                def build(
                    slow: dict[str, float], variant: DemoParams = variant, yaw: float = yaw
                ) -> tuple[MotionScript, float]:
                    return planner.attempt_script(start, instruction, variant, grasp_yaw=yaw, slow=slow)

                plan = planner.checked(
                    build, q, dt, kind="attempt", target=instruction.object, bucket=instruction.bucket
                )
                plan.yaw_grasp = yaw
                if plan.ok:
                    return plan
                if best is None or _severity(plan) < _severity(best):
                    best = plan
                if (
                    len(self.arms) > 1
                    and self._may_run(plan)
                    and (eligible is None or _severity(plan) < _severity(eligible))
                ):
                    eligible = plan
        if eligible is not None:
            best = eligible  # a plan that would be refused must not hide one that may run
        if report:
            self._report_problems(best)  # type: ignore[arg-type]
        return best  # type: ignore[return-value]

    @staticmethod
    def _may_run(plan: JointPlan) -> bool:
        """Return whether a best-effort plan of several arms may run: near its path and clear of the other arm.

        Near is within :data:`OUT_OF_REACH`; clear is :data:`CLEARANCE` ``arm``.

        Returns:
            ``True`` for a plan that misses some check but neither of these.
        """
        return plan.pos_error <= OUT_OF_REACH and plan.clearance.get("arm", np.inf) >= CLEARANCE["arm"]

    def _too_close_to_an_arm(self, plan: JointPlan) -> bool:
        """Return whether *plan* brings the working arm closer than :data:`CLEARANCE` ``arm`` to an arm holding still.

        Every plan runs through :meth:`_run`, the re-planned approach and place and the way home
        included, so none moves the working arm into the other.
        """
        return len(self.arms) > 1 and plan.clearance.get("arm", np.inf) < CLEARANCE["arm"]

    def _stop_and_fail(
        self,
        data: object,
        dt: float,
        brake: np.ndarray | None,
        reason: str,
        *,
        freeze_at: np.ndarray | None = None,
    ) -> None:
        """Stop and hold; the episode fails and the scene resets after :data:`FAILURE_HOLD_S`.

        The arm stops where *brake* ends (checked clear of the other arm), else at once on *freeze_at*
        (the target before a braking that was not clear), else where its targets are. No other plan
        runs from here: one that goes home could come as close to the other arm.
        """
        episode = self._episode
        if brake is not None and len(brake):
            q = np.vstack([brake, brake[-1:]])
            self._q = self._braking_from  # this tick still outputs where the arm is now
        else:
            self._q = np.array(freeze_at if freeze_at is not None else self._q, dtype=np.float64)
            q = self._q[None, :].copy()
        ticks = len(q)
        state = self._start_state()
        stop = JointPlan(
            kind="home",
            dt=dt,
            q=q,
            grip=np.full(ticks, self._grip),
            phases=["failed"] * ticks,
            cartesian=np.repeat(state[None, :], ticks, axis=0),
        )
        if episode is not None:
            if episode.outcome != "failed":  # a failed episode on its way home keeps its reason
                self.stats.failures += 1
                episode.outcome, episode.failure = "failed", reason
            episode.reasons.append(reason)
        logger.warning("Demonstrator: {}; stopping, the scene resets next", reason)
        self._plan = stop
        self.plans.append(stop)
        self._tick = 0
        self._plan_end = None
        self._lag_since = None
        self._lift_check_since = None
        self._set_phase("failed", data)

    def _report_problems(self, plan: JointPlan) -> None:
        self.stats.plan_problems += 1
        logger.warning("Demonstrator: no plan passed every check ({}); running the best one", "; ".join(plan.problems))

    def _run(self, plan: JointPlan, data: object, brake: np.ndarray | None = None) -> None:
        """Start *plan*, after *brake*; with several arms, every plan passes one gate here first.

        The braking and the plan must keep clear of the arms holding still, and an attempt or place
        plan must stay within :data:`OUT_OF_REACH` of its path; otherwise the arm stops
        (:meth:`_stop_and_fail`). One arm runs its plans as before: the FR3's best-effort plans included.
        """
        if len(self.arms) > 1:
            braking = brake is not None and len(brake) > 0
            brake_clearance = self.planner.arm_clearance(brake, SceneSnapshot.take(data)) if braking else np.inf
            needed = CLEARANCE["arm"] * 1000
            if brake_clearance < CLEARANCE["arm"]:
                # Braking along the current speed would reach into the other arm: stop on the last
                # target instead, which a checked plan (or the hold) put there.
                reason = (
                    f"braking would come {brake_clearance * 1000:.0f} mm from the other arm (at least {needed:.0f} mm)"
                )
                self._stop_and_fail(data, plan.dt, None, reason, freeze_at=self._braking_from)
                return
            if self._too_close_to_an_arm(plan):
                distance = plan.clearance["arm"] * 1000
                reason = (
                    f"the {plan.kind} plan would come {distance:.0f} mm from the other arm (at least {needed:.0f} mm)"
                )
                self._stop_and_fail(data, plan.dt, brake, reason)
                return
            if plan.kind != "home" and plan.pos_error > OUT_OF_REACH:
                # A re-planned approach or place out of reach would close on nothing or let go outside
                # the bucket; the checked braking still stops the arm.
                reason = (
                    f"the {plan.kind} plan misses its path by {plan.pos_error * 1000:.0f} mm "
                    f"(at most {OUT_OF_REACH * 1000:.0f} mm): out of reach"
                )
                self._stop_and_fail(data, plan.dt, brake, reason)
                return
        if brake is not None and len(brake):
            # Prefix the braking ticks; the plan's own ticks shift by as many.
            n = len(brake)
            plan.q = np.vstack([brake, plan.q])
            plan.grip = np.concatenate([np.full(n, self._grip), plan.grip])
            plan.cartesian = np.vstack([np.repeat(plan.cartesian[:1], n, axis=0), plan.cartesian])
            plan.phases = [plan.phases[0]] * n + plan.phases
            for name in ("lift_check", "hold_until", "carry_start"):
                tick = getattr(plan, name)
                if tick is not None:
                    setattr(plan, name, tick + n)
            self._q = self._braking_from  # this tick still outputs where the arm is now
        self._plan = plan
        self.plans.append(plan)
        self._tick = 0
        self._plan_end = None
        self._lag_since = None
        self._lift_check_since = None
        self._set_phase(plan.phases[0], data)


__all__ = [
    "ACTUATORS",
    "FR3_DDDQ_MAX",
    "FR3_DDQ_MAX",
    "FR3_DQ_MAX",
    "HOME_Q",
    "BucketSortDemonstrator",
    "DemoParams",
    "FrankaIK",
    "JointPlan",
    "MotionScript",
    "Planner",
    "SceneSnapshot",
    "hand_rotation",
    "joint_smoothness",
    "min_jerk",
    "move_duration",
    "object_shape",
    "wrap_near",
]
