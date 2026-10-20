# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Franka Research 3 driver for PhysicalAI.

Controls the 7-DOF FR3 arm over the Franka Control Interface (FCI) through the
official ``pylibfranka`` Python bindings. The 1 kHz FCI loop runs in its own
process and follows the latest target submitted by :meth:`FrankaFR3.send_action`
through a jerk-limited tracker. By default it sends torques with the joint
impedance law of Franka's own GELLO controller; the Franka Hand runs in a
second process.

Why processes: ``pylibfranka`` keeps the GIL while it waits for the next robot
packet, so a 1 kHz loop in a thread starves (and is starved by) everything else
in the interpreter, such as a second arm, the gripper or a teleoperation leader.
Why torques: in joint position mode a late cycle can trip the motion
generator's ``joint_motion_generator_*_discontinuity`` reflexes; in torque mode
the robot simply keeps the last torque for that cycle.
"""

# The whole module operates against the untyped ``pylibfranka`` pybind11
# binding, so ``Any`` is the honest annotation at that boundary.
# ruff: noqa: ANN401

from __future__ import annotations

import contextlib
import gc
import importlib
import multiprocessing
import os
import queue
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from loguru import logger

from physicalai.config import export_config
from physicalai_franka_plugin.constants import (
    FR3_GRIPPER_DEFAULT_FORCE_N,
    FR3_GRIPPER_DEFAULT_SPEED_M_S,
    FR3_GRIPPER_WIDTH_RANGE_M,
    FR3_IMPEDANCE_DAMPING_NM_S_RAD,
    FR3_IMPEDANCE_STIFFNESS_NM_RAD,
    FR3_IMPEDANCE_VELOCITY_FILTER_ALPHA,
    FR3_JOINT_MAX_ACCELERATION_RAD_S2,
    FR3_JOINT_MAX_EFFORT_NM,
    FR3_JOINT_MAX_JERK_RAD_S3,
    FR3_JOINT_MAX_TORQUE_RATE_NM_S,
    FR3_JOINT_MAX_VELOCITY_RAD_S,
    FR3_JOINT_ORDER,
    VALID_CONTROL_MODES,
    VALID_REALTIME_CONFIGS,
    fr3_joint_limits_deg,
)

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import RobotObservation

_GRIPPER_COMMAND_EPSILON_M = 5e-4
_ARM_READY_TIMEOUT_S = 15.0
_DISCONNECT_JOIN_TIMEOUT_S = 5.0
_HISTORY = 500  # cycles kept for the post-mortem dump
_PERIOD_S = 0.001

# Shared-memory layout of an arm process's state: q, dq, tau_J, tau_ext (7 each), then counters.
_STATE_Q, _STATE_DQ, _STATE_TAU, _STATE_EXT = (slice(i * 7, i * 7 + 7) for i in range(4))
_STATE_TICKS, _STATE_OVERRUNS, _STATE_ALIVE = 28, 29, 30
_STATE_SIZE = 31


@dataclass
class FrankaFRObservation:
    """Observation data for the Franka Research 3 arm.

    Attributes:
        joint_positions: Arm joint positions in degrees, plus the Franka Hand
            gripper width in meters as a final element when the gripper is
            enabled.
        timestamp: Monotonic time of the observation.
        sensor_data: Optional dict with velocities (deg/s), torques (N·m), and
            filtered external torques (N·m) for the arm joints.
        images: Optional camera frames.
    """

    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None

    @property
    def state(self) -> np.ndarray:
        """Alias for joint positions, matching the Robot protocol."""
        return self.joint_positions


class JerkLimitedTracker:
    """Track a (stepwise) joint target with bounded velocity, acceleration and jerk.

    Per joint, the desired velocity points at the target and is capped by the
    velocity limit, by a braking profile ``sqrt(2 a_brake |e|)`` that can still
    stop in time, and by a linear ``kp |e|`` term for a smooth arrival. The
    acceleration follows the desired velocity, clamped to the acceleration
    limit, and may only change by ``jerk * dt`` per step; integrating that gives
    a C2 position command.
    """

    def __init__(
        self,
        q0: np.ndarray,
        max_velocity: np.ndarray,
        max_acceleration: np.ndarray,
        max_jerk: np.ndarray,
        *,
        kp: float = 8.0,
        kv: float = 25.0,
        brake_fraction: float = 0.5,
    ) -> None:
        """Start at rest at ``q0`` (rad) with per-joint limits."""
        self.q = np.asarray(q0, dtype=np.float64).copy()
        self.v = np.zeros_like(self.q)
        self.a = np.zeros_like(self.q)
        self._v_max = np.asarray(max_velocity, dtype=np.float64)
        self._a_max = np.asarray(max_acceleration, dtype=np.float64)
        self._j_max = np.asarray(max_jerk, dtype=np.float64)
        self._a_brake = brake_fraction * self._a_max
        self._kp = kp
        self._kv = kv

    def step(self, target: np.ndarray, dt: float) -> np.ndarray:
        """Advance one control period of ``dt`` seconds.

        Returns:
            The new position command (rad).
        """
        error = np.asarray(target, dtype=np.float64) - self.q
        magnitude = np.minimum.reduce([
            self._v_max,
            np.sqrt(2.0 * self._a_brake * np.abs(error)),
            self._kp * np.abs(error),
        ])
        v_desired = np.sign(error) * magnitude
        a_desired = np.clip(self._kv * (v_desired - self.v), -self._a_max, self._a_max)
        jerk_step = self._j_max * dt
        self.a += np.clip(a_desired - self.a, -jerk_step, jerk_step)
        self.v += self.a * dt
        self.q += self.v * dt
        return self.q.copy()

    def advance(self, target: np.ndarray, periods: int, period_s: float = _PERIOD_S) -> np.ndarray:
        """Advance ``periods`` robot cycles.

        When command packets were missed, the robot extrapolated those cycles
        itself with constant acceleration. Coasting the same way for the missed
        cycles, then taking one normal step, keeps the command continuous with
        what the robot executed.

        Returns:
            The position command (rad) for the last cycle.
        """
        for _ in range(max(periods, 1) - 1):
            self.v += self.a * period_s
            self.q += self.v * period_s
        return self.step(target, period_s if periods >= 1 else 0.0)


class SeqBuffer:
    """Lock-free single-writer buffer of doubles in shared memory (a sequence lock).

    The writer makes the counter odd, writes, then makes it even again; a reader
    copies the data and retries if the counter was odd or changed meanwhile.
    Neither side ever blocks, so the real-time loop never waits on a
    normal-priority process (no priority inversion).
    """

    def __init__(self, raw: Any) -> None:
        """Wrap a ``multiprocessing`` ``RawArray("d", n + 1)``."""
        self.raw = raw
        self._view = np.frombuffer(raw, dtype=np.float64)

    @classmethod
    def create(cls, context: Any, size: int) -> SeqBuffer:
        """Allocate a buffer for ``size`` doubles in ``context``'s shared memory.

        Returns:
            The new buffer.
        """
        return cls(context.RawArray("d", size + 1))

    def write(self, values: np.ndarray) -> None:
        """Publish ``values`` (writer side)."""
        view = self._view
        view[0] += 1.0  # odd: write in progress
        view[1 : 1 + len(values)] = values
        view[0] += 1.0  # even: consistent

    def read(self, attempts: int = 100) -> np.ndarray | None:
        """Return a consistent copy, or ``None`` if every attempt overlapped a write."""
        view = self._view
        for _ in range(attempts):
            before = view[0]
            if int(before) % 2 == 0:
                data = view[1:].copy()
                if view[0] == before:
                    return data
        return None


def _isolate_process(cpu: int | None) -> None:
    """Pin this process to one CPU and lock its memory (best effort, Linux)."""
    if cpu is not None and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {cpu})
        except (OSError, ValueError) as exc:
            logger.warning("Could not pin to CPU {}: {}", cpu, exc)
    with contextlib.suppress(OSError, AttributeError):
        import ctypes  # ruff: ignore[PLC0415]

        libc = ctypes.CDLL(None, use_errno=True)
        if libc.mlockall(1) != 0:  # MCL_CURRENT; fails if `ulimit -l` is below the process size
            logger.warning("mlockall failed (errno {}); raise memlock for the realtime group", ctypes.get_errno())


def _set_realtime_priority(priority: int) -> None:
    """Give the calling thread SCHED_FIFO priority (Linux); warn if not permitted."""
    if not hasattr(os, "sched_setscheduler"):
        return
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
    except PermissionError:
        logger.warning("No permission for SCHED_FIFO; the 1 kHz loop may miss packets. Join the 'realtime' group.")


def _arm_process(  # ruff: ignore[C901, PLR0912, PLR0914, PLR0915] - one inline 1 kHz loop
    host: str,
    realtime_config: str,
    dynamics_factor: float,
    rt_priority: int,
    auto_recover: bool,  # ruff: ignore[FBT001] - Process(args=...) is positional
    target_raw: Any,
    state_raw: Any,
    stop: Any,
    ready: Any,
    errors: Any,
    *,
    prepared: Any = None,
    go: Any = None,
    cpu: int | None = None,
    pylibfranka: Any = None,
    control_mode: str = "impedance",
) -> None:
    """Run one FR3's 1 kHz FCI loop in its own process.

    ``target_raw`` holds the 7 joint targets (rad, written by the parent);
    ``state_raw`` gets the robot state (``_STATE_*`` layout). ``prepared`` is set
    once connected, before control starts; the loop starts when ``go`` is set,
    so the heavy start-up of one arm never overlaps another arm's running loop.

    ``control_mode="impedance"`` sends ``tau = K (q_cmd - q) - D dq_filtered``
    with Franka's GELLO gains, clipped to the FR3 torque and torque-rate limits
    (the robot adds gravity compensation). ``"position"`` sends ``q_cmd`` as a
    joint position command instead.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent stops us via ``stop``
    if pylibfranka is None:
        pylibfranka = importlib.import_module("pylibfranka")
    _isolate_process(cpu)
    target = SeqBuffer(target_raw)
    state = SeqBuffer(state_raw)
    published = np.zeros(_STATE_SIZE)
    robot = None
    stiffness = np.asarray(FR3_IMPEDANCE_STIFFNESS_NM_RAD)
    damping = np.asarray(FR3_IMPEDANCE_DAMPING_NM_S_RAD)
    max_torque = np.asarray(FR3_JOINT_MAX_EFFORT_NM)
    ticks = 0
    max_gap_s, max_gap_tick = 0.0, 0
    # last cycles: robot dt, command (7), measured (7: q in impedance, q_d in position mode),
    # wall-clock time since the previous read, torque sent (7; zeros in position mode)
    history = np.full((_HISTORY, 23), np.nan)
    try:  # any FCI error ends the loop and is reported
        # Warm up the tracker's code paths before the 1 kHz deadline starts.
        JerkLimitedTracker(np.zeros(7), np.ones(7), np.ones(7), np.ones(7)).step(np.ones(7), _PERIOD_S)
        robot = pylibfranka.Robot(host, getattr(pylibfranka.RealtimeConfig, realtime_config))
        initial = robot.read_once()
        if auto_recover and initial.current_errors:
            robot.automatic_error_recovery()
            initial = robot.read_once()
        # Same thresholds as Franka's pylibfranka examples (setDefaultBehaviour).
        robot.set_collision_behavior([20.0] * 7, [40.0] * 7, [10.0] * 6, [20.0] * 6)
        robot.set_joint_impedance([3000.0, 3000.0, 3000.0, 2500.0, 2500.0, 2000.0, 2000.0])
        if prepared is not None:
            prepared.set()
        while go is not None and not go.wait(0.05):
            if stop.is_set():
                return
        initial = robot.read_once()
        impedance = control_mode == "impedance"
        if impedance:
            control = robot.start_torque_control()
        else:
            control = robot.start_joint_position_control(pylibfranka.ControllerMode.JointImpedance)
        goal = np.asarray(initial.q, dtype=np.float64).copy()
        target.write(goal)
        _set_realtime_priority(rt_priority)
        tracker: JerkLimitedTracker | None = None
        overruns = 0
        dq_filtered = np.zeros(7)
        tau = np.zeros(7)
        last_read = time.monotonic()
        # The loop makes no reference cycles; a cyclic GC pass mid-loop would only add jitter.
        gc.collect()
        gc.freeze()
        gc.disable()
        while not stop.is_set():
            robot_state, duration = control.readOnce()
            now = time.monotonic()
            wall_s = now - last_read
            if ticks and wall_s > max_gap_s:
                max_gap_s, max_gap_tick = wall_s, ticks
            last_read = now
            dt = float(duration.to_sec())
            periods = round(dt / _PERIOD_S)
            if tracker is None:  # start exactly where the robot is (impedance) or is commanded, at rest
                tracker = JerkLimitedTracker(
                    np.asarray(robot_state.q if impedance else robot_state.q_d, dtype=np.float64),
                    dynamics_factor * np.asarray(FR3_JOINT_MAX_VELOCITY_RAD_S),
                    dynamics_factor * np.asarray(FR3_JOINT_MAX_ACCELERATION_RAD_S2),
                    dynamics_factor * np.asarray(FR3_JOINT_MAX_JERK_RAD_S3),
                )
                periods = 0
            elif periods > 1:  # command packets were missed; the robot extrapolated meanwhile
                overruns += 1
            latest = target.read(attempts=3)  # never wait: keep the previous goal on a collision
            if latest is not None:
                goal = latest
            command = tracker.advance(goal, periods)
            if impedance:
                q = np.asarray(robot_state.q, dtype=np.float64)
                dq = np.asarray(robot_state.dq, dtype=np.float64)
                dq_filtered = (
                    1.0 - FR3_IMPEDANCE_VELOCITY_FILTER_ALPHA
                ) * dq_filtered + FR3_IMPEDANCE_VELOCITY_FILTER_ALPHA * dq
                tau_new = stiffness * (command - q) - damping * dq_filtered
                step = FR3_JOINT_MAX_TORQUE_RATE_NM_S * _PERIOD_S * max(periods, 1)
                tau = np.clip(np.clip(tau_new, tau - step, tau + step), -max_torque, max_torque)
                control.writeOnce(pylibfranka.Torques(tau))
                measured = q
            else:
                control.writeOnce(pylibfranka.JointPositions(command))
                measured = robot_state.q_d
            history[ticks % _HISTORY] = (dt, *command, *measured, wall_s, *tau)
            ticks += 1
            published[_STATE_Q] = robot_state.q
            published[_STATE_DQ] = robot_state.dq
            published[_STATE_TAU] = robot_state.tau_J
            published[_STATE_EXT] = robot_state.tau_ext_hat_filtered
            published[_STATE_TICKS] = ticks
            published[_STATE_OVERRUNS] = overruns
            published[_STATE_ALIVE] = 1.0
            state.write(published)
            if ticks == 1:
                ready.set()
        _save_history(host, "last", history, ticks)
    except Exception as exc:  # ruff: ignore[BLE001] - report any FCI error (reflex, network) to the parent
        timing = f" [after {ticks} ticks; longest gap {max_gap_s * 1000:.1f} ms at tick {max_gap_tick}]"
        dump = _save_history(host, "abort", history, ticks)
        if dump is not None:
            timing += f" (last cycles saved to {dump})"
        errors.put(f"{type(exc).__name__}: {exc}{timing}")
    finally:
        gc.enable()
        gc.unfreeze()
        published[_STATE_ALIVE] = 0.0
        state.write(published)
        if robot is not None:
            with contextlib.suppress(Exception):
                robot.stop()
        if prepared is not None:
            prepared.set()  # unblock a parent still waiting in prepare()
        ready.set()  # unblock a parent still waiting in connect()


def _save_history(host: str, kind: str, history: np.ndarray, ticks: int) -> Path | None:
    """Save the last cycles (oldest first) to ``/tmp/franka-<host>-<kind>.npz`` for post-mortems.

    Returns:
        The file written, or ``None`` if it could not be written.
    """
    order = np.roll(np.arange(_HISTORY), -(ticks % _HISTORY))
    path = Path(f"/tmp/franka-{host}-{kind}.npz")  # ruff: ignore[S108] - fixed, documented name
    try:
        np.savez(path, history=history[order], ticks=ticks)
    except OSError:
        return None
    return path


def _gripper_process(
    host: str,
    commands: Any,
    width: Any,
    speed_m_s: float,
    force_n: float,
    *,
    factory: Any = None,
) -> None:
    """Drive one Franka Hand in its own process (``pylibfranka`` blocks with the GIL held).

    Commands are target widths in meters; ``None`` ends the process. Closing
    grasps with ``force_n`` (anything between fully closed and open counts as
    grasped, so an object is held), opening moves to the width.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent sends ``None`` to stop
    try:
        gripper = factory() if factory is not None else importlib.import_module("pylibfranka").Gripper(host)
        width.value = float(gripper.read_once().width)
    except Exception as exc:  # ruff: ignore[BLE001] - without a hand the arm keeps working
        logger.warning("Franka Hand at {} unavailable ({}); continuing arm-only", host, exc)
        return
    open_m = FR3_GRIPPER_WIDTH_RANGE_M[1]
    while True:
        target = commands.get()
        with contextlib.suppress(queue.Empty):
            while True:  # only the latest request matters
                target = commands.get_nowait()
        if target is None:
            with contextlib.suppress(Exception):
                gripper.stop()
            return
        try:
            if target < float(width.value) - _GRIPPER_COMMAND_EPSILON_M:
                gripper.grasp(target, speed_m_s, force_n, open_m, open_m)
            else:
                gripper.move(target, speed_m_s)
            width.value = float(gripper.read_once().width)
        except Exception as exc:  # ruff: ignore[BLE001] - a gripper fault must not stop the arm
            logger.warning("Franka Hand {} command failed: {}", host, exc)


@export_config(class_path="physicalai_franka_plugin.robot.FrankaFR3")
class FrankaFR3:
    """Driver for the Franka Research 3 robot arm.

    Connects to the robot controller over the FCI and runs the 1 kHz control
    loop in a separate process (and the Franka Hand in another), following the
    latest target through a jerk-limited tracker. Positions are exposed in
    degrees at the driver boundary; the Franka Hand gripper, when enabled, is
    commanded by width in meters as an extra action element.

    Args:
        host: IP address or hostname of the robot controller.
        gripper_enabled: Whether a Franka Hand gripper is attached and the
            action/observation vectors include a gripper element.
        realtime_config: ``"kEnforce"`` for real-time scheduling requirements
            (production) or ``"kIgnore"`` for testing/development.
        goal_time: Accepted for configuration compatibility. The timing now
            comes from the jerk-limited tracker (see ``dynamics_factor``).
        auto_recover_on_error: Attempt automatic error recovery during
            ``connect`` when the robot reports recoverable errors.
        gripper_speed_m_s: Gripper movement speed in meters per second.
        control_mode: ``"impedance"`` (default) sends torques with the joint
            impedance gains of Franka's GELLO controller; ``"position"`` sends
            joint positions.
        dynamics_factor: Fraction of the FR3 velocity, acceleration and jerk
            limits the tracker may use, in ``(0, 1]``.
        gripper_force_n: Grasp force in newtons when the gripper closes.
        joint_limits: ``"hardware"`` (FR3 specification minus
            ``limit_margin_deg``), ``"suggested"`` or ``"urdf"``.
        limit_margin_deg: Margin inside the hardware limits, in degrees.
        cpu: CPU core to pin the control process to (Linux), or ``None``.
        rt_priority: SCHED_FIFO priority of the control loop (Linux).

    Raises:
        ValueError: If any parameter has an invalid value.
    """

    JOINT_ORDER: ClassVar[list[str]] = list(FR3_JOINT_ORDER)
    NUM_JOINTS: ClassVar[int] = len(JOINT_ORDER)

    def __init__(
        self,
        host: str,
        *,
        gripper_enabled: bool = True,
        realtime_config: str = "kEnforce",
        goal_time: float = 0.1,
        auto_recover_on_error: bool = True,
        gripper_speed_m_s: float = FR3_GRIPPER_DEFAULT_SPEED_M_S,
        control_mode: str = "impedance",
        dynamics_factor: float = 0.3,
        gripper_force_n: float = FR3_GRIPPER_DEFAULT_FORCE_N,
        joint_limits: str = "hardware",
        limit_margin_deg: float = 5.0,
        cpu: int | None = None,
        rt_priority: int = 80,
    ) -> None:
        """Initialize the Franka Research 3 driver.

        Raises:
            ValueError: If any parameter has an invalid value.
        """
        if not host:
            msg = "host must be a non-empty IP address or hostname."
            raise ValueError(msg)
        if realtime_config not in VALID_REALTIME_CONFIGS:
            msg = f"Invalid realtime_config {realtime_config!r}. Must be one of {sorted(VALID_REALTIME_CONFIGS)}."
            raise ValueError(msg)
        if goal_time <= 0.0:
            msg = "goal_time must be positive."
            raise ValueError(msg)
        if gripper_speed_m_s <= 0.0:
            msg = "gripper_speed_m_s must be positive."
            raise ValueError(msg)
        if control_mode not in VALID_CONTROL_MODES:
            msg = f"Invalid control_mode {control_mode!r}. Must be one of {sorted(VALID_CONTROL_MODES)}."
            raise ValueError(msg)
        if not 0.0 < dynamics_factor <= 1.0:
            msg = "dynamics_factor must be in (0, 1]."
            raise ValueError(msg)
        if gripper_force_n <= 0.0:
            msg = "gripper_force_n must be positive."
            raise ValueError(msg)

        self._host = host
        self._gripper_enabled = gripper_enabled
        self._realtime_config = realtime_config
        self._goal_time = goal_time
        self._auto_recover_on_error = auto_recover_on_error
        self._gripper_speed_m_s = gripper_speed_m_s
        self._control_mode = control_mode
        self._dynamics_factor = dynamics_factor
        self._gripper_force_n = gripper_force_n
        self._limits_deg = fr3_joint_limits_deg(joint_limits, limit_margin_deg)
        self._cpu = cpu
        self._rt_priority = rt_priority

        # Overridable for tests (a "fork" context lets the children use injected fakes).
        self._mp_context: Any = multiprocessing.get_context("spawn")
        self._pylibfranka: Any = None
        self._gripper_factory: Any = None
        self._arm: Any = None
        self._gripper: Any = None
        self._target: SeqBuffer | None = None
        self._state: SeqBuffer | None = None
        self._stop: Any = None
        self._ready: Any = None
        self._errors: Any = None
        self._prepared: Any = None
        self._go: Any = None
        self._gripper_queue: Any = None
        self._gripper_width: Any = None
        self._last_gripper_command: float | None = None
        self._last_error: str | None = None

    @property
    def host(self) -> str:
        """IP address or hostname of the robot controller."""
        return self._host

    @property
    def gripper_enabled(self) -> bool:
        """Whether the Franka Hand gripper is part of the action/observation layout."""
        return self._gripper_enabled

    @property
    def realtime_config(self) -> str:
        """Real-time scheduling configuration (``"kEnforce"`` or ``"kIgnore"``)."""
        return self._realtime_config

    @property
    def goal_time(self) -> float:
        """Kept for configuration compatibility; the tracker sets the timing."""
        return self._goal_time

    @property
    def control_mode(self) -> str:
        """``"impedance"`` (torques) or ``"position"`` (joint positions)."""
        return self._control_mode

    @property
    def limits_deg(self) -> np.ndarray:
        """Joint limits (7 x 2, degrees) that targets are clipped to."""
        return self._limits_deg.copy()

    @property
    def joint_names(self) -> list[str]:
        """Ordered joint names; includes ``"gripper"`` when the gripper is enabled."""
        if self._gripper_enabled:
            return [*self.JOINT_ORDER, "gripper"]
        return self.JOINT_ORDER

    @property
    def device_ids(self) -> tuple[str, ...]:
        """The FCI connection is a network resource, not an exclusively-owned device."""
        return ()

    @property
    def last_error(self) -> str | None:
        """The last error reported by the control process (for example a reflex), if any."""
        return self._last_error

    @property
    def loop_overruns(self) -> int:
        """Cycles in which the robot reported missed command packets."""
        # The shared state outlives disconnect(), which is when callers usually report this.
        values = self._state.read() if self._state is not None else None
        return int(values[_STATE_OVERRUNS]) if values is not None else 0

    def _check_pylibfranka(self) -> None:
        if self._pylibfranka is not None:
            return
        try:
            importlib.import_module("pylibfranka")
        except ImportError as exc:
            msg = (
                "pylibfranka is required to control a Franka FR3. Install it with: "
                "pip install 'physicalai-franka-plugin[franka]'. Note: pylibfranka ships "
                "wheels for Python 3.9-3.12 only; use Python <= 3.12 to control the robot."
            )
            raise ImportError(msg) from exc

    def prepare(self) -> None:
        """Start the arm (and hand) processes and connect to the robot, without starting control.

        Call it for every arm before :meth:`connect`, so no process start-up
        overlaps another arm's running 1 kHz loop.

        Raises:
            ConnectionError: If the robot cannot be reached.
        """
        if self._arm is not None:
            return
        self._check_pylibfranka()
        ctx = self._mp_context
        self._target = SeqBuffer.create(ctx, 7)
        self._state = SeqBuffer.create(ctx, _STATE_SIZE)
        self._stop, self._ready, self._errors = ctx.Event(), ctx.Event(), ctx.Queue()
        self._prepared, self._go = ctx.Event(), ctx.Event()
        self._last_error = None
        if self._gripper_enabled:
            self._gripper_queue = ctx.Queue()
            self._gripper_width = ctx.Value("d", float("nan"), lock=False)
            self._last_gripper_command = None
            self._gripper = ctx.Process(
                target=_gripper_process,
                args=(
                    self._host,
                    self._gripper_queue,
                    self._gripper_width,
                    self._gripper_speed_m_s,
                    self._gripper_force_n,
                ),
                kwargs={"factory": self._gripper_factory},
                name=f"franka-gripper-{self._host}",
                daemon=True,
            )
            self._gripper.start()
        self._arm = ctx.Process(
            target=_arm_process,
            args=(
                self._host,
                self._realtime_config,
                self._dynamics_factor,
                self._rt_priority,
                self._auto_recover_on_error,
                self._target.raw,
                self._state.raw,
                self._stop,
                self._ready,
                self._errors,
            ),
            kwargs={
                "prepared": self._prepared,
                "go": self._go,
                "cpu": self._cpu,
                "pylibfranka": self._pylibfranka,
                "control_mode": self._control_mode,
            },
            name=f"franka-arm-{self._host}",
            daemon=True,
        )
        self._arm.start()
        if not self._prepared.wait(_ARM_READY_TIMEOUT_S) or not self._arm.is_alive():
            error = self._read_error() or f"no connection within {_ARM_READY_TIMEOUT_S:.0f} s"
            self.disconnect()
            msg = f"FR3 {self._host}: {error}"
            raise ConnectionError(msg)

    def connect(self) -> None:
        """Start the 1 kHz control loop (preparing first if needed); return once it runs.

        Raises:
            ConnectionError: If the loop does not deliver a robot state in time.
        """
        if self.is_connected():
            return
        self.prepare()
        self._go.set()
        if not self._ready.wait(_ARM_READY_TIMEOUT_S) or not self.is_connected():
            error = self._read_error() or f"no robot state within {_ARM_READY_TIMEOUT_S:.0f} s"
            self.disconnect()
            msg = f"FR3 {self._host}: {error}"
            raise ConnectionError(msg)
        logger.info(
            "FrankaFR3 {} connected ({} control, loop in process {})", self._host, self._control_mode, self._arm.pid
        )

    def _read_error(self) -> str | None:
        if self._errors is not None:
            with contextlib.suppress(queue.Empty, OSError, ValueError):
                self._last_error = self._errors.get(timeout=0.5)
        return self._last_error

    def disconnect(self) -> None:
        """Stop the control loop and the hand process; the robot holds its pose."""
        if self._arm is not None:
            self._stop.set()
            self._arm.join(timeout=_DISCONNECT_JOIN_TIMEOUT_S)
            if self._arm.is_alive():
                self._arm.terminate()
            self._arm = None
            logger.info("FrankaFR3 disconnected from {}", self._host)
        if self._gripper is not None:
            with contextlib.suppress(Exception):
                self._gripper_queue.put(None)
            self._gripper.join(timeout=_DISCONNECT_JOIN_TIMEOUT_S)
            if self._gripper.is_alive():
                self._gripper.terminate()
            self._gripper = None

    def is_connected(self) -> bool:
        """Return whether the control loop is running and accepting commands."""
        if self._arm is None or not self._arm.is_alive() or self._state is None:
            return False
        values = self._state.read()
        return values is not None and bool(values[_STATE_ALIVE])  # published as 0.0 or 1.0

    def recover(self) -> None:
        """Restart the control loop with automatic error recovery (for example after a reflex)."""
        auto_recover = self._auto_recover_on_error
        self.disconnect()
        self._auto_recover_on_error = True
        try:
            self.connect()
        finally:
            self._auto_recover_on_error = auto_recover

    def _not_running_message(self) -> str:
        error = self._read_error() if self._arm is not None else None
        detail = f": {error}" if error else ". Call connect() first."
        return f"FR3 {self._host} control loop is not running{detail}"

    def get_observation(self) -> RobotObservation:
        """Return the latest state published by the control loop.

        Returns:
            A ``FrankaFRObservation`` with arm joint positions in degrees and,
            when enabled, the gripper width in meters as the final element.

        Raises:
            ConnectionError: If the control loop is not running.
        """
        if not self.is_connected() or self._state is None:
            msg = self._not_running_message()
            raise ConnectionError(msg)
        values = self._state.read(attempts=10_000)
        if values is None:
            msg = f"FR3 {self._host}: could not read a consistent state"
            raise ConnectionError(msg)
        positions = np.degrees(values[_STATE_Q])
        if self._gripper_enabled:
            width = float(self._gripper_width.value) if self._gripper_width is not None else float("nan")
            positions = np.append(positions, width if np.isfinite(width) else FR3_GRIPPER_WIDTH_RANGE_M[0])
        return FrankaFRObservation(
            joint_positions=positions.astype(np.float32),
            timestamp=time.monotonic(),
            sensor_data={
                "velocities": np.degrees(values[_STATE_DQ]).astype(np.float32),
                "torques": values[_STATE_TAU].astype(np.float32),
                "external_torques": values[_STATE_EXT].astype(np.float32),
            },
        )

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Publish a joint target to the control loop.

        The arm joints are interpreted in degrees and clipped to the joint
        limits; the jerk-limited tracker moves there. When the gripper is
        enabled, an extra width element in meters commands the Franka Hand
        whenever it changes materially (closing grasps with ``gripper_force_n``).

        Args:
            action: Array of arm joint targets in degrees, plus the gripper
                width in meters when ``gripper_enabled``.
            goal_time: Accepted for API compatibility; the tracker sets the timing.

        Raises:
            ConnectionError: If the control loop is not running.
            ValueError: If the action shape does not match the joint layout.
        """
        if goal_time <= 0.0:
            msg = "goal_time must be positive."
            raise ValueError(msg)
        if not self.is_connected() or self._target is None:
            msg = self._not_running_message()
            raise ConnectionError(msg)
        action = np.asarray(action, dtype=np.float64)
        expected_shape = (len(self.joint_names),)
        if action.shape != expected_shape:
            msg = f"Expected action shape {expected_shape}, got {action.shape}"
            raise ValueError(msg)
        arm_deg = np.clip(action[: self.NUM_JOINTS], self._limits_deg[:, 0], self._limits_deg[:, 1])
        self._target.write(np.radians(arm_deg))
        if self._gripper_enabled and self._gripper_queue is not None:
            width = float(np.clip(action[self.NUM_JOINTS], *FR3_GRIPPER_WIDTH_RANGE_M))
            last = self._last_gripper_command
            if last is None or abs(width - last) > _GRIPPER_COMMAND_EPSILON_M:
                self._last_gripper_command = width
                self._gripper_queue.put(width)
