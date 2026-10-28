# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Joint-position follower with a single-owner RTDE servo loop."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from loguru import logger

from physicalai.config import export_config

from .profiles import JOINT_NAMES, Model, joint_limits
from .settings import RobotSettings

if TYPE_CHECKING:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    from physicalai.capture.frame import Frame

_ROBOT_MODE_RUNNING = 7
_SAFE_MODES = {1, 2}


@dataclass
class URObservation:
    """Six joint positions in degrees, sampled by the RTDE owner thread."""

    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None

    @property
    def state(self) -> np.ndarray:
        """Position-only inference state."""
        return self.joint_positions


@dataclass(frozen=True)
class _Target:
    positions: np.ndarray
    goal_time: float
    accepted_at: float


@dataclass
class _Trajectory:
    start: np.ndarray
    end: np.ndarray
    duration: float
    elapsed: float = 0.0

    def step(self, dt: float) -> np.ndarray:
        self.elapsed = min(self.duration, self.elapsed + dt)
        return self.start + (self.end - self.start) * (self.elapsed / self.duration)


@dataclass
class _Session:
    control: RTDEControlInterface | None = None
    receive: RTDEReceiveInterface | None = None
    servo_active: bool = False
    target: _Target | None = None
    trajectory: _Trajectory | None = None
    command: np.ndarray | None = None
    controller_time: float | None = None
    fresh_at: float = field(default_factory=time.monotonic)
    watchdog_at: float = field(default_factory=time.monotonic)
    previous_cycle: float = field(default_factory=time.monotonic)


def _checked(result: object, operation: str) -> None:
    if not result:
        msg = f"RTDE rejected {operation}."
        raise RuntimeError(msg)


@export_config(class_path="physicalai_ur5e_plugin.UniversalRobot")
class UniversalRobot:
    """Six-axis UR e-Series joint-position follower.

    All public positions/actions are degrees; RTDE uses radians internally.
    ``send_action`` queues the latest target, interpolated over at least
    ``goal_time`` and lengthened to respect the configured setpoint speed.
    Commands must continue arriving within ``command_timeout`` even after the
    target is reached. A timeout or controller fault latches an error until
    explicit disconnect/reconnect; it never automatically resumes motion.

    Connection uploads/starts the RTDE script in remote mode, but never homes,
    powers on, releases brakes, or resets a safety stop. SDK control calls and
    cleanup belong exclusively to one worker because RTDEControl is not
    thread-safe. The Python loop is not a hard-real-time or safety-rated system.

    Args:
        ip: Literal controller IPv4 address on a trusted robot-cell network.
        model: Model profile, not automatic hardware identification.
        control_mode: PolyScope 5 remote mode or manually started External Control URCap.
        external_control_port: External Control URCap port, when selected.
        frequency: RTDE servo frequency in Hz (50 to 500).
        command_timeout: Maximum command and state age in seconds.
        max_velocity: Maximum interpolated setpoint speed in degrees/s.
        max_relative_target: Maximum target/tracking offset in degrees.
        lookahead_time: UR servo lookahead in seconds (0.03 to 0.2).
        gain: UR servo gain (100 to 2000).
    """

    def __init__(
        self,
        ip: str,
        model: Model = "ur5e",
        *,
        control_mode: Literal["remote", "external_control"] = "remote",
        external_control_port: int = 50002,
        frequency: float = 500.0,
        command_timeout: float = 0.5,
        max_velocity: float = 30.0,
        max_relative_target: float = 10.0,
        lookahead_time: float = 0.1,
        gain: float = 300.0,
    ) -> None:
        """Validate settings and construct a disconnected driver without hardware I/O."""
        self._settings = RobotSettings(
            ip=ip,
            model=model,
            control_mode=control_mode,
            external_control_port=external_control_port,
            frequency=frequency,
            command_timeout=command_timeout,
            max_velocity=max_velocity,
            max_relative_target=max_relative_target,
            lookahead_time=lookahead_time,
            gain=gain,
        )
        limits = joint_limits(self._settings.model)
        self._lower = np.array([-np.inf if value is None else value for value in limits["lower"]])
        self._upper = np.array([np.inf if value is None else value for value in limits["upper"]])
        self._velocity = np.minimum(np.array(limits["velocity"]), np.deg2rad(self._settings.max_velocity))
        self._max_offset = float(np.deg2rad(self._settings.max_relative_target))
        self._lock = threading.Lock()
        self._lifecycle = threading.Lock()
        self._stop = threading.Event()
        self._started = threading.Event()
        self._thread: threading.Thread | None = None
        self._fault: Exception | None = None
        self._observation: URObservation | None = None
        self._pending: _Target | None = None
        self._ready = False

    @property
    def ip(self) -> str:
        """Controller IPv4 address, validated without hardware I/O."""
        return self._settings.ip

    @property
    def model(self) -> Model:
        """Configured model profile."""
        return self._settings.model

    @property
    def joint_names(self) -> list[str]:
        """Joint order shared by observations, actions, and official URDFs."""
        return list(JOINT_NAMES)

    @property
    def device_ids(self) -> tuple[str, ...]:
        """Exclusive identity shared across all model/control-mode profiles."""
        return (f"tcp:{self.ip}",)

    def connect(self) -> None:
        """Start the RTDE owner; never move the arm during connection.

        Raises:
            ConnectionError: Startup failed or a previous session needs disconnecting.
            TimeoutError: The SDK did not finish starting within ten seconds.
        """
        with self._lifecycle:
            if self.is_connected():
                return
            if self._thread is not None:
                msg = "Disconnect the previous RTDE session before reconnecting."
                raise ConnectionError(msg) from self._fault
            self._stop.clear()
            self._started.clear()
            self._fault = None
            self._observation = None
            self._pending = None
            self._thread = threading.Thread(target=self._run, name=f"ur-rtde-{self.ip}", daemon=True)
            self._thread.start()
            if not self._started.wait(timeout=10.0):
                self._stop.set()
                msg = "RTDE startup timed out; check remote mode/URCap, then call disconnect()."
                raise TimeoutError(msg)
            if not self.is_connected():
                msg = f"Unable to start RTDE session: {self._fault}"
                raise ConnectionError(msg) from self._fault

    def disconnect(self) -> None:
        """Stop servoing and the script, then close both RTDE connections.

        If the SDK blocks, report the failure without closing its sockets from
        another thread. The controller watchdog remains the fallback; an
        interrupted network cannot guarantee delivery of a software stop.

        Raises:
            TimeoutError: The owner remains blocked in the SDK after five seconds.
            RuntimeError: The session or one of its cleanup operations failed.
        """
        with self._lifecycle:
            if self._thread is None:
                return
            self._stop.set()
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                msg = "RTDE owner did not stop; check the robot locally. Controller watchdog remains enabled."
                raise TimeoutError(msg)
            self._thread = None
            with self._lock:
                self._pending = None
                self._observation = None
                self._ready = False
            if self._fault is not None:
                msg = f"RTDE session stopped with an error: {self._fault}"
                raise RuntimeError(msg) from self._fault

    def is_connected(self) -> bool:
        """Return whether the owner has fresh state and no latched fault."""
        with self._lock:
            return (
                self._ready
                and self._fault is None
                and not self._stop.is_set()
                and self._observation is not None
                and time.monotonic() - self._observation.timestamp <= self._settings.command_timeout
            )

    def _require_observation(self) -> URObservation:
        # Called with the state lock held; all public reads return copies.
        if self._fault is not None:
            msg = f"RTDE fault: {self._fault}. Disconnect and inspect the robot before reconnecting."
            raise RuntimeError(msg) from self._fault
        if not self._ready or self._stop.is_set() or self._observation is None:
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)
        if time.monotonic() - self._observation.timestamp > self._settings.command_timeout:
            self._stop.set()
            msg = "RTDE state is stale."
            raise ConnectionError(msg)
        return self._observation

    def get_observation(self) -> URObservation:
        """Return a fresh position-only observation, in degrees."""
        with self._lock:
            observation = self._require_observation()
            return URObservation(observation.joint_positions.copy(), observation.timestamp)

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Queue exactly six finite degree targets; reject rather than clip.

        Controller-side safety checks happen asynchronously in the owner. Any
        rejection latches a fault, stops servoing, and is surfaced on the next
        API call. This method never calls the SDK from the caller's thread.
        State validation raises ConnectionError when disconnected/stale, or
        RuntimeError for latched controller/communication faults.

        Raises:
            ValueError: The target, shape, or goal time violates configured bounds.
        """
        positions = np.asarray(action, dtype=np.float64)
        if positions.shape != (6,) or not np.isfinite(positions).all():
            msg = "Action must have shape (6,) with finite joint positions in degrees."
            raise ValueError(msg)
        if not np.isfinite(goal_time) or goal_time <= 0:
            msg = "goal_time must be finite and positive."
            raise ValueError(msg)
        target = np.deg2rad(positions)
        if np.any(target < self._lower) or np.any(target > self._upper):
            msg = "Action exceeds the configured model's joint limits."
            raise ValueError(msg)
        with self._lock:
            observation = self._require_observation()
            if np.any(np.abs(positions - observation.joint_positions) > self._settings.max_relative_target):
                msg = "Action exceeds max_relative_target from the measured position."
                raise ValueError(msg)
            self._pending = _Target(target.copy(), goal_time, time.monotonic())

    def _record_fault(self, error: Exception) -> None:
        with self._lock:
            if self._fault is None:
                self._fault = error
            self._ready = False
        logger.error("UR RTDE session at {} stopped: {}", self.ip, error)

    def _run(self) -> None:
        session = _Session()
        try:
            self._open(session)
            self._stream(session)
        except Exception as error:  # ruff: ignore[BLE001] -- latch every failure before stopping/cleanup.
            self._record_fault(error)
        finally:
            with self._lock:
                self._ready = False
            if session.control is not None:
                self._cleanup(session.control, session.receive, servo_active=session.servo_active)
            elif session.receive is not None:
                try:
                    session.receive.disconnect()
                except Exception as error:  # ruff: ignore[BLE001] -- report partial startup cleanup failure.
                    self._record_fault(error)
            self._started.set()

    def _open(self, session: _Session) -> None:
        # Vendor bindings stay out of package import, catalog building, and config export.
        from rtde_control import RTDEControlInterface  # ruff: ignore[PLC0415]
        from rtde_receive import RTDEReceiveInterface  # ruff: ignore[PLC0415]

        settings = self._settings
        flags = (
            RTDEControlInterface.FLAGS_DEFAULT
            if settings.control_mode == "remote"
            else RTDEControlInterface.FLAG_USE_EXT_UR_CAP
        )
        session.receive = RTDEReceiveInterface(self.ip, settings.frequency)
        session.control = RTDEControlInterface(
            self.ip,
            settings.frequency,
            flags=flags,
            ur_cap_port=settings.external_control_port,
        )
        _checked(session.control.setWatchdog(1.0 / settings.command_timeout), "setWatchdog")
        session.previous_cycle = session.fresh_at = session.watchdog_at = time.monotonic()

    def _stream(self, session: _Session) -> None:
        control, receive = session.control, session.receive
        if control is None or receive is None:
            msg = "Internal RTDE session was not initialized."
            raise RuntimeError(msg)
        while not self._stop.is_set():
            cycle = control.initPeriod()
            now = time.monotonic()
            measured = self._sample(session, control, receive, now)
            with self._lock:
                self._observation = URObservation(np.rad2deg(measured), session.fresh_at)
                self._ready = True
                pending, self._pending = self._pending, None
            self._started.set()
            if pending is not None:
                self._set_target(session, control, measured, pending)
            if self._stop.is_set():
                break
            self._servo(session, control, measured, time.monotonic())
            if now - session.watchdog_at >= self._settings.command_timeout / 4.0:
                _checked(control.kickWatchdog(), "kickWatchdog")
                session.watchdog_at = now
            control.waitPeriod(cycle)

    def _sample(
        self,
        session: _Session,
        control: RTDEControlInterface,
        receive: RTDEReceiveInterface,
        now: float,
    ) -> np.ndarray:
        if now - session.previous_cycle > self._settings.command_timeout:
            msg = "RTDE servo loop missed its deadline."
            raise TimeoutError(msg)
        session.previous_cycle = now
        measured = self._read_state(control, receive)
        stamp = float(receive.getTimestamp())
        if not np.isfinite(stamp):
            msg = "Invalid controller timestamp."
            raise RuntimeError(msg)
        if session.controller_time is not None and stamp < session.controller_time:
            msg = "Controller timestamp moved backwards."
            raise RuntimeError(msg)
        if stamp != session.controller_time:
            session.controller_time, session.fresh_at = stamp, now
        if now - session.fresh_at > self._settings.command_timeout:
            msg = "RTDE receive stream is stale."
            raise TimeoutError(msg)
        return measured

    def _set_target(
        self,
        session: _Session,
        control: RTDEControlInterface,
        measured: np.ndarray,
        pending: _Target,
    ) -> None:
        if np.any(np.abs(pending.positions - measured) > self._max_offset):
            msg = "Queued target is too far from the measured position."
            raise RuntimeError(msg)
        _checked(control.isJointsWithinSafetyLimits(pending.positions.tolist()), "target safety check")
        start = measured.copy() if session.command is None else session.command.copy()
        duration = max(pending.goal_time, float(np.max(np.abs(pending.positions - start) / self._velocity)))
        session.target = pending
        session.trajectory = _Trajectory(start, pending.positions, duration)

    def _servo(self, session: _Session, control: RTDEControlInterface, measured: np.ndarray, now: float) -> None:
        if session.target is None or session.trajectory is None:
            return
        if now - session.target.accepted_at > self._settings.command_timeout:
            msg = "No fresh action arrived before command_timeout."
            raise TimeoutError(msg)
        dt = 1.0 / self._settings.frequency
        session.command = session.trajectory.step(dt)
        if np.any(np.abs(session.command - measured) > self._max_offset):
            msg = "Robot tracking error exceeds max_relative_target."
            raise RuntimeError(msg)
        session.servo_active = True
        _checked(
            control.servoJ(
                session.command.tolist(),
                0.0,
                0.0,
                dt,
                self._settings.lookahead_time,
                self._settings.gain,
            ),
            "servoJ",
        )

    @staticmethod
    def _read_state(control: RTDEControlInterface, receive: RTDEReceiveInterface) -> np.ndarray:
        if not control.isConnected() or not receive.isConnected() or not control.isProgramRunning():
            msg = "RTDE connection or control program stopped."
            raise ConnectionError(msg)
        # UR safety modes 1 and 2 are normal/reduced; robot mode 7 is running.
        if receive.getSafetyMode() not in _SAFE_MODES or receive.getRobotMode() != _ROBOT_MODE_RUNNING:
            msg = "Robot is not running in normal/reduced safety mode; inspect the teach pendant."
            raise RuntimeError(msg)
        positions = np.asarray(receive.getActualQ(), dtype=np.float64)
        if positions.shape != (6,) or not np.isfinite(positions).all():
            msg = "RTDE returned invalid joint positions."
            raise RuntimeError(msg)
        return positions

    def _cleanup(
        self,
        control: RTDEControlInterface,
        receive: RTDEReceiveInterface | None,
        *,
        servo_active: bool,
    ) -> None:
        # Stop and close independently so a rejected stop cannot skip socket cleanup.
        operations = []
        if servo_active:
            operations.append(("servoStop", control.servoStop, True))
        # The Python binding's stopScript returns None, unlike servoStop's bool.
        operations.extend((
            ("stopScript", control.stopScript, False),
            ("control disconnect", control.disconnect, False),
        ))
        if receive is not None:
            operations.append(("receive disconnect", receive.disconnect, False))
        for name, operation, check in operations:
            try:
                result = operation()
                if check:
                    _checked(result, name)
            except Exception as error:  # ruff: ignore[BLE001] -- attempt every remaining cleanup operation.
                self._record_fault(error)
