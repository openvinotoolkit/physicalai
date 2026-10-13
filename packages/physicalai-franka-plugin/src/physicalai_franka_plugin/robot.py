# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Franka Research 3 driver for PhysicalAI.

Controls the 7-DOF FR3 arm over the Franka Control Interface (FCI) through
the official ``pylibfranka`` Python bindings. A background 1 kHz control loop
reads the robot state and streams interpolated joint position commands toward
the latest target submitted by :meth:`FrankaFR3.send_action`.
"""

# The whole module operates against the untyped ``pylibfranka`` pybind11
# binding, so ``Any`` is the honest annotation at that boundary.
# ruff: file-ignore[any-type]

from __future__ import annotations

import contextlib
import importlib
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from loguru import logger

from physicalai.config import export_config
from physicalai_franka_plugin.constants import (
    FR3_GRIPPER_DEFAULT_SPEED_M_S,
    FR3_GRIPPER_WIDTH_RANGE_M,
    FR3_JOINT_LIMITS_RAD,
    FR3_JOINT_ORDER,
    VALID_REALTIME_CONFIGS,
)

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import RobotObservation

_GRIPPER_READ_INTERVAL_S = 1.0
_GRIPPER_COMMAND_EPSILON_M = 5e-4
_DISCONNECT_JOIN_TIMEOUT_S = 5.0
_FIRST_OBSERVATION_TIMEOUT_S = 5.0


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


@export_config(class_path="physicalai_franka_plugin.robot.FrankaFR3")
class FrankaFR3:
    """Driver for the Franka Research 3 robot arm.

    Connects to the robot controller over the FCI and runs a 1 kHz joint
    position control loop in a background thread. Positions are exposed in
    degrees at the driver boundary; the Franka Hand gripper, when enabled, is
    commanded by width in meters as an extra action element.

    Args:
        host: IP address or hostname of the robot controller.
        gripper_enabled: Whether a Franka Hand gripper is attached and the
            action/observation vectors include a gripper element.
        realtime_config: ``"kEnforce"`` for real-time scheduling requirements
            (production) or ``"kIgnore"`` for testing/development.
        goal_time: Default time in seconds to reach a commanded target.
        auto_recover_on_error: Attempt automatic error recovery during
            ``connect`` when the robot reports recoverable errors.
        gripper_speed_m_s: Gripper movement speed in meters per second.

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

        self._host = host
        self._gripper_enabled = gripper_enabled
        self._realtime_config = realtime_config
        self._goal_time = goal_time
        self._auto_recover_on_error = auto_recover_on_error
        self._gripper_speed_m_s = gripper_speed_m_s

        self._pylibfranka: Any = None
        self._robot: Any = None
        self._gripper: Any = None
        self._control: Any = None
        self._loop_thread: threading.Thread | None = None
        self._stop_loop = threading.Event()
        self._loop_alive = False

        self._lock = threading.Lock()
        self._target: np.ndarray = np.zeros(self.NUM_JOINTS, dtype=np.float64)
        self._commanded: np.ndarray = np.zeros(self.NUM_JOINTS, dtype=np.float64)
        self._latest_observation: FrankaFRObservation | None = None
        self._last_gripper_width: float | None = None
        self._last_gripper_read_monotonic = 0.0
        self._first_observation = threading.Event()

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
        """Default time in seconds to reach a commanded target."""
        return self._goal_time

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

    def _load_pylibfranka(self) -> Any:
        if self._pylibfranka is None:
            try:
                self._pylibfranka = importlib.import_module("pylibfranka")
            except ImportError as exc:
                msg = (
                    "pylibfranka is required to control a Franka FR3. Install it with: "
                    "pip install 'physicalai-franka-plugin[franka]'. Note: pylibfranka ships "
                    "wheels for Python 3.9-3.12 only; use Python <= 3.12 to control the robot."
                )
                raise ImportError(msg) from exc
        return self._pylibfranka

    def connect(self) -> None:
        """Connect to the robot, recover from errors, and start the 1 kHz control loop.

        Blocks until the first robot state has been received.

        Raises:
            ConnectionError: If no robot state arrives within a timeout.
        """
        if self.is_connected():
            return
        self._first_observation.clear()
        pylibfranka = self._load_pylibfranka()
        robot, gripper, control, initial = self._establish_connection(pylibfranka)

        with self._lock:
            self._robot = robot
            self._gripper = gripper
            self._control = control
            self._target = np.asarray(initial.q, dtype=np.float64).copy()
            self._commanded = np.asarray(initial.q, dtype=np.float64).copy()
            if gripper is not None:
                with contextlib.suppress(Exception):
                    self._last_gripper_width = float(gripper.read_once().width)
            else:
                self._last_gripper_width = None
            self._loop_alive = True
        self._stop_loop.clear()
        self._loop_thread = threading.Thread(
            target=self._run_control_loop,
            name="franka-fr3-control-loop",
            daemon=True,
        )
        self._loop_thread.start()
        if not self._first_observation.wait(timeout=_FIRST_OBSERVATION_TIMEOUT_S):
            self._stop_loop.set()
            with contextlib.suppress(Exception):
                robot.stop()
            thread = self._loop_thread
            if thread is not None:
                thread.join(timeout=_DISCONNECT_JOIN_TIMEOUT_S)
            with self._lock:
                self._loop_alive = False
                self._robot = None
                self._control = None
                self._loop_thread = None
            msg = f"FrankaFR3 did not deliver an initial state within {_FIRST_OBSERVATION_TIMEOUT_S}s"
            raise ConnectionError(msg)
        logger.info("FrankaFR3 connected to {}", self._host)

    def _establish_connection(self, pylibfranka: Any) -> tuple[Any, Any, Any, Any]:
        robot = pylibfranka.Robot(self._host, getattr(pylibfranka.RealtimeConfig, self._realtime_config))
        try:
            initial = self._read_initial_state(robot)
            gripper = self._connect_gripper(pylibfranka)
            control = robot.start_joint_position_control(pylibfranka.ControllerMode.JointImpedance)
        except Exception:
            with contextlib.suppress(Exception):
                robot.stop()
            raise
        return robot, gripper, control, initial

    def _read_initial_state(self, robot: Any) -> Any:
        initial = robot.read_once()
        if self._auto_recover_on_error and initial.current_errors:
            robot.automatic_error_recovery()
            initial = robot.read_once()
        return initial

    def _connect_gripper(self, pylibfranka: Any) -> Any:
        if not self._gripper_enabled:
            return None
        try:
            return pylibfranka.Gripper(self._host)
        except Exception:  # ruff: ignore[blind-except] - a missing gripper must not block arm control
            logger.warning("Franka Hand gripper unavailable at {}; continuing arm-only", self._host)
            return None

    def disconnect(self) -> None:
        """Stop the control loop and leave the robot holding in a safe state."""
        with self._lock:
            robot = self._robot
            gripper = self._gripper
        if robot is None:
            return

        self._stop_loop.set()
        with contextlib.suppress(Exception):
            robot.stop()
        thread = self._loop_thread
        if thread is not None:
            thread.join(timeout=_DISCONNECT_JOIN_TIMEOUT_S)
        if gripper is not None:
            with contextlib.suppress(Exception):
                gripper.stop()

        with self._lock:
            self._robot = None
            self._gripper = None
            self._control = None
            self._loop_thread = None
            self._latest_observation = None
        logger.info("FrankaFR3 disconnected from {}", self._host)

    def is_connected(self) -> bool:
        """Return whether the control loop is active and accepting commands."""
        with self._lock:
            return self._robot is not None and self._loop_alive

    def recover(self) -> None:
        """Attempt automatic error recovery on the connected robot.

        Raises:
            ConnectionError: If the robot is not connected.
        """
        robot = self._robot
        if robot is None:
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)
        robot.automatic_error_recovery()

    def get_observation(self) -> RobotObservation:
        """Return the latest state captured by the control loop.

        Returns:
            A ``FrankaFRObservation`` with arm joint positions in degrees and,
            when enabled, the gripper width in meters as the final element.

        Raises:
            ConnectionError: If the robot is not connected or no state has been received yet.
        """
        with self._lock:
            observation = self._latest_observation
        if observation is None:
            msg = "Robot is not connected or no state has been received yet. Call connect() first."
            raise ConnectionError(msg)
        return observation

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Publish a joint position target to the control loop.

        The arm joints are interpreted in degrees and clipped to the FR3 joint
        limits. When the gripper is enabled, an extra width element in meters
        commands the Franka Hand whenever it changes materially.

        Args:
            action: Array of arm joint targets in degrees, plus the gripper
                width in meters when ``gripper_enabled``.
            goal_time: Time in seconds to reach the targets.

        Raises:
            ConnectionError: If the robot is not connected.
            ValueError: If the action shape does not match the joint layout.
        """
        if not self.is_connected():
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)
        if goal_time <= 0.0:
            msg = "goal_time must be positive."
            raise ValueError(msg)
        action = np.asarray(action, dtype=np.float64)
        expected_shape = (len(self.joint_names),)
        if action.shape != expected_shape:
            msg = f"Expected action shape {expected_shape}, got {action.shape}"
            raise ValueError(msg)

        arm_target_deg = self._clip_to_limits(action[: self.NUM_JOINTS])
        with self._lock:
            self._target = np.radians(arm_target_deg)
            self._goal_time = goal_time
        if self._gripper_enabled and action.shape[0] > self.NUM_JOINTS:
            self._command_gripper(float(action[self.NUM_JOINTS]))

    def _clip_to_limits(self, action_deg: np.ndarray) -> np.ndarray:
        clipped = np.empty_like(action_deg)
        for i, name in enumerate(self.JOINT_ORDER):
            low_deg = float(np.degrees(FR3_JOINT_LIMITS_RAD[name][0]))
            high_deg = float(np.degrees(FR3_JOINT_LIMITS_RAD[name][1]))
            clipped[i] = float(np.clip(action_deg[i], low_deg, high_deg))
        return clipped

    def _command_gripper(self, width_m: float) -> None:
        width_m = float(np.clip(width_m, *FR3_GRIPPER_WIDTH_RANGE_M))
        with self._lock:
            gripper = self._gripper
            last_width = self._last_gripper_width
        if gripper is None:
            return
        if last_width is None or abs(width_m - last_width) > _GRIPPER_COMMAND_EPSILON_M:
            with contextlib.suppress(Exception):
                gripper.move(width_m, self._gripper_speed_m_s)
                with self._lock:
                    self._last_gripper_width = width_m

    def _run_control_loop(self) -> None:
        pylibfranka = self._pylibfranka
        with self._lock:
            control = self._control
            commanded = self._commanded.copy()
        try:
            while not self._stop_loop.is_set():
                commanded = self._step_control_loop(control, pylibfranka, commanded)
        except Exception:  # ruff: ignore[blind-except] - the loop must terminate cleanly on any FCI error
            if not self._stop_loop.is_set():
                logger.exception("Franka FR3 control loop terminated unexpectedly")
        finally:
            with self._lock:
                self._loop_alive = False

    def _step_control_loop(self, control: Any, pylibfranka: Any, commanded: np.ndarray) -> np.ndarray:
        state, duration = control.readOnce()
        dt = max(1e-6, float(duration.to_sec()))
        with self._lock:
            target = self._target.copy()
            goal_time = self._goal_time
        step_frac = min(1.0, dt / goal_time)
        commanded += (target - commanded) * step_frac
        control.writeOnce(pylibfranka.JointPositions(commanded))
        self._cache_observation(state)
        return commanded

    def _cache_observation(self, state: Any) -> None:
        gripper_width: float | None = None
        now = time.monotonic()
        if self._gripper_enabled:
            with self._lock:
                gripper = self._gripper
                last_read = self._last_gripper_read_monotonic
            if gripper is not None and now - last_read >= _GRIPPER_READ_INTERVAL_S:
                with contextlib.suppress(Exception):
                    gripper_width = float(gripper.read_once().width)
                with self._lock:
                    self._last_gripper_read_monotonic = now
        observation = self._build_observation(state, gripper_width=gripper_width)
        with self._lock:
            if gripper_width is not None:
                self._last_gripper_width = gripper_width
            self._latest_observation = observation
            self._first_observation.set()

    def _build_observation(self, state: Any, *, gripper_width: float | None = None) -> FrankaFRObservation:
        positions = np.degrees(np.asarray(state.q, dtype=np.float64))
        if self._gripper_enabled:
            width = gripper_width if gripper_width is not None else self._last_gripper_width
            if width is None:
                width = FR3_GRIPPER_WIDTH_RANGE_M[0]
            positions = np.concatenate([positions, np.asarray([width])])
        return FrankaFRObservation(
            joint_positions=positions.astype(np.float32),
            timestamp=time.monotonic(),
            sensor_data={
                "velocities": np.degrees(np.asarray(state.dq, dtype=np.float64)).astype(np.float32),
                "torques": np.asarray(state.tau_J, dtype=np.float64).astype(np.float32),
                "external_torques": np.asarray(state.tau_ext_hat_filtered, dtype=np.float64).astype(np.float32),
            },
        )
