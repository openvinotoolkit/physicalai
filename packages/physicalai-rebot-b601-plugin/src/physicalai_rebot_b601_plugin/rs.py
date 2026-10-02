# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RobStride motor driver for the reBot B601 robot arm.

Uses the ``motorbridge`` SDK to communicate with RobStride RS-series motors
over SocketCAN. All joints run in MIT mode; the gripper uses impedance
control with velocity-limited torque output.

Observations and actions share one joint frame: ``send_action`` multiplies
each target by ``REBOT_B601_RS_JOINT_DIRECTIONS`` to reach the motor frame, and
``get_observation`` divides measured motor positions by the same factors. Echoing
an observation back as an action therefore holds the arm in place.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
from loguru import logger

from physicalai.config import export_config
from physicalai_rebot_b601_plugin.constants import (
    REBOT_B601_RS_JOINT_DIRECTIONS,
    REBOT_B601_RS_JOINT_LIMITS_DEG,
    REBOT_B601_RS_JOINT_ORDER,
    REBOT_B601_RS_MIT_KD,
    REBOT_B601_RS_MIT_KP,
    REBOT_B601_RS_MOTOR_IDS,
    REBOT_B601_RS_MOTOR_MODELS,
    VALID_ROLES,
    VALID_RS_CAN_ADAPTERS,
)

if TYPE_CHECKING:
    from motorbridge import Motor, MotorState

    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import RobotObservation

from motorbridge import Controller, Mode

ReBotRSCanAdapter = Literal["socketcan", "robstride"]
ReBotRole = Literal["follower"]
RSMITGain = float | dict[str, float]

# Gripper stall detection, matching the Seeed LeRobot RS follower.
_GRIPPER_STALL_VEL_RAD_S = 0.25
_GRIPPER_OPEN_STALL_ERROR_RAD = 0.05
_GRIPPER_OPEN_STALL_CYCLES = 5


def _validate_mit_gain(gain: RSMITGain | None, name: str) -> None:
    """Validate an MIT gain override for the six position joints.

    Args:
        gain: ``None`` (use defaults), a scalar applied to every position joint, or a
            mapping from position-joint name to gain. Joints omitted from the mapping
            keep their default gain.
        name: Parameter name used in error messages.

    Raises:
        ValueError: If the gain names an unknown joint or any value is negative or not finite.
    """
    if gain is None:
        return
    values = gain if isinstance(gain, dict) else {"*": gain}
    if isinstance(gain, dict):
        unknown = set(gain) - set(REBOT_B601_RS_MIT_KP)
        if unknown:
            msg = f"{name} may only contain position joints {sorted(REBOT_B601_RS_MIT_KP)}; got {sorted(unknown)}"
            raise ValueError(msg)
    for joint, value in values.items():
        if not math.isfinite(value) or value < 0.0:
            label = name if joint == "*" else f"{name}[{joint!r}]"
            msg = f"{label} must be a non-negative finite value, got {value!r}"
            raise ValueError(msg)


@dataclass
class ReBotB601RSObservation:
    """Observation data for the RobStride reBot B601 arm.

    Attributes:
        joint_positions: Measured joint positions in degrees.
        timestamp: Monotonic time of the observation.
        sensor_data: Optional dict of velocity, torque, temperature, and status arrays.
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


@export_config
class ReBotB601RS:
    """RobStride motor driver for the reBot B601 robot arm.

    Controls 7 RS-series motors (6-DOF + gripper) in MIT mode. The 6
    position joints use per-joint stiffness/damping gains; the gripper
    uses impedance control with a velocity-limited torque command.
    """

    JOINT_ORDER: ClassVar[list[str]] = list(REBOT_B601_RS_JOINT_ORDER)
    NUM_JOINTS: ClassVar[int] = len(JOINT_ORDER)

    def __init__(
        self,
        port: str = "can0",
        *,
        can_adapter: ReBotRSCanAdapter = "socketcan",
        role: ReBotRole = "follower",
        disable_torque_on_disconnect: bool = True,
        mit_kp: RSMITGain | None = None,
        mit_kd: RSMITGain | None = None,
        gripper_mit_kp: float = 12.0,
        gripper_mit_kd: float = 0.05,
        gripper_mit_torque_limit: float = 3.5,
        gripper_mit_hold_torque_limit: float = 1.0,
        max_relative_target: float | None = None,
    ) -> None:
        """Initialize the RobStride motor driver.

        Args:
            port: SocketCAN channel (e.g. ``"can0"``).
            can_adapter: ``"socketcan"`` for native SocketCAN; ``"robstride"`` raises.
            role: Currently only ``"follower"`` is supported.
            disable_torque_on_disconnect: Whether to disable all motors on disconnect.
            mit_kp: MIT stiffness for the six position joints: a single value for all of
                them, or a per-joint mapping overriding :data:`REBOT_B601_RS_MIT_KP`.
                Lower stiffness holds more quietly but sags more under load.
            mit_kd: MIT damping for the six position joints, in the same form as ``mit_kp``,
                overriding :data:`REBOT_B601_RS_MIT_KD`.
            gripper_mit_kp: MIT stiffness gain for gripper impedance control.
            gripper_mit_kd: MIT damping gain for gripper impedance control.
            gripper_mit_torque_limit: Maximum torque (N·m) for gripper impedance while moving.
            gripper_mit_hold_torque_limit: Maximum gripper torque (N·m) once it stalls, e.g. on a grasped
                object.
            max_relative_target: Optional maximum allowed change (joint degrees, i.e. the action frame)
                between the current and commanded position per step; limits how far the arm lunges on a
                single command.

        Raises:
            ValueError: If any parameter has an invalid value.
        """
        if role not in VALID_ROLES:
            msg = f"Invalid role {role!r}. ReBotB601RS currently supports only {sorted(VALID_ROLES)}."
            raise ValueError(msg)
        if can_adapter not in VALID_RS_CAN_ADAPTERS:
            msg = f"Invalid can_adapter {can_adapter!r}. Must be one of {sorted(VALID_RS_CAN_ADAPTERS)}."
            raise ValueError(msg)
        gripper_values = (gripper_mit_kp, gripper_mit_kd, gripper_mit_torque_limit, gripper_mit_hold_torque_limit)
        if not all(math.isfinite(value) and value >= 0.0 for value in gripper_values):
            msg = "gripper MIT gains and torque limits must be non-negative finite values."
            raise ValueError(msg)
        if max_relative_target is not None and (not math.isfinite(max_relative_target) or max_relative_target <= 0.0):
            msg = f"max_relative_target must be a finite positive value, got {max_relative_target!r}"
            raise ValueError(msg)
        _validate_mit_gain(mit_kp, "mit_kp")
        _validate_mit_gain(mit_kd, "mit_kd")

        self._port = port
        self._can_adapter = can_adapter
        self._role = role
        self._disable_torque_on_disconnect = disable_torque_on_disconnect
        self._mit_kp = mit_kp
        self._mit_kd = mit_kd
        self._gripper_mit_kp = gripper_mit_kp
        self._gripper_mit_kd = gripper_mit_kd
        self._gripper_mit_torque_limit = gripper_mit_torque_limit
        self._gripper_mit_hold_torque_limit = gripper_mit_hold_torque_limit
        self._max_relative_target = max_relative_target
        self._controller: Controller | None = None
        self._motors: dict[str, Motor] = {}
        self._gripper_prev_target_pos: float | None = None
        self._gripper_prev_filtered_target_vel: float | None = None
        self._gripper_prev_state_pos: float | None = None
        self._gripper_open_stall_count = 0

    @property
    def joint_names(self) -> list[str]:
        """Ordered list of joint names matching the expected action/observation layout."""
        return self.JOINT_ORDER

    @property
    def device_ids(self) -> tuple[str, ...]:
        """Configured CAN transport identity without opening it."""
        return (f"rebot-rs:{self._can_adapter}:{self._port}",)

    @property
    def port(self) -> str:
        """SocketCAN channel the driver is configured for."""
        return self._port

    @property
    def can_adapter(self) -> ReBotRSCanAdapter:
        """CAN adapter type (``"socketcan"`` or ``"robstride"``)."""
        return self._can_adapter

    @property
    def role(self) -> ReBotRole:
        """Role of this driver instance (``"follower"``)."""
        return self._role

    @property
    def max_relative_target(self) -> float | None:
        """Maximum per-step position change in degrees, or ``None`` when unlimited."""
        return self._max_relative_target

    @property
    def disable_torque_on_disconnect(self) -> bool:
        """Whether torque is automatically disabled on disconnect."""
        return self._disable_torque_on_disconnect

    @disable_torque_on_disconnect.setter
    def disable_torque_on_disconnect(self, value: bool) -> None:
        self._disable_torque_on_disconnect = value

    def _require_controller(self) -> Controller:
        controller = self._controller
        if controller is None:
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)
        return controller

    def connect(self) -> None:
        """Open the controller, register motors, and enable MIT mode on all joints."""
        if self.is_connected():
            return

        try:
            controller = self._open_controller()
            self._controller = controller
            self._motors = self._register_motors(controller)
            self._configure_motors()
        except Exception:
            with contextlib.suppress(Exception):
                self._cleanup_connection()
            raise

        logger.info(f"ReBotB601RS connected on {self.port} (adapter={self.can_adapter})")

    def _open_controller(self) -> Controller:
        if self.can_adapter == "robstride":
            msg = "can_adapter='robstride' is not supported by the current MotorBridge Python SDK. Use socketcan."
            raise NotImplementedError(msg)
        return Controller(channel=self.port)

    def _register_motors(self, controller: Controller) -> dict[str, Motor]:
        motors: dict[str, Motor] = {}
        for name in self.JOINT_ORDER:
            motor_id, feedback_id = REBOT_B601_RS_MOTOR_IDS[name]
            motors[name] = controller.add_robstride_motor(motor_id, feedback_id, REBOT_B601_RS_MOTOR_MODELS[name])
        return motors

    def disconnect(self) -> None:
        """Disable torque (if configured), clear errors, close motors, and release the controller."""
        if self._controller is None:
            return

        try:
            self._disconnect_motors()
        finally:
            self._cleanup_connection()

        logger.info(f"ReBotB601RS disconnected from {self.port}")

    def _disconnect_motors(self) -> None:
        if self.disable_torque_on_disconnect and self._controller is not None:
            with contextlib.suppress(Exception):
                self._controller.disable_all()
        for name, motor in self._motors.items():
            with contextlib.suppress(Exception):
                motor.clear_error()
            with contextlib.suppress(Exception):
                motor.close()
                logger.debug(f"Closed reBot RS motor {name}")

    def _cleanup_connection(self) -> None:
        controller = self._controller
        self._controller = None
        self._motors = {}
        if controller is not None:
            with contextlib.suppress(Exception):
                controller.close()

    def is_connected(self) -> bool:
        """Return whether the controller connection is active."""
        return self._controller is not None

    def _configure_motors(self) -> None:
        controller = self._require_controller()
        controller.disable_all()

        for motor in self._motors.values():
            motor.ensure_mode(Mode.MIT)

        controller.enable_all()

    def disable_torque(self) -> None:
        """Disable torque on all motors."""
        self._require_controller().disable_all()

    def enable_torque(self) -> None:
        """Enable torque on all motors."""
        self._require_controller().enable_all()

    def set_zero_position(self) -> None:
        """Store the arm's current pose as zero on every motor.

        Torque is disabled first, as the motors require; call :meth:`enable_torque` to resume control.
        """
        controller = self._require_controller()
        controller.disable_all()
        for motor in self._motors.values():
            motor.set_zero_position()
        self._gripper_prev_target_pos = None
        self._gripper_prev_filtered_target_vel = None
        self._gripper_prev_state_pos = None
        self._gripper_open_stall_count = 0

    def _read_motor_states(self) -> list[MotorState]:
        if not self.is_connected():
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)
        controller = self._require_controller()
        for motor in self._motors.values():
            motor.request_feedback()
        controller.poll_feedback_once()

        states: list[MotorState] = []
        for name in self.JOINT_ORDER:
            state = self._motors[name].get_state()
            if state is None:
                msg = f"No feedback received for motor '{name}'"
                raise ConnectionError(msg)
            states.append(state)
        return states

    def get_observation(self) -> RobotObservation:
        """Read joint positions, velocities, torques, and temperatures from all motors.

        Positions and velocities are reported in the action frame (motor value divided by
        :data:`REBOT_B601_RS_JOINT_DIRECTIONS`), so passing ``joint_positions`` to
        :meth:`send_action` holds the current pose. Torques remain in the motor frame.

        Returns:
            A ``ReBotB601RSObservation`` with joint positions in degrees and
            sensor data arrays for velocities, torques, temperatures, and status codes.

        Raises:
            ConnectionError: If the robot is not connected.
        """
        if not self.is_connected():
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)
        states = self._read_motor_states()

        positions = np.empty(self.NUM_JOINTS, dtype=np.float32)
        velocities = np.empty(self.NUM_JOINTS, dtype=np.float32)
        torques = np.empty(self.NUM_JOINTS, dtype=np.float32)
        mos_temperatures = np.empty(self.NUM_JOINTS, dtype=np.float32)
        rotor_temperatures = np.empty(self.NUM_JOINTS, dtype=np.float32)
        status_codes = np.empty(self.NUM_JOINTS, dtype=np.int32)

        for i, (name, state) in enumerate(zip(self.JOINT_ORDER, states, strict=True)):
            direction = REBOT_B601_RS_JOINT_DIRECTIONS[name]
            positions[i] = math.degrees(float(state.pos)) / direction
            velocities[i] = math.degrees(float(state.vel)) / direction
            torques[i] = float(state.torq)
            mos_temperatures[i] = float(state.t_mos)
            rotor_temperatures[i] = float(state.t_rotor)
            status_codes[i] = int(state.status_code)

        return ReBotB601RSObservation(
            joint_positions=positions,
            timestamp=time.monotonic(),
            sensor_data={
                "velocities": velocities,
                "torques": torques,
                "mos_temperatures": mos_temperatures,
                "rotor_temperatures": rotor_temperatures,
                "status_codes": status_codes,
            },
        )

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Send MIT position commands to all joints.

        The gripper receives an impedance torque command; the other joints
        receive a pure position command with per-joint stiffness/damping.

        Args:
            action: Array of 7 joint position targets in degrees.
            goal_time: Ignored; present for protocol compatibility.

        Raises:
            ConnectionError: If the robot is not connected.
            ValueError: If the action shape does not match ``NUM_JOINTS``.
        """
        _ = goal_time
        if not self.is_connected():
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)
        self._require_controller()
        expected_shape = (self.NUM_JOINTS,)
        if action.shape != expected_shape:
            msg = f"Expected action shape {expected_shape}, got {action.shape}"
            raise ValueError(msg)

        max_relative_target = self._max_relative_target
        present_deg: list[float] | None = None
        if max_relative_target is not None:
            present_deg = [math.degrees(float(s.pos)) for s in self._read_motor_states()]

        for i, name in enumerate(self.JOINT_ORDER):
            target_deg = self._map_and_clip_action(name, float(action[i]))
            if present_deg is not None and max_relative_target is not None:
                # Targets are in the motor frame; scale the action-frame limit to match (gripper is 6x).
                max_step_deg = max_relative_target * abs(REBOT_B601_RS_JOINT_DIRECTIONS[name])
                delta = target_deg - present_deg[i]
                if abs(delta) > max_step_deg:
                    limited = present_deg[i] + math.copysign(max_step_deg, delta)
                    min_deg, max_deg = REBOT_B601_RS_JOINT_LIMITS_DEG[name]
                    target_deg = float(np.clip(limited, min_deg, max_deg))
            target_rad = math.radians(target_deg)
            motor = self._motors[name]

            if name == "gripper":
                tau = self._gripper_output_torque(target_rad)
                motor.send_mit(0.0, 0.0, 0.0, 1.5, tau)
            else:
                motor.send_mit(target_rad, 0.0, self._mit_gain(name, "kp"), self._mit_gain(name, "kd"), 0.0)

    def _mit_gain(self, name: str, which: Literal["kp", "kd"]) -> float:
        """Resolve a position joint's MIT gain from the configured override or defaults.

        Args:
            name: Position-joint name.
            which: ``"kp"`` for stiffness or ``"kd"`` for damping.

        Returns:
            The gain to send for this joint.
        """
        override = self._mit_kp if which == "kp" else self._mit_kd
        defaults = REBOT_B601_RS_MIT_KP if which == "kp" else REBOT_B601_RS_MIT_KD
        if override is None:
            return defaults[name]
        if isinstance(override, dict):
            return override.get(name, defaults[name])
        return override

    @staticmethod
    def _map_and_clip_action(name: str, action_deg: float) -> float:
        mapped = action_deg * REBOT_B601_RS_JOINT_DIRECTIONS[name]
        min_deg, max_deg = REBOT_B601_RS_JOINT_LIMITS_DEG[name]
        return float(np.clip(mapped, min_deg, max_deg))

    def _gripper_output_torque(self, target_rad: float) -> float:
        motor = self._motors["gripper"]
        self._require_controller().poll_feedback_once()
        state = motor.get_state()
        if state is None:
            return 0.0

        control_dt_s = 0.02
        if self._gripper_prev_target_pos is None:
            target_vel = 0.0
        else:
            target_vel = (target_rad - self._gripper_prev_target_pos) / control_dt_s
        self._gripper_prev_target_pos = target_rad

        lpf_alpha = 0.3
        target_vel_max = 3.0
        if self._gripper_prev_filtered_target_vel is None:
            filtered_target_vel = target_vel
        else:
            filtered_target_vel = lpf_alpha * target_vel + (1.0 - lpf_alpha) * self._gripper_prev_filtered_target_vel
        target_vel = float(np.clip(filtered_target_vel, -target_vel_max, target_vel_max))
        self._gripper_prev_filtered_target_vel = target_vel

        if self._gripper_prev_state_pos is None:
            state_vel = 0.0
        else:
            state_vel = (state.pos - self._gripper_prev_state_pos) / control_dt_s
        self._gripper_prev_state_pos = state.pos

        impedance_torque = self._gripper_mit_kp * (target_rad - state.pos) + self._gripper_mit_kd * (
            target_vel - state.vel
        )

        # Opening (positive) against a stop for several cycles: cut torque instead of pushing it.
        stalled = abs(state_vel) < _GRIPPER_STALL_VEL_RAD_S
        if target_rad - state.pos > _GRIPPER_OPEN_STALL_ERROR_RAD and stalled:
            self._gripper_open_stall_count += 1
        else:
            self._gripper_open_stall_count = 0
        if self._gripper_open_stall_count >= _GRIPPER_OPEN_STALL_CYCLES:
            # Warn once per stall; torque stays cut on every later tick until the stall clears.
            if self._gripper_open_stall_count == _GRIPPER_OPEN_STALL_CYCLES:
                logger.warning("reBot RS gripper open-stall detected, cutting torque")
            return 0.0

        # Stalled while closing means an object is grasped: hold with the lower torque limit.
        max_torque = self._gripper_mit_hold_torque_limit if stalled else self._gripper_mit_torque_limit
        return float(np.clip(impedance_torque, -max_torque, max_torque))
