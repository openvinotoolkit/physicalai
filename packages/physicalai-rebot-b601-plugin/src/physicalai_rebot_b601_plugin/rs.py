"""RobStride motor driver for the reBot B601 robot arm.

Uses the ``motorbridge`` SDK to communicate with RobStride RS-series motors
over SocketCAN. All joints run in MIT mode; the gripper uses impedance
control with velocity-limited torque output.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
from loguru import logger

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
    from motorbridge import Motor
    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import RobotObservation

from motorbridge import Controller, Mode

ReBotRSCanAdapter = Literal["socketcan", "robstride"]
ReBotRole = Literal["follower"]


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
        gripper_mit_kp: float = 12.0,
        gripper_mit_kd: float = 0.05,
        gripper_mit_torque_limit: float = 10.0,
    ) -> None:
        """Initialize the RobStride motor driver.

        Args:
            port: SocketCAN channel (e.g. ``"can0"``).
            can_adapter: ``"socketcan"`` for native SocketCAN; ``"robstride"`` raises.
            role: Currently only ``"follower"`` is supported.
            disable_torque_on_disconnect: Whether to disable all motors on disconnect.
            gripper_mit_kp: MIT stiffness gain for gripper impedance control.
            gripper_mit_kd: MIT damping gain for gripper impedance control.
            gripper_mit_torque_limit: Maximum torque (N·m) for gripper impedance.

        Raises:
            ValueError: If any parameter has an invalid value.
        """
        if role not in VALID_ROLES:
            msg = f"Invalid role {role!r}. ReBotB601RS currently supports only {sorted(VALID_ROLES)}."
            raise ValueError(msg)
        if can_adapter not in VALID_RS_CAN_ADAPTERS:
            msg = f"Invalid can_adapter {can_adapter!r}. Must be one of {sorted(VALID_RS_CAN_ADAPTERS)}."
            raise ValueError(msg)
        if gripper_mit_kp < 0.0 or gripper_mit_kd < 0.0 or gripper_mit_torque_limit < 0.0:
            msg = "gripper MIT gains and torque limit must be non-negative."
            raise ValueError(msg)

        self._port = port
        self._can_adapter = can_adapter
        self._role = role
        self._disable_torque_on_disconnect = disable_torque_on_disconnect
        self._gripper_mit_kp = gripper_mit_kp
        self._gripper_mit_kd = gripper_mit_kd
        self._gripper_mit_torque_limit = gripper_mit_torque_limit
        self._controller: Controller | None = None
        self._motors: dict[str, Motor] = {}
        self._gripper_prev_target_pos: float | None = None
        self._gripper_prev_filtered_target_vel: float | None = None

    @property
    def joint_names(self) -> list[str]:
        """Ordered list of joint names matching the expected action/observation layout."""
        return self.JOINT_ORDER

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

    def get_observation(self) -> RobotObservation:
        """Read joint positions, velocities, torques, and temperatures from all motors.

        Returns:
            A ``ReBotB601RSObservation`` with joint positions in degrees and
            sensor data arrays for velocities, torques, temperatures, and status codes.

        Raises:
            ConnectionError: If the robot is not connected.
        """
        if not self.is_connected():
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)
        controller = self._require_controller()

        for motor in self._motors.values():
            motor.request_feedback()
        controller.poll_feedback_once()

        positions = np.empty(self.NUM_JOINTS, dtype=np.float32)
        velocities = np.empty(self.NUM_JOINTS, dtype=np.float32)
        torques = np.empty(self.NUM_JOINTS, dtype=np.float32)
        mos_temperatures = np.empty(self.NUM_JOINTS, dtype=np.float32)
        rotor_temperatures = np.empty(self.NUM_JOINTS, dtype=np.float32)
        status_codes = np.empty(self.NUM_JOINTS, dtype=np.int32)

        for i, name in enumerate(self.JOINT_ORDER):
            state = self._motors[name].get_state()
            if state is None:
                msg = f"No feedback received for motor '{name}'"
                raise ConnectionError(msg)
            positions[i] = math.degrees(float(state.pos))
            velocities[i] = math.degrees(float(state.vel))
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

        for i, name in enumerate(self.JOINT_ORDER):
            target_deg = self._map_and_clip_action(name, float(action[i]))
            target_rad = math.radians(target_deg)
            motor = self._motors[name]

            if name == "gripper":
                tau = self._gripper_output_torque(target_rad)
                motor.send_mit(0.0, 0.0, 0.0, 1.5, tau)
            else:
                motor.send_mit(target_rad, 0.0, REBOT_B601_RS_MIT_KP[name], REBOT_B601_RS_MIT_KD[name], 0.0)

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

        impedance_torque = self._gripper_mit_kp * (target_rad - state.pos) + self._gripper_mit_kd * (
            target_vel - state.vel
        )
        return float(np.clip(impedance_torque, -self._gripper_mit_torque_limit, self._gripper_mit_torque_limit))
