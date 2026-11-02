# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Damiao motor driver for the reBot B601 robot arm.

Uses the ``motorbridge`` SDK to communicate with Damiao DM-series motors over
serial (Damiao CAN adapter) or native SocketCAN. By default, the gripper runs
in FORCE_POS mode and all other joints run in POS_VEL mode.

In MIT mode the motors run a plain PD law, ``tau = kp*(q_ref - q) + kd*(dq_ref - dq) + tau_ff``.
The driver fills the two feedforward terms so the gains only have to correct tracking errors:

* ``tau_ff`` is the gravity torque of the commanded pose (:mod:`physicalai_rebot_b601_plugin.gravity`),
  so the arm does not sag by ``gravity / kp``.
* ``dq_ref`` is the velocity of the commanded targets, so ``kd`` damps tracking errors instead of
  holding the arm back by ``kd / kp`` seconds.

Both variants also ease into the first command after the motors are enabled, so a leader or policy
that starts away from the arm's pose does not make it lunge, and fold the arm into its rest pose
before torque is released on disconnect, so it does not fall.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
from loguru import logger
from motorbridge import Controller, Mode

from physicalai.config import export_config
from physicalai_rebot_b601_plugin._urdf import get_urdf_path
from physicalai_rebot_b601_plugin.constants import (
    REBOT_B601_ARM_JOINT_COUNT,
    REBOT_B601_DM_GRAVITY_URDF,
    REBOT_B601_DM_JOINT_DIRECTIONS,
    REBOT_B601_DM_JOINT_LIMITS_DEG,
    REBOT_B601_DM_JOINT_ORDER,
    REBOT_B601_DM_MIT_KD,
    REBOT_B601_DM_MIT_KP,
    REBOT_B601_DM_MOTOR_IDS,
    REBOT_B601_DM_MOTOR_MODELS,
    REBOT_B601_DM_POS_VEL_DEG_S,
    REBOT_B601_MAX_ACCEL_DEG_S2,
    REBOT_B601_MAX_SPEED_DEG_S,
    REBOT_B601_MIT_GAIN_LIMITS,
    REBOT_B601_PARK_SPEED_DEG_S,
    REBOT_B601_PARK_STAGES,
    REBOT_B601_URDF_ARM_JOINTS,
    VALID_CAN_ADAPTERS,
    VALID_ROLES,
)
from physicalai_rebot_b601_plugin.gravity import GravityModel

ReBotB601DMControlMode = Literal["pos_vel", "mit"]
ReBotB601DMGripperControlMode = Literal["force_pos", "mit"]
DMMITGain = float | dict[str, float]

# Interval between commands used for velocity feedforward. The measured interval is clamped to
# this range so a stalled or bursty caller cannot produce a huge velocity estimate.
_NOMINAL_COMMAND_DT_S = 1.0 / 30.0
_MIN_COMMAND_DT_S = 0.005
_MAX_COMMAND_DT_S = 0.1
# Exponential smoothing of the velocity feedforward (weight of the newest sample).
_VELOCITY_FILTER_ALPHA = 0.5
# Acceleration of the start-up approach and the rest-pose move, in degrees per second squared.
_APPROACH_ACCEL_DEG_S2 = 180.0
# The approach hands over to normal tracking once every joint is this close to its target.
_APPROACH_DONE_DEG = 0.5
# A command that would move an arm joint faster than this (and by more than the minimum below)
# cannot be a real motion: a policy restarted, a leader was swapped, or a chunk boundary jumped.
# The driver eases into it with the start-up approach instead of snapping to it. The interval is
# capped, so a command after a pause in the stream cannot jump further than at 20 Hz.
_JUMP_SPEED_DEG_S = 480.0
_JUMP_MAX_DT_S = 0.05
_JUMP_MIN_DEG = 10.0
# The motors have stopped driving (lost power, or reset after a supply dropout) when every arm joint
# reports less torque than this while the arm is this far from its command, for this many reads.
_DROPOUT_TORQUE_NM = 0.05
_DROPOUT_ERROR_DEG = 5.0
_DROPOUT_READS = 10
# Reading the pose to hold right after enabling torque: attempts and the pause between them.
_HOLD_READ_ATTEMPTS = 3
_HOLD_READ_RETRY_S = 0.01
# Rest-pose (park) move: command rate, minimum duration per stage, and hold time at the end.
_PARK_RATE_HZ = 50.0
_PARK_MIN_STAGE_S = 1.0
_PARK_SETTLE_S = 0.3
# Soft release at the end of the park move (MIT mode): fade stiffness and gravity feedforward over
# this long, then let the arm settle on damping alone before torque is released:
_RELEASE_FADE_S = 1.0
# ...then wait until every arm joint has been slower than _RELEASE_STILL_DEG_S for _RELEASE_STILL_S,
# for at most _RELEASE_SETTLE_MAX_S.
_RELEASE_STILL_DEG_S = 2.0
_RELEASE_STILL_S = 0.3
_RELEASE_SETTLE_MAX_S = 4.0
# Abort the park move if a joint lags its command by more than this (e.g. a collision).
_PARK_MAX_ERROR_DEG = 25.0


def _validate_per_joint_gains(gains: dict[str, float], name: str) -> None:
    """Ensure a per-joint gain dict covers exactly the expected joint set.

    Args:
        gains: Mapping from joint name to gain value.
        name: Human-readable gain name used in error messages.

    Raises:
        ValueError: If ``gains`` is missing/extra a joint or any value is invalid.
    """
    missing = set(REBOT_B601_DM_JOINT_ORDER) - set(gains)
    extra = set(gains) - set(REBOT_B601_DM_JOINT_ORDER)
    if missing or extra:
        msg = f"{name} must cover exactly {list(REBOT_B601_DM_JOINT_ORDER)}; got {sorted(gains)}"
        raise ValueError(msg)
    for joint, value in gains.items():
        if not math.isfinite(value) or value < 0.0:
            msg = f"{name}[{joint!r}] must be a non-negative finite value, got {value!r}"
            raise ValueError(msg)


def _normalize_max_relative_target(value: float | None) -> float | None:
    """Normalize the disabled sentinel and validate an optional step limit.

    Returns:
        ``None`` when disabled, otherwise the positive finite limit.

    Raises:
        ValueError: If the value is neither ``-1``, ``None``, nor finite and positive.
    """
    if value is None:
        return None
    if not math.isfinite(value) or value < -1.0 or -1.0 < value <= 0.0:
        msg = f"max_relative_target must be -1, None, or a finite positive value, got {value!r}"
        raise ValueError(msg)
    return None if value <= -1.0 else value


def _normalize_speed(value: float | None, name: str) -> float | None:
    """Validate an optional joint speed in degrees per second; ``None`` or ``-1`` disables it.

    Returns:
        ``None`` when disabled, otherwise the positive finite speed.

    Raises:
        ValueError: If the value is neither ``None``, ``-1``, nor finite and positive.
    """
    if value is None:
        return None
    if not math.isfinite(value) or value < -1.0 or -1.0 < value <= 0.0:
        msg = f"{name} must be -1, None, or a finite positive speed in degrees per second, got {value!r}"
        raise ValueError(msg)
    return None if value <= -1.0 else value


def _minimum_jerk(s: float) -> float:
    """Minimum-jerk position profile on ``s`` in ``[0, 1]``; zero velocity and acceleration at both ends.

    Returns:
        The fraction of the move completed at ``s``.
    """
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


if TYPE_CHECKING:
    from motorbridge import Motor, MotorState

    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import RobotObservation

ReBotCANAdapter = Literal["damiao", "socketcan"]
ReBotRole = Literal["follower"]


@dataclass
class ReBotB601DMObservation:
    """Observation data for the Damiao reBot B601 arm.

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
class ReBotB601DM:
    """Damiao motor driver for the reBot B601 robot arm.

    Controls 7 DM-series motors (6-DOF + gripper). The position joints run in
    POS_VEL mode (default) or MIT mode, and the gripper runs in FORCE_POS mode
    (default) or MIT mode. MIT mode commands a target position with
    per-joint stiffness/damping gains so the motors drive to the target using
    available torque instead of a fixed velocity ceiling, giving much snappier
    motion than the velocity-limited POS_VEL mode. In MIT mode the driver adds
    gravity and velocity feedforward (see the module docstring).
    """

    JOINT_ORDER: ClassVar[list[str]] = list(REBOT_B601_DM_JOINT_ORDER)
    NUM_JOINTS: ClassVar[int] = len(JOINT_ORDER)
    DEVICE_PREFIX: ClassVar[str] = "rebot-dm"
    MOTOR_IDS: ClassVar = REBOT_B601_DM_MOTOR_IDS
    MOTOR_MODELS: ClassVar = REBOT_B601_DM_MOTOR_MODELS
    JOINT_DIRECTIONS: ClassVar = REBOT_B601_DM_JOINT_DIRECTIONS
    JOINT_LIMITS_DEG: ClassVar = REBOT_B601_DM_JOINT_LIMITS_DEG
    OBSERVATION_CLASS: ClassVar = ReBotB601DMObservation
    VALID_CAN_ADAPTERS: ClassVar = VALID_CAN_ADAPTERS
    GRAVITY_URDF: ClassVar[str] = REBOT_B601_DM_GRAVITY_URDF

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        *,
        can_adapter: ReBotCANAdapter = "damiao",
        dm_serial_baud: int = 921600,
        role: ReBotRole = "follower",
        disable_torque_on_disconnect: bool = True,
        force_pos_torque_ratio: float = 0.1,
        control_mode: ReBotB601DMControlMode = "pos_vel",
        gripper_control_mode: ReBotB601DMGripperControlMode = "force_pos",
        mit_kp: DMMITGain | None = None,
        mit_kd: DMMITGain | None = None,
        max_relative_target: float | None = None,
        gravity_compensation: bool = True,
        velocity_feedforward: bool = True,
        approach_speed: float | None = 30.0,
        park_on_disconnect: bool = True,
    ) -> None:
        """Initialize the Damiao motor driver.

        Args:
            port: Serial port (``/dev/ttyACM0``) or SocketCAN channel (``can0``).
            can_adapter: ``"damiao"`` for Damiao USB-CAN, ``"socketcan"`` for native SocketCAN.
            dm_serial_baud: Baud rate for Damiao serial adapter.
            role: Currently only ``"follower"`` is supported.
            disable_torque_on_disconnect: Whether to disable all motors on disconnect.
            force_pos_torque_ratio: FORCE_POS torque ratio in ``[0, 1]`` for gripper.
            control_mode: ``"pos_vel"`` (default) for velocity-limited position control, or
                ``"mit"`` for stiffness-controlled position on the 6-DOF joints.
            gripper_control_mode: ``"force_pos"`` (default) for force-limited position control,
                or ``"mit"`` for impedance-controlled gripper control.
            mit_kp: MIT stiffness gain, either a single value for all joints or a per-joint dict;
                defaults to :data:`REBOT_B601_DM_MIT_KP` when ``None``.
            mit_kd: MIT damping gain, either a single value for all joints or a per-joint dict;
                defaults to :data:`REBOT_B601_DM_MIT_KD` when ``None``.
            max_relative_target: Optional maximum allowed change (degrees) between the current and
                commanded position per step; limits how far the arm lunges on a single command.
            gravity_compensation: In MIT mode, send the gravity torque of the commanded pose as
                feedforward so the arm does not sag. Ignored in POS_VEL mode.
            velocity_feedforward: In MIT mode, send the velocity of the commanded targets so the
                damping gain does not hold the arm back while it follows a moving target.
            approach_speed: Peak joint speed (degrees per second) for the first move after the
                motors are enabled: the arm eases from its measured pose to the commanded one,
                then tracks normally. ``None`` or ``-1`` jumps straight to the first command.
            park_on_disconnect: Move the arm into its folded rest (zero) pose at
                ``REBOT_B601_PARK_SPEED_DEG_S`` before disconnecting, so it does not fall when torque
                is released. Skipped while torque is disabled.

        Raises:
            ValueError: If any parameter has an invalid value.
        """
        if role not in VALID_ROLES:
            msg = f"Invalid role {role!r}. ReBotB601DM currently supports only {sorted(VALID_ROLES)}."
            raise ValueError(msg)
        if can_adapter not in self.VALID_CAN_ADAPTERS:
            msg = f"Invalid can_adapter {can_adapter!r}. Must be one of {sorted(self.VALID_CAN_ADAPTERS)}."
            raise ValueError(msg)
        if dm_serial_baud <= 0:
            msg = f"dm_serial_baud must be a positive integer, got {dm_serial_baud!r}"
            raise ValueError(msg)
        if not (0.0 <= force_pos_torque_ratio <= 1.0):
            msg = f"force_pos_torque_ratio must be in [0, 1], got {force_pos_torque_ratio!r}"
            raise ValueError(msg)
        if control_mode not in {"pos_vel", "mit"}:
            msg = f"Invalid control_mode {control_mode!r}. Must be 'pos_vel' or 'mit'."
            raise ValueError(msg)
        if gripper_control_mode not in {"force_pos", "mit"}:
            msg = f"Invalid gripper_control_mode {gripper_control_mode!r}. Must be 'force_pos' or 'mit'."
            raise ValueError(msg)
        self._validate_gains(mit_kp, mit_kd)
        max_relative_target = _normalize_max_relative_target(max_relative_target)
        approach_speed = _normalize_speed(approach_speed, "approach_speed")

        self._port = port
        self._can_adapter = can_adapter
        self._dm_serial_baud = dm_serial_baud
        self._role = role
        self._disable_torque_on_disconnect = disable_torque_on_disconnect
        self._force_pos_torque_ratio = force_pos_torque_ratio
        self._control_mode: ReBotB601DMControlMode = control_mode
        self._gripper_control_mode: ReBotB601DMGripperControlMode = gripper_control_mode
        self._mit_kp = mit_kp
        self._mit_kd = mit_kd
        self._max_relative_target: float | None = max_relative_target
        self._gravity_compensation = gravity_compensation
        self._velocity_feedforward = velocity_feedforward
        self._approach_speed: float | None = approach_speed
        self._park_on_disconnect = park_on_disconnect
        self._gravity_model: GravityModel | None = None
        self._controller: Controller | None = None
        self._motors: dict[str, Motor] = {}
        self._torque_enabled = False
        self._reset_command_state()

    def _validate_gains(self, mit_kp: DMMITGain | None, mit_kd: DMMITGain | None) -> None:
        """Validate MIT gains: full per-joint dicts within each motor's encodable range.

        Raises:
            ValueError: If a gain is negative, non-finite, missing a joint, or out of range.
        """
        for name, gain in (("mit_kp", mit_kp), ("mit_kd", mit_kd)):
            if gain is None:
                continue
            if isinstance(gain, dict):
                _validate_per_joint_gains(gain, name)
            elif not math.isfinite(gain) or gain < 0.0:
                msg = f"{name} must be a non-negative finite value, got {gain!r}"
                raise ValueError(msg)
            self._check_gain_limits(gain, name)

    def _check_gain_limits(self, gain: float | dict[str, float], name: str) -> None:
        """Reject gains the motor's MIT frame cannot encode; motorbridge would clamp them silently.

        Raises:
            ValueError: If a gain exceeds the motor model's MIT range.
        """
        index = 0 if name == "mit_kp" else 1
        values = gain if isinstance(gain, dict) else dict.fromkeys(self.JOINT_ORDER, gain)
        for joint, value in values.items():
            model = self.MOTOR_MODELS[joint]
            limit = REBOT_B601_MIT_GAIN_LIMITS[model][index]
            if value > limit:
                msg = f"{name}[{joint!r}] = {value} exceeds the {model} MIT limit of {limit}"
                raise ValueError(msg)

    @property
    def joint_names(self) -> list[str]:
        """Ordered list of joint names matching the expected action/observation layout."""
        return self.JOINT_ORDER

    @property
    def device_ids(self) -> tuple[str, ...]:
        """Configured CAN transport identity without opening it."""
        return (f"{self.DEVICE_PREFIX}:{self._can_adapter}:{self._port}",)

    @property
    def port(self) -> str:
        """Serial port or CAN channel the driver is configured for."""
        return self._port

    @property
    def can_adapter(self) -> ReBotCANAdapter:
        """CAN adapter type (``"damiao"`` or ``"socketcan"``)."""
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

    @property
    def control_mode(self) -> ReBotB601DMControlMode:
        """Position-joint control mode (``"mit"`` or ``"pos_vel"``)."""
        return self._control_mode

    @property
    def gripper_control_mode(self) -> ReBotB601DMGripperControlMode:
        """Gripper control mode (``"mit"`` or ``"force_pos"``)."""
        return self._gripper_control_mode

    @property
    def max_relative_target(self) -> float | None:
        """Maximum allowed per-step position change in degrees, if configured."""
        return self._max_relative_target

    def _require_controller(self) -> Controller:
        controller = self._controller
        if controller is None:
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)
        return controller

    def connect(self) -> None:
        """Open the controller connection, register motors, and configure control modes."""
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

        logger.info(f"{self.__class__.__name__} connected on {self.port} (adapter={self.can_adapter})")

    def _open_controller(self) -> Controller:
        if self.can_adapter == "damiao":
            return Controller.from_dm_serial(serial_port=self.port, baud=self._dm_serial_baud)
        return Controller(channel=self.port)

    def _register_motors(self, controller: Controller) -> dict[str, Motor]:
        motors: dict[str, Motor] = {}
        for name in self.JOINT_ORDER:
            motor_id, feedback_id = self.MOTOR_IDS[name]
            motors[name] = controller.add_damiao_motor(motor_id, feedback_id, self.MOTOR_MODELS[name])
        return motors

    def disconnect(self) -> None:
        """Park the arm (if configured), disable torque (if configured), and release the controller."""
        if self._controller is None:
            return

        try:
            if self._park_on_disconnect and self._torque_enabled:
                try:
                    self.park()
                except Exception:  # noqa: BLE001 - disconnect must always release the bus
                    logger.opt(exception=True).warning("Could not park the reBot arm before disconnecting")
            self._disconnect_motors()
        finally:
            self._cleanup_connection()

        logger.info(f"{self.__class__.__name__} disconnected from {self.port}")

    def _disconnect_motors(self) -> None:
        if self.disable_torque_on_disconnect and self._controller is not None:
            with contextlib.suppress(Exception):
                self._controller.disable_all()
                self._torque_enabled = False
        for name, motor in self._motors.items():
            with contextlib.suppress(Exception):
                motor.clear_error()
            with contextlib.suppress(Exception):
                motor.close()
                logger.debug(f"Closed reBot motor {name}")

    def _cleanup_connection(self) -> None:
        controller = self._controller
        self._controller = None
        self._motors = {}
        self._torque_enabled = False
        self._reset_control_state()
        if controller is not None:
            with contextlib.suppress(Exception):
                controller.close()

    def is_connected(self) -> bool:
        """Return whether the controller connection is active."""
        return self._controller is not None

    def _configure_motors(self) -> None:
        controller = self._require_controller()
        controller.disable_all()
        self._torque_enabled = False

        for name, motor in self._motors.items():
            if name == "gripper":
                target_mode = Mode.MIT if self._gripper_control_mode == "mit" else Mode.FORCE_POS
            else:
                target_mode = Mode.MIT if self._control_mode == "mit" else Mode.POS_VEL
            motor.ensure_mode(target_mode)

        controller.enable_all()
        self._torque_enabled = True
        self._hold_after_enable()

    def _hold_after_enable(self) -> None:
        """Hold the measured pose right after the motors are enabled.

        An enabled motor runs its last command, which is zero stiffness and zero torque after power-up:
        until the first action arrives the arm would hang limp and could fall. Holding the measured
        pose (with gravity feedforward) keeps it where it is; the first action then eases in from there.
        """
        self._reset_control_state()
        present = None
        for _ in range(_HOLD_READ_ATTEMPTS):
            try:
                present = self._measured_public_deg()
                break
            except ConnectionError:
                time.sleep(_HOLD_READ_RETRY_S)
        if present is None:
            logger.warning("No motor feedback right after enabling torque; the arm is not held until the first command")
            return
        self._send_command(present, _NOMINAL_COMMAND_DT_S, goal_time=_NOMINAL_COMMAND_DT_S)
        self._reference_velocity_deg_s = np.zeros(self.NUM_JOINTS)
        self._approaching = self._approach_speed is not None
        self._approach_progress_speed = 0.0

    def disable_torque(self) -> None:
        """Disable torque (brake) on all motors."""
        self._require_controller().disable_all()
        self._torque_enabled = False
        self._reset_control_state()

    def enable_torque(self) -> None:
        """Enable torque on all motors and hold the measured pose; the next command eases in from it."""
        self._require_controller().enable_all()
        self._torque_enabled = True
        self._hold_after_enable()

    def configure_position_control(self) -> None:
        """Reconfigure all motors to their position-control modes."""
        self._configure_motors()

    def set_zero_position(self) -> None:
        """Disable torque and store the current pose as zero on every motor."""
        controller = self._require_controller()
        controller.disable_all()
        self._torque_enabled = False
        for motor in self._motors.values():
            motor.set_zero_position()
        self._reset_control_state()

    def _reset_control_state(self) -> None:
        """Forget the command history so the next command is seeded from the measured pose.

        Subclasses extend this to reset their own controller state.
        """
        self._reset_command_state()

    def _reset_command_state(self) -> None:
        self._last_command_deg: np.ndarray | None = None
        self._last_command_time: float | None = None
        self._reference_velocity_deg_s = np.zeros(self.NUM_JOINTS)
        self._approaching = False
        self._approach_progress_speed = 0.0
        self._previous_target_deg: np.ndarray | None = None
        self._limit_velocity_deg_s = np.zeros(self.NUM_JOINTS)
        self._dropout_reads = 0

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

    def _check_driving(self, positions: np.ndarray, torques: np.ndarray) -> None:
        """Raise if the motors answer but have stopped driving the arm.

        A supply dropout resets the motors: they keep reporting positions but no longer push, so the
        arm goes limp while commands keep arriving. That shows as zero torque on every arm joint while
        the arm drifts away from its command.

        Raises:
            ConnectionError: If that has lasted for ``_DROPOUT_READS`` consecutive reads.
        """
        last = self._last_command_deg
        arm = slice(0, REBOT_B601_ARM_JOINT_COUNT)
        if not self._torque_enabled or last is None:
            self._dropout_reads = 0
            return
        error = float(np.max(np.abs(last[arm] - positions[arm])))
        limp = float(np.max(np.abs(torques[arm]))) < _DROPOUT_TORQUE_NM and error > _DROPOUT_ERROR_DEG
        self._dropout_reads = self._dropout_reads + 1 if limp else 0
        if self._dropout_reads >= _DROPOUT_READS:
            self._dropout_reads = 0
            msg = (
                f"The reBot motors stopped driving the arm: no joint pushes any torque while the arm is "
                f"{error:.0f} degrees from its command. They probably lost power or reset (a very fast swing "
                "can overload the power supply). Check the supply, then reconnect."
            )
            raise ConnectionError(msg)

    def _measured_public_deg(self, states: list[MotorState] | None = None) -> np.ndarray:
        states = self._read_motor_states() if states is None else states
        return np.array(
            [
                self._motor_to_public_position(name, math.degrees(float(state.pos)))
                for name, state in zip(self.JOINT_ORDER, states, strict=True)
            ],
        )

    def get_observation(self) -> RobotObservation:
        """Read joint positions, velocities, torques, and temperatures from all motors.

        Returns:
            A ``ReBotB601DMObservation`` with joint positions in degrees and
            sensor data arrays for velocities, torques, temperatures, and status codes.
            Positions, velocities and torques are in the public joint frame.

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
            direction = self.JOINT_DIRECTIONS[name]
            positions[i] = math.degrees(float(state.pos)) / direction
            velocities[i] = math.degrees(float(state.vel)) / direction
            torques[i] = float(state.torq) * direction
            mos_temperatures[i] = float(state.t_mos)
            rotor_temperatures[i] = float(state.t_rotor)
            status_codes[i] = int(state.status_code)

        self._check_driving(positions, torques)
        return self.OBSERVATION_CLASS(
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
        """Send a target position command to each joint.

        Targets are clipped to the joint limits (and, if configured, to ``max_relative_target``
        of the measured pose). Right after the motors are enabled the arm eases from its measured
        pose toward the target at ``approach_speed`` before it tracks targets directly.

        In MIT mode the 6-DOF joints (and, optionally, the gripper) are sent the
        target with per-joint stiffness/damping gains plus velocity and gravity
        feedforward. In POS_VEL mode the joints are sent a position with a velocity
        limit calculated to reach the target within ``goal_time`` and capped by
        ``REBOT_B601_DM_POS_VEL_DEG_S``. The gripper uses
        FORCE_POS mode unless ``gripper_control_mode`` is ``"mit"``.

        Args:
            action: Array of 7 joint position targets in degrees.
            goal_time: Requested time to reach each target in seconds (POS_VEL mode only).

        Raises:
            ConnectionError: If the robot is not connected.
            ValueError: If the action shape does not match ``NUM_JOINTS``.
        """
        if not self.is_connected():
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)
        if not math.isfinite(goal_time) or goal_time <= 0.0:
            msg = f"goal_time must be a finite positive value, got {goal_time!r}"
            raise ValueError(msg)
        self._require_controller()
        expected_shape = (self.NUM_JOINTS,)
        if action.shape != expected_shape:
            msg = f"Expected action shape {expected_shape}, got {action.shape}"
            raise ValueError(msg)

        targets = np.array(
            [self._clip_public_action(name, float(action[i])) for i, name in enumerate(self.JOINT_ORDER)],
        )
        present: np.ndarray | None = None
        seed = self._last_command_deg is None and self._approach_speed is not None
        if self._max_relative_target is not None or seed:
            present = self._measured_public_deg()
        if self._max_relative_target is not None and present is not None:
            step = self._max_relative_target
            targets = present + np.clip(targets - present, -step, step)

        now = time.monotonic()
        if seed and present is not None:
            # First command since the motors were enabled: ease in from where the arm is.
            self._last_command_deg = present.copy()
            self._last_command_time = None
            self._approaching = True
            self._approach_progress_speed = 0.0
            self._limit_velocity_deg_s = np.zeros(self.NUM_JOINTS)
        dt = self._command_interval(now)
        if not self._approaching and self._is_jump(targets, dt):
            self._approaching = self._approach_speed is not None
            self._approach_progress_speed = 0.0
            self._limit_velocity_deg_s = np.zeros(self.NUM_JOINTS)
        command = self._approach(targets, dt) if self._approaching else self._limit_motion(targets, dt)
        self._previous_target_deg = targets.copy()
        self._send_command(command, dt, goal_time)
        self._last_command_time = now

    def _command_interval(self, now: float) -> float:
        if self._last_command_time is None:
            return _NOMINAL_COMMAND_DT_S
        return min(max(now - self._last_command_time, _MIN_COMMAND_DT_S), _MAX_COMMAND_DT_S)

    def _is_jump(self, targets: np.ndarray, dt: float) -> bool:
        """Whether ``targets`` jumps further from the previous target than any real motion could.

        The previous target, not the last command: the speed limits let the command fall behind a
        fast but continuous stream, which is not a jump.

        Returns:
            ``True`` if an arm joint would have to move faster than ``_JUMP_SPEED_DEG_S``.
        """
        last = self._previous_target_deg if self._previous_target_deg is not None else self._last_command_deg
        if last is None:
            return False
        step = float(np.max(np.abs(targets[:REBOT_B601_ARM_JOINT_COUNT] - last[:REBOT_B601_ARM_JOINT_COUNT])))
        if step <= max(_JUMP_MIN_DEG, _JUMP_SPEED_DEG_S * min(dt, _JUMP_MAX_DT_S)):
            return False
        logger.warning(f"reBot command jumped {step:.0f} degrees in one step; easing into it")
        return True

    def _limit_motion(self, targets: np.ndarray, dt: float) -> np.ndarray:
        """Follow ``targets`` with the arm joints within the speed and acceleration limits.

        Below the limits the command is the target itself. A joint that falls behind catches up no
        faster than the target moves plus what it can still brake from before reaching it, so it does
        not overshoot a target that holds still. The gripper is not limited.

        Returns:
            The command for this step.
        """
        last = self._last_command_deg
        arm = slice(0, REBOT_B601_ARM_JOINT_COUNT)
        if last is None:
            self._limit_velocity_deg_s = np.zeros(self.NUM_JOINTS)
            return targets
        speed_limit, accel_limit = self._motion_limits()
        previous = self._previous_target_deg if self._previous_target_deg is not None else last
        target_speed = np.minimum(np.abs(targets[arm] - previous[arm]) / dt, speed_limit)
        error = targets[arm] - last[arm]
        # Fastest speed from which the joint can still stop within the remaining error, braking at the
        # acceleration limit in steps of dt.
        half_step = 0.5 * accel_limit * dt
        braking = np.sqrt(half_step**2 + 2.0 * accel_limit * np.abs(error)) - half_step
        catch_up = np.minimum(target_speed + braking, speed_limit)
        desired = np.clip(error / dt, -catch_up, catch_up)
        velocity = self._limit_velocity_deg_s[arm].copy()
        velocity += np.clip(desired - velocity, -accel_limit * dt, accel_limit * dt)
        step = velocity * dt
        # Land on a target instead of passing it once the joint is nearly stopped.
        landing = (
            (np.sign(step) == np.sign(error))
            & (np.abs(step) > np.abs(error))
            & (np.abs(velocity) <= 2 * accel_limit * dt)
        )
        step = np.where(landing, error, step)
        self._limit_velocity_deg_s[arm] = step / dt
        command = targets.copy()
        command[arm] = last[arm] + step
        return command

    def _motion_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-joint speed and acceleration limits of the arm joints, from their motor models.

        Returns:
            Speed limits (deg/s) and acceleration limits (deg/s²), one per arm joint.
        """
        models = [self.MOTOR_MODELS[name] for name in self.JOINT_ORDER[:REBOT_B601_ARM_JOINT_COUNT]]
        return (
            np.array([REBOT_B601_MAX_SPEED_DEG_S[m] for m in models]),
            np.array([REBOT_B601_MAX_ACCEL_DEG_S2[m] for m in models]),
        )

    def _approach(self, targets: np.ndarray, dt: float) -> np.ndarray:
        """Move the arm joints toward ``targets`` along a straight joint-space line at a bounded speed.

        The speed ramps up at ``_APPROACH_ACCEL_DEG_S2`` and slows down before the target, so the
        arm starts and stops smoothly. The gripper is not slowed (it is force-limited instead).

        Returns:
            The command for this step.
        """
        last = self._last_command_deg
        speed_limit = self._approach_speed
        if last is None or speed_limit is None:
            self._approaching = False
            return targets
        arm = slice(0, REBOT_B601_ARM_JOINT_COUNT)
        delta = targets[arm] - last[arm]
        distance = float(np.max(np.abs(delta)))
        if distance <= _APPROACH_DONE_DEG:
            self._approaching = False
            logger.debug("reBot approach finished; tracking commands directly")
            return targets
        speed = min(
            speed_limit,
            self._approach_progress_speed + _APPROACH_ACCEL_DEG_S2 * dt,
            math.sqrt(2.0 * _APPROACH_ACCEL_DEG_S2 * distance),
        )
        self._approach_progress_speed = speed
        fraction = min(1.0, speed * dt / distance)
        command = targets.copy()
        command[arm] = last[arm] + fraction * delta
        return command

    def _send_command(self, command_deg: np.ndarray, dt: float, goal_time: float, *, hold_scale: float = 1.0) -> None:
        """Send public-frame targets with velocity and gravity feedforward, and remember them.

        ``hold_scale`` scales the MIT stiffness and gravity feedforward of the arm joints (damping is
        kept); the soft release fades it to zero.
        """
        last = self._last_command_deg
        velocity = np.zeros(self.NUM_JOINTS)
        if self._velocity_feedforward and last is not None:
            raw = (command_deg - last) / dt
            velocity = _VELOCITY_FILTER_ALPHA * raw + (1.0 - _VELOCITY_FILTER_ALPHA) * self._reference_velocity_deg_s
        self._reference_velocity_deg_s = velocity
        torques = hold_scale * self._gravity_torques(command_deg)

        for i, name in enumerate(self.JOINT_ORDER):
            direction = self.JOINT_DIRECTIONS[name]
            target_rad = math.radians(self._public_to_motor_position(name, float(command_deg[i])))
            velocity_rad_s = math.radians(float(velocity[i]) * direction)
            torque_nm = float(torques[i]) / direction
            self._send_joint(
                name,
                target_rad,
                goal_time,
                velocity_rad_s=velocity_rad_s,
                torque_nm=torque_nm,
                kp_scale=hold_scale,
            )
        self._last_command_deg = command_deg.copy()

    def _gravity_torques(self, command_deg: np.ndarray) -> np.ndarray:
        """Public-frame gravity torques of the commanded arm pose (zero for the gripper or when disabled).

        Returns:
            One torque in N·m per joint.
        """
        torques = np.zeros(self.NUM_JOINTS)
        if not self._gravity_compensation or self._control_mode != "mit":
            return torques
        arm = REBOT_B601_ARM_JOINT_COUNT
        torques[:arm] = self._gravity().torques(np.radians(command_deg[:arm]))
        return torques

    def _gravity(self) -> GravityModel:
        if self._gravity_model is None:
            self._gravity_model = GravityModel(get_urdf_path() / self.GRAVITY_URDF, REBOT_B601_URDF_ARM_JOINTS)
        return self._gravity_model

    def _send_joint(
        self,
        name: str,
        target_rad: float,
        goal_time: float,
        *,
        velocity_rad_s: float = 0.0,
        torque_nm: float = 0.0,
        kp_scale: float = 1.0,
    ) -> None:
        motor = self._motors[name]
        if name == "gripper":
            if self._gripper_control_mode == "mit":
                motor.send_mit(target_rad, 0.0, self._mit_gain(name, "kp"), self._mit_gain(name, "kd"), 0.0)
                return
            velocity_limit = self._pos_vel_velocity_rad_s(name, target_rad, goal_time)
            motor.send_force_pos(target_rad, velocity_limit, self._force_pos_torque_ratio)
            return
        if self._control_mode == "mit":
            motor.send_mit(
                target_rad,
                velocity_rad_s,
                kp_scale * self._mit_gain(name, "kp"),
                self._mit_gain(name, "kd"),
                torque_nm,
            )
        else:
            velocity_limit = self._pos_vel_velocity_rad_s(name, target_rad, goal_time)
            motor.send_pos_vel(target_rad, velocity_limit)

    def park(self, speed: float = REBOT_B601_PARK_SPEED_DEG_S) -> None:
        """Fold the arm into its rest (zero) pose with a smooth, gravity-compensated move.

        The wrist and base turn back first, then the shoulder and elbow fold down, so the gripper
        does not sweep through the arm. The gripper holds its opening. The move aborts with a
        warning if a joint lags far behind its command, for example after a collision.

        Args:
            speed: Peak joint speed of the move, in degrees per second.

        Raises:
            ConnectionError: If the robot is not connected.
            ValueError: If ``speed`` is not a finite positive value.
        """
        if not math.isfinite(speed) or speed <= 0.0:
            msg = f"speed must be a finite positive value in degrees per second, got {speed!r}"
            raise ValueError(msg)
        if not self.is_connected():
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)
        present = self._measured_public_deg()
        if self._last_command_deg is None:
            self._last_command_deg = present.copy()
        start = present.copy()
        logger.info(f"Parking {self.__class__.__name__} in its rest pose")
        for stage in REBOT_B601_PARK_STAGES:
            indices = [self.JOINT_ORDER.index(name) for name in stage]
            goal = start.copy()
            goal[indices] = 0.0
            if not self._move_to(start, goal, speed):
                return
            start = goal
        time.sleep(_PARK_SETTLE_S)
        self._release_softly()

    def _release_softly(self) -> None:
        """Fade stiffness and gravity feedforward to zero while keeping damping (MIT mode only).

        The rest pose does not support every link (the wrist, for one, rests on whatever is below the
        gripper). Fading the holding torque under damping lets such a link sink onto its support at a
        walking pace, ``gravity torque / kd``, instead of dropping when torque is released.
        """
        if self._control_mode != "mit" or self._last_command_deg is None:
            return
        command = self._last_command_deg.copy()
        steps = max(1, round(_RELEASE_FADE_S * _PARK_RATE_HZ))
        for step in range(1, steps + 1):
            self._send_command(
                command, 1.0 / _PARK_RATE_HZ, goal_time=1.0 / _PARK_RATE_HZ, hold_scale=1.0 - step / steps
            )
            time.sleep(1.0 / _PARK_RATE_HZ)
        self._wait_until_still()

    def _wait_until_still(self) -> None:
        """Wait (up to ``_RELEASE_SETTLE_MAX_S``) until every arm joint has been still for a moment."""
        dt = 1.0 / _PARK_RATE_HZ
        still_for = 0.0
        waited = 0.0
        while waited < _RELEASE_SETTLE_MAX_S and still_for < _RELEASE_STILL_S:
            time.sleep(dt)
            waited += dt
            states = self._read_motor_states()[:REBOT_B601_ARM_JOINT_COUNT]
            moving = max(abs(math.degrees(float(state.vel))) for state in states) > _RELEASE_STILL_DEG_S
            still_for = 0.0 if moving else still_for + dt

    def _move_to(self, start: np.ndarray, goal: np.ndarray, speed: float) -> bool:
        """Stream a minimum-jerk move from ``start`` to ``goal`` peaking at ``speed`` degrees per second.

        Returns:
            ``False`` if the move was aborted because a joint lagged too far behind.
        """
        distance = float(np.max(np.abs(goal - start)))
        if distance < _APPROACH_DONE_DEG:
            return True
        # Minimum-jerk peak speed is 1.875 * distance / duration.
        duration = max(_PARK_MIN_STAGE_S, 1.875 * distance / speed)
        dt = 1.0 / _PARK_RATE_HZ
        steps = max(1, math.ceil(duration / dt))
        self._approaching = False
        for step in range(1, steps + 1):
            command = start + (goal - start) * _minimum_jerk(step / steps)
            self._send_command(command, dt, goal_time=dt)
            self._last_command_time = time.monotonic()
            time.sleep(dt)
            error = np.abs(self._measured_public_deg() - command)[:REBOT_B601_ARM_JOINT_COUNT]
            worst = int(np.argmax(error))
            if error[worst] > _PARK_MAX_ERROR_DEG:
                logger.warning(
                    f"Stopped parking the reBot arm: {self.JOINT_ORDER[worst]} is {error[worst]:.0f} degrees "
                    "behind its command (blocked?)",
                )
                return False
        return True

    def _mit_gain(self, name: str, which: Literal["kp", "kd"]) -> float:
        """Resolve the MIT gain for a joint from the configured value or defaults.

        Args:
            name: Joint name whose gain is requested.
            which: ``"kp"`` for stiffness or ``"kd"`` for damping.

        Returns:
            The resolved gain value for the requested joint.
        """
        value = self._mit_kp if which == "kp" else self._mit_kd
        if value is None:
            source = REBOT_B601_DM_MIT_KP if which == "kp" else REBOT_B601_DM_MIT_KD
            return source[name]
        if isinstance(value, dict):
            return value[name]
        return value

    def _pos_vel_velocity_rad_s(self, name: str, target_rad: float, goal_time: float) -> float:
        """Compute the POS_VEL velocity limit (rad/s) for a joint target.

        Args:
            name: Joint name being commanded.
            target_rad: Target position in radians.
            goal_time: Requested time to reach the target in seconds.

        Returns:
            The velocity limit in rad/s, capped by the joint's maximum and the
            time-to-target.
        """
        idx = self.JOINT_ORDER.index(name)
        max_velocity_rad_s = math.radians(REBOT_B601_DM_POS_VEL_DEG_S[idx])
        state = self._motors[name].get_state()
        if state is None:
            return max_velocity_rad_s
        return min(max_velocity_rad_s, abs(target_rad - float(state.pos)) / goal_time)

    def _clip_public_action(self, name: str, action_deg: float) -> float:
        min_deg, max_deg = self.JOINT_LIMITS_DEG[name]
        return float(np.clip(action_deg, min_deg, max_deg))

    def _public_to_motor_position(self, name: str, position_deg: float) -> float:
        return position_deg * self.JOINT_DIRECTIONS[name]

    def _motor_to_public_position(self, name: str, position_deg: float) -> float:
        return position_deg / self.JOINT_DIRECTIONS[name]
