# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RobStride specialization of the reBot B601 follower driver."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
from loguru import logger
from motorbridge import Controller, Mode

from physicalai.config import export_config
from physicalai_rebot_b601_plugin.constants import (
    REBOT_B601_MIT_GAIN_LIMITS,
    REBOT_B601_RS_GRAVITY_URDF,
    REBOT_B601_RS_GRIPPER_MAX_TORQUE_NM,
    REBOT_B601_RS_JOINT_DIRECTIONS,
    REBOT_B601_RS_JOINT_LIMITS_DEG,
    REBOT_B601_RS_JOINT_ORDER,
    REBOT_B601_RS_MIT_KD,
    REBOT_B601_RS_MIT_KP,
    REBOT_B601_RS_MOTOR_IDS,
    REBOT_B601_RS_MOTOR_MODELS,
    VALID_RS_CAN_ADAPTERS,
)
from physicalai_rebot_b601_plugin.dm import ReBotB601DM, ReBotB601DMObservation

ReBotRSCanAdapter = Literal["socketcan", "robstride"]
ReBotRole = Literal["follower"]
RSMITGain = float | dict[str, float]

_GRIPPER_STALL_VEL_RAD_S = 0.25
_GRIPPER_OPEN_STALL_ERROR_RAD = 0.05
_GRIPPER_OPEN_STALL_CYCLES = 5
_GRIPPER_DAMPING = 1.5
_GRIPPER_LPF_ALPHA = 0.3
_GRIPPER_TARGET_VEL_MAX_RAD_S = 3.0
# The gripper's velocity estimates use the measured interval between commands, clamped to this
# range; the first command after a reset uses the nominal interval.
_GRIPPER_NOMINAL_DT_S = 0.02
_GRIPPER_MIN_DT_S = 0.005
_GRIPPER_MAX_DT_S = 0.1


def _validate_mit_gain(gain: RSMITGain | None, name: str) -> None:
    if gain is None:
        return
    values = gain if isinstance(gain, dict) else dict.fromkeys(REBOT_B601_RS_MIT_KP, gain)
    if isinstance(gain, dict):
        unknown = set(gain) - set(REBOT_B601_RS_MIT_KP)
        if unknown:
            msg = f"{name} may only contain position joints {sorted(REBOT_B601_RS_MIT_KP)}; got {sorted(unknown)}"
            raise ValueError(msg)
    index = 0 if name == "mit_kp" else 1
    for joint, value in values.items():
        if not math.isfinite(value) or value < 0.0:
            label = name if not isinstance(gain, dict) else f"{name}[{joint!r}]"
            msg = f"{label} must be a non-negative finite value, got {value!r}"
            raise ValueError(msg)
        model = REBOT_B601_RS_MOTOR_MODELS[joint]
        limit = REBOT_B601_MIT_GAIN_LIMITS[model][index]
        if value > limit:
            msg = f"{name}[{joint!r}] = {value} exceeds the {model} MIT limit of {limit}"
            raise ValueError(msg)


@dataclass
class ReBotB601RSObservation(ReBotB601DMObservation):
    """Observation data for the RobStride reBot B601 arm."""


@export_config
class ReBotB601RS(ReBotB601DM):
    """RobStride B601 follower using MIT control and an impedance gripper.

    The arm joints run MIT mode with gravity and velocity feedforward (see
    :mod:`physicalai_rebot_b601_plugin.dm`); the gripper runs a force-limited impedance loop.
    """

    JOINT_ORDER: ClassVar[list[str]] = list(REBOT_B601_RS_JOINT_ORDER)
    NUM_JOINTS: ClassVar[int] = len(JOINT_ORDER)
    DEVICE_PREFIX: ClassVar[str] = "rebot-rs"
    MOTOR_IDS: ClassVar = REBOT_B601_RS_MOTOR_IDS
    MOTOR_MODELS: ClassVar = REBOT_B601_RS_MOTOR_MODELS
    JOINT_DIRECTIONS: ClassVar = REBOT_B601_RS_JOINT_DIRECTIONS
    JOINT_LIMITS_DEG: ClassVar = REBOT_B601_RS_JOINT_LIMITS_DEG
    OBSERVATION_CLASS: ClassVar = ReBotB601RSObservation
    VALID_CAN_ADAPTERS: ClassVar = VALID_RS_CAN_ADAPTERS
    GRAVITY_URDF: ClassVar[str] = REBOT_B601_RS_GRAVITY_URDF

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
        gravity_compensation: bool = True,
        velocity_feedforward: bool = True,
        approach_speed: float | None = 30.0,
        park_on_disconnect: bool = True,
    ) -> None:
        """Initialize the RobStride follower and its impedance controller.

        Args:
            port: SocketCAN channel, e.g. ``can0``.
            can_adapter: ``"socketcan"``; ``"robstride"`` is reserved for the vendor adapter.
            role: Currently only ``"follower"`` is supported.
            disable_torque_on_disconnect: Whether to disable all motors on disconnect.
            mit_kp: MIT stiffness for the arm joints, a single value or a partial per-joint dict;
                unset joints use :data:`REBOT_B601_RS_MIT_KP`.
            mit_kd: MIT damping for the arm joints, like ``mit_kp``; defaults to
                :data:`REBOT_B601_RS_MIT_KD`.
            gripper_mit_kp: Gripper impedance stiffness (N·m/rad).
            gripper_mit_kd: Gripper impedance damping (N·m·s/rad).
            gripper_mit_torque_limit: Gripper torque limit while it moves (N·m).
            gripper_mit_hold_torque_limit: Gripper torque limit once it stalls, e.g. on an object (N·m).
            max_relative_target: Optional per-command step limit in degrees from the measured pose;
                ``None`` or ``-1`` disables it.
            gravity_compensation: Send the gravity torque of the commanded pose as feedforward.
            velocity_feedforward: Send the velocity of the commanded targets as feedforward.
            approach_speed: Peak joint speed (degrees per second) for the first move after the
                motors are enabled; ``None`` or ``-1`` jumps straight to the first command.
            park_on_disconnect: Fold the arm into its rest pose before disconnecting.

        Raises:
            ValueError: If a control setting, gain, torque limit, or relative target is invalid.
        """
        gripper_values = (
            gripper_mit_kp,
            gripper_mit_kd,
            gripper_mit_torque_limit,
            gripper_mit_hold_torque_limit,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in gripper_values):
            msg = "gripper MIT gains and torque limits must be non-negative finite values."
            raise ValueError(msg)
        torque_limits = (gripper_mit_torque_limit, gripper_mit_hold_torque_limit)
        if any(limit > REBOT_B601_RS_GRIPPER_MAX_TORQUE_NM for limit in torque_limits):
            msg = f"gripper torque limits must not exceed {REBOT_B601_RS_GRIPPER_MAX_TORQUE_NM} N·m."
            raise ValueError(msg)
        _validate_mit_gain(mit_kp, "mit_kp")
        _validate_mit_gain(mit_kd, "mit_kd")

        super().__init__(
            port=port,
            can_adapter=can_adapter,  # pyrefly: ignore[bad-argument-type]
            role=role,
            disable_torque_on_disconnect=disable_torque_on_disconnect,
            control_mode="mit",
            gripper_control_mode="mit",
            max_relative_target=max_relative_target,
            gravity_compensation=gravity_compensation,
            velocity_feedforward=velocity_feedforward,
            approach_speed=approach_speed,
            park_on_disconnect=park_on_disconnect,
        )
        self._mit_kp = mit_kp
        self._mit_kd = mit_kd
        self._gripper_mit_kp = gripper_mit_kp
        self._gripper_mit_kd = gripper_mit_kd
        self._gripper_mit_torque_limit = gripper_mit_torque_limit
        self._gripper_mit_hold_torque_limit = gripper_mit_hold_torque_limit
        self._reset_control_state()

    def _open_controller(self) -> Controller:
        if self.can_adapter == "robstride":
            msg = "can_adapter='robstride' is not supported by the current MotorBridge Python SDK. Use socketcan."
            raise NotImplementedError(msg)
        return Controller(channel=self.port)

    def _register_motors(self, controller: Controller) -> dict:
        motors = {}
        for name in self.JOINT_ORDER:
            motor_id, feedback_id = self.MOTOR_IDS[name]
            motors[name] = controller.add_robstride_motor(motor_id, feedback_id, self.MOTOR_MODELS[name])
        return motors

    def _configure_motors(self) -> None:
        controller = self._require_controller()
        controller.disable_all()
        self._torque_enabled = False
        for motor in self._motors.values():
            motor.ensure_mode(Mode.MIT)
        controller.enable_all()
        self._torque_enabled = True
        self._reset_control_state()

    def _reset_control_state(self) -> None:
        super()._reset_control_state()
        self._gripper_prev_target_pos: float | None = None
        self._gripper_prev_filtered_target_vel: float | None = None
        self._gripper_prev_state_pos: float | None = None
        self._gripper_prev_time: float | None = None
        self._gripper_open_stall_count = 0

    def _mit_gain(self, name: str, which: Literal["kp", "kd"]) -> float:
        override = self._mit_kp if which == "kp" else self._mit_kd
        defaults = REBOT_B601_RS_MIT_KP if which == "kp" else REBOT_B601_RS_MIT_KD
        if override is None:
            return defaults[name]
        if isinstance(override, dict):
            return override.get(name, defaults[name])
        return override

    def _send_joint(
        self,
        name: str,
        target_rad: float,
        goal_time: float,
        *,
        velocity_rad_s: float = 0.0,
        torque_nm: float = 0.0,
    ) -> None:
        _ = goal_time
        motor = self._motors[name]
        if name == "gripper":
            tau = self._gripper_output_torque(target_rad)
            motor.send_mit(0.0, 0.0, 0.0, _GRIPPER_DAMPING, tau)
            return
        motor.send_mit(target_rad, velocity_rad_s, self._mit_gain(name, "kp"), self._mit_gain(name, "kd"), torque_nm)

    def _gripper_control_dt(self) -> float:
        now = time.monotonic()
        previous = self._gripper_prev_time
        self._gripper_prev_time = now
        if previous is None:
            return _GRIPPER_NOMINAL_DT_S
        return min(max(now - previous, _GRIPPER_MIN_DT_S), _GRIPPER_MAX_DT_S)

    def _gripper_output_torque(self, target_rad: float) -> float:
        motor = self._motors["gripper"]
        self._require_controller().poll_feedback_once()
        state = motor.get_state()
        if state is None:
            return 0.0

        control_dt_s = self._gripper_control_dt()
        if self._gripper_prev_target_pos is None:
            target_vel = 0.0
        else:
            target_vel = (target_rad - self._gripper_prev_target_pos) / control_dt_s
        self._gripper_prev_target_pos = target_rad

        if self._gripper_prev_filtered_target_vel is None:
            filtered_target_vel = target_vel
        else:
            filtered_target_vel = (
                _GRIPPER_LPF_ALPHA * target_vel + (1.0 - _GRIPPER_LPF_ALPHA) * self._gripper_prev_filtered_target_vel
            )
        target_vel = float(np.clip(filtered_target_vel, -_GRIPPER_TARGET_VEL_MAX_RAD_S, _GRIPPER_TARGET_VEL_MAX_RAD_S))
        self._gripper_prev_filtered_target_vel = target_vel

        if self._gripper_prev_state_pos is None:
            state_vel = 0.0
        else:
            state_vel = (state.pos - self._gripper_prev_state_pos) / control_dt_s
        self._gripper_prev_state_pos = state.pos

        impedance_torque = self._gripper_mit_kp * (target_rad - state.pos) + self._gripper_mit_kd * (
            target_vel - state.vel
        )
        stalled = abs(state_vel) < _GRIPPER_STALL_VEL_RAD_S
        if target_rad - state.pos > _GRIPPER_OPEN_STALL_ERROR_RAD and stalled:
            self._gripper_open_stall_count += 1
        else:
            self._gripper_open_stall_count = 0
        if self._gripper_open_stall_count >= _GRIPPER_OPEN_STALL_CYCLES:
            if self._gripper_open_stall_count == _GRIPPER_OPEN_STALL_CYCLES:
                logger.warning("reBot RS gripper open-stall detected, cutting torque")
            return 0.0
        max_torque = self._gripper_mit_hold_torque_limit if stalled else self._gripper_mit_torque_limit
        return float(np.clip(impedance_torque, -max_torque, max_torque))
