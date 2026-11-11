# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RobStride specialization of the reBot B601 follower driver."""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

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
from physicalai_rebot_b601_plugin.dm import ReBotB601DM, ReBotB601DMObservation, _normalize_speed

if TYPE_CHECKING:
    from motorbridge import Motor

ReBotRSCanAdapter = Literal["socketcan", "robstride"]
ReBotRole = Literal["follower"]
RSMITGain = float | dict[str, float]

_GRIPPER_STALL_VEL_RAD_S = 0.25
_GRIPPER_OPEN_STALL_ERROR_RAD = 0.05
# Stall timing in seconds, not command counts: a gripper starting from rest needs 50-100 ms before
# its measured speed passes the stall threshold (measured at 30-100 Hz), so a count of commands
# misfires at Studio's 100 Hz. Hold force applies after this long without moving...
_GRIPPER_HOLD_AFTER_S = 0.1
# ...and a gripper that is told to open but has not moved for this long is at its open stop.
_GRIPPER_OPEN_STALL_S = 0.5
# RobStride ``limit_torque``: the firmware caps every MIT command's torque at this value, so the
# gripper's stiff on-board servo can never squeeze harder than the configured limit.
_LIMIT_TORQUE_PARAM = 0x700B
# The limit is written without an acknowledgement; write it again this often in case one was lost.
_GRIPPER_LIMIT_REFRESH_S = 0.5
# The gripper's velocity estimate uses the measured interval between commands, clamped to this
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

    def __init__(  # noqa: PLR0913 - one flat option per control feature, for YAML and Studio
        self,
        port: str = "can0",
        *,
        can_adapter: ReBotRSCanAdapter = "socketcan",
        role: ReBotRole = "follower",
        disable_torque_on_disconnect: bool = True,
        mit_kp: RSMITGain | None = None,
        mit_kd: RSMITGain | None = None,
        gripper_mit_kp: float = 30.0,
        gripper_mit_kd: float = 0.8,
        gripper_mit_torque_limit: float = 1.5,
        gripper_mit_hold_torque_limit: float = 0.5,
        gripper_max_speed: float | None = 400.0,
        max_relative_target: float | None = None,
        gravity_compensation: float | bool = 1.0,
        velocity_feedforward: float | bool = 0.3,
        approach_speed: float | None = 30.0,
        park_on_disconnect: bool = True,
        motion_limits: bool = True,
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
            gripper_mit_kp: Gripper stiffness (N·m/rad) of the motor's own MIT servo.
            gripper_mit_kd: Gripper damping (N·m·s/rad) of the motor's own MIT servo.
            gripper_mit_torque_limit: Gripper torque limit while it moves (N·m), enforced by the
                motor firmware (``limit_torque``).
            gripper_mit_hold_torque_limit: Gripper torque limit once it stalls, e.g. on an object (N·m).
            gripper_max_speed: Fastest the gripper's target moves (degrees per second), which sets how
                hard the jaws hit an object; ``None`` or ``-1`` follows the commands as they come.
            max_relative_target: Optional per-command step limit in degrees from the measured pose;
                ``None`` or ``-1`` disables it.
            gravity_compensation: Share (0 to 1) of the commanded pose's gravity torque sent as
                feedforward.
            velocity_feedforward: Share (0 to 1) of the commanded targets' velocity sent as the
                velocity reference; above about 0.5 the arm overshoots fast stops.
            approach_speed: Peak joint speed (degrees per second) for the first move after the
                motors are enabled; ``None`` or ``-1`` jumps straight to the first command.
            park_on_disconnect: Fold the arm into its rest pose before disconnecting.
            motion_limits: Follow each command within the per-motor speed and acceleration limits;
                ``False`` sends commands as they come, like LeRobot's follower.

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
        gripper_max_speed = _normalize_speed(gripper_max_speed, "gripper_max_speed")
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
            motion_limits=motion_limits,
        )
        self._mit_kp = mit_kp
        self._mit_kd = mit_kd
        self._gripper_mit_kp = gripper_mit_kp
        self._gripper_mit_kd = gripper_mit_kd
        self._gripper_mit_torque_limit = gripper_mit_torque_limit
        self._gripper_mit_hold_torque_limit = gripper_mit_hold_torque_limit
        self._gripper_max_speed_rad_s = None if gripper_max_speed is None else math.radians(gripper_max_speed)
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
        self._enable_and_hold()

    def _reset_control_state(self) -> None:
        super()._reset_control_state()
        self._gripper_prev_state_pos: float | None = None
        self._gripper_prev_time: float | None = None
        self._gripper_stalled_since: float | None = None
        self._gripper_open_stalled_since: float | None = None
        self._gripper_open_stalled = False
        # Unknown after a reset (a new connection, or another program may have changed it).
        self._gripper_holding: bool | None = None
        self._gripper_target_rad: float | None = None
        self._gripper_limit_written_at = -math.inf

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
        kp_scale: float = 1.0,
    ) -> None:
        _ = goal_time
        motor = self._motors[name]
        if name == "gripper":
            self._send_gripper(motor, target_rad)
            return
        kp = kp_scale * self._mit_gain(name, "kp")
        motor.send_mit(target_rad, velocity_rad_s, kp, self._mit_gain(name, "kd"), torque_nm)

    def _gripper_control_dt(self) -> float:
        now = time.monotonic()
        previous = self._gripper_prev_time
        self._gripper_prev_time = now
        if previous is None:
            return _GRIPPER_NOMINAL_DT_S
        return min(max(now - previous, _GRIPPER_MIN_DT_S), _GRIPPER_MAX_DT_S)

    def _send_gripper(self, motor: Motor, target_rad: float) -> None:
        """Servo the gripper on the motor itself, with the firmware capping its torque.

        The motor's MIT loop runs much faster than the commands arrive, so the gripper follows a
        fast leader; ``limit_torque`` keeps its force at the moving limit, or at the hold limit once
        it stalls (for example on a grasped object). A gripper that cannot open, at its open stop,
        is let go with damping only.
        """
        kd = self._gripper_mit_kd
        self._require_controller().poll_feedback_once()
        state = motor.get_state()
        if state is None:
            # Not driven while there is no feedback: the open-stall clock and the speed estimate start over. The
            # still clock is kept, so a gripper holding an object keeps the hold limit when feedback returns.
            self._gripper_prev_state_pos = None
            self._gripper_open_stalled_since = None
            motor.send_mit(0.0, 0.0, 0.0, kd, 0.0)
            return

        control_dt_s = self._gripper_control_dt()
        if self._gripper_prev_state_pos is None:
            state_vel = 0.0
        else:
            state_vel = (state.pos - self._gripper_prev_state_pos) / control_dt_s
        self._gripper_prev_state_pos = state.pos
        target_rad = self._slew_gripper_target(target_rad, state.pos, control_dt_s)

        now = time.monotonic()
        moving = abs(state_vel) >= _GRIPPER_STALL_VEL_RAD_S
        # Hysteresis: once the open-stall clock runs, an opening error that dips just under the threshold
        # (a jittery leader at the open stop) does not restart it.
        open_threshold = _GRIPPER_OPEN_STALL_ERROR_RAD / (2.0 if self._gripper_open_stalled_since is not None else 1.0)
        opening = target_rad - state.pos > open_threshold
        if moving:
            self._gripper_stalled_since = None
        elif self._gripper_stalled_since is None:
            self._gripper_stalled_since = now
        # The open-stall clock runs only while the gripper is told to open and does not move. A gripper that
        # sat still (closed, or holding an object) and is then told to open gets the full time to start.
        if opening and not moving:
            if self._gripper_open_stalled_since is None:
                self._gripper_open_stalled_since = now
        else:
            self._gripper_open_stalled_since = None
        if (
            self._gripper_open_stalled_since is not None
            and now - self._gripper_open_stalled_since >= _GRIPPER_OPEN_STALL_S
        ):
            if not self._gripper_open_stalled:
                logger.warning("reBot RS gripper open-stall detected, cutting torque")
            self._gripper_open_stalled = True
            motor.send_mit(state.pos, 0.0, 0.0, kd, 0.0)
            return
        self._gripper_open_stalled = False
        stalled_s = 0.0 if self._gripper_stalled_since is None else now - self._gripper_stalled_since
        # The hold limit is for gripping; opening (also from rest) always gets the moving limit to break free.
        holding = not opening and stalled_s >= _GRIPPER_HOLD_AFTER_S
        self._send_gripper_reference(motor, target_rad, state.pos, holding=holding)

    def _send_gripper_reference(self, motor: Motor, target_rad: float, measured_rad: float, *, holding: bool) -> None:
        """Send the gripper's reference and its torque cap, in the order that never squeezes harder than either."""
        kd = self._gripper_mit_kd
        previous = self._gripper_holding
        if previous is not None and self._gripper_limit(holding=holding) > self._gripper_limit(holding=previous):
            # Raising the cap: replace the reference first, so a gripper on an object that is told to open is
            # not squeezed at the higher limit.
            motor.send_mit(target_rad, 0.0, self._gripper_mit_kp, kd, 0.0)
            self._set_gripper_torque_limit(motor, holding=holding)
            return
        if previous is None:
            # The cap is unknown (after a reconnect or re-enable) and may be about to rise while the motor still
            # runs an old reference: let go (damping only) before writing it.
            motor.send_mit(measured_rad, 0.0, 0.0, kd, 0.0)
        # Lowering or keeping the cap, or setting an unknown one: cap first.
        self._set_gripper_torque_limit(motor, holding=holding)
        motor.send_mit(target_rad, 0.0, self._gripper_mit_kp, kd, 0.0)

    def _slew_gripper_target(self, target_rad: float, measured_rad: float, dt: float) -> float:
        """Move the gripper's target toward ``target_rad`` no faster than ``gripper_max_speed``.

        Returns:
            The target to send this command, starting from the measured position after a reset.
        """
        speed = self._gripper_max_speed_rad_s
        if speed is None:
            return target_rad
        previous = self._gripper_target_rad if self._gripper_target_rad is not None else measured_rad
        step = speed * dt
        self._gripper_target_rad = previous + min(max(target_rad - previous, -step), step)
        return self._gripper_target_rad

    def _gripper_limit(self, *, holding: bool) -> float:
        return self._gripper_mit_hold_torque_limit if holding else self._gripper_mit_torque_limit

    def _set_gripper_torque_limit(self, motor: Motor, *, holding: bool) -> None:
        """Write the gripper's firmware torque limit (hold or moving) when it changes, and now and then."""
        now = time.monotonic()
        if holding == self._gripper_holding and now - self._gripper_limit_written_at < _GRIPPER_LIMIT_REFRESH_S:
            return
        motor.robstride_write_param_f32(_LIMIT_TORQUE_PARAM, self._gripper_limit(holding=holding))
        self._gripper_holding = holding
        self._gripper_limit_written_at = now

    def _disconnect_motors(self) -> None:
        """Give the gripper back its full firmware torque range before the motors are disabled.

        The gripper is let go (damping only) first, so the higher limit never squeezes anything. With
        ``disable_torque_on_disconnect=False`` the motors stay on and the limit is kept.
        """
        gripper = self._motors.get("gripper")
        if self.disable_torque_on_disconnect and gripper is not None:
            with contextlib.suppress(Exception):
                gripper.send_mit(0.0, 0.0, 0.0, self._gripper_mit_kd, 0.0)
                gripper.robstride_write_param_f32(_LIMIT_TORQUE_PARAM, REBOT_B601_RS_GRIPPER_MAX_TORQUE_NM)
        super()._disconnect_motors()
