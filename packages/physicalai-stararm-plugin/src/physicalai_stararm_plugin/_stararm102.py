# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared FashionStar UART implementation for Star Arm 102 variants."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

import numpy as np
from loguru import logger

from physicalai_stararm_plugin.constants import (
    STAR_ARM_102_JOINT_IDS,
    STAR_ARM_102_JOINT_ORDER,
    STAR_ARM_102_JOINT_RANGES_DEG,
)

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import RobotObservation

from motorbridge_smart_servo import FashionStarServo

_GLITCH_JUMP_DEG = 90.0
_GLITCH_MAX_HELD_SAMPLES = 3


class _ServoMonitor(Protocol):
    angle_deg: float
    reliable: bool


class _FashionStarBus(Protocol):
    def ping(self, servo_id: int) -> bool: ...
    def unlock(self, servo_id: int) -> None: ...
    def reset_multi_turn(self, servo_id: int) -> None: ...
    def set_origin_point(self, servo_id: int) -> None: ...
    def sync_monitor(self, servo_ids: list[int]) -> dict[int, _ServoMonitor | None]: ...
    def set_angle(self, servo_id: int, angle_deg: float, *, multi_turn: bool = False, interval_ms: int = 0) -> None: ...
    def close(self) -> None: ...


@dataclass
class _StarArm102Observation:
    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None

    @property
    def state(self) -> np.ndarray:
        """Alias for joint positions, matching the Robot protocol."""
        return self.joint_positions


class _StarArm102:
    """Common servo transport, observation, and command behavior."""

    JOINT_ORDER: ClassVar[list[str]] = list(STAR_ARM_102_JOINT_ORDER)
    NUM_JOINTS: ClassVar[int] = len(JOINT_ORDER)
    MODEL_NAME: ClassVar[str]
    DEVICE_PREFIX: ClassVar[str]
    OBSERVATION_CLASS: ClassVar[type[_StarArm102Observation]] = _StarArm102Observation

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        *,
        baudrate: int = 1_000_000,
        unlock_on_connect: bool = True,
        reset_multi_turn_on_connect: bool = True,
        zero_on_connect: bool = False,
        command_interval_ms: int = 10,
    ) -> None:
        """Initialize shared Star Arm servo communication and state.

        Args:
            port: UART serial device connected to the arm.
            baudrate: UART communication speed in bits per second.
            unlock_on_connect: Unlock every servo when connecting.
            reset_multi_turn_on_connect: Reset each servo's accumulated turn count when connecting.
            zero_on_connect: Store the current servo positions as their origins when connecting.
            command_interval_ms: Minimum duration passed to servo position commands.

        Raises:
            ValueError: If ``baudrate`` is not positive or ``command_interval_ms`` is negative.
        """
        if baudrate <= 0:
            msg = f"baudrate must be a positive integer, got {baudrate!r}"
            raise ValueError(msg)
        if command_interval_ms < 0:
            msg = f"command_interval_ms must be >= 0, got {command_interval_ms!r}"
            raise ValueError(msg)

        self._port = port
        self._baudrate = baudrate
        self._unlock_on_connect = unlock_on_connect
        self._reset_multi_turn_on_connect = reset_multi_turn_on_connect
        self._zero_on_connect = zero_on_connect
        self._command_interval_ms = command_interval_ms
        self._bus: _FashionStarBus | None = None
        # Samples are cached in the native Star Arm frame. Subclasses may
        # transform only the public observation returned to callers.
        self._last_positions: np.ndarray | None = None
        self._last_raw_positions: np.ndarray | None = None
        self._last_reliable: np.ndarray | None = None
        self._glitch_held = np.zeros(self.NUM_JOINTS, dtype=np.int32)

    @property
    def joint_names(self) -> list[str]:
        """Ordered list of joint names matching the observation/action layout."""
        return self.JOINT_ORDER

    @property
    def device_ids(self) -> tuple[str, ...]:
        """Configured UART transport identity without opening it."""
        return (f"{self.DEVICE_PREFIX}:{self._port}",)

    @property
    def port(self) -> str:
        """UART serial port the driver is configured for."""
        return self._port

    @property
    def baudrate(self) -> int:
        """Serial baud rate for the FashionStar bus."""
        return self._baudrate

    @property
    def zero_on_connect(self) -> bool:
        """Whether the current position is set as origin point on connect."""
        return self._zero_on_connect

    @property
    def command_interval_ms(self) -> int:
        """Minimum command interval in milliseconds."""
        return self._command_interval_ms

    def _require_bus(self) -> _FashionStarBus:
        bus = self._bus
        if bus is None:
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)
        return bus

    def connect(self) -> None:
        """Open the UART bus, ping all servos, and configure them."""
        if self.is_connected():
            return

        bus = FashionStarServo(self.port, baudrate=self.baudrate)
        try:
            self._ping_servos(bus)
            self._bus = bus
            self._configure_servos(bus)
        except Exception:
            with contextlib.suppress(Exception):
                bus.close()
            self._bus = None
            raise

        self._reset_samples()
        logger.info(f"{self.__class__.__name__} connected on {self.port}")

    def disconnect(self) -> None:
        """Close the UART bus and release resources."""
        bus = self._bus
        if bus is None:
            return
        self._bus = None
        self._reset_samples()
        bus.close()
        logger.info(f"{self.__class__.__name__} disconnected from {self.port}")

    def is_connected(self) -> bool:
        """Return whether the UART bus connection is active."""
        return self._bus is not None

    def _reset_samples(self) -> None:
        self._last_positions = None
        self._last_raw_positions = None
        self._last_reliable = None
        self._glitch_held[:] = 0

    def _ping_servos(self, bus: _FashionStarBus) -> None:
        for name in self.JOINT_ORDER:
            servo_id = STAR_ARM_102_JOINT_IDS[name]
            if not bus.ping(servo_id):
                msg = f"Servo '{name}' (ID {servo_id}) did not respond on {self.port}."
                raise ConnectionError(msg)

    def _configure_servos(self, bus: _FashionStarBus) -> None:
        for name in self.JOINT_ORDER:
            servo_id = STAR_ARM_102_JOINT_IDS[name]
            if self._unlock_on_connect:
                bus.unlock(servo_id)
            if self._zero_on_connect:
                bus.set_origin_point(servo_id)
            if self._reset_multi_turn_on_connect:
                bus.reset_multi_turn(servo_id)

    def disable_torque(self) -> None:
        """Unlock every servo so the arm can be moved by hand."""
        bus = self._require_bus()
        for name in self.JOINT_ORDER:
            bus.unlock(STAR_ARM_102_JOINT_IDS[name])

    def set_zero_position(self) -> None:
        """Store the arm's current pose as every servo's origin."""
        bus = self._require_bus()
        for name in self.JOINT_ORDER:
            servo_id = STAR_ARM_102_JOINT_IDS[name]
            bus.set_origin_point(servo_id)
            if self._reset_multi_turn_on_connect:
                bus.reset_multi_turn(servo_id)
        self._reset_samples()

    def get_observation(self) -> RobotObservation:
        """Read joint positions from all servos.

        Returns:
            Observation containing filtered/raw positions and reliability flags.

        Raises:
            ConnectionError: If no prior sample exists and reading fails.
        """
        bus = self._require_bus()
        try:
            positions, raw_positions, reliable = self._read_positions(bus)
            self._last_positions = positions
            self._last_raw_positions = raw_positions
            self._last_reliable = reliable
        except Exception as e:
            if self._last_positions is None or self._last_raw_positions is None or self._last_reliable is None:
                msg = f"Failed to read {self.MODEL_NAME} positions: {e}"
                raise ConnectionError(msg) from e
            logger.warning(f"Failed to read {self.MODEL_NAME} positions; using last valid sample: {e}")
            positions = self._last_positions.copy()
            raw_positions = self._last_raw_positions.copy()
            reliable = np.zeros(self.NUM_JOINTS, dtype=np.float32)

        return self.OBSERVATION_CLASS(
            joint_positions=self._observation_positions(positions),
            timestamp=time.monotonic(),
            sensor_data={"raw_positions": raw_positions, "reliable": reliable},
        )

    def _read_positions(self, bus: _FashionStarBus) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.empty(self.NUM_JOINTS, dtype=np.float32)
        raw_positions = np.empty(self.NUM_JOINTS, dtype=np.float32)
        reliable = np.empty(self.NUM_JOINTS, dtype=np.float32)
        monitors = bus.sync_monitor([STAR_ARM_102_JOINT_IDS[name] for name in self.JOINT_ORDER])
        last_positions = self._last_positions

        for i, name in enumerate(self.JOINT_ORDER):
            servo_id = STAR_ARM_102_JOINT_IDS[name]
            monitor = monitors.get(servo_id)
            if monitor is None:
                msg = f"Servo '{name}' (ID {servo_id}) has never responded on {self.port}."
                raise ConnectionError(msg)
            range_min, range_max = STAR_ARM_102_JOINT_RANGES_DEG[name]
            unwrapped, _ = self._round_to_valid_range(float(monitor.angle_deg), range_min, range_max)
            positions[i] = float(np.clip(unwrapped, range_min, range_max))
            raw_positions[i] = float(monitor.angle_deg)
            reliable[i] = 1.0 if monitor.reliable else 0.0

            if (
                last_positions is not None
                and abs(positions[i] - last_positions[i]) > _GLITCH_JUMP_DEG
                and self._glitch_held[i] < _GLITCH_MAX_HELD_SAMPLES
            ):
                self._glitch_held[i] += 1
                positions[i] = last_positions[i]
                reliable[i] = 0.0
            else:
                self._glitch_held[i] = 0

        return positions, raw_positions, reliable

    def _observation_positions(self, positions: np.ndarray) -> np.ndarray:  # ruff: ignore[PLR6301]
        """Convert native positions to this variant's public observation frame.

        Returns:
            Joint positions in the variant's public frame.
        """
        return positions.copy()

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Command joints to the given absolute targets in degrees."""
        self._send_action_internal(action, goal_time=goal_time)

    def _send_action_internal(self, action: np.ndarray, *, goal_time: float) -> None:
        bus = self._require_bus()
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.shape != (self.NUM_JOINTS,):
            msg = f"Expected action shape ({self.NUM_JOINTS},), got {tuple(action_arr.shape)}"
            raise ValueError(msg)

        interval_ms = self._goal_time_to_interval_ms(goal_time)
        for i, name in enumerate(self.JOINT_ORDER):
            servo_id = STAR_ARM_102_JOINT_IDS[name]
            range_min, range_max = STAR_ARM_102_JOINT_RANGES_DEG[name]
            target = float(np.clip(float(action_arr[i]), range_min, range_max))
            bus.set_angle(servo_id, target, multi_turn=True, interval_ms=interval_ms)

    def _goal_time_to_interval_ms(self, goal_time: float) -> int:
        if goal_time <= 0.0:
            return self._command_interval_ms
        return max(int(goal_time * 1000.0), self._command_interval_ms)

    @staticmethod
    def _round_to_valid_range(value: float, min_value: float, max_value: float) -> tuple[float, int]:
        center = (min_value + max_value) / 2.0
        low = center - 180.0
        high = center + 180.0
        for k in range(4096):
            candidate_plus = value + k * 360.0
            if low <= candidate_plus <= high:
                return candidate_plus, k
            candidate_minus = value - k * 360.0
            if low <= candidate_minus <= high:
                return candidate_minus, k
        return value - round((value - center) / 360.0) * 360.0, 4096
