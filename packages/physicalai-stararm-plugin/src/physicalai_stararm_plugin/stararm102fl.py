"""Star Arm 102-FL follower driver."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

import numpy as np
from loguru import logger
from physicalai.config import export_config

from physicalai_stararm_plugin.constants import (
    STAR_ARM_102_JOINT_IDS,
    STAR_ARM_102_JOINT_ORDER,
    STAR_ARM_102_JOINT_RANGES_DEG,
)

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import RobotObservation

from motorbridge_smart_servo import FashionStarServo


class _AngleSample(Protocol):
    raw_deg: float
    filtered_deg: float
    reliable: bool


class _FashionStarBus(Protocol):
    def ping(self, servo_id: int) -> bool: ...
    def unlock(self, servo_id: int) -> None: ...
    def reset_multi_turn(self, servo_id: int) -> None: ...
    def set_origin_point(self, servo_id: int) -> None: ...
    def read_angle(self, servo_id: int, *, multi_turn: bool = True) -> _AngleSample: ...
    def set_angle(self, servo_id: int, angle_deg: float, *, multi_turn: bool = False, interval_ms: int = 0) -> None: ...
    def close(self) -> None: ...


@dataclass
class StarArm102FLFollowerObservation:
    """Observation data for the Star Arm 102-FL follower."""

    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None

    @property
    def state(self) -> np.ndarray:
        """Alias for joint positions, matching the Robot protocol."""
        return self.joint_positions


@export_config
class StarArm102FLFollower:
    """FashionStar UART follower arm driver (position control)."""

    JOINT_ORDER: ClassVar[list[str]] = list(STAR_ARM_102_JOINT_ORDER)
    NUM_JOINTS: ClassVar[int] = len(JOINT_ORDER)

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
        """Initialize the Star Arm 102-FL follower driver.

        Raises:
            ValueError: If baudrate or command_interval_ms is invalid.
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

    @property
    def joint_names(self) -> list[str]:
        """Ordered list of joint names matching the expected action layout."""
        return self.JOINT_ORDER

    @property
    def device_ids(self) -> tuple[str, ...]:
        """Configured UART transport identity without opening it."""
        return (f"stararm102-fl:{self._port}",)

    @property
    def port(self) -> str:
        """UART serial port the driver is configured for."""
        return self._port

    @property
    def baudrate(self) -> int:
        """Serial baud rate for the FashionStar bus."""
        return self._baudrate

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

        logger.info(f"StarArm102FLFollower connected on {self.port}")

    def disconnect(self) -> None:
        """Close the UART bus and release resources."""
        bus = self._bus
        if bus is None:
            return
        self._bus = None
        bus.close()
        logger.info(f"StarArm102FLFollower disconnected from {self.port}")

    def is_connected(self) -> bool:
        """Return whether the UART bus connection is active."""
        return self._bus is not None

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

    def get_observation(self) -> RobotObservation:
        """Read and return the current follower joint positions.

        Returns:
            Observation containing filtered/raw positions and reliability flags.
        """
        bus = self._require_bus()
        positions = np.empty(self.NUM_JOINTS, dtype=np.float32)
        raw_positions = np.empty(self.NUM_JOINTS, dtype=np.float32)
        reliable = np.empty(self.NUM_JOINTS, dtype=np.float32)

        for i, name in enumerate(self.JOINT_ORDER):
            servo_id = STAR_ARM_102_JOINT_IDS[name]
            sample = bus.read_angle(servo_id, multi_turn=True)
            range_min, range_max = STAR_ARM_102_JOINT_RANGES_DEG[name]
            positions[i] = float(np.clip(float(sample.filtered_deg), range_min, range_max))
            raw_positions[i] = float(sample.raw_deg)
            reliable[i] = 1.0 if sample.reliable else 0.0

        return StarArm102FLFollowerObservation(
            joint_positions=positions,
            timestamp=time.monotonic(),
            sensor_data={
                "raw_positions": raw_positions,
                "reliable": reliable,
            },
        )

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Command follower joints to the given absolute targets in degrees.

        Raises:
            ValueError: If ``action`` does not match the expected joint vector shape.
        """
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
