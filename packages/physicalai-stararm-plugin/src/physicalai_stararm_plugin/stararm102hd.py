"""Star Arm 102-HD leader driver."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

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
    """Protocol for a single angle-read sample from a FashionStar servo."""

    raw_deg: float
    filtered_deg: float
    reliable: bool


class _FashionStarBus(Protocol):
    """Protocol for the FashionStar UART servo bus."""

    def ping(self, servo_id: int) -> bool: ...
    def unlock(self, servo_id: int) -> None: ...
    def reset_multi_turn(self, servo_id: int) -> None: ...
    def set_origin_point(self, servo_id: int) -> None: ...
    def read_angle(self, servo_id: int, *, multi_turn: bool = True) -> _AngleSample: ...
    def set_angle(self, servo_id: int, angle_deg: float, *, multi_turn: bool = False, interval_ms: int = 0) -> None: ...
    def close(self) -> None: ...


@dataclass
class StarArm102HDLeaderObservation:
    """Observation data for the Star Arm 102-HD leader."""

    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None

    @property
    def state(self) -> np.ndarray:
        """Alias for joint positions, matching the Robot protocol."""
        return self.joint_positions


@export_config
class StarArm102HDLeader:
    """FashionStar UART leader arm driver.

    Connects to the Star Arm 102-HD leader arm via a UART-to-USB adapter.
    By default this driver runs in passive mode and only reads joint angles.
    In assist mode it can also accept position commands and hold poses.
    """

    JOINT_ORDER: ClassVar[list[str]] = list(STAR_ARM_102_JOINT_ORDER)
    NUM_JOINTS: ClassVar[int] = len(JOINT_ORDER)
    MODEL_NAME: ClassVar[str] = "Star Arm 102-HD"
    DEVICE_PREFIX: ClassVar[str] = "stararm102-hd"
    VALID_CONTROL_MODES: ClassVar[frozenset[str]] = frozenset({"passive", "assist"})

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        *,
        baudrate: int = 1_000_000,
        unlock_on_connect: bool = True,
        reset_multi_turn_on_connect: bool = True,
        zero_on_connect: bool = False,
        control_mode: Literal["passive", "assist"] = "passive",
        command_interval_ms: int = 10,
    ) -> None:
        """Initialize the FashionStar leader arm driver.

        Raises:
            ValueError: If baudrate/control_mode/command_interval_ms is invalid.
        """
        if baudrate <= 0:
            msg = f"baudrate must be a positive integer, got {baudrate!r}"
            raise ValueError(msg)
        if control_mode not in self.VALID_CONTROL_MODES:
            msg = f"Invalid control_mode {control_mode!r}. Must be one of {sorted(self.VALID_CONTROL_MODES)}."
            raise ValueError(msg)
        if command_interval_ms < 0:
            msg = f"command_interval_ms must be >= 0, got {command_interval_ms!r}"
            raise ValueError(msg)

        self._port = port
        self._baudrate = baudrate
        self._unlock_on_connect = unlock_on_connect
        self._reset_multi_turn_on_connect = reset_multi_turn_on_connect
        self._zero_on_connect = zero_on_connect
        self._control_mode = control_mode
        self._command_interval_ms = command_interval_ms
        self._bus: _FashionStarBus | None = None
        self._last_positions: np.ndarray | None = None
        self._last_raw_positions: np.ndarray | None = None
        self._last_reliable: np.ndarray | None = None
        self._holding = False

    @property
    def joint_names(self) -> list[str]:
        """Ordered list of joint names matching the expected observation layout."""
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
    def control_mode(self) -> Literal["passive", "assist"]:
        """Current control mode for this leader."""
        return self._control_mode

    @property
    def command_interval_ms(self) -> int:
        """Minimum command interval in milliseconds for assist/hold commands."""
        return self._command_interval_ms

    @property
    def is_holding(self) -> bool:
        """Whether hold mode is currently active."""
        return self._holding

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

        logger.info(f"{self.__class__.__name__} connected on {self.port}")

    def disconnect(self) -> None:
        """Close the UART bus and release resources."""
        bus = self._bus
        if bus is None:
            return
        self._holding = False
        self._bus = None
        bus.close()
        logger.info(f"{self.__class__.__name__} disconnected from {self.port}")

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
                msg = f"Failed to read {self.MODEL_NAME} leader positions: {e}"
                raise ConnectionError(msg) from e
            logger.warning(f"Failed to read {self.MODEL_NAME} leader positions; using last valid sample: {e}")
            positions = self._last_positions.copy()
            raw_positions = self._last_raw_positions.copy()
            reliable = np.zeros(self.NUM_JOINTS, dtype=np.float32)

        return StarArm102HDLeaderObservation(
            joint_positions=positions,
            timestamp=time.monotonic(),
            sensor_data={
                "raw_positions": raw_positions,
                "reliable": reliable,
            },
        )

    def _read_positions(self, bus: _FashionStarBus) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.empty(self.NUM_JOINTS, dtype=np.float32)
        raw_positions = np.empty(self.NUM_JOINTS, dtype=np.float32)
        reliable = np.empty(self.NUM_JOINTS, dtype=np.float32)

        for i, name in enumerate(self.JOINT_ORDER):
            servo_id = STAR_ARM_102_JOINT_IDS[name]
            sample = bus.read_angle(servo_id, multi_turn=True)
            range_min, range_max = STAR_ARM_102_JOINT_RANGES_DEG[name]
            unwrapped, _ = self._round_to_valid_range(float(sample.filtered_deg), range_min, range_max)
            positions[i] = float(np.clip(unwrapped, range_min, range_max))
            raw_positions[i] = float(sample.raw_deg)
            reliable[i] = 1.0 if sample.reliable else 0.0

        return positions, raw_positions, reliable

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Optionally command the HD leader in assist mode."""
        if self._control_mode != "assist":
            return
        self._send_action_internal(action, goal_time=goal_time)

    def hold_position(self, *, goal_time: float = 0.2) -> None:
        """Capture the current pose and command the HD leader to hold it."""
        obs = self.get_observation()
        self._send_action_internal(np.asarray(obs.joint_positions, dtype=np.float32), goal_time=goal_time)
        self._holding = True

    def release_hold(self) -> None:
        """Release hold mode and return to manual guidance."""
        self._holding = False

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
