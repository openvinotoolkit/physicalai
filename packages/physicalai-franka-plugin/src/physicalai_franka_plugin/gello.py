# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ruff: file-ignore[docstring-missing-exception, docstring-missing-returns, magic-value-comparison, no-self-use, raise-within-try, too-many-statements-in-try-clause, type-check-without-type-error, undocumented-public-init, undocumented-public-method]

"""Native drivers for the Franka seven-axis GELLO and GELLO Duo leaders."""

from __future__ import annotations

import importlib
import json
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from physicalai.config import export_config
from physicalai_franka_plugin.constants import FR3_JOINT_LIMITS_RAD

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import RobotObservation

_RAD_PER_COUNT = float(2.0 * np.pi / 4095.0)
_PRESENT_POSITION_ADDRESS = 132
_TORQUE_ENABLE_ADDRESS = 64
_COMM_SUCCESS = 0
_NUM_ARM_JOINTS = 7
_NUM_TOTAL_JOINTS = 8


@dataclass(frozen=True)
class GelloCalibration:
    """FR3 GELLO assembly calibration, expressed in the official ROS2 convention."""

    joint_signs: tuple[int, int, int, int, int, int, int]
    assembly_offsets_rad: tuple[float, float, float, float, float, float, float]
    gripper_range_rad: tuple[float, float]
    baudrate: int = 57_600

    @classmethod
    def default(cls) -> GelloCalibration:
        """Return the calibration published for preassembled Franka GELLO hardware."""
        return cls(
            joint_signs=(1, -1, 1, -1, 1, 1, 1),
            assembly_offsets_rad=(0.0, 0.0, math.pi, math.pi, math.pi, 4.712, 0.0),
            gripper_range_rad=(2.00, 3.22),
        )

    @classmethod
    def from_json(cls, value: str) -> GelloCalibration:
        """Parse a custom calibration JSON string from advanced configuration."""
        try:
            data = json.loads(value)
        except json.JSONDecodeError as exc:
            msg = "GELLO calibration must be valid JSON"
            raise ValueError(msg) from exc
        if not isinstance(data, dict):
            msg = "GELLO calibration JSON must be an object"
            raise ValueError(msg)
        signs = data.get("joint_signs")
        offsets = data.get("assembly_offsets_rad")
        gripper_range = data.get("gripper_range_rad")
        if not isinstance(signs, list) or not isinstance(offsets, list) or not isinstance(gripper_range, list):
            msg = "GELLO calibration requires joint_signs, assembly_offsets_rad, and gripper_range_rad arrays"
            raise ValueError(msg)
        if len(signs) != 7 or len(offsets) != 7 or len(gripper_range) != 2:
            msg = "GELLO calibration requires 7 joint signs, 7 offsets, and 2 gripper endpoints"
            raise ValueError(msg)
        return cls(
            joint_signs=tuple(int(item) for item in signs),  # type: ignore[arg-type]
            assembly_offsets_rad=tuple(float(item) for item in offsets),  # type: ignore[arg-type]
            gripper_range_rad=(float(gripper_range[0]), float(gripper_range[1])),
            baudrate=int(data.get("baudrate", 57_600)),
        )


@dataclass
class GelloObservation:
    """Observation from one GELLO arm or a left/right GELLO Duo."""

    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None

    @property
    def state(self) -> np.ndarray:
        """Joint positions used as the leader state vector."""
        return self.joint_positions


@export_config(class_path="physicalai_franka_plugin.GelloLeader")
class GelloLeader:
    """Passive, seven-axis Franka GELLO leader on one Dynamixel serial bus.

    The default calibration is for the preassembled Franka GELLO. Supply a
    custom ``calibration`` only for a self-assembled or modified leader.
    """

    JOINT_ORDER: ClassVar[tuple[str, ...]] = (*tuple(f"joint_{index}" for index in range(7)), "gripper")
    _LIMITS_RAD: ClassVar[np.ndarray] = np.array([FR3_JOINT_LIMITS_RAD[f"fr3_joint{index}"] for index in range(1, 8)])

    def __init__(
        self, port: str, *, calibration: GelloCalibration | str | None = None, smoothing: float = 0.85
    ) -> None:
        if not port or not 0.0 < smoothing <= 1.0:
            msg = "port must be non-empty and smoothing must be in (0, 1]"
            raise ValueError(msg)
        self._port = port
        self._smoothing = smoothing
        resolved_calibration = GelloCalibration.from_json(calibration) if isinstance(calibration, str) else calibration
        self._calibration: GelloCalibration = resolved_calibration or GelloCalibration.default()
        self._port_handler: Any = None
        self._packet_handler: Any = None
        self._previous_raw: np.ndarray | None = None
        self._previous_positions: np.ndarray | None = None

    @property
    def port(self) -> str:
        return self._port

    @property
    def joint_names(self) -> list[str]:
        return list(self.JOINT_ORDER)

    @property
    def device_ids(self) -> tuple[str, ...]:
        return (f"serial:{self._port}",)

    def connect(self) -> None:
        if self.is_connected():
            return
        sdk = importlib.import_module("dynamixel_sdk")
        port_handler = sdk.PortHandler(self._port)
        packet_handler = sdk.PacketHandler(2.0)
        if not port_handler.openPort() or not port_handler.setBaudRate(self._calibration.baudrate):
            port_handler.closePort()
            msg = f"Could not open GELLO serial port {self._port!r} at {self._calibration.baudrate} baud"
            raise ConnectionError(msg)
        try:
            for motor_id in range(1, _NUM_TOTAL_JOINTS + 1):
                _model, comm_result, error = packet_handler.ping(port_handler, motor_id)
                if comm_result != _COMM_SUCCESS or error:
                    msg = f"GELLO motor ID {motor_id} did not respond on {self._port!r}"
                    raise ConnectionError(msg)
                comm_result, error = packet_handler.write1ByteTxRx(port_handler, motor_id, _TORQUE_ENABLE_ADDRESS, 0)
                if comm_result != _COMM_SUCCESS or error:
                    msg = f"Could not disable torque for GELLO motor ID {motor_id}"
                    raise ConnectionError(msg)
        except Exception:
            port_handler.closePort()
            raise
        self._port_handler = port_handler
        self._packet_handler = packet_handler
        raw = self._read_raw_rad()
        self._previous_raw = raw[:_NUM_ARM_JOINTS]
        self._previous_positions = self._normalize_initial(raw[:_NUM_ARM_JOINTS])

    def disconnect(self) -> None:
        if self._port_handler is not None:
            self._port_handler.closePort()
        self._port_handler = None
        self._packet_handler = None
        self._previous_raw = None
        self._previous_positions = None

    def is_connected(self) -> bool:
        return self._port_handler is not None

    def _read_raw_rad(self) -> np.ndarray:
        if self._port_handler is None or self._packet_handler is None:
            msg = "GELLO is not connected. Call connect() first."
            raise ConnectionError(msg)
        values = []
        for motor_id in range(1, _NUM_TOTAL_JOINTS + 1):
            value, comm_result, error = self._packet_handler.read4ByteTxRx(
                self._port_handler, motor_id, _PRESENT_POSITION_ADDRESS
            )
            if comm_result != _COMM_SUCCESS or error:
                msg = f"Failed to read GELLO motor ID {motor_id}"
                raise ConnectionError(msg)
            values.append(float(value) * _RAD_PER_COUNT)
        return np.asarray(values, dtype=np.float32)

    def _normalize_initial(self, raw: np.ndarray) -> np.ndarray:
        offsets = np.asarray(self._calibration.assembly_offsets_rad)
        signs = np.asarray(self._calibration.joint_signs)
        midpoint = self._LIMITS_RAD.mean(axis=1)
        return np.mod((raw - offsets) * signs - midpoint, 2 * np.pi) - np.pi + midpoint

    def get_observation(self) -> RobotObservation:
        raw = self._read_raw_rad()
        if self._previous_raw is None or self._previous_positions is None:
            msg = "GELLO has no initial state. Reconnect the leader."
            raise ConnectionError(msg)
        delta = np.mod(raw[:_NUM_ARM_JOINTS] - self._previous_raw + np.pi, 2 * np.pi) - np.pi
        positions = np.clip(
            self._previous_positions + delta * np.asarray(self._calibration.joint_signs),
            self._LIMITS_RAD[:, 0],
            self._LIMITS_RAD[:, 1],
        )
        self._previous_raw = raw[:_NUM_ARM_JOINTS]
        self._previous_positions = positions
        closed, opened = self._calibration.gripper_range_rad
        trigger = float(np.clip((raw[-1] - closed) / (opened - closed), 0.0, 1.0))
        output = np.concatenate([np.degrees(positions), [trigger]]).astype(np.float32)
        return GelloObservation(
            joint_positions=output, timestamp=time.monotonic(), sensor_data={"raw_positions_rad": raw}
        )

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        _ = action, goal_time
        msg = "Cannot send actions to a GELLO leader. It is read-only for teleoperation."
        raise RuntimeError(msg)


@export_config(class_path="physicalai_franka_plugin.BimanualGelloLeader")
class BimanualGelloLeader:
    """Left/right composition of two :class:`GelloLeader` instances."""

    def __init__(
        self,
        left_port: str,
        right_port: str,
        *,
        left_calibration: GelloCalibration | str | None = None,
        right_calibration: GelloCalibration | str | None = None,
        smoothing: float = 0.85,
    ) -> None:
        if not left_port or not right_port or left_port == right_port:
            msg = "left_port and right_port must be distinct non-empty serial paths"
            raise ValueError(msg)
        self._left = GelloLeader(left_port, calibration=left_calibration, smoothing=smoothing)
        self._right = GelloLeader(right_port, calibration=right_calibration, smoothing=smoothing)

    @property
    def joint_names(self) -> list[str]:
        return [
            *(f"left_{name}" for name in self._left.joint_names),
            *(f"right_{name}" for name in self._right.joint_names),
        ]

    @property
    def device_ids(self) -> tuple[str, ...]:
        return (*self._left.device_ids, *self._right.device_ids)

    def connect(self) -> None:
        self._left.connect()
        try:
            self._right.connect()
        except Exception:
            self._left.disconnect()
            raise

    def disconnect(self) -> None:
        self._left.disconnect()
        self._right.disconnect()

    def is_connected(self) -> bool:
        return self._left.is_connected() and self._right.is_connected()

    def get_observation(self) -> RobotObservation:
        left = self._left.get_observation()
        right = self._right.get_observation()
        if left.sensor_data is None or right.sensor_data is None:
            msg = "GELLO child observations are missing raw encoder readings"
            raise RuntimeError(msg)
        return GelloObservation(
            joint_positions=np.concatenate([left.joint_positions, right.joint_positions]),
            timestamp=min(left.timestamp, right.timestamp),
            sensor_data={
                "raw_positions_rad": np.concatenate([
                    left.sensor_data["raw_positions_rad"],
                    right.sensor_data["raw_positions_rad"],
                ])
            },
        )

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        _ = action, goal_time
        msg = "Cannot send actions to a GELLO leader. It is read-only for teleoperation."
        raise RuntimeError(msg)
