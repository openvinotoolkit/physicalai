# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ruff: file-ignore[docstring-missing-exception, docstring-missing-returns, magic-value-comparison, no-self-use, raise-within-try, too-many-statements-in-try-clause]

"""Native USB Dynamixel driver for a bimanual GELLO DUO leader.

The hardware and calibration math follow the Apache-2.0 GELLO LeRobot plugin
from F-Fer/lerobot_ur5e_gello. Unlike that plugin, this module exposes the
PhysicalAI ``Robot`` protocol directly and never prompts interactively during a
runtime connection.
"""

from __future__ import annotations

import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from physicalai.config import export_config

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import RobotObservation

_RAD_PER_COUNT = float(2.0 * np.pi / 4095.0)
_PRESENT_POSITION_ADDRESS = 132
_PRESENT_POSITION_LENGTH = 4
_TORQUE_ENABLE_ADDRESS = 64
_COMM_SUCCESS = 0


@dataclass(frozen=True)
class GelloCalibration:
    """Encoder offsets and trigger endpoints for one six-axis GELLO arm."""

    joint_offsets: tuple[int, int, int, int, int, int]
    gripper_open_position: int
    gripper_closed_position: int

    @classmethod
    def from_file(cls, path: str | Path) -> GelloCalibration:
        """Load the calibration JSON written by the referenced GELLO plugin."""
        with Path(path).open(encoding="utf-8") as calibration_file:
            data = json.load(calibration_file)
        offsets = data.get("joint_offsets")
        if isinstance(offsets, dict):
            offsets = [offsets[f"joint_{index}"] for index in range(6)]
        if not isinstance(offsets, list) or len(offsets) != 6:
            msg = "GELLO calibration joint_offsets must contain exactly six encoder counts"
            raise ValueError(msg)
        typed_offsets = tuple(int(value) for value in offsets)
        return cls(
            joint_offsets=(
                typed_offsets[0],
                typed_offsets[1],
                typed_offsets[2],
                typed_offsets[3],
                typed_offsets[4],
                typed_offsets[5],
            ),
            gripper_open_position=int(data["gripper_open_position"]),
            gripper_closed_position=int(data["gripper_closed_position"]),
        )


@dataclass
class GelloObservation:
    """Observation from one GELLO arm or a left/right GELLO DUO."""

    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None

    @property
    def state(self) -> np.ndarray:
        """Joint positions used as the leader state vector."""
        return self.joint_positions


class _GelloArm:
    """One passive six-axis GELLO arm on a dedicated USB Dynamixel bus."""

    def __init__(
        self,
        port: str,
        calibration_path: str,
        *,
        baudrate: int,
        joint_signs: tuple[int, int, int, int, int, int],
        calibration_position_rad: tuple[float, float, float, float, float, float],
        smoothing: float,
    ) -> None:
        self.port = port
        self.calibration_path = calibration_path
        self.baudrate = baudrate
        self.joint_signs = joint_signs
        self.calibration_position_rad = calibration_position_rad
        self.smoothing = smoothing
        self._sdk: Any = None
        self._port_handler: Any = None
        self._packet_handler: Any = None
        self._calibration: GelloCalibration | None = None
        self._last_positions: np.ndarray | None = None

    def connect(self) -> None:
        """Open the bus, verify every motor, and release torque for hand guidance."""
        if self._port_handler is not None:
            return
        sdk = importlib.import_module("dynamixel_sdk")
        port_handler = sdk.PortHandler(self.port)
        packet_handler = sdk.PacketHandler(2.0)
        if not port_handler.openPort():
            msg = f"Could not open GELLO serial port {self.port!r}"
            raise ConnectionError(msg)
        try:
            if not port_handler.setBaudRate(self.baudrate):
                msg = f"Could not set GELLO serial port {self.port!r} to {self.baudrate} baud"
                raise ConnectionError(msg)
            self._calibration = GelloCalibration.from_file(self.calibration_path)
            for motor_id in range(1, 8):
                _model, comm_result, error = packet_handler.ping(port_handler, motor_id)
                if comm_result != _COMM_SUCCESS or error:
                    msg = f"GELLO motor ID {motor_id} did not respond on {self.port!r}"
                    raise ConnectionError(msg)
                _result, comm_result, error = packet_handler.write1ByteTxRx(
                    port_handler, motor_id, _TORQUE_ENABLE_ADDRESS, 0
                )
                if comm_result != _COMM_SUCCESS or error:
                    msg = f"Could not disable torque for GELLO motor ID {motor_id} on {self.port!r}"
                    raise ConnectionError(msg)
        except Exception:
            port_handler.closePort()
            raise
        self._sdk = sdk
        self._port_handler = port_handler
        self._packet_handler = packet_handler
        self._last_positions = None

    def disconnect(self) -> None:
        """Close the serial port; torque remains disabled for the passive leader."""
        if self._port_handler is not None:
            self._port_handler.closePort()
        self._sdk = None
        self._port_handler = None
        self._packet_handler = None
        self._calibration = None
        self._last_positions = None

    def is_connected(self) -> bool:
        """Whether this USB bus is currently open."""
        return self._port_handler is not None

    def read_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Read six joint angles in degrees and normalized trigger position."""
        if self._port_handler is None or self._packet_handler is None or self._calibration is None:
            msg = "GELLO is not connected. Call connect() first."
            raise ConnectionError(msg)
        raw = np.empty(7, dtype=np.float32)
        for index, motor_id in enumerate(range(1, 8)):
            value, comm_result, error = self._packet_handler.read4ByteTxRx(
                self._port_handler, motor_id, _PRESENT_POSITION_ADDRESS
            )
            if comm_result != _COMM_SUCCESS or error:
                msg = f"Failed to read GELLO motor ID {motor_id} on {self.port!r}"
                raise ConnectionError(msg)
            raw[index] = value
        angles_rad = np.array(
            [
                sign * (raw[index] - offset) * _RAD_PER_COUNT + reference
                for index, (sign, offset, reference) in enumerate(
                    zip(self.joint_signs, self._calibration.joint_offsets, self.calibration_position_rad, strict=True)
                )
            ],
            dtype=np.float32,
        )
        denominator = self._calibration.gripper_closed_position - self._calibration.gripper_open_position
        if denominator == 0:
            msg = f"Invalid GELLO calibration at {self.calibration_path!r}: gripper endpoints are identical"
            raise ValueError(msg)
        positions = np.concatenate([
            np.degrees(angles_rad),
            np.array([(raw[6] - self._calibration.gripper_open_position) / denominator]),
        ]).astype(np.float32)
        if self._last_positions is not None:
            positions = self.smoothing * positions + (1.0 - self.smoothing) * self._last_positions
        self._last_positions = positions.copy()
        return positions, raw


@export_config(class_path="physicalai_franka_plugin.BimanualGelloLeader")
class BimanualGelloLeader:
    """Read-only GELLO DUO leader backed by two USB Dynamixel buses.

    Positions are left arm then right arm. Each arm has six joint angles in
    degrees followed by a normalized ``0..1`` trigger value. This follows the
    reference plugin's calibration and sign convention while matching FR3 arm
    angle units at the PhysicalAI boundary.
    """

    JOINT_ORDER: ClassVar[tuple[str, ...]] = (
        "joint_0",
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "gripper",
    )
    _DEFAULT_SIGNS: ClassVar[tuple[int, int, int, int, int, int]] = (1, 1, -1, 1, 1, 1)
    _DEFAULT_REFERENCE_RAD: ClassVar[tuple[float, float, float, float, float, float]] = (
        0.0,
        -1.57,
        1.57,
        -1.57,
        -1.57,
        -1.57,
    )

    def __init__(
        self,
        left_port: str,
        right_port: str,
        *,
        left_calibration_path: str,
        right_calibration_path: str,
        baudrate: int = 57_600,
        joint_signs: tuple[int, int, int, int, int, int] = _DEFAULT_SIGNS,
        calibration_position_rad: tuple[float, float, float, float, float, float] = _DEFAULT_REFERENCE_RAD,
        smoothing: float = 0.85,
    ) -> None:
        """Configure the two serial buses and their explicit calibration files."""
        if not left_port or not right_port or left_port == right_port:
            msg = "left_port and right_port must be distinct non-empty serial paths"
            raise ValueError(msg)
        if not left_calibration_path or not right_calibration_path:
            msg = "A calibration JSON path is required for each GELLO arm"
            raise ValueError(msg)
        if baudrate <= 0 or not 0.0 < smoothing <= 1.0:
            msg = "baudrate must be positive and smoothing must be in (0, 1]"
            raise ValueError(msg)
        if len(joint_signs) != 6 or len(calibration_position_rad) != 6:
            msg = "joint_signs and calibration_position_rad must each contain six values"
            raise ValueError(msg)
        arm_args = {
            "baudrate": baudrate,
            "joint_signs": joint_signs,
            "calibration_position_rad": calibration_position_rad,
            "smoothing": smoothing,
        }
        self._left = _GelloArm(left_port, left_calibration_path, **arm_args)
        self._right = _GelloArm(right_port, right_calibration_path, **arm_args)

    @property
    def joint_names(self) -> list[str]:
        """Left-prefixed joint names followed by right-prefixed joint names."""
        return [*(f"left_{name}" for name in self.JOINT_ORDER), *(f"right_{name}" for name in self.JOINT_ORDER)]

    @property
    def device_ids(self) -> tuple[str, ...]:
        """Stable serial identities for the two exclusively owned USB buses."""
        return (f"serial:{self._left.port}", f"serial:{self._right.port}")

    def connect(self) -> None:
        """Connect both serial buses, rolling the left bus back if needed."""
        if self.is_connected():
            return
        self._left.connect()
        try:
            self._right.connect()
        except Exception:
            self._left.disconnect()
            raise

    def disconnect(self) -> None:
        """Close both GELLO buses."""
        self._left.disconnect()
        self._right.disconnect()

    def is_connected(self) -> bool:
        """Whether both GELLO arms are connected."""
        return self._left.is_connected() and self._right.is_connected()

    def get_observation(self) -> RobotObservation:
        """Read left/right joint positions and raw Dynamixel encoder counts."""
        left_positions, left_raw = self._left.read_positions()
        right_positions, right_raw = self._right.read_positions()
        return GelloObservation(
            joint_positions=np.concatenate([left_positions, right_positions]),
            timestamp=time.monotonic(),
            sensor_data={"raw_positions": np.concatenate([left_raw, right_raw])},
        )

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Reject actions because GELLO is a passive teleoperation leader.

        Raises:
            RuntimeError: Always; leaders do not receive position commands.
        """
        _ = action, goal_time
        msg = "Cannot send actions to a GELLO leader. It is read-only for teleoperation."
        raise RuntimeError(msg)
