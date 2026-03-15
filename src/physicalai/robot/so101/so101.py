# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SO-101 robot arm driver.

Concrete implementation of the :class:`~physicalai.robot.protocol.Robot` protocol
for the SO-101 robot arm (6-DOF, Feetech STS3215 servos).

Requires the ``feetech-servo-sdk`` package::

    pip install physicalai[so101]

The driver supports two roles:

* **follower** (default) — torque enabled, used for inference / deployment.
* **leader** — torque disabled, used for teleoperation (read-only).

Calibration data can optionally be loaded from a JSON file so that joint
positions are reported in radians rather than raw servo ticks.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TICKS_PER_REVOLUTION = 4096
"""STS3215 encoder resolution: 4096 ticks per full 360° revolution."""

_RADIANS_PER_TICK = 2.0 * np.pi / _TICKS_PER_REVOLUTION

_VALID_ROLES = frozenset({"leader", "follower"})

# Feetech STS3215 control table addresses
_ADDR_TORQUE_ENABLE = 40
_ADDR_GOAL_POSITION = 42
_ADDR_PRESENT_POSITION = 56

# Byte widths for sync read / sync write
_LEN_GOAL_POSITION = 2
_LEN_PRESENT_POSITION = 2

# Protocol version for STS / SCS bus
_PROTOCOL_VERSION = 0


class SO101:
    """Driver for the SO-101 robot arm (6-DOF, Feetech STS3215 servos).

    Args:
        port: Serial port path, e.g. ``"/dev/ttyUSB0"`` or ``"/dev/ttyACM0"``.
        baudrate: Serial baudrate. Defaults to 1 000 000 (STS3215 factory default).
        role: ``"follower"`` (torque enabled, full control) or ``"leader"``
            (torque disabled, read-only for teleoperation).
        servo_ids: Optional mapping from joint name to servo ID.  Defaults to
            IDs 1-6 in ``JOINT_ORDER``.
        calibration_path: Optional path to a JSON calibration file.  When
            provided, joint positions are reported in radians.  When ``None``,
            positions are raw servo ticks (0-4095) and a warning is logged once.
    """

    JOINT_ORDER: ClassVar[list[str]] = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    """Canonical joint ordering (index 0 → first element of state vector)."""

    NUM_JOINTS: ClassVar[int] = 6
    """Number of joints / servos on the SO-101."""

    def __init__(
        self,
        port: str,
        baudrate: int = 1_000_000,
        role: str = "follower",
        servo_ids: dict[str, int] | None = None,
        calibration_path: str | Path | None = None,
    ) -> None:
        """Initialize the SO-101 driver (does not open the connection).

        Raises:
            ValueError: If ``role`` is not ``"leader"`` or ``"follower"``.
        """
        if role not in _VALID_ROLES:
            msg = f"Invalid role {role!r}. Must be one of {sorted(_VALID_ROLES)}."
            raise ValueError(msg)

        self.port = port
        self.baudrate = baudrate
        self.role = role

        # Servo ID mapping — default 1..6 in JOINT_ORDER
        self.servo_ids: dict[str, int] = servo_ids or {
            name: idx + 1 for idx, name in enumerate(self.JOINT_ORDER)
        }

        # Calibration -------------------------------------------------------
        self._calibration: dict[str, dict[str, Any]] | None = None
        self._warned_uncalibrated = False
        if calibration_path is not None:
            self._calibration = self._load_calibration(Path(calibration_path))
            # Use servo IDs from calibration when none were explicitly given.
            if servo_ids is None:
                self.servo_ids = {
                    name: self._calibration[name]["id"]
                    for name in self.JOINT_ORDER
                }

        # Connection state (set during connect()) --------------------------
        self._port_handler: Any | None = None
        self._packet_handler: Any | None = None
        self._group_sync_read: Any | None = None
        self._group_sync_write: Any | None = None

    def connect(self) -> None:
        """Open the serial port, ping all servos, and configure torque.

        Idempotent: calling ``connect()`` on an already-connected robot is a
        no-op.

        Raises:
            ImportError: If ``feetech-servo-sdk`` is not installed.
            ConnectionError: If the serial port cannot be opened or a servo
                does not respond to ping.
        """
        if self._port_handler is not None:
            return  # already connected

        # Lazy import — only pull in the SDK when actually connecting.
        try:
            from scservo_sdk import (  # type: ignore[import-untyped]  # noqa: PLC0415
                GroupSyncRead,
                GroupSyncWrite,
                PacketHandler,
                PortHandler,
            )
        except ImportError:
            msg = (
                "feetech-servo-sdk is required for SO-101 support. "
                "Install it with:  pip install physicalai[so101]"
            )
            raise ImportError(msg) from None

        # Open port ---------------------------------------------------------
        self._port_handler = PortHandler(self.port)
        if not self._port_handler.openPort():
            self._port_handler = None
            msg = f"Failed to open serial port {self.port}"
            raise ConnectionError(msg)

        # Set a packet timeout so pings/reads don't block forever.
        self._port_handler.setPacketTimeoutMillis(50.0)

        if not self._port_handler.setBaudRate(self.baudrate):
            self._port_handler.closePort()
            self._port_handler = None
            msg = f"Failed to set baudrate {self.baudrate} on {self.port}"
            raise ConnectionError(msg)

        self._packet_handler = PacketHandler(_PROTOCOL_VERSION)

        # Ping all servos ---------------------------------------------------
        self._ping_servos()

        # Sync read / write groups -----------------------------------------
        self._group_sync_read = GroupSyncRead(
            self._port_handler,
            self._packet_handler,
            _ADDR_PRESENT_POSITION,
            _LEN_PRESENT_POSITION,
        )
        for servo_id in self.servo_ids.values():
            if not self._group_sync_read.addParam(servo_id):
                msg = f"Failed to add servo {servo_id} to sync read group"
                raise ConnectionError(msg)

        self._group_sync_write = GroupSyncWrite(
            self._port_handler,
            self._packet_handler,
            _ADDR_GOAL_POSITION,
            _LEN_GOAL_POSITION,
        )

        # Configure torque based on role ------------------------------------
        self._set_torque(enabled=self.role == "follower")

        logger.info(
            "SO-101 connected on %s (role=%s, servos=%s)",
            self.port,
            self.role,
            self.servo_ids,
        )

    def disconnect(self) -> None:
        """Disconnect from the robot, leaving it in a safe state.

        * **Follower**: torque remains enabled (arm holds position).
        * **Leader**: torque stays disabled.

        Idempotent: calling ``disconnect()`` when not connected is a no-op.
        """
        if self._port_handler is None:
            return  # not connected

        if self.role == "follower":
            self._hold_position()

        self._group_sync_read = None
        self._group_sync_write = None
        self._packet_handler = None
        self._port_handler.closePort()
        self._port_handler = None

        logger.info("SO-101 disconnected from %s", self.port)

    def get_observation(self) -> dict[str, Any]:
        """Read current joint positions from all servos.

        Returns:
            A dict with:

            * ``"state"``: ``np.ndarray`` of shape ``(6,)`` — joint positions
              in radians (if calibrated) or raw ticks (if uncalibrated).
            * ``"timestamp"``: ``float`` from ``time.monotonic()``.
        """
        raw_positions = self._read_joint_positions()

        if self._calibration is not None:
            state = self._ticks_to_radians(raw_positions)
        else:
            if not self._warned_uncalibrated:
                logger.warning(
                    "No calibration file provided. Joint positions are in raw "
                    "servo ticks (0-4095), not radians.",
                )
                self._warned_uncalibrated = True
            state = raw_positions.astype(np.float32)

        return {
            "state": state,
            "timestamp": time.monotonic(),
        }

    def send_action(self, action: np.ndarray) -> None:
        """Send joint position commands to all servos.

        Args:
            action: Array of shape ``(6,)`` with target joint positions in
                radians (calibrated) or raw ticks (uncalibrated).

        Raises:
            RuntimeError: If the robot is in ``"leader"`` role.
            ValueError: If the action shape does not match ``(6,)``.
        """
        if self.role == "leader":
            msg = (
                "Cannot send actions to a leader arm. "
                "Leader arms are read-only for teleoperation."
            )
            raise RuntimeError(msg)

        expected_shape = (self.NUM_JOINTS,)
        if action.shape != expected_shape:
            msg = f"Expected action shape {expected_shape}, got {action.shape}"
            raise ValueError(msg)

        ticks = self._radians_to_ticks(action) if self._calibration is not None else np.round(action).astype(np.int32)
        self._write_joint_positions(ticks)

    @staticmethod
    def _load_calibration(path: Path) -> dict[str, dict[str, Any]]:
        """Load and validate a calibration JSON file.

        Supports the LeRobot calibration format::

            {
                "<joint_name>": {
                    "id": <int>,
                    "drive_mode": <0 | 1>,
                    "homing_offset": <int>,
                    "range_min": <int>,
                    "range_max": <int>
                },
                ...
            }

        ``drive_mode`` controls direction: ``0`` = normal, ``1`` = reversed.
        ``range_min`` and ``range_max`` are in raw servo ticks.

        Args:
            path: Path to the JSON calibration file.

        Returns:
            The joints dict from the calibration file.

        Raises:
            TypeError: If the calibration data is not a dict.
            ValueError: If joints are missing or required keys are absent.
        """
        joints = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(joints, dict):
            msg = "Calibration file must be a JSON object mapping joint names to calibration data"
            raise TypeError(msg)

        required_joints = set(SO101.JOINT_ORDER)
        missing = required_joints - joints.keys()
        if missing:
            msg = f"Calibration file is missing joints: {sorted(missing)}"
            raise ValueError(msg)

        for name, cal in joints.items():
            for key in ("id", "drive_mode", "homing_offset", "range_min", "range_max"):
                if key not in cal:
                    msg = f"Joint '{name}' missing required calibration key '{key}'"
                    raise ValueError(msg)
            if cal["drive_mode"] not in {0, 1}:
                msg = f"Joint '{name}' drive_mode must be 0 or 1, got {cal['drive_mode']}"
                raise ValueError(msg)

        return joints

    def _ticks_to_radians(self, ticks: np.ndarray) -> np.ndarray:
        """Convert raw servo ticks to radians using calibration data.

        Args:
            ticks: Integer tick values, shape ``(6,)``.

        Returns:
            Float32 array of joint positions in radians, shape ``(6,)``.
        """
        assert self._calibration is not None  # noqa: S101
        result = np.empty(self.NUM_JOINTS, dtype=np.float32)
        for i, name in enumerate(self.JOINT_ORDER):
            cal = self._calibration[name]
            direction = -1 if cal["drive_mode"] == 1 else 1
            result[i] = (ticks[i] - cal["homing_offset"]) * direction * _RADIANS_PER_TICK
        return result

    def _radians_to_ticks(self, radians: np.ndarray) -> np.ndarray:
        """Convert radians to raw servo ticks, clamping to calibration range.

        Args:
            radians: Float joint positions in radians, shape ``(6,)``.

        Returns:
            Int32 array of tick values, shape ``(6,)``.
        """
        assert self._calibration is not None  # noqa: S101
        result = np.empty(self.NUM_JOINTS, dtype=np.int32)
        for i, name in enumerate(self.JOINT_ORDER):
            cal = self._calibration[name]
            direction = -1 if cal["drive_mode"] == 1 else 1
            ticks_val = round(radians[i] / (direction * _RADIANS_PER_TICK) + cal["homing_offset"])
            result[i] = int(np.clip(ticks_val, cal["range_min"], cal["range_max"]))
        return result

    def _ping_servos(self) -> None:
        """Ping every servo and raise on failure.

        Raises:
            ConnectionError: If a servo does not respond.
        """
        for name, servo_id in self.servo_ids.items():
            assert self._packet_handler is not None  # noqa: S101
            assert self._port_handler is not None  # noqa: S101
            _, comm_result, error = self._packet_handler.ping(self._port_handler, servo_id)
            if comm_result != 0:
                msg = (
                    f"Servo '{name}' (ID {servo_id}) did not respond on {self.port}. "
                    f"Comm result: {comm_result}"
                )
                raise ConnectionError(msg)
            if error != 0:
                logger.warning("Servo '%s' (ID %d) returned error: %d", name, servo_id, error)

    def _set_torque(self, *, enabled: bool) -> None:
        """Enable or disable torque on all servos."""
        assert self._packet_handler is not None  # noqa: S101
        assert self._port_handler is not None  # noqa: S101
        value = 1 if enabled else 0
        for name, servo_id in self.servo_ids.items():
            comm_result, error = self._packet_handler.write1ByteTxRx(
                self._port_handler,
                servo_id,
                _ADDR_TORQUE_ENABLE,
                value,
            )
            if comm_result != 0:
                logger.warning("Failed to set torque on servo '%s' (ID %d): comm=%d", name, servo_id, comm_result)
            if error != 0:
                logger.warning("Torque write error on servo '%s' (ID %d): err=%d", name, servo_id, error)

    def _hold_position(self) -> None:
        """Command all servos to hold their current position.

        Reads the current positions and writes them back as goal positions,
        then ensures torque is enabled.  This prevents the arm from dropping
        under gravity when the connection is closed.
        """
        raw = self._read_joint_positions()
        self._write_joint_positions(raw.astype(np.int32))
        self._set_torque(enabled=True)

    def _read_joint_positions(self) -> np.ndarray:
        """Bulk-read present positions from all servos via sync read.

        Returns:
            Int32 array of raw tick positions, shape ``(6,)``.

        Raises:
            ConnectionError: If sync read fails.
        """
        assert self._group_sync_read is not None  # noqa: S101
        comm_result = self._group_sync_read.txRxPacket()
        if comm_result != 0:
            msg = f"Sync read failed with comm result {comm_result}"
            raise ConnectionError(msg)

        positions = np.empty(self.NUM_JOINTS, dtype=np.int32)
        for i, name in enumerate(self.JOINT_ORDER):
            servo_id = self.servo_ids[name]
            if not self._group_sync_read.isAvailable(servo_id, _ADDR_PRESENT_POSITION, _LEN_PRESENT_POSITION):
                msg = f"Servo '{name}' (ID {servo_id}) data not available in sync read"
                raise ConnectionError(msg)
            positions[i] = self._group_sync_read.getData(
                servo_id,
                _ADDR_PRESENT_POSITION,
                _LEN_PRESENT_POSITION,
            )
        return positions

    def _write_joint_positions(self, ticks: np.ndarray) -> None:
        """Bulk-write goal positions to all servos via sync write.

        Args:
            ticks: Int32 array of goal tick positions, shape ``(6,)``.

        Raises:
            ConnectionError: If sync write fails.
        """
        assert self._group_sync_write is not None  # noqa: S101
        self._group_sync_write.clearParam()

        for i, name in enumerate(self.JOINT_ORDER):
            servo_id = self.servo_ids[name]
            position = int(ticks[i])
            # STS3215 goal position is 2 bytes, little-endian
            param = [position & 0xFF, (position >> 8) & 0xFF]
            if not self._group_sync_write.addParam(servo_id, param):
                msg = f"Failed to add servo '{name}' (ID {servo_id}) to sync write"
                raise ConnectionError(msg)

        comm_result = self._group_sync_write.txPacket()
        if comm_result != 0:
            msg = f"Sync write failed with comm result {comm_result}"
            raise ConnectionError(msg)
