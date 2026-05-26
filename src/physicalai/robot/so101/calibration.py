# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SO-101 calibration data types.

Calibration maps raw servo ticks to radians using per-joint homing offsets,
drive-mode direction, and valid tick ranges.  The LeRobot JSON format is
supported natively via :meth:`SO101Calibration.from_path`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from physicalai.robot.so101.constants import SO101_JOINT_ORDER, TICKS_PER_REVOLUTION


@dataclass(frozen=True)
class SO101JointCalibration:
    """Calibration data for a single SO-101 joint."""

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int

    @property
    def direction(self) -> int:
        """Direction multiplier derived from drive mode."""
        return -1 if self.drive_mode == 1 else 1


@dataclass(frozen=True)
class SO101Calibration:
    """Calibration data for all SO-101 joints."""

    joints: dict[str, SO101JointCalibration]

    @classmethod
    def from_path(cls, path: str | Path) -> SO101Calibration:
        """Load and validate a calibration JSON file from disk.

        Returns:
            Validated calibration object.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: object) -> SO101Calibration:
        """Build a calibration object from parsed JSON data.

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

        Returns:
            Validated calibration object.

        Raises:
            TypeError: If the calibration data is not a dict.
            ValueError: If joints are missing, required keys are absent,
                tick ranges are invalid or out of bounds, or servo IDs are
                not positive / unique.
        """
        if not isinstance(data, dict):
            msg = "Calibration file must be a JSON object mapping joint names to calibration data"
            raise TypeError(msg)

        required_joints = set(SO101_JOINT_ORDER)
        missing = required_joints - data.keys()
        if missing:
            msg = f"Calibration file is missing joints: {sorted(missing)}"
            raise ValueError(msg)

        joints: dict[str, SO101JointCalibration] = {}
        for name in SO101_JOINT_ORDER:
            cal = data[name]
            if not isinstance(cal, dict):
                msg = f"Joint '{name}' calibration must be a dict"
                raise TypeError(msg)
            for key in ("id", "drive_mode", "homing_offset", "range_min", "range_max"):
                if key not in cal:
                    msg = f"Joint '{name}' missing required calibration key '{key}'"
                    raise ValueError(msg)
            if cal["drive_mode"] not in {0, 1}:
                msg = f"Joint '{name}' drive_mode must be 0 or 1, got {cal['drive_mode']}"
                raise ValueError(msg)

            joint = SO101JointCalibration(
                id=int(cal["id"]),
                drive_mode=int(cal["drive_mode"]),
                homing_offset=int(cal["homing_offset"]),
                range_min=int(cal["range_min"]),
                range_max=int(cal["range_max"]),
            )
            # Validate range ordering and hardware encoder bounds.
            if joint.range_min >= joint.range_max:
                msg = f"Joint '{name}': range_min ({joint.range_min}) must be less than range_max ({joint.range_max})"
                raise ValueError(msg)
            if not (0 <= joint.range_min < TICKS_PER_REVOLUTION):
                msg = (
                    f"Joint '{name}': range_min ({joint.range_min}) is outside the valid "
                    f"STS3215 encoder range [0, {TICKS_PER_REVOLUTION - 1}]"
                )
                raise ValueError(msg)
            if not (0 <= joint.range_max < TICKS_PER_REVOLUTION):
                msg = (
                    f"Joint '{name}': range_max ({joint.range_max}) is outside the valid "
                    f"STS3215 encoder range [0, {TICKS_PER_REVOLUTION - 1}]"
                )
                raise ValueError(msg)
            joints[name] = joint

        # Validate servo IDs are positive and unique across all joints.
        ids = [j.id for j in joints.values()]
        if any(servo_id <= 0 for servo_id in ids):
            bad = {n: j.id for n, j in joints.items() if j.id <= 0}
            msg = f"All servo IDs must be positive integers, got: {bad}"
            raise ValueError(msg)
        if len(set(ids)) != len(ids):
            msg = "All servo IDs must be unique across joints."
            raise ValueError(msg)

        return cls(joints=joints)
