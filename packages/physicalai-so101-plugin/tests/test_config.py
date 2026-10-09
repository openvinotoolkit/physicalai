# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration-export tests for the SO-101 driver plugin."""

from __future__ import annotations

from typing import Any

from physicalai.config import Config
from physicalai.robot import Robot

SAMPLE_CALIBRATION: dict[str, Any] = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 2048, "range_min": 707, "range_max": 3439},
    "shoulder_lift": {"id": 2, "drive_mode": 1, "homing_offset": 1024, "range_min": 669, "range_max": 3292},
    "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": 2048, "range_min": 846, "range_max": 3069},
    "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": 2048, "range_min": 956, "range_max": 3311},
    "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 2048, "range_min": 59, "range_max": 3946},
    "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 2048, "range_min": 2026, "range_max": 3074},
}


def test_driver_exports_plugin_class_path() -> None:
    from physicalai_so101_plugin import SO101

    robot = SO101(port="/dev/ttyUSB0", calibration=SAMPLE_CALIBRATION)
    assert isinstance(robot, Robot)
    assert Config.from_instance(robot).to_dict() == {
        "class_path": "physicalai_so101_plugin.SO101",
        "init_args": {"port": "/dev/ttyUSB0", "calibration": SAMPLE_CALIBRATION},
    }


def test_uncalibrated_driver_exports_explicit_ticks_mode() -> None:
    from physicalai_so101_plugin import SO101

    recipe = Config.from_instance(SO101.uncalibrated(port="/dev/ttyUSB0")).to_dict()
    assert recipe["class_path"] == "physicalai_so101_plugin.SO101"
    assert recipe["init_args"]["allow_uncalibrated"] is True
    assert recipe["init_args"]["unit"] == "ticks"
