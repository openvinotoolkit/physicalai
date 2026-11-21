# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration-export tests for the SO-101 driver plugin."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

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


def _assert_construction_round_trip(robot: object) -> dict[str, Any]:
    """Serialize and reconstruct a robot configuration recipe."""
    assert Config.is_exportable(robot)
    assert isinstance(robot, Robot)
    config = Config.from_instance(robot)
    wire: dict[str, Any] = json.loads(json.dumps(config.to_dict()))
    restored = Config.from_dict(wire).instantiate()
    assert type(restored) is type(robot)
    assert Config.from_instance(restored) == wire
    assert not restored.is_connected()  # type: ignore[union-attr]
    return wire


def test_driver_exports_plugin_class_path() -> None:
    from physicalai_so101_plugin import SO101

    robot = SO101(port="/dev/ttyUSB0", calibration=SAMPLE_CALIBRATION)
    assert isinstance(robot, Robot)
    assert Config.from_instance(robot).to_dict() == {
        "class_path": "physicalai_so101_plugin.SO101",
        "init_args": {"port": "/dev/ttyUSB0", "calibration": SAMPLE_CALIBRATION},
    }


def test_legacy_so101_package_exports_plugin_driver() -> None:
    from physicalai.robot.so101 import SO101 as LegacySO101
    from physicalai_so101_plugin import SO101 as PluginSO101

    assert LegacySO101 is PluginSO101


def test_calibration_object_encodes_to_dict() -> None:
    """Accept typed calibration objects in exportable SO-101 recipes."""
    from physicalai_so101_plugin import SO101
    from physicalai_so101_plugin.calibration import SO101Calibration

    calibration = SO101Calibration.from_dict(SAMPLE_CALIBRATION)
    robot = SO101(port="/dev/ttyACM0", calibration=calibration)
    wire = _assert_construction_round_trip(robot)
    assert wire["init_args"]["calibration"] == SAMPLE_CALIBRATION


def test_path_calibration_keeps_relative_string(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve relative calibration strings in the serialized recipe."""
    from physicalai_so101_plugin import SO101

    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(SAMPLE_CALIBRATION), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    wire = _assert_construction_round_trip(SO101(port="/dev/ttyUSB0", calibration="calibration.json"))
    assert wire["init_args"]["calibration"] == "calibration.json"


def test_path_object_keeps_relative_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve relative calibration paths in the serialized recipe."""
    from physicalai_so101_plugin import SO101

    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(SAMPLE_CALIBRATION), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    wire = _assert_construction_round_trip(SO101(port="/dev/ttyUSB0", calibration=Path("calibration.json")))
    assert wire["init_args"]["calibration"] == "calibration.json"


def test_absolute_path_calibration_stays_absolute(tmp_path: Path) -> None:
    """Preserve absolute calibration paths in the serialized recipe."""
    from physicalai_so101_plugin import SO101

    calibration_path = (tmp_path / "calibration.json").resolve()
    calibration_path.write_text(json.dumps(SAMPLE_CALIBRATION), encoding="utf-8")

    wire = _assert_construction_round_trip(SO101(port="/dev/ttyUSB0", calibration=calibration_path))
    assert wire["init_args"]["calibration"] == str(calibration_path)


def test_uncalibrated_driver_exports_explicit_ticks_mode() -> None:
    from physicalai_so101_plugin import SO101

    recipe = Config.from_instance(SO101.uncalibrated(port="/dev/ttyUSB0")).to_dict()
    assert recipe["class_path"] == "physicalai_so101_plugin.SO101"
    assert recipe["init_args"]["allow_uncalibrated"] is True
    assert recipe["init_args"]["unit"] == "ticks"


def test_uncalibrated_driver_recipe_is_jsonargparse_loadable() -> None:
    """Keep the uncalibrated recipe valid for ``physicalai run`` configuration loading."""
    from jsonargparse import ArgumentParser

    from physicalai_so101_plugin import SO101

    wire = _assert_construction_round_trip(SO101.uncalibrated(port="/dev/ttyUSB0"))
    assert wire["init_args"]["calibration"] is None
    assert wire["init_args"]["allow_uncalibrated"] is True
    assert wire["init_args"]["unit"] == "ticks"

    parser = ArgumentParser()
    parser.add_class_arguments(SO101, "robot")
    namespace = parser.parse_object({"robot": wire["init_args"]})
    assert namespace.robot.calibration is None
    assert namespace.robot.allow_uncalibrated is True
