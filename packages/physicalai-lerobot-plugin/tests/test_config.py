# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Restore typed LeRobot configs after JSON and local worker serialization."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from physicalai_lerobot_plugin import lerobot_adapter, studio_catalog


@dataclass
class ArmConfig:
    calibration_dir: Path | None = None


@dataclass
class DeviceConfig:
    calibration_dir: Path | None
    arms: list[ArmConfig]


@pytest.mark.parametrize("leader", [False, True])
@pytest.mark.parametrize("calibration_dir", [None, "/calibration/stararm"])
def test_worker_rebuilds_typed_config(
    monkeypatch: pytest.MonkeyPatch, *, leader: bool, calibration_dir: str | None
) -> None:
    import lerobot.robots
    import lerobot.teleoperators

    kwargs = json.loads(json.dumps({
        "calibration_dir": calibration_dir,
        "arms": [{"calibration_dir": "/calibration/nested"}],
    }))
    factory = Mock()
    lookup = "_teleoperator_config_class" if leader else "_robot_config_class"
    monkeypatch.setattr(lerobot_adapter, lookup, lambda _: DeviceConfig)
    module = lerobot.teleoperators if leader else lerobot.robots
    factory_name = "make_teleoperator_from_config" if leader else "make_robot_from_config"
    monkeypatch.setattr(module, factory_name, factory)
    adapter_cls = lerobot_adapter.LeRobotTeleoperatorAdapter if leader else lerobot_adapter.LeRobotAdapter
    adapter = adapter_cls("test_config", kwargs)
    # Trusted local pickle roundtrip simulates the multiprocessing spawn boundary.
    restored: Any = pickle.loads(pickle.dumps(adapter))  # noqa: S301

    if leader:
        restored._ensure_teleoperator()  # noqa: SLF001
    else:
        restored._ensure_robot()  # noqa: SLF001

    config = factory.call_args.args[0]
    assert config.calibration_dir == (None if calibration_dir is None else Path(calibration_dir))
    assert isinstance(config.arms[0], ArmConfig)
    assert config.arms[0].calibration_dir == Path("/calibration/nested")
    assert kwargs["calibration_dir"] == calibration_dir
    assert kwargs["arms"][0]["calibration_dir"] == "/calibration/nested"


def test_catalog_materializes_paths_and_nested_configs() -> None:
    config = studio_catalog._materialize_dataclass(  # noqa: SLF001
        DeviceConfig,
        {"calibration_dir": "/calibration/stararm", "arms": [{"calibration_dir": "/calibration/nested"}]},
    )

    assert config.calibration_dir == Path("/calibration/stararm")
    assert isinstance(config.arms[0], ArmConfig)
    assert config.arms[0].calibration_dir == Path("/calibration/nested")
