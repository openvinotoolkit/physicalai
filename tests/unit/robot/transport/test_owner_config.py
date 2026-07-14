# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pickle
import sys
from unittest.mock import MagicMock, patch

import pytest

from physicalai.robot.transport._owner_config import RobotOwnerConfig, normalize_robot_class

from .fake import FakeRobot


class _Outer:
    class Inner: ...


class _RecordingRobot:
    """Stands in for a vendor driver class resolved via a stubbed module."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class TestNormalizeRobotClass:
    def test_string_passthrough(self) -> None:
        assert normalize_robot_class("pkg.mod.Cls") == "pkg.mod.Cls"

    def test_class_object(self) -> None:
        assert normalize_robot_class(FakeRobot) == "tests.unit.robot.transport.fake.FakeRobot"

    def test_nested_qualname(self) -> None:
        path = normalize_robot_class(_Outer.Inner)
        assert path.endswith(".test_owner_config._Outer.Inner")

    def test_local_class_raises(self) -> None:
        class _Local: ...

        with pytest.raises(ValueError, match="local class"):
            normalize_robot_class(_Local)

    def test_non_class_non_string_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a class or a dotted path string"):
            normalize_robot_class(123)  # type: ignore[arg-type]


class TestRobotOwnerConfig:
    def test_picklable(self) -> None:
        config = RobotOwnerConfig(name="left-arm", robot_class="pkg.mod.Cls", robot_kwargs={"port": "/dev/ttyUSB0"})
        restored = pickle.loads(pickle.dumps(config))
        assert restored == config

    def test_json_roundtrip(self) -> None:
        config = RobotOwnerConfig(name="left-arm", robot_class="pkg.mod.Cls", robot_kwargs={"ip": "10.0.0.2"})
        assert RobotOwnerConfig.from_json_dict(json.loads(json.dumps(config.to_json_dict()))) == config

    def test_defaults(self) -> None:
        config = RobotOwnerConfig(name="left-arm", robot_class="pkg.mod.Cls")
        assert config.robot_kwargs == {}
        assert config.allow_remote is False
        assert config.rate_hz == 100.0
        assert config.idle_timeout == 10.0

    def test_zero_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="rate_hz must be finite"):
            RobotOwnerConfig(name="left-arm", robot_class="pkg.mod.Cls", rate_hz=0.0)

    def test_negative_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="rate_hz must be finite"):
            RobotOwnerConfig(name="left-arm", robot_class="pkg.mod.Cls", rate_hz=-1.0)

    def test_infinite_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="rate_hz must be finite"):
            RobotOwnerConfig(name="left-arm", robot_class="pkg.mod.Cls", rate_hz=float("inf"))

    def test_non_serializable_kwargs_raises(self) -> None:
        with pytest.raises(ValueError, match="JSON-serializable"):
            RobotOwnerConfig(name="left-arm", robot_class="pkg.mod.Cls", robot_kwargs={"calibration": object()})

    def test_build_known_class(self) -> None:
        config = RobotOwnerConfig(
            name="left-arm",
            robot_class="tests.unit.robot.transport.fake.FakeRobot",
            robot_kwargs={"port": "/dev/ttyUSB0"},
        )
        robot = config.build()
        assert isinstance(robot, FakeRobot)

    def test_build_arbitrary_module_level_class(self) -> None:
        # No registry to update: any importable class works, mirroring how
        # the vendor SDK is stubbed for so101 elsewhere in this test suite.
        mock_module = MagicMock()
        mock_module.SO101 = _RecordingRobot
        config = RobotOwnerConfig(name="left-arm", robot_class="physicalai.robot.so101.SO101", robot_kwargs={"port": "x"})
        with patch.dict(sys.modules, {"physicalai.robot.so101": mock_module}):
            robot = config.build()
        assert isinstance(robot, _RecordingRobot)
        assert robot.kwargs == {"port": "x"}

    def test_build_non_class_path_raises(self) -> None:
        config = RobotOwnerConfig(name="left-arm", robot_class="physicalai.robot.transport._ids.KEY_PREFIX")
        with pytest.raises(TypeError, match="does not resolve to a class"):
            config.build()

    def test_build_unknown_path_raises(self) -> None:
        config = RobotOwnerConfig(name="left-arm", robot_class="totally.unknown.module.Cls")
        with pytest.raises(ValueError, match="could not import"):
            config.build()
