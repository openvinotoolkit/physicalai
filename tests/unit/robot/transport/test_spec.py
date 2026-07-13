# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
import sys
from unittest.mock import MagicMock, patch

import pytest

from physicalai.robot.transport._spec import RobotSpec, default_rate_hz


class TestRobotSpec:
    def test_picklable(self) -> None:
        spec = RobotSpec(robot_type="so101", robot_kwargs={"port": "/dev/ttyUSB0", "role": "follower"})
        restored = pickle.loads(pickle.dumps(spec))
        assert restored == spec

    def test_json_roundtrip(self) -> None:
        spec = RobotSpec(robot_type="widowxai", robot_kwargs={"ip": "10.0.0.2"})
        assert RobotSpec.from_json_dict(spec.to_json_dict()) == spec

    def test_default_kwargs_empty(self) -> None:
        assert RobotSpec("so101").robot_kwargs == {}

    def test_build_so101(self) -> None:
        # The so101 module needs the vendor SDK; stub it out so build()
        # delegation is testable without the extra installed.
        mock_module = MagicMock()
        spec = RobotSpec("so101", {"port": "/dev/ttyUSB0", "calibration": None, "_allow_uncalibrated": True})
        with patch.dict(sys.modules, {"physicalai.robot.so101": mock_module}):
            spec.build()
        mock_module.SO101.assert_called_once_with(port="/dev/ttyUSB0", calibration=None, _allow_uncalibrated=True)

    def test_build_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown robot_type"):
            RobotSpec("hexapod").build()


class TestDefaultRate:
    def test_per_robot_defaults(self) -> None:
        assert default_rate_hz("so101") == 100.0
        assert default_rate_hz("widowxai") == 200.0

    def test_fallback(self) -> None:
        assert default_rate_hz("unknown") == 100.0
