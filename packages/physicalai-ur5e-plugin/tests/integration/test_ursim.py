# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Explicitly opt-in URSim smoke test. Never target a real robot."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from physicalai_ur5e_plugin import UniversalRobot

pytestmark = [
    pytest.mark.ursim,
    pytest.mark.skipif(
        os.environ.get("PHYSICALAI_URSIM_ONLY") != "1" or not os.environ.get("PHYSICALAI_URSIM_IP"),
        reason="Requires an explicitly configured URSim instance; no hardware tests by default.",
    ),
]


def test_ursim_joint_stream():
    robot = UniversalRobot(
        ip=os.environ["PHYSICALAI_URSIM_IP"],
        model=os.environ.get("PHYSICALAI_URSIM_MODEL", "ur5e"),
        max_velocity=5.0,
    )
    try:
        robot.connect()
        assert robot.is_connected()
        target = robot.get_observation().joint_positions.copy()
        target[0] += 0.5
        for _ in range(25):
            robot.send_action(target, goal_time=0.1)
            time.sleep(0.02)
        np.testing.assert_allclose(robot.get_observation().joint_positions, target, atol=0.3)
    finally:
        robot.disconnect()
