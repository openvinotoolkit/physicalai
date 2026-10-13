# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Joint and control constants for the Franka Research 3 robot arm.

Joint limits, maximum velocities, and efforts are taken from the FR3
description bundled with the plugin (``urdf/franka_description``), which is
generated from the upstream ``franka_description`` repository.
"""

from __future__ import annotations

from typing import Final

FR3_JOINT_ORDER: Final = (
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
)

FR3_JOINT_LIMITS_RAD: Final = {
    "fr3_joint1": (-2.3093, 2.3093),
    "fr3_joint2": (-1.5133, 1.5133),
    "fr3_joint3": (-2.4937, 2.4937),
    "fr3_joint4": (-2.7478, -0.4461),
    "fr3_joint5": (-2.48, 2.48),
    "fr3_joint6": (0.8521, 4.2094),
    "fr3_joint7": (-2.6895, 2.6895),
}

FR3_JOINT_MAX_VELOCITY_RAD_S: Final = (2.0, 1.0, 1.5, 1.25, 3.0, 1.5, 3.0)

FR3_JOINT_MAX_EFFORT_NM: Final = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)

FR3_GRIPPER_WIDTH_RANGE_M: Final = (0.0, 0.08)

FR3_GRIPPER_DEFAULT_SPEED_M_S: Final = 0.05

VALID_REALTIME_CONFIGS: Final = frozenset({"kEnforce", "kIgnore"})
