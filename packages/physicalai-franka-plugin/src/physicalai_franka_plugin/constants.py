# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Joint and control constants for the Franka Research 3 robot arm.

``FR3_JOINT_LIMITS_RAD``, the maximum velocities and the efforts are taken from
the FR3 description bundled with the plugin (``urdf/franka_description``),
which is generated from the upstream ``franka_description`` repository. The
hardware and suggested joint limits, the motion limits and the torque limits
come from the FR3 specification:
https://frankarobotics.github.io/docs/robot_specifications.html
"""

from __future__ import annotations

from typing import Final

import numpy as np

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

# FR3 specification: hardware joint limits (rad), per joint.
FR3_HARDWARE_JOINT_LIMITS_RAD: Final = (
    (-2.9007, 2.9007),
    (-1.8361, 1.8361),
    (-2.9007, 2.9007),
    (-3.0770, -0.1169),
    (-2.8763, 2.8763),
    (0.4398, 4.6216),
    (-3.0508, 3.0508),
)

# FR3 specification: "suggested rectangular position-velocity limits" (rad), per joint.
FR3_SUGGESTED_JOINT_LIMITS_RAD: Final = (
    (-2.3476, 2.3476),
    (-1.5454, 1.5454),
    (-2.4937, 2.4937),
    (-2.7714, -0.4226),
    (-2.5100, 2.5100),
    (0.7773, 4.2841),
    (-2.7045, 2.7045),
)

JOINT_LIMIT_KINDS: Final = ("hardware", "suggested", "urdf")

FR3_JOINT_MAX_VELOCITY_RAD_S: Final = (2.0, 1.0, 1.5, 1.25, 3.0, 1.5, 3.0)

FR3_JOINT_MAX_ACCELERATION_RAD_S2: Final = (15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0)

FR3_JOINT_MAX_JERK_RAD_S3: Final = (5000.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0)

FR3_JOINT_MAX_EFFORT_NM: Final = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)

FR3_JOINT_MAX_TORQUE_RATE_NM_S: Final = 1000.0

# Torque-mode joint impedance of Franka's GELLO controller (gello_software, ros2
# franka_fr3_arm_controllers: joint_impedance_controller.cpp and controllers.yaml):
# tau = K (q_goal - q) - D dq_filtered, with dq_filtered = (1 - alpha) dq_filtered + alpha dq.
FR3_IMPEDANCE_STIFFNESS_NM_RAD: Final = (240.0, 240.0, 240.0, 240.0, 100.0, 60.0, 20.0)
FR3_IMPEDANCE_DAMPING_NM_S_RAD: Final = (20.0, 20.0, 20.0, 10.0, 10.0, 10.0, 5.0)
FR3_IMPEDANCE_VELOCITY_FILTER_ALPHA: Final = 0.99

FR3_GRIPPER_WIDTH_RANGE_M: Final = (0.0, 0.08)

FR3_GRIPPER_DEFAULT_SPEED_M_S: Final = 0.05

FR3_GRIPPER_DEFAULT_FORCE_N: Final = 30.0

VALID_REALTIME_CONFIGS: Final = frozenset({"kEnforce", "kIgnore"})

VALID_CONTROL_MODES: Final = frozenset({"impedance", "position"})


def fr3_joint_limits_deg(kind: str = "hardware", margin_deg: float = 5.0) -> np.ndarray:
    """Return FR3 joint limits as a 7 x 2 array in degrees.

    ``"hardware"`` is the FR3 specification shrunk by ``margin_deg`` on both
    sides: near its hard limits the robot only allows slow motion towards them.
    ``"suggested"`` (Franka's rectangular limits) and ``"urdf"`` (the bundled
    description) are returned as they are. All three share the same midpoints,
    so a GELLO's wrap-around window does not depend on the choice.

    Raises:
        ValueError: If ``kind`` is unknown or the margin leaves no range.
    """
    if kind == "urdf":
        return np.degrees(np.array([FR3_JOINT_LIMITS_RAD[name] for name in FR3_JOINT_ORDER]))
    if kind == "suggested":
        return np.degrees(np.array(FR3_SUGGESTED_JOINT_LIMITS_RAD))
    if kind == "hardware":
        limits = np.degrees(np.array(FR3_HARDWARE_JOINT_LIMITS_RAD)) + np.array([margin_deg, -margin_deg])
        if np.any(limits[:, 0] >= limits[:, 1]):
            msg = f"limit margin {margin_deg} deg leaves no joint range"
            raise ValueError(msg)
        return limits
    msg = f"joint limits must be one of {JOINT_LIMIT_KINDS}, got {kind!r}"
    raise ValueError(msg)
