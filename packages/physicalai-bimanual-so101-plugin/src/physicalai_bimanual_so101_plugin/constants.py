"""Bimanual SO-101 hardware constants (dual STS3215 arms, left + right)."""

from __future__ import annotations

from typing import Final

BIMANUAL_SO101_JOINT_ORDER: Final = (
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
    "left_gripper",
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
    "right_gripper",
)

LEFT_ARM_JOINTS: Final = BIMANUAL_SO101_JOINT_ORDER[:6]
RIGHT_ARM_JOINTS: Final = BIMANUAL_SO101_JOINT_ORDER[6:]

NUM_BIMANUAL_JOINTS: Final = 12
NUM_SINGLE_ARM_JOINTS: Final = 6

BIMANUAL_SO101_JOINT_LIMITS_DEG: Final = {
    "left_shoulder_pan": (-110.0, 110.0),
    "left_shoulder_lift": (-100.0, 100.0),
    "left_elbow_flex": (-97.0, 97.0),
    "left_wrist_flex": (-95.0, 95.0),
    "left_wrist_roll": (-157.0, 163.0),
    "left_gripper": (-10.0, 100.0),
    "right_shoulder_pan": (-110.0, 110.0),
    "right_shoulder_lift": (-100.0, 100.0),
    "right_elbow_flex": (-97.0, 97.0),
    "right_wrist_flex": (-95.0, 95.0),
    "right_wrist_roll": (-157.0, 163.0),
    "right_gripper": (-10.0, 100.0),
}

VALID_ROLES: Final = frozenset({"leader", "follower"})
