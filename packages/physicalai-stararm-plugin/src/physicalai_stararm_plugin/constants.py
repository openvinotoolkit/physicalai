"""Motor and joint constants for Star Arm 102 variants."""

from __future__ import annotations

from typing import Final

STAR_ARM_102_JOINT_ORDER: Final = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)

STAR_ARM_102_JOINT_IDS: Final = {
    "shoulder_pan": 0,
    "shoulder_lift": 1,
    "elbow_flex": 2,
    "wrist_flex": 3,
    "wrist_yaw": 4,
    "wrist_roll": 5,
    "gripper": 6,
}

STAR_ARM_102_JOINT_RANGES_DEG: Final = {
    "shoulder_pan": (-150.0, 150.0),
    "shoulder_lift": (-1.0, 170.0),
    "elbow_flex": (-200.0, 1.0),
    "wrist_flex": (-80.0, 90.0),
    "wrist_yaw": (-90.0, 90.0),
    "wrist_roll": (-90.0, 90.0),
    "gripper": (-0.0, 270.0),
}
