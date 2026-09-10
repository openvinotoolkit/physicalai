"""Motor and joint constants for reBot B601 robot arms.

Defines joint orders, motor IDs, model numbers, joint limits, direction
signs, and control gains for the Damiao (DM) and RobStride (RS) variants.
"""

from __future__ import annotations

from typing import Final

REBOT_B601_DM_JOINT_ORDER: Final = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)

REBOT_B601_DM_MOTOR_IDS: Final = {
    "shoulder_pan": (0x01, 0x11),
    "shoulder_lift": (0x02, 0x12),
    "elbow_flex": (0x03, 0x13),
    "wrist_flex": (0x04, 0x14),
    "wrist_yaw": (0x05, 0x15),
    "wrist_roll": (0x06, 0x16),
    "gripper": (0x07, 0x17),
}

REBOT_B601_DM_MOTOR_MODELS: Final = {
    "shoulder_pan": "4340P",
    "shoulder_lift": "4340P",
    "elbow_flex": "4340P",
    "wrist_flex": "4310",
    "wrist_yaw": "4310",
    "wrist_roll": "4310",
    "gripper": "4310",
}

REBOT_B601_DM_JOINT_LIMITS_DEG: Final = {
    "shoulder_pan": (-145.0, 145.0),
    "shoulder_lift": (-170.0, 0.0),
    "elbow_flex": (-200.0, 0.0),
    "wrist_flex": (-80.0, 90.0),
    "wrist_yaw": (-90.0, 90.0),
    "wrist_roll": (-90.0, 90.0),
    "gripper": (-270.0, 0.0),
}

REBOT_B601_DM_JOINT_DIRECTIONS: Final = {
    "shoulder_pan": -1.0,
    "shoulder_lift": -1.0,
    "elbow_flex": 1.0,
    "wrist_flex": 1.0,
    "wrist_yaw": 1.0,
    "wrist_roll": -1.0,
    "gripper": -6.0,
}

REBOT_B601_DM_POS_VEL_DEG_S: Final = (250.0, 250.0, 250.0, 200.0, 200.0, 200.0, 200.0)

REBOT_B601_DM_MIT_KP: Final = {
    "shoulder_pan": 50.0,
    "shoulder_lift": 150.0,
    "elbow_flex": 150.0,
    "wrist_flex": 50.0,
    "wrist_yaw": 50.0,
    "wrist_roll": 50.0,
    "gripper": 12.0,
}

REBOT_B601_DM_MIT_KD: Final = {
    "shoulder_pan": 3.0,
    "shoulder_lift": 10.0,
    "elbow_flex": 10.0,
    "wrist_flex": 5.0,
    "wrist_yaw": 4.0,
    "wrist_roll": 4.0,
    "gripper": 0.05,
}

VALID_CAN_ADAPTERS: Final = frozenset({"damiao", "socketcan"})
VALID_RS_CAN_ADAPTERS: Final = frozenset({"socketcan", "robstride"})
VALID_ROLES: Final = frozenset({"follower"})

REBOT_B601_RS_JOINT_ORDER: Final = REBOT_B601_DM_JOINT_ORDER

REBOT_B601_RS_MOTOR_IDS: Final = {
    "shoulder_pan": (0x01, 0xFD),
    "shoulder_lift": (0x02, 0xFD),
    "elbow_flex": (0x03, 0xFD),
    "wrist_flex": (0x04, 0xFD),
    "wrist_yaw": (0x05, 0xFD),
    "wrist_roll": (0x06, 0xFD),
    "gripper": (0x07, 0xFD),
}

REBOT_B601_RS_MOTOR_MODELS: Final = {
    "shoulder_pan": "rs-06",
    "shoulder_lift": "rs-06",
    "elbow_flex": "rs-06",
    "wrist_flex": "rs-00",
    "wrist_yaw": "rs-00",
    "wrist_roll": "rs-00",
    "gripper": "rs-00",
}

REBOT_B601_RS_JOINT_LIMITS_DEG: Final = {
    "shoulder_pan": (-0.0, 145.0),
    "shoulder_lift": (-0.0, 170.0),
    "elbow_flex": (-0.0, 200.0),
    "wrist_flex": (-80.0, 90.0),
    "wrist_yaw": (-90.0, 90.0),
    "wrist_roll": (-90.0, 90.0),
    "gripper": (-0.0, 270.0),
}

REBOT_B601_RS_JOINT_DIRECTIONS: Final = {
    "shoulder_pan": 1.0,
    "shoulder_lift": 1.0,
    "elbow_flex": -1.0,
    "wrist_flex": -1.0,
    "wrist_yaw": -1.0,
    "wrist_roll": 1.0,
    "gripper": 6.0,
}

REBOT_B601_RS_MIT_KP: Final = {
    "shoulder_pan": 50.0,
    "shoulder_lift": 150.0,
    "elbow_flex": 150.0,
    "wrist_flex": 50.0,
    "wrist_yaw": 50.0,
    "wrist_roll": 50.0,
}

REBOT_B601_RS_MIT_KD: Final = {
    "shoulder_pan": 3.0,
    "shoulder_lift": 10.0,
    "elbow_flex": 10.0,
    "wrist_flex": 5.0,
    "wrist_yaw": 4.0,
    "wrist_roll": 4.0,
}
