# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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

# The first six joints are the arm; the seventh is the gripper.
REBOT_B601_ARM_JOINT_COUNT: Final = 6

# URDF names of the six arm joints, in driver order, in the gravity-model URDFs below.
REBOT_B601_URDF_ARM_JOINTS: Final = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")

# URDFs (relative to ``get_urdf_path()``) whose joint frames match the public frame; the drivers
# compute gravity feedforward from their link masses. joint1 turns about +z, joint2 about -z, and
# joint3 to joint6 about +z in both, matching MuJoCo Menagerie's ``seeed_rebot_devarm``.
REBOT_B601_DM_GRAVITY_URDF: Final = "rebot-b601-dm/urdf/reBot-DevArm_fixend.urdf"
REBOT_B601_RS_GRAVITY_URDF: Final = "rebot-b601-rs/urdf/00-arm-rs_asm-v3_joint_frame.urdf"

# Peak joint speed of the rest-pose (park) move, in degrees per second; Seeed's own drivers return
# to zero at 25-30.
REBOT_B601_PARK_SPEED_DEG_S: Final = 30.0

# Joints that move back to zero in each stage of the rest-pose (park) move: the base and wrist
# first, then the shoulder and elbow fold the arm down onto its rest.
REBOT_B601_PARK_STAGES: Final = (
    ("shoulder_pan", "wrist_flex", "wrist_yaw", "wrist_roll"),
    ("shoulder_lift", "elbow_flex"),
)

# Largest MIT stiffness (N·m/rad) and damping (N·m·s/rad) each motor model's command frame can
# encode. motorbridge clamps larger values silently, so the drivers reject them instead.
# Speed and acceleration limits for every arm command, by motor model (degrees per second, and per
# second squared). A light leader arm can be swung far faster than the follower should move: on a
# B601-RS with its 48 V 12.5 A supply, a 400 deg/s shoulder swing that reversed hard reset every motor
# at once. The limits bound how hard the motors accelerate and brake the arm; real teleoperation stays
# below them about 99% of the time.
REBOT_B601_MAX_SPEED_DEG_S: Final = {"rs-06": 150.0, "rs-00": 300.0, "4340P": 150.0, "4310": 300.0}
REBOT_B601_MAX_ACCEL_DEG_S2: Final = {"rs-06": 1000.0, "rs-00": 2000.0, "4340P": 1000.0, "4310": 2000.0}

REBOT_B601_MIT_GAIN_LIMITS: Final = {
    "4340P": (500.0, 5.0),
    "4310": (500.0, 5.0),
    "rs-06": (5000.0, 100.0),
    "rs-00": (500.0, 5.0),
}

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
    "shoulder_pan": (-150.0, 150.0),
    "shoulder_lift": (-200.0, 1.0),
    "elbow_flex": (-200.0, 1.0),
    "wrist_flex": (-80.0, 90.0),
    "wrist_yaw": (-90.0, 90.0),
    "wrist_roll": (-90.0, 90.0),
    "gripper": (-270.0, 0.0),
}

REBOT_B601_DM_JOINT_DIRECTIONS: Final = {
    "shoulder_pan": 1.0,
    "shoulder_lift": 1.0,
    "elbow_flex": 1.0,
    "wrist_flex": 1.0,
    "wrist_yaw": 1.0,
    "wrist_roll": 1.0,
    "gripper": 1.0,
}

REBOT_B601_DM_POS_VEL_DEG_S: Final = (250.0, 250.0, 250.0, 200.0, 200.0, 200.0, 200.0)

# DM MIT gains: Seeed's reBotArm_control_py DM values (kp 120 / 18, kd 8 / 2, gripper 8 / 1) with
# the shoulder/elbow damping cut to the 5 N·m·s/rad a Damiao MIT frame can encode and the wrists
# stiffened to 30, which tracked best in simulation once gravity and velocity feedforward are on.
REBOT_B601_DM_MIT_KP: Final = {
    "shoulder_pan": 120.0,
    "shoulder_lift": 120.0,
    "elbow_flex": 120.0,
    "wrist_flex": 30.0,
    "wrist_yaw": 30.0,
    "wrist_roll": 30.0,
    "gripper": 8.0,
}

REBOT_B601_DM_MIT_KD: Final = {
    "shoulder_pan": 5.0,
    "shoulder_lift": 5.0,
    "elbow_flex": 5.0,
    "wrist_flex": 2.0,
    "wrist_yaw": 2.0,
    "wrist_roll": 2.0,
    "gripper": 1.0,
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

# Peak torque (N·m) of the RobStride RS-00 driving the gripper, from its MIT-mode torque range.
REBOT_B601_RS_GRIPPER_MAX_TORQUE_NM: Final = 14.0

REBOT_B601_RS_JOINT_LIMITS_DEG: Final = {
    "shoulder_pan": (-145.0, 145.0),
    "shoulder_lift": (-170.0, 0.0),
    "elbow_flex": (-200.0, 0.0),
    "wrist_flex": (-90.0, 80.0),
    "wrist_yaw": (-90.0, 90.0),
    "wrist_roll": (-90.0, 90.0),
    "gripper": (-270.0, 0.0),
}

REBOT_B601_RS_JOINT_DIRECTIONS: Final = {
    "shoulder_pan": -1.0,
    "shoulder_lift": -1.0,
    "elbow_flex": -1.0,
    "wrist_flex": -1.0,
    "wrist_yaw": -1.0,
    "wrist_roll": -1.0,
    "gripper": -1.0,
}

# RS MIT gains: Seeed's and LeRobot's RobStride defaults, validated on hardware. With gravity and
# velocity feedforward they track a real teleop recording to about 0.5 degrees RMS in simulation,
# against about 2.3 degrees with the PD law alone.
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
