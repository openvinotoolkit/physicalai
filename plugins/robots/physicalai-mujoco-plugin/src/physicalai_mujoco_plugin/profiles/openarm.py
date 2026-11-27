# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenArm v2 bimanual twin: Enactic's MJCF under LeRobot's ``bi_openarm_follower`` names and units.

The model is Enactic's ``v2/openarm_bimanual.xml`` (``external_models.OPENARM_V2``): both 7-DOF arms
with their parallel grippers, hanging from a shoulder frame, one position actuator per joint and one
per gripper (the second finger follows through an equality). Like ALOHA's, it is one model with both
arms, so it attaches at a single ``robot_mount`` frame; the ``openarm_cell`` scene puts that frame
where the OpenArm Cell's lifter holds the shoulders.

LeRobot's ``BiOpenArmFollower`` reports ``left_joint_1.pos`` ... ``left_joint_7.pos``,
``left_gripper.pos`` and the same for ``right_``, in degrees from the calibration pose: arms hanging,
grippers closed, which is the model's zero pose. ``physicalai-lerobot-plugin`` strips ``.pos`` and
sorts the joints by name, so the channels here are in that order: each side's gripper first.

The gripper is the motor angle, 0 closed to -65 degrees open on both sides; the model's finger joint
opens from 0 to +45 degrees on the left and to -45 on the right. The linear map between them is an
assumption until it is measured on the arm, and so are the signs of the symmetric joints (1, 3, 5, 6, 7);
the asymmetric limits of joints 2 and 4 agree between LeRobot and the model. Hence ``provisional``.
"""

# MjSpec is supplied by the MuJoCo C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from physicalai_mujoco_plugin.external_models import OPENARM_V2
from physicalai_mujoco_plugin.profiles._types import ChannelOverride, EndEffector, RobotProfile

if TYPE_CHECKING:
    import mujoco

OPENARM_SIDES = ("left", "right")
OPENARM_GRIPPER_OPEN_DEG = 65.0
"""How far LeRobot's OpenArm gripper motor turns from closed (0) to open (-65 degrees)."""
OPENARM_FINGER_OPEN_DEG = 45.0
"""How far the model's finger joint turns from closed (0) to open: +45 degrees left, -45 right."""
OPENARM_GRIPPER_SCALES = {
    "left": -OPENARM_GRIPPER_OPEN_DEG / OPENARM_FINGER_OPEN_DEG,
    "right": OPENARM_GRIPPER_OPEN_DEG / OPENARM_FINGER_OPEN_DEG,
}
"""LeRobot gripper degrees per model finger degree, by side; both map open to -65."""
OPENARM_HOME_DEG = (0.0, 0.0, 0.0, 90.0, 0.0, 0.0, 0.0)
"""Joints 1-7 at home, both sides: elbows bent 90 degrees, as Enactic's cell scene starts."""
OPENARM_CAMERAS = {"camera_wrist_left": "left_wrist", "camera_wrist_right": "right_wrist"}
"""The model's gripper cameras and the names the profile publishes them under."""
_FINGER_TIP = (-0.01, 0.0, -0.168)
"""Between the finger tips, in the gripper base frame: the mean of the finger mesh vertices within
5 mm of the tips. The fingers point along the frame's -z."""


def _channels() -> tuple[ChannelOverride, ...]:
    channels: list[ChannelOverride] = []
    for side in OPENARM_SIDES:
        channels.append(
            ChannelOverride(
                name=f"{side}_gripper",
                actuators=(f"{side}_finger1_ctrl",),
                unit="degrees",
                scale=OPENARM_GRIPPER_SCALES[side],
            )
        )
        channels.extend(
            ChannelOverride(name=f"{side}_joint_{index}", actuators=(f"{side}_joint{index}_ctrl",))
            for index in range(1, 8)
        )
    return tuple(channels)


def customize_openarm(spec: mujoco.MjSpec) -> None:
    """Name the gripper cameras like the other twins, drop the origin marker and add the home pose."""
    for old, new in OPENARM_CAMERAS.items():
        spec.camera(old).name = new
    spec.delete(spec.site("arm_origin"))
    for key in list(spec.keys):
        spec.delete(key)
    arm = [math.radians(value) for value in OPENARM_HOME_DEG]
    qpos: list[float] = []
    ctrl: list[float] = []
    for side in OPENARM_SIDES:
        open_finger = math.radians(OPENARM_FINGER_OPEN_DEG) * (1.0 if side == "left" else -1.0)
        qpos += [*arm, open_finger, open_finger]
        ctrl += [*arm, open_finger]
    spec.add_key(name="home", qpos=qpos, ctrl=ctrl)


OPENARM_PROFILE = RobotProfile(
    name="openarm",
    display_name="OpenArm v2",
    menagerie_model=OPENARM_V2.name,
    menagerie_entry="openarm_bimanual",
    tier="twin",
    provisional=True,
    channels=_channels(),
    default_unit="degrees",
    customize=customize_openarm,
    default_scene="openarm_cell",
    end_effectors=tuple(
        EndEffector(
            body=f"openarm_{side}_ee_base_link",
            pos=_FINGER_TIP,
            axis=(0.0, 0.0, -1.0),
            gripper=f"{side}_gripper",
            gripper_opens_up=side == "left",  # the right finger joint opens toward -45 degrees
        )
        for side in OPENARM_SIDES
    ),
)

__all__ = [
    "OPENARM_CAMERAS",
    "OPENARM_FINGER_OPEN_DEG",
    "OPENARM_GRIPPER_OPEN_DEG",
    "OPENARM_GRIPPER_SCALES",
    "OPENARM_HOME_DEG",
    "OPENARM_PROFILE",
    "OPENARM_SIDES",
    "customize_openarm",
]
