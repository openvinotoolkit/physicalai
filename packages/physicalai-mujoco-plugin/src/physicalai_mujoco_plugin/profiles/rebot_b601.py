# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""reBot B601 profile: Menagerie's ``seeed_rebot_devarm`` in the LeRobot-style joint frame of ``ReBotB601RS``.

PROVISIONAL, for two reasons:

- The frame is the one PR #360 introduces (issue #343): every RS direction -1, the gripper -270
  (open) to 0 (closed) in degrees, no x6 scaling. Until #360 merges, the B601 plugin on ``main``
  still uses the previous frame (directions +1, +1, -1, -1, -1, +1, gripper /6, 0..45), so this
  profile and that driver disagree in sign on shoulder_pan, shoulder_lift and wrist_roll and in
  the gripper's sign and scale. Data recorded in either frame does not carry over to the other.
- No real arm has confirmed the mapping yet (design open issue 16.1).

How the mapping is derived:

- Menagerie's DevArm has the link origins and frames of the B601-RS URDF, so both share one zero
  pose: the folded rest pose in which the RS arm is zero-calibrated. Every offset is therefore 0.
- #360's ``00-arm-rs_asm-v3_joint_frame.urdf``, which Studio previews the real arm with, has the
  same joint axes as Menagerie (joint1 +z, joint2 -z, joint3 to joint6 +z), so every arm joint
  maps one to one (``scale`` 1); both go negative from the fold on shoulder_lift and elbow_flex.
- The driver clips actions to its public limits (``REBOT_B601_RS_JOINT_LIMITS_DEG`` in #360),
  which are tighter than Menagerie's ranges on shoulder_pan (145 vs 160 degrees), shoulder_lift
  (-170 vs -180), wrist_flex (-90..80 vs -103..97) and wrist_roll (90 vs 180). Each channel's
  ``range`` is the overlap of the two, so the simulation stops where the real arm stops. The one
  exception is elbow_flex: the driver allows -200 degrees, the model only -180 (its geometry), so
  the simulation stops 20 degrees short there.
- Menagerie's two fingers slide 0..0.05 m each from closed to open, coupled 1:1. #360's URDF
  drives them from a ``gripper_drive`` joint of -270 degrees (open) to 0, with the first finger at
  0.05 m when fully open. The gripper channel drives both fingers and reports
  ``-5400 * finger`` metres in the driver's degrees (0.05 m -> -270), velocities likewise.

The wrist camera is an Intel RealSense D405 on Seeed's printed mount (reBot-DevArm
``hardware/reBot_B601_DM/3D_Printed_Parts/D405_305_Mount.step``). Its clamp (57.3 mm bore) sits on
the 57-60 mm neck of the gripper housing, after the wrist-roll joint, so the camera rolls with the
gripper. The mount holds the 42 mm camera 104 mm off the roll axis, tilted 30 degrees toward it, so
the view centre crosses the axis just past the finger tips. :data:`REBOT_B601_WRIST_CAMERA` is
that pose in ``link6``, taken from the STEP geometry. Which side of the gripper the mount faces is
symmetric in the model and assumed (``link6`` +x); check it on the arm with the joint signs.
"""

from __future__ import annotations

from math import radians

from physicalai_mujoco_plugin.profiles._types import CameraSpec, ChannelOverride, EndEffector, RobotProfile

REBOT_B601_GRIPPER_SCALE = -270.0 / 0.05
"""Driver gripper degrees (#360's frame: -270 open, 0 closed) per metre of finger travel. Provisional."""

REBOT_B601_JOINT_RANGES_DEG: dict[str, tuple[float, float] | None] = {
    "shoulder_pan": (-145.0, 145.0),
    "shoulder_lift": (-170.0, 0.0),
    "elbow_flex": None,  # the model's -180..0; the driver allows -200
    "wrist_flex": (-90.0, 80.0),
    "wrist_yaw": None,  # the model's -89.95..89.95; the driver allows -90..90
    "wrist_roll": (-90.0, 90.0),
}
"""Arm joint ranges in the driver's public degrees: #360's limits where tighter than Menagerie's, else ``None``."""


def _range(name: str) -> tuple[float, float] | None:
    limits = REBOT_B601_JOINT_RANGES_DEG[name]
    return None if limits is None else (radians(limits[0]), radians(limits[1]))


REBOT_B601_WRIST_CAMERA = CameraSpec(
    name="wrist",
    body="link6",
    # The D405's front face centre: 104 mm off the roll axis (link6 +z), 39 mm along it. The camera
    # looks along (-0.5, 0, 0.866) in link6, toward the axis and the fingers, with image up along
    # (0.866, 0, 0.5), away from the gripper; its x axis is link6 +y.
    pos=(0.10414, 0.0, 0.03915),
    quat=(0.18301270189221933, 0.6830127018922193, 0.6830127018922193, 0.18301270189221933),
    fovy=58.0,  # the D405's vertical field of view
)
"""The wrist D405 on Seeed's ``D405_305_Mount``, from the mount's STEP geometry; the side is assumed."""

REBOT_B601_PROFILE = RobotProfile(
    name="rebot_b601",
    display_name="reBot B601",
    menagerie_model="seeed_rebot_devarm",
    menagerie_entry="seeed_rebot_devarm",
    tier="twin",
    # Names and order of REBOT_B601_RS_JOINT_ORDER, in #360's frame; provisional (see the module docstring).
    provisional=True,
    channels=(
        ChannelOverride(name="shoulder_pan", actuators=("joint1",), range=_range("shoulder_pan")),
        ChannelOverride(name="shoulder_lift", actuators=("joint2",), range=_range("shoulder_lift")),
        ChannelOverride(name="elbow_flex", actuators=("joint3",), range=_range("elbow_flex")),
        ChannelOverride(name="wrist_flex", actuators=("joint4",), range=_range("wrist_flex")),
        ChannelOverride(name="wrist_yaw", actuators=("joint5",), range=_range("wrist_yaw")),
        ChannelOverride(name="wrist_roll", actuators=("joint6",), range=_range("wrist_roll")),
        # Both fingers, so neither actuator fights the model's finger equality. Like every RS
        # channel, the driver reports the gripper in degrees (-270 open to 0 closed), so the unit
        # is pinned to degrees: the slide's metres stay unconverted and the scale turns them into
        # driver degrees, also under unit="normalized".
        ChannelOverride(
            name="gripper",
            actuators=("joint_left", "joint_right"),
            unit="degrees",
            scale=REBOT_B601_GRIPPER_SCALE,
            member_scales=(1.0,),
        ),
    ),
    default_unit="degrees",
    cameras=(REBOT_B601_WRIST_CAMERA,),
    default_scene="single_pick_place",
    # The model has no tool site: the point between the finger tips, on the roll axis (link6 +z).
    end_effectors=(EndEffector(body="link6", pos=(0.0, 0.0, 0.163)),),
    reach=0.544,
)

__all__ = ["REBOT_B601_GRIPPER_SCALE", "REBOT_B601_JOINT_RANGES_DEG", "REBOT_B601_PROFILE", "REBOT_B601_WRIST_CAMERA"]
