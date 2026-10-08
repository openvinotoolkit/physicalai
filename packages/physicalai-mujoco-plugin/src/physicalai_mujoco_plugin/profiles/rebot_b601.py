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
  same joint axes and limits as Menagerie (joint1 +z, joint2 -z with range -pi..0, joint3 to
  joint6 +z), so every arm joint maps one to one (``scale`` 1). Menagerie's ranges agree with
  #360's limits in sign: shoulder_lift and elbow_flex go negative from the fold.
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

from physicalai_mujoco_plugin.profiles._types import CameraSpec, ChannelOverride, EndEffector, RobotProfile

REBOT_B601_GRIPPER_SCALE = -270.0 / 0.05
"""Driver gripper degrees (#360's frame: -270 open, 0 closed) per metre of finger travel. Provisional."""

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
        ChannelOverride(name="shoulder_pan", actuators=("joint1",)),
        ChannelOverride(name="shoulder_lift", actuators=("joint2",)),
        ChannelOverride(name="elbow_flex", actuators=("joint3",)),
        ChannelOverride(name="wrist_flex", actuators=("joint4",)),
        ChannelOverride(name="wrist_yaw", actuators=("joint5",)),
        ChannelOverride(name="wrist_roll", actuators=("joint6",)),
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

__all__ = ["REBOT_B601_GRIPPER_SCALE", "REBOT_B601_PROFILE", "REBOT_B601_WRIST_CAMERA"]
