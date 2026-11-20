# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""reBot B601 profile: Menagerie's ``seeed_rebot_devarm`` in the LeRobot-style joint frame of ``ReBotB601RS``.

The frame is the B601 driver's (every RS direction -1, the gripper -270 open to 0 closed, in
degrees), confirmed on a real B601-RS. The arm responds like that driver: each joint's servo runs its
MIT law ``kp * (target - q) - kd * qdot`` with the driver's default gains, plus full gravity
compensation within the motor's torque limit, and the joints have the friction and damping fitted
to real RS teleoperation (``prateeks-iu/rebot_b601_rs_sugar``). The driver's 0.3 velocity
feedforward is not modelled, so the twin lags a moving target a little more than the real arm.

How the mapping is derived:

- Menagerie's DevArm has the link origins and frames of the B601-RS URDF, so both share one zero
  pose: the folded rest pose in which the RS arm is zero-calibrated. Every offset is therefore 0.
- The driver's ``00-arm-rs_asm-v3_joint_frame.urdf``, which Studio previews the real arm with, has the
  same joint axes as Menagerie (joint1 +z, joint2 -z, joint3 to joint6 +z), so every arm joint
  maps one to one (``scale`` 1); both go negative from the fold on shoulder_lift and elbow_flex.
- The driver clips actions to its public limits (``REBOT_B601_RS_JOINT_LIMITS_DEG``),
  which are tighter than Menagerie's ranges on shoulder_pan (145 vs 160 degrees), shoulder_lift
  (-170 vs -180), wrist_flex (-90..80 vs -103..97) and wrist_roll (90 vs 180). Each channel's
  ``range`` is the overlap of the two, so the simulation stops where the real arm stops. The one
  exception is elbow_flex: the driver allows -200 degrees, the model only -180 (its geometry), so
  the simulation stops 20 degrees short there.
- Menagerie's two fingers slide 0..0.05 m each from closed to open, coupled 1:1. The driver's URDF
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
from typing import TYPE_CHECKING

from physicalai_mujoco_plugin.profiles._types import CameraSpec, ChannelOverride, EndEffector, RobotProfile

if TYPE_CHECKING:
    import mujoco

REBOT_B601_GRIPPER_SCALE = -270.0 / 0.05
"""Driver gripper degrees (-270 open, 0 closed) per metre of finger travel."""

REBOT_B601_JOINT_RANGES_DEG: dict[str, tuple[float, float] | None] = {
    "shoulder_pan": (-145.0, 145.0),
    "shoulder_lift": (-170.0, 0.0),
    "elbow_flex": None,  # the model's -180..0; the driver allows -200
    "wrist_flex": (-90.0, 80.0),
    "wrist_yaw": None,  # the model's -89.95..89.95; the driver allows -90..90
    "wrist_roll": (-90.0, 90.0),
}
"""Arm joint ranges in the driver's public degrees: its limits where tighter than Menagerie's, else ``None``."""


def _range(name: str) -> tuple[float, float] | None:
    limits = REBOT_B601_JOINT_RANGES_DEG[name]
    return None if limits is None else (radians(limits[0]), radians(limits[1]))


_ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")

REBOT_B601_MIT_KP = dict(zip(_ARM, (50.0, 150.0, 150.0, 50.0, 50.0, 50.0), strict=True))
"""The RS driver's default MIT stiffness per arm joint (N·m/rad), Seeed's and LeRobot's (``REBOT_B601_RS_MIT_KP``)."""

REBOT_B601_MIT_KD = dict(zip(_ARM, (3.0, 10.0, 10.0, 5.0, 4.0, 4.0), strict=True))
"""The RS driver's default MIT damping per arm joint (N·m·s/rad) (``REBOT_B601_RS_MIT_KD``)."""

REBOT_B601_JOINT_FRICTION_NM = (0.2, 0.5, 0.5, 0.3, 0.2, 0.2)
"""Coulomb friction per arm joint (N·m), fitted to real RS teleoperation; joint damping is 0.05 N·m·s/rad."""


def customize_rebot_b601(spec: mujoco.MjSpec) -> None:
    """Make the arm respond like the RS driver: its gains, gravity compensation, the real joints' friction.

    Menagerie's servos (kp 900 on the large joints) become the driver's MIT gains and keep their
    torque limits (36 and 14 N·m, as the RS-06 and RS-00). Every body is gravity-compensated through
    the arm's actuators, so the compensation counts against those limits, like the driver's gravity
    feedforward. The gripper keeps its servo.
    """
    for body in spec.bodies:
        if body.name != "world":
            body.gravcomp = 1.0
    for index, (name, friction) in enumerate(zip(_ARM, REBOT_B601_JOINT_FRICTION_NM, strict=True), start=1):
        spec.actuator(f"joint{index}").set_to_position(kp=REBOT_B601_MIT_KP[name], kv=REBOT_B601_MIT_KD[name])
        joint = spec.joint(f"joint{index}")
        joint.actgravcomp = True
        joint.frictionloss = friction
        try:
            joint.damping = 0.05
        except TypeError:  # MuJoCo >= 3.4: [linear, polynomial terms]
            joint.damping = [0.05, 0.0, 0.0]


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
    # Names and order of REBOT_B601_RS_JOINT_ORDER, in the driver's frame.
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
    customize=customize_rebot_b601,
    cameras=(REBOT_B601_WRIST_CAMERA,),
    default_scene="single_pick_place",
    # The model has no tool site: the point between the finger tips, on the roll axis (link6 +z).
    end_effectors=(EndEffector(body="link6", pos=(0.0, 0.0, 0.163), gripper="gripper"),),
    reach=0.544,
)

__all__ = [
    "REBOT_B601_GRIPPER_SCALE",
    "REBOT_B601_JOINT_FRICTION_NM",
    "REBOT_B601_JOINT_RANGES_DEG",
    "REBOT_B601_MIT_KD",
    "REBOT_B601_MIT_KP",
    "REBOT_B601_PROFILE",
    "REBOT_B601_WRIST_CAMERA",
    "customize_rebot_b601",
]
