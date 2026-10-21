# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Franka FR3 twins: the FR3 with its Franka Hand, and the two-arm FR3 Duo.

They follow the names and units of ``physicalai-franka-plugin``: ``FrankaFR3(gripper_enabled=True)``
reports ``fr3_joint1`` ... ``fr3_joint7`` in degrees and ``gripper``, the hand's opening width in
metres (0 to 0.08); ``BimanualFrankaFR3`` prefixes them ``left_`` and ``right_``.

Menagerie's FR3 has no hand. Both profiles mount the Franka Hand of Menagerie's Panda package
(``hand.xml``; the FR3 and the Panda share it) on the FR3 flange as Menagerie's Panda does:
0.107 m out along link 7, turned -45 deg about it. The hand's actuator then drives the finger
tendon as the width the real hand reports.

The Duo is one model, like ALOHA's, with both arms where Franka's FR3 Duo mount puts them, so it
attaches at a single ``robot_mount`` frame and its joint-space poses match the real Duo and the
Franka GELLO Duo one to one.
"""

# MjSpec is supplied by the MuJoCo C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from physicalai_mujoco_plugin.profiles._types import CameraSpec, EndEffector, RobotProfile
from physicalai_mujoco_plugin.profiles.dataset import FRANKA_PANDA_PROFILE, exclude_franka_self_collisions

if TYPE_CHECKING:
    import mujoco

FRANKA_HAND_WIDTH_M = 0.08
"""The Franka Hand's full opening width: each finger travels 0.04 m."""
_HAND_POS = (0.0, 0.0, 0.107)
_HAND_QUAT = (0.9238795, 0.0, 0.0, -0.3826834)  # -45 deg about link 7, as in Menagerie's panda.xml
# Menagerie's hand actuator pulls a tendon of half the finger sum with kp 100, kv 10 and +-100 N.
# The tendon here is the full width (twice as long), so these halve per-finger force and damping
# back to the same values at each finger joint.
_HAND_KP, _HAND_KV, _HAND_FORCE_N = 25.0, 2.5, 50.0

# franka_description accessories/fr3_duo_mount_v0_3/kinematics.yaml (release 2.9.0): each arm base is
# 50.18 mm either side of the mount origin, rolled outward 51.2 deg, pitched -10 deg, yawed 26.5 deg.
FRANKA_DUO_ARM_Y = 0.05018
FRANKA_DUO_RPY = (0.89334809, -0.17456074, 0.46334506)
"""Right arm's mount roll, pitch, yaw (rad); the left arm's roll and yaw are negated."""
FRANKA_DUO_HEIGHT = 0.30
"""Mount origin above the ``robot_mount`` frame (m): an estimate of the Duo's column, which the CAD leaves open."""
FRANKA_DUO_HOME_DEG = {
    "left": (0.0, 0.0, 0.0, -90.0, 0.0, 90.0, -45.0),
    "right": (0.0, 0.0, 0.0, -90.0, 0.0, 90.0, 45.0),
}
"""Franka's home pose for the right arm, mirrored (J1, J3, J5, J7 negated) for the left."""


def _rpy_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    """Convert URDF fixed-axis roll-pitch-yaw (``Rz(yaw) Ry(pitch) Rx(roll)``).

    Returns:
        The rotation as a ``wxyz`` quaternion.
    """
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def _load_hand() -> mujoco.MjSpec:
    import mujoco  # noqa: PLC0415

    from physicalai_mujoco_plugin.compose import fetch_profile  # noqa: PLC0415 - compose imports the profiles

    return mujoco.MjSpec.from_file(str(fetch_profile(FRANKA_PANDA_PROFILE).parent / "hand.xml"))


def attach_franka_hand(arm: mujoco.MjSpec) -> None:
    """Mount the Franka Hand on an FR3 arm spec, with a ``gripper`` actuator in width metres."""
    import mujoco  # noqa: PLC0415

    hand = _load_hand()
    taken = {material.name for material in arm.materials}
    for material in hand.materials:  # both models name materials black and white
        if material.name in taken:
            old, material.name = material.name, f"hand_{material.name}"
            for geom in hand.geoms:
                if geom.material == old:
                    geom.material = material.name
    hand.delete(hand.actuator("actuator8"))
    frame = arm.body("fr3_link7").add_frame(pos=list(_HAND_POS), quat=list(_HAND_QUAT))
    frame.attach_body(hand.body("hand"), "", "").quat = [1.0, 0.0, 0.0, 0.0]
    width = arm.add_tendon(name="gripper_width")
    width.wrap_joint("finger_joint1", 1.0)
    width.wrap_joint("finger_joint2", 1.0)
    split = next(tendon for tendon in arm.tendons if tendon.name == "split")
    arm.delete(split)
    actuator = arm.add_actuator(name="gripper", target="gripper_width", trntype=mujoco.mjtTrn.mjTRN_TENDON)
    actuator.set_to_position(kp=_HAND_KP, kv=_HAND_KV)
    actuator.ctrlrange = [0.0, FRANKA_HAND_WIDTH_M]
    actuator.forcerange = [-_HAND_FORCE_N, _HAND_FORCE_N]
    actuator.ctrllimited = 1
    actuator.forcelimited = 1


def customize_franka_fr3(spec: mujoco.MjSpec) -> None:
    """FR3 with the Franka Hand; Franka's home pose with the hand open."""
    attach_franka_hand(spec)
    exclude_franka_self_collisions(spec, link_prefix="fr3_")
    for key in list(spec.keys):
        spec.delete(key)
    home = [math.radians(v) for v in FRANKA_DUO_HOME_DEG["right"]]
    spec.add_key(name="home", qpos=[*home, 0.04, 0.04], ctrl=[*home, FRANKA_HAND_WIDTH_M])


def customize_franka_duo(spec: mujoco.MjSpec) -> None:
    """Replace the loaded FR3 with two FR3s and hands on the Duo mount, named ``left_*`` and ``right_*``."""
    arm = spec.copy()
    customize_franka_fr3(arm)
    for element in (*spec.keys, *spec.actuators, *spec.excludes):
        spec.delete(element)
    spec.delete(spec.body("base"))
    roll, pitch, yaw = FRANKA_DUO_RPY
    for side, sign in (("left", 1.0), ("right", -1.0)):
        quat = _rpy_quat(-roll, pitch, -yaw) if side == "left" else _rpy_quat(roll, pitch, yaw)
        frame = spec.worldbody.add_frame(pos=[0.0, sign * FRANKA_DUO_ARM_Y, FRANKA_DUO_HEIGHT], quat=quat)
        frame.attach_body(arm.copy().body("base"), f"{side}_", "")
    qpos, ctrl = [], []
    for side in ("left", "right"):
        home = [math.radians(v) for v in FRANKA_DUO_HOME_DEG[side]]
        qpos += [*home, 0.04, 0.04]
        ctrl += [*home, FRANKA_HAND_WIDTH_M]
    spec.add_key(name="home", qpos=qpos, ctrl=ctrl)


_HAND_TIP = (0.0, 0.0, 0.1034)
"""Between the finger tips in the hand frame: Franka's TCP, 103.4 mm from the hand's base."""


def _wrist_camera(name: str, body: str) -> CameraSpec:
    """Place the plugin's generated Franka Hand wrist camera (as the Panda profile derives it) on *body*.

    Returns:
        The camera, named *name*.
    """
    return CameraSpec(
        name=name,
        body=body,
        pos=(-0.0823, 0.0, 0.07),
        quat=(-0.18574699792052746, -0.6822741771190755, 0.6822741771190755, 0.18574699792052746),
        fovy=75.0,
    )


FRANKA_FR3_PROFILE = RobotProfile(
    name="franka_fr3",
    display_name="Franka Research 3",
    menagerie_model="franka_fr3",
    menagerie_entry="fr3",
    tier="twin",
    default_unit="degrees",
    customize=customize_franka_fr3,
    default_scene="single_pick_place",
    end_effectors=(EndEffector(body="hand", pos=_HAND_TIP),),
    reach=0.855,
)
FRANKA_DUO_PROFILE = RobotProfile(
    name="franka_duo",
    display_name="Franka Duo",
    menagerie_model="franka_fr3",
    menagerie_entry="fr3",
    tier="twin",
    default_unit="degrees",
    cameras=(_wrist_camera("left_wrist", "left_hand"), _wrist_camera("right_wrist", "right_hand")),
    customize=customize_franka_duo,
    default_scene="single_pick_place",
    end_effectors=(
        EndEffector(body="left_hand", pos=_HAND_TIP),
        EndEffector(body="right_hand", pos=_HAND_TIP),
    ),
    reach=0.815,  # from the tilted mount origin; one arm alone reaches 0.855 m
)

__all__ = [
    "FRANKA_DUO_ARM_Y",
    "FRANKA_DUO_HEIGHT",
    "FRANKA_DUO_HOME_DEG",
    "FRANKA_DUO_PROFILE",
    "FRANKA_DUO_RPY",
    "FRANKA_FR3_PROFILE",
    "FRANKA_HAND_WIDTH_M",
    "attach_franka_hand",
    "customize_franka_duo",
    "customize_franka_fr3",
]
