# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Franka FR3 twins: the FR3 with its Franka Hand, and the two-arm FR3 Duo.

They follow the names and units of ``physicalai-franka-plugin``: ``FrankaFR3(gripper_enabled=True)``
reports ``fr3_joint1`` ... ``fr3_joint7`` in degrees and ``gripper``, the hand's opening width in
metres (0 to 0.08); ``BimanualFrankaFR3`` prefixes them ``left_`` and ``right_``.

Menagerie's FR3 has no hand. Both profiles mount the Franka Hand of Menagerie's Panda package
(``hand.xml``; the FR3 and the Panda share it) on the FR3 flange as Menagerie's Panda does:
0.107 m out along link 7, turned -45 deg about it. The hand's actuator then drives the finger
tendon as the width the real hand reports, and grasps like the real hand: 70 N on each finger,
at most 0.2 m/s (:func:`~physicalai_mujoco_plugin.profiles.dataset.tune_franka_hand_fingers`).

The Duo is one model, like ALOHA's, with both arms where Franka's FR3 Duo mount puts them, so it
attaches at a single ``robot_mount`` frame and its joint-space poses match the real Duo and the
Franka GELLO Duo one to one. The arms stand on Franka's own stand, from ``franka_description``
(``scripts/build_franka_duo_meshes.py``): the FR3 Duo mount (``fr3_duo_mount_v0_3``) on the
carriage of the mobile FR3 Duo's column (``franka_spine_v0_1``), which stands on the floor behind
the mount. The column's carriage slides; the CAD leaves its height open within its 0.85 m travel.
"""

# MjSpec is supplied by the MuJoCo C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import math
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

from physicalai_mujoco_plugin.profiles._types import CameraSpec, EndEffector, RobotProfile
from physicalai_mujoco_plugin.profiles.dataset import (
    FRANKA_HAND_GRASP_FORCE_N,
    FRANKA_HAND_PEAK_FORCE_N,
    FRANKA_PANDA_PROFILE,
    customize_franka_panda,
    exclude_franka_self_collisions,
    franka_hand_stiffness,
    tune_franka_hand_fingers,
)

if TYPE_CHECKING:
    import mujoco

FRANKA_HAND_WIDTH_M = 0.08
"""The Franka Hand's full opening width: each finger travels 0.04 m."""
_HAND_POS = (0.0, 0.0, 0.107)
_HAND_QUAT = (0.9238795, 0.0, 0.0, -0.3826834)  # -45 deg about link 7, as in Menagerie's panda.xml

# franka_description accessories/fr3_duo_mount_v0_3/kinematics.yaml (release 2.9.0): each arm base is
# 50.18 mm either side of the mount origin, rolled outward 51.2 deg, pitched -10 deg, yawed 26.5 deg.
FRANKA_DUO_ARM_Y = 0.05018
FRANKA_DUO_RPY = (0.89334809, -0.17456074, 0.46334506)
"""Right arm's mount roll, pitch, yaw (rad); the left arm's roll and yaw are negated."""
FRANKA_DUO_MOUNT_X = -0.025
"""The mount origin (between the arm bases) along x from the ``robot_mount`` frame (m): far enough
back that the column clears a table edge 0.17 m behind the frame (bucket_sort's)."""
DUO_MOUNTING_POINT = (-0.037, 0.0, -0.05)
"""The mount's mounting point (its frame, the bracket's foot on the carriage) from its origin
(``fr3_duo_mount_v0_3/kinematics.yaml``)."""
DUO_COVER_Z = 0.068
"""The mount's cover above its mounting point (``fr3_duo_mount.xacro``)."""
SPINE_CARRIAGE = (0.266711, 0.0, 0.1)
"""The column's carriage, where the mounting point sits, from the column's foot at zero travel
(``franka_spine_v0_1/kinematics.yaml``)."""
SPINE_MAX_TRAVEL = 0.85
"""How far the carriage slides up (``franka_spine.xacro``)."""
FRANKA_DUO_HEIGHT = 0.25
"""Mount origin above the ``robot_mount`` frame (m), from the column's carriage.

The carriage sits ``SPINE_CARRIAGE`` (0.10 m) plus its travel above the column's foot, and the arm
bases 0.05 m above the carriage: 0.15 m plus the travel above the floor. On a table 0.75 m above
the floor (bucket_sort), where the column stands on the floor behind it, the full 0.85 m travel puts
the arm bases 0.25 m above the table top; standing on the mount's own plane the column takes 0.10 m
of travel for the same 0.25 m, so the Duo is the same robot in every scene."""
SPINE_TRAVEL = FRANKA_DUO_HEIGHT + DUO_MOUNTING_POINT[2] - SPINE_CARRIAGE[2]
"""The carriage's travel with the column standing on the mount's plane (0.10 m)."""
DUO_STAND_BODY = "duo_spine"
"""The column's body: a scene with its floor below the mount lowers it onto the floor (``RobotProfile.stand``)."""
DUO_STAND_RGBA = {"black": (0.05, 0.05, 0.05, 1.0), "grey": (0.42, 0.42, 0.42, 1.0), "white": (0.95, 0.95, 0.95, 1.0)}
"""The stand's colours: the sources' black, grey (the mount's metal) and white."""
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


def attach_franka_hand(arm: mujoco.MjSpec, grip_force_n: float = FRANKA_HAND_GRASP_FORCE_N) -> None:
    """Mount the Franka Hand on an FR3 arm spec, with a ``gripper`` actuator in width metres.

    Each finger grips with up to *grip_force_n* newtons.
    """
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
    # The tendon is the full finger sum, so its force is the force on each finger.
    actuator.set_to_position(kp=franka_hand_stiffness(grip_force_n))
    actuator.ctrlrange = [0.0, FRANKA_HAND_WIDTH_M]
    actuator.forcerange = [-grip_force_n, grip_force_n]
    actuator.ctrllimited = 1
    actuator.forcelimited = 1
    tune_franka_hand_fingers(arm, grip_force_n=grip_force_n)


def customize_franka_fr3(spec: mujoco.MjSpec, grip_force_n: float = FRANKA_HAND_GRASP_FORCE_N) -> None:
    """FR3 with the Franka Hand; Franka's home pose with the hand open."""
    attach_franka_hand(spec, grip_force_n)
    exclude_franka_self_collisions(spec, link_prefix="fr3_")
    for key in list(spec.keys):
        spec.delete(key)
    home = [math.radians(v) for v in FRANKA_DUO_HOME_DEG["right"]]
    spec.add_key(name="home", qpos=[*home, 0.04, 0.04], ctrl=[*home, FRANKA_HAND_WIDTH_M])


def _add_stand_part(
    spec: mujoco.MjSpec, body: mujoco.MjsBody, part: str, colours: tuple[str, ...], z: float = 0.0
) -> None:
    """Give *body* a stand part's visual meshes (one per colour) and its collision hull, *z* up in the body.

    The meshes are the plugin's own (``urdf/franka_duo/assets``), from ``franka_description``.
    """
    import mujoco  # noqa: PLC0415

    from physicalai_mujoco_plugin._urdf import get_urdf_path  # noqa: PLC0415

    assets = get_urdf_path() / "franka_duo" / "assets"
    for colour in colours:
        mesh = spec.add_mesh(name=f"{part}_{colour}", file=str(assets / f"{part}_{colour}.msh"))
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh.name,
            pos=[0.0, 0.0, z],
            rgba=list(DUO_STAND_RGBA[colour]),
            contype=0,
            conaffinity=0,
            group=2,
            density=0.0,
        )
    hull = spec.add_mesh(name=f"{part}_collision", file=str(assets / f"{part}_collision.msh"))
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_MESH, meshname=hull.name, pos=[0.0, 0.0, z], group=3, density=0.0, rgba=[0.5] * 4
    )


def _add_duo_stand(spec: mujoco.MjSpec) -> None:
    """Add the Duo's mount and the column under it, standing on the ``robot_mount`` plane."""
    x, _, z = DUO_MOUNTING_POINT
    mount = spec.worldbody.add_body(name="duo_mount", pos=[FRANKA_DUO_MOUNT_X + x, 0.0, FRANKA_DUO_HEIGHT + z])
    _add_stand_part(spec, mount, "duo_mount", ("grey",))
    _add_stand_part(spec, mount, "duo_cover", ("white",), DUO_COVER_Z)
    foot = (FRANKA_DUO_MOUNT_X + x - SPINE_CARRIAGE[0], 0.0, 0.0)
    _add_stand_part(
        spec, spec.worldbody.add_body(name=DUO_STAND_BODY, pos=list(foot)), "spine", ("black", "grey", "white")
    )


def customize_franka_duo(spec: mujoco.MjSpec, grip_force_n: float = FRANKA_HAND_GRASP_FORCE_N) -> None:
    """Replace the loaded FR3 with two FR3s and hands on the Duo's stand, named ``left_*`` and ``right_*``."""
    arm = spec.copy()
    customize_franka_fr3(arm, grip_force_n)
    for element in (*spec.keys, *spec.actuators, *spec.excludes):
        spec.delete(element)
    spec.delete(spec.body("base"))
    roll, pitch, yaw = FRANKA_DUO_RPY
    for side, sign in (("left", 1.0), ("right", -1.0)):
        quat = _rpy_quat(-roll, pitch, -yaw) if side == "left" else _rpy_quat(roll, pitch, yaw)
        frame = spec.worldbody.add_frame(
            pos=[FRANKA_DUO_MOUNT_X, sign * FRANKA_DUO_ARM_Y, FRANKA_DUO_HEIGHT], quat=quat
        )
        frame.attach_body(arm.copy().body("base"), f"{side}_", "")
    _add_duo_stand(spec)
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
    stand=DUO_STAND_BODY,
    end_effectors=(
        EndEffector(body="left_hand", pos=_HAND_TIP),
        EndEffector(body="right_hand", pos=_HAND_TIP),
    ),
    # From the tilted mount origin, 0.25 m up on its column (measured 0.837 m; 0.829 keeps the scaled spawn
    # arc's outer corner within 15 degrees of straight down). One arm alone reaches 0.855 m.
    reach=0.829,
)

_FRANKA_HAND_CUSTOMIZERS = (customize_franka_fr3, customize_franka_duo, customize_franka_panda)


def with_grip_force(profile: RobotProfile, grip_force_n: float) -> RobotProfile:
    """Return *profile* with its Franka Hand gripping with *grip_force_n* newtons per finger.

    Returns:
        A copy of the profile whose ``customize`` builds the hand for that force.

    Raises:
        ValueError: If the force is not above 0 and at most the hand's 140 N peak, or the profile
            has no Franka Hand.
    """
    if not 0.0 < grip_force_n <= FRANKA_HAND_PEAK_FORCE_N:  # also rejects NaN
        msg = f"The grip force must be above 0 N and at most {FRANKA_HAND_PEAK_FORCE_N:g} N, got {grip_force_n!r}"
        raise ValueError(msg)
    if profile.customize not in _FRANKA_HAND_CUSTOMIZERS:
        msg = f"Profile {profile.name!r} has no Franka Hand, so --grip-force does not apply to it"
        raise ValueError(msg)
    return replace(profile, customize=partial(profile.customize, grip_force_n=grip_force_n))


__all__ = [
    "FRANKA_DUO_ARM_Y",
    "FRANKA_DUO_HEIGHT",
    "FRANKA_DUO_HOME_DEG",
    "FRANKA_DUO_MOUNT_X",
    "FRANKA_DUO_PROFILE",
    "FRANKA_DUO_RPY",
    "FRANKA_FR3_PROFILE",
    "FRANKA_HAND_WIDTH_M",
    "SPINE_MAX_TRAVEL",
    "SPINE_TRAVEL",
    "attach_franka_hand",
    "customize_franka_duo",
    "customize_franka_fr3",
    "with_grip_force",
]
