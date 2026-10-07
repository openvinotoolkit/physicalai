# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Profile registry: hand-written profiles first, then any MuJoCo Menagerie model by name."""

from __future__ import annotations

from physicalai_mujoco_plugin.profiles._types import CameraSpec, EndEffector, RobotProfile, validate_profile
from physicalai_mujoco_plugin.profiles.dataset import DATASET_PROFILES
from physicalai_mujoco_plugin.profiles.rebot_b601 import REBOT_B601_PROFILE
from physicalai_mujoco_plugin.profiles.so101 import SO101_PROFILE
from physicalai_mujoco_plugin.profiles.trossen_wxai import TROSSEN_WXAI_PROFILE

UR5E_PROFILE = RobotProfile(
    name="ur5e",
    display_name="Universal Robots UR5e",
    menagerie_model="universal_robots_ur5e",
    menagerie_entry="ur5e",
    tier="dataset",
    default_scene="single_pick_place",
    end_effectors=(EndEffector(site="attachment_site"),),
    reach=0.945,
)

# Experimental floating-base robots: derived channels on floor scenes, no balance controller.
UNITREE_G1_PROFILE = RobotProfile(
    name="unitree_g1",
    display_name="Unitree G1",
    menagerie_model="unitree_g1",
    menagerie_entry="g1",
    tier="experimental",
    default_scene="floor_flat",
    cameras=(
        # The G1's head RealSense D435 (Unitree g1_29dof_rev_1_0.urdf, d435_joint on torso_link: xyz
        # 0.0576 0.0175 0.4299, pitched 0.831 rad down). The quaternion turns that pose into MuJoCo's
        # camera axes (looks along -z, +y up); fovy is the D435 colour sensor's vertical field of view.
        CameraSpec(
            name="head",
            body="torso_link",
            pos=(0.0576235, 0.01753, 0.42987),
            quat=(-0.6592524821011074, -0.2557071857487173, 0.2557071857487173, 0.6592524821011074),
            fovy=42.0,
        ),
    ),
)
UNITREE_GO2_PROFILE = RobotProfile(
    name="unitree_go2",
    display_name="Unitree Go2",
    menagerie_model="unitree_go2",
    menagerie_entry="go2",
    tier="experimental",
    default_scene="floor_flat",
)
BOSTON_DYNAMICS_SPOT_PROFILE = RobotProfile(
    name="boston_dynamics_spot",
    display_name="Boston Dynamics Spot",
    menagerie_model="boston_dynamics_spot",
    menagerie_entry="spot",
    tier="experimental",
    default_scene="floor_flat",
)

PROFILES: dict[str, RobotProfile] = {
    profile.name: profile
    for profile in (
        SO101_PROFILE,
        TROSSEN_WXAI_PROFILE,
        REBOT_B601_PROFILE,
        UR5E_PROFILE,
        *DATASET_PROFILES,
        UNITREE_G1_PROFILE,
        UNITREE_GO2_PROFILE,
        BOSTON_DYNAMICS_SPOT_PROFILE,
    )
}
"""Hand-written profiles by name; ``physicalai-mujoco prefetch`` downloads all of them."""

for _profile in PROFILES.values():
    validate_profile(_profile)


def get_profile(name: str) -> RobotProfile:
    """Return the hand-written profile *name*, or a derived profile for the Menagerie model *name*.

    Args:
        name: A profile name (``so101``) or a MuJoCo Menagerie model name (``unitree_go2``).

    Returns:
        The profile. Menagerie models without a hand-written profile get ``tier="unsupported"``.

    Raises:
        KeyError: If *name* is neither a profile nor a Menagerie model.
    """
    registered = PROFILES.get(name)
    if registered is not None:
        return registered
    import mujoco_menagerie  # noqa: PLC0415

    try:
        robot = mujoco_menagerie.get(name)
    except mujoco_menagerie.MenagerieError as exc:
        msg = f"Unknown robot profile {name!r}: not a profile ({', '.join(PROFILES)}) nor a Menagerie model"
        raise KeyError(msg) from exc
    return RobotProfile(name=name, display_name=robot.display_name, menagerie_model=name)


def list_profiles(*, include_unsupported: bool = False) -> tuple[RobotProfile, ...]:
    """Return the hand-written profiles, optionally followed by every other Menagerie model.

    Returns:
        Profiles in registry order, then Menagerie order.
    """
    if not include_unsupported:
        return tuple(PROFILES.values())
    import mujoco_menagerie  # noqa: PLC0415

    extra = (get_profile(name) for name in mujoco_menagerie.names() if name not in PROFILES)
    return (*PROFILES.values(), *extra)


__all__ = [
    "BOSTON_DYNAMICS_SPOT_PROFILE",
    "PROFILES",
    "UNITREE_G1_PROFILE",
    "UNITREE_GO2_PROFILE",
    "UR5E_PROFILE",
    "get_profile",
    "list_profiles",
]
