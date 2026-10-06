# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Profile registry: hand-written profiles first, then any MuJoCo Menagerie model by name."""

from __future__ import annotations

from physicalai_mujoco_plugin.profiles._types import RobotProfile, validate_profile
from physicalai_mujoco_plugin.profiles.so101 import SO101_PROFILE

UR5E_PROFILE = RobotProfile(
    name="ur5e",
    display_name="Universal Robots UR5e",
    menagerie_model="universal_robots_ur5e",
    menagerie_entry="ur5e",
    tier="dataset",
    default_scene="single_pick_place",
)

PROFILES: dict[str, RobotProfile] = {profile.name: profile for profile in (SO101_PROFILE, UR5E_PROFILE)}
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


__all__ = ["PROFILES", "UR5E_PROFILE", "get_profile", "list_profiles"]
