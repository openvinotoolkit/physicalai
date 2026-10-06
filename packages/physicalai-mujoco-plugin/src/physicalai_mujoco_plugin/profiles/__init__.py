# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Robot profiles: what the plugin knows about a robot beyond its MuJoCo model.

Importing this package does not import MuJoCo, so the Studio catalog and the CLI can use it.
"""

from physicalai_mujoco_plugin.profiles._types import (
    ChannelOverride,
    DefaultUnit,
    PDOverride,
    ProfileTier,
    RobotProfile,
    Unit,
    validate_profile,
)
from physicalai_mujoco_plugin.profiles.registry import PROFILES, UR5E_PROFILE, get_profile, list_profiles
from physicalai_mujoco_plugin.profiles.so101 import SO101_PROFILE

__all__ = [
    "PROFILES",
    "SO101_PROFILE",
    "UR5E_PROFILE",
    "ChannelOverride",
    "DefaultUnit",
    "PDOverride",
    "ProfileTier",
    "RobotProfile",
    "Unit",
    "get_profile",
    "list_profiles",
    "validate_profile",
]
