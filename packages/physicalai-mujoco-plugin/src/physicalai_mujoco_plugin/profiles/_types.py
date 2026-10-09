# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Profile value types and static validation. Importing this module does not import MuJoCo."""

# MjSpec is supplied by the MuJoCo C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import mujoco

Unit = Literal["degrees", "metres", "normalized", "raw"]
"""Public unit of one channel. ``degrees`` keeps slide joints and tendons in metres."""
DefaultUnit = Literal["normalized", "degrees"]
"""Robot-wide unit choice (``MuJoCoRobot(unit=...)``); channel overrides may pin their own."""
ProfileTier = Literal["twin", "dataset", "experimental", "unsupported"]


@dataclass(frozen=True)
class ChannelOverride:
    """One public channel: a single model actuator, or a group of actuators that move together.

    ``public = offset + scale * value``, where ``value`` is the model value in the channel's unit
    (or its normalized position). A group writes ``value * member_scales[k - 1]`` to member ``k``
    and reads member 0.
    """

    name: str
    actuators: tuple[str, ...]
    unit: Unit | None = None
    """``None`` follows the robot-wide unit."""
    scale: float = 1.0
    offset: float = 0.0
    member_scales: tuple[float, ...] = ()
    gripper: bool = False
    """Normalized span ``[0, 100]`` instead of ``[-100, 100]``."""
    range: tuple[float, float] | None = None
    """Joint range in model units. It replaces the model's range and pins the normalized span."""
    velocity_unit: Literal["public", "model"] = "public"


@dataclass(frozen=True)
class PDOverride:
    """Software-PD settings for torque-controlled channels."""

    bandwidth_hz: float = 20.0
    kp: Mapping[str, float] | None = None
    kd: Mapping[str, float] | None = None


@dataclass(frozen=True)
class RobotProfile:
    """Everything the plugin knows about one robot beyond what its model says.

    An empty profile (only the Menagerie name) is valid: channels, the home pose (the model's home
    keyframe, else ``qpos0``), sensors and cameras are then derived from the model. Fields for
    floating bases, cameras and twins arrive with the pull requests that use them.
    """

    name: str
    display_name: str
    menagerie_model: str
    menagerie_entry: str | None = None
    """Robot-only entry point of the Menagerie model; ``None`` uses the model's default."""
    tier: ProfileTier = "unsupported"
    channels: tuple[ChannelOverride, ...] = ()
    """Public layout in order; empty means one derived channel per actuator."""
    default_unit: DefaultUnit = "degrees"
    pd: PDOverride | None = None
    actuator_forcerange: tuple[float, float] | None = None
    """Force range applied to every actuator named in ``channels``."""
    customize: Callable[[mujoco.MjSpec], None] | None = None
    """Last-resort edits of the loaded robot spec, before it is attached."""
    renames: tuple[tuple[str, str], ...] = ()
    default_scene: str | None = None


def validate_profile(profile: RobotProfile) -> None:
    """Check a profile's values without loading its model.

    Model references (actuators, joints, camera bodies) are checked when the profile is bound
    to a compiled model, in :class:`~physicalai_mujoco_plugin.channels.ArmChannels`.

    Raises:
        ValueError: If channel names or actuators repeat, or a factor, range or gain is invalid.
    """
    seen_channels: set[str] = set()
    seen_actuators: set[str] = set()
    for channel in profile.channels:
        where = f"Profile {profile.name!r} channel {channel.name!r}"
        if not channel.name or channel.name in seen_channels:
            msg = f"{where}: names must be non-empty and unique"
            raise ValueError(msg)
        seen_channels.add(channel.name)
        if not channel.actuators:
            msg = f"{where} has no actuators"
            raise ValueError(msg)
        if len(channel.member_scales) not in {0, len(channel.actuators) - 1}:
            msg = f"{where} has {len(channel.member_scales)} member scales for {len(channel.actuators)} actuators"
            raise ValueError(msg)
        factors = (channel.scale, channel.offset, *channel.member_scales)
        if not all(isfinite(value) for value in factors) or channel.scale == 0:
            msg = f"{where} has an invalid scale, offset or member scale"
            raise ValueError(msg)
        if channel.range is not None:
            lower, upper = channel.range
            if not (isfinite(lower) and isfinite(upper) and lower < upper):
                msg = f"{where} has an invalid range {channel.range!r}"
                raise ValueError(msg)
        for actuator in channel.actuators:
            if actuator in seen_actuators:
                msg = f"Profile {profile.name!r} assigns actuator {actuator!r} to two channels"
                raise ValueError(msg)
            seen_actuators.add(actuator)
    if profile.actuator_forcerange is not None:
        low, high = profile.actuator_forcerange
        if not (isfinite(low) and isfinite(high) and low < high):
            msg = f"Profile {profile.name!r} has an invalid actuator force range {profile.actuator_forcerange!r}"
            raise ValueError(msg)
    if profile.pd is not None and not (isfinite(profile.pd.bandwidth_hz) and profile.pd.bandwidth_hz > 0):
        msg = f"Profile {profile.name!r}: the PD bandwidth must be finite and positive"
        raise ValueError(msg)


__all__ = [
    "ChannelOverride",
    "DefaultUnit",
    "PDOverride",
    "ProfileTier",
    "RobotProfile",
    "Unit",
    "validate_profile",
]
