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
CameraSource = Literal["override", "model", "default"]
"""Where a robot camera comes from: the profile, the model's own sensor camera, or a generated default."""


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
class CameraSpec:
    """A robot camera: its unprefixed public name and its pose in a robot body.

    The camera looks along its frame's -z axis with +y up, as every MuJoCo camera does.
    """

    name: str
    """Unprefixed public name, e.g. ``wrist``; a robot attached with prefix ``left_`` publishes ``left_wrist``."""
    body: str
    """Parent body, by its unprefixed model name."""
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    """Orientation in the body frame, ``wxyz``."""
    fovy: float
    """Vertical field of view in degrees."""
    source: CameraSource = "override"


@dataclass(frozen=True)
class EndEffector:
    """One arm's tool frame: the point between its finger tips, or its flange, and the way it points.

    It is either a site of the robot model, or a point ``pos`` in a body's frame. The approach axis
    ``axis`` is the direction, in that site's or body's frame, that points out of the gripper
    (or out of the flange of an arm without one); grasping from above points it straight down.
    """

    site: str | None = None
    """Site at the tool point, by its unprefixed model name."""
    body: str | None = None
    """Body that carries the tool point when the model has no site there, by its unprefixed name."""
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Tool point in ``body``'s frame; only used with ``body``."""
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Approach axis in the site's or body's frame."""


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
    floating bases and twins arrive with the pull requests that use them.
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
    cameras: tuple[CameraSpec, ...] | None = None
    """Robot cameras; ``None`` takes the model's sensor cameras, else generated defaults (CAM-2)."""
    pd: PDOverride | None = None
    actuator_forcerange: tuple[float, float] | None = None
    """Force range applied to every actuator named in ``channels``."""
    customize: Callable[[mujoco.MjSpec], None] | None = None
    """Last-resort edits of the loaded robot spec, before it is attached."""
    renames: tuple[tuple[str, str], ...] = ()
    default_scene: str | None = None
    end_effectors: tuple[EndEffector, ...] = ()
    """Tool frame of each arm, in the model's arm order; ALOHA's one model lists both arms."""
    reach: float | None = None
    """Reach from above in metres; ``layout="reach"`` scenes scale with ``reach / SO-101 reach`` (SCN-6).

    It is the farthest horizontal distance from the arm's base, toward the scene's spawn area, at
    which the end effector reaches a point 2 cm above the table within 5 mm, with its approach axis
    within 15 degrees of straight down (the SO-101 needs 12 degrees at the far edge of its own spawn
    area). The reachability test (``tests/test_reach.py``) measures it and checks the stored value;
    running that file as a script prints the measurements.
    """


MIN_CHANNEL_SCALE, MAX_CHANNEL_SCALE = 1e-3, 1e4
"""Range of ``|ChannelOverride.scale|`` that :func:`validate_profile` accepts."""
MAX_OFFSET_PER_SCALE = 1e3
"""``|offset| / |scale|`` limit: float32 observations then still resolve 1e-4 of a unit before scaling."""
MIN_MEMBER_SCALE, MAX_MEMBER_SCALE = 1e-3, 1e3
"""Range of ``|member scale|`` that :func:`validate_profile` accepts."""


def _validate_factors(where: str, channel: ChannelOverride) -> None:
    """Check a channel's scale, offset and member scales.

    Raises:
        ValueError: If one is not finite or outside the accepted range.
    """
    factors = (channel.scale, channel.offset, *channel.member_scales)
    if not all(isfinite(value) for value in factors) or channel.scale == 0:
        msg = f"{where} has an invalid scale, offset or member scale"
        raise ValueError(msg)
    # Unit conversions stay well inside float32 observations: degrees, metres and driver units
    # differ by a few orders of magnitude at most (a driver's 900 per metre is the largest so far),
    # and an offset must not swamp the scaled value in float32.
    if not MIN_CHANNEL_SCALE <= abs(channel.scale) <= MAX_CHANNEL_SCALE:
        msg = f"{where} needs {MIN_CHANNEL_SCALE:g} <= |scale| <= {MAX_CHANNEL_SCALE:g}"
        raise ValueError(msg)
    if abs(channel.offset) > MAX_OFFSET_PER_SCALE * abs(channel.scale):
        msg = f"{where} needs |offset| <= {MAX_OFFSET_PER_SCALE:g} * |scale|"
        raise ValueError(msg)
    if not all(MIN_MEMBER_SCALE <= abs(scale) <= MAX_MEMBER_SCALE for scale in channel.member_scales):
        msg = f"{where} needs {MIN_MEMBER_SCALE:g} <= |member scale| <= {MAX_MEMBER_SCALE:g}"
        raise ValueError(msg)


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
        _validate_factors(where, channel)
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
    _validate_cameras(profile)
    _validate_reach(profile)


def _validate_cameras(profile: RobotProfile) -> None:
    seen: set[str] = set()
    for camera in profile.cameras or ():
        where = f"Profile {profile.name!r} camera {camera.name!r}"
        if not camera.name or camera.name in seen:
            msg = f"{where}: names must be non-empty and unique"
            raise ValueError(msg)
        seen.add(camera.name)
        if not camera.body:
            msg = f"{where} has no body"
            raise ValueError(msg)
        if len(camera.pos) != 3 or not all(isfinite(value) for value in camera.pos):  # noqa: PLR2004
            msg = f"{where} has an invalid position {camera.pos!r}"
            raise ValueError(msg)
        norm = sum(value * value for value in camera.quat)
        if len(camera.quat) != 4 or not all(isfinite(value) for value in camera.quat) or norm == 0:  # noqa: PLR2004
            msg = f"{where} has an invalid quaternion {camera.quat!r}"
            raise ValueError(msg)
        if not (isfinite(camera.fovy) and 0 < camera.fovy < 180):  # noqa: PLR2004
            msg = f"{where} needs a field of view between 0 and 180 degrees, got {camera.fovy!r}"
            raise ValueError(msg)


def _validate_reach(profile: RobotProfile) -> None:
    """Check the end effectors (a site, or a body and a finite point) and the reach.

    Raises:
        ValueError: If an end effector names neither or both of a site and a body, has a non-finite
            point or axis, or a zero axis, or the reach is not a finite positive length.
    """
    for index, effector in enumerate(profile.end_effectors):
        where = f"Profile {profile.name!r} end effector {index}"
        if (effector.site is None) == (effector.body is None):
            msg = f"{where} needs exactly one of a site or a body"
            raise ValueError(msg)
        if len(effector.pos) != 3 or not all(isfinite(value) for value in effector.pos):  # noqa: PLR2004
            msg = f"{where} has an invalid position {effector.pos!r}"
            raise ValueError(msg)
        norm = sum(value * value for value in effector.axis)
        if len(effector.axis) != 3 or not all(isfinite(value) for value in effector.axis) or norm == 0:  # noqa: PLR2004
            msg = f"{where} has an invalid approach axis {effector.axis!r}"
            raise ValueError(msg)
    if profile.reach is not None and not (isfinite(profile.reach) and profile.reach > 0):
        msg = f"Profile {profile.name!r} has an invalid reach {profile.reach!r}"
        raise ValueError(msg)


__all__ = [
    "CameraSource",
    "CameraSpec",
    "ChannelOverride",
    "DefaultUnit",
    "EndEffector",
    "PDOverride",
    "ProfileTier",
    "RobotProfile",
    "Unit",
    "validate_profile",
]
