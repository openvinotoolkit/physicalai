# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Public channels of one attached robot: unit conversion, actuator writes and software PD.

One :class:`ArmChannels` covers one robot (one scene anchor); the driver concatenates them in anchor
order. Conversions follow CHN-3 of the design: ``read`` goes model value → unit value → public
value, ``write`` goes back the same way and then to ``ctrl`` per the actuator's control kind.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from dataclasses import dataclass
from itertools import starmap
from math import degrees, isfinite, radians, sqrt
from typing import TYPE_CHECKING, Literal

import numpy as np

from physicalai_mujoco_plugin.profiles import ChannelOverride

if TYPE_CHECKING:
    from collections.abc import Mapping

    import mujoco

    from physicalai_mujoco_plugin.profiles import DefaultUnit, RobotProfile, Unit
    from physicalai_mujoco_plugin.profiles.derive import DerivedChannel, DerivedLayout

TorqueMode = Literal["pd", "raw"]
"""``pd``: torque actuators track position targets through a software PD. ``raw``: actions are ``ctrl``."""


@dataclass(frozen=True)
class Channel:
    """One public vector element, backed by one actuator or a group of actuators."""

    name: str
    members: tuple[DerivedChannel, ...]
    unit: Unit
    scale: float
    offset: float
    member_scales: tuple[float, ...]
    range: tuple[float, float] | None
    """Model-unit range that ``normalized`` spans."""
    public_range: tuple[float, float]
    """``(-100, 100)``, or ``(0, 100)`` for grippers; only used by ``normalized``."""
    velocity_unit: Literal["public", "model"]

    @property
    def identity(self) -> bool:
        """Whether ``scale`` and ``offset`` leave values unchanged, so they are skipped bit-exactly."""
        return self.scale == 1.0 and self.offset == 0.0  # noqa: RUF069 - exact defaults, not computed values

    @property
    def first(self) -> DerivedChannel:
        """The member that is read back."""
        return self.members[0]

    @property
    def hinge(self) -> bool:
        """Whether the channel's model value is an angle in radians."""
        return self.first.unit == "degrees"


class ArmChannels:
    """Bind a profile's public channels to one attached robot of a compiled model."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        layout: DerivedLayout,
        profile: RobotProfile,
        *,
        prefix: str = "",
        unit: DefaultUnit | None = None,
        torque_mode: TorqueMode = "pd",
    ) -> None:
        """Resolve the channels and the PD gains (at the current pose, normally home).

        Args:
            model: The compiled (composed) model.
            data: Its data.
            layout: The robot's layout, bound to ``model`` (prefixed names, composed ids).
            profile: The robot's profile; its channel overrides use unprefixed names.
            prefix: The robot's anchor prefix.
            unit: Robot-wide unit; ``None`` uses the profile's default.
            torque_mode: How torque actuators are driven (CHN-9, CHN-10).

        Raises:
            ValueError: If an override names a missing actuator, a group mixes control kinds, or a
                ``normalized`` channel has no finite range.
        """
        if torque_mode not in {"pd", "raw"}:
            msg = f"Unsupported torque_mode {torque_mode!r}; expected 'pd' or 'raw'"
            raise ValueError(msg)
        self.model = model
        self.data = data
        self.profile = profile
        self.torque_mode: TorqueMode = torque_mode
        self.channels = _bind(layout, profile, prefix, unit or profile.default_unit, torque_mode)
        self.names = tuple(channel.name for channel in self.channels)
        self._pd_targets = np.array([self._read_model(channel) for channel in self.channels], dtype=np.float64)
        self._pd_rows = self._pd_gains() if torque_mode == "pd" else ()

    def __len__(self) -> int:
        """Number of public channels.

        Returns:
            The channel count.
        """
        return len(self.channels)

    @property
    def units(self) -> tuple[str, ...]:
        """Public unit of each channel."""
        return tuple(channel.unit for channel in self.channels)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read_positions(self) -> np.ndarray:
        """Return the current positions in public units, in channel order (``float64``)."""
        return np.array([self.to_public(channel, self._read_model(channel)) for channel in self.channels])

    def read_velocities(self) -> np.ndarray:
        """Return the current velocities in the positions' units per second (CHN-8), as ``float64``."""
        return np.array([self._velocity(channel) for channel in self.channels])

    def model_targets(self) -> np.ndarray:
        """Return each channel's current target in model units (radians, metres, ``ctrl`` for raw).

        Position and velocity targets are read back from ``ctrl``, so targets written by scene
        automation (the conveyor autopilot) are included.
        """
        return np.array(list(starmap(self._target, enumerate(self.channels))))

    def targets_to_public(self, model_values: np.ndarray) -> np.ndarray:
        """Convert model-unit values, one per channel, to public units.

        Returns:
            Public values in channel order.
        """
        return np.array([
            self.to_public(channel, float(value)) for channel, value in zip(self.channels, model_values, strict=True)
        ])

    @staticmethod
    def to_public(channel: Channel, value: float) -> float:
        """Convert one model value of *channel* to its public unit.

        Returns:
            The public value; ``normalized`` is clamped to the channel's span.
        """
        if channel.unit == "normalized":
            lower, upper = channel.range  # type: ignore[misc]
            public_low, public_high = channel.public_range
            fraction = min(max((value - lower) / (upper - lower), 0.0), 1.0)
            value = public_low + fraction * (public_high - public_low)
        elif channel.unit == "degrees" and channel.hinge:
            value = degrees(value)
        return _affine(channel, value)

    @staticmethod
    def to_model(channel: Channel, public: float) -> float:
        """Convert one public value of *channel* to model units.

        Returns:
            The model value; ``normalized`` is clamped to the channel's range.
        """
        value = public if channel.identity else (public - channel.offset) / channel.scale
        if channel.unit == "normalized":
            lower, upper = channel.range  # type: ignore[misc]
            public_low, public_high = channel.public_range
            fraction = min(max((value - public_low) / (public_high - public_low), 0.0), 1.0)
            return lower + fraction * (upper - lower)
        if channel.unit == "degrees" and channel.hinge:
            return radians(value)
        return value

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write(self, action: np.ndarray) -> None:
        """Set every channel's target from one public action vector.

        Raises:
            ValueError: If the vector has the wrong shape or a non-finite value.
        """
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (len(self.channels),):
            msg = f"Expected action shape ({len(self.channels)},), got {values.shape}"
            raise ValueError(msg)
        if not np.isfinite(values).all():
            msg = "Action values must be finite"
            raise ValueError(msg)
        for index, (channel, public) in enumerate(zip(self.channels, values, strict=True)):
            self._write_model(index, channel, self.to_model(channel, float(public)))

    def write_model_targets(self, model_values: Mapping[str, float]) -> None:
        """Set targets in model units, by joint name of each channel's first member; others keep theirs."""
        for index, channel in enumerate(self.channels):
            joint = channel.first.joint
            if joint is not None and joint in model_values:
                self._write_model(index, channel, float(model_values[joint]))

    def place(self, qpos: Mapping[str, float]) -> None:
        """Teleport each channel's joint to ``qpos[joint]`` at rest, and hold it there.

        Positions are clipped to the joint ranges. Channels whose joint is not in *qpos* keep
        their position and target.
        """
        model, data = self.model, self.data
        for index, channel in enumerate(self.channels):
            joint, joint_id = channel.first.joint, channel.first.joint_id
            if joint is None or joint_id is None or joint not in qpos:
                continue
            value = float(qpos[joint])
            if bool(model.jnt_limited[joint_id]):
                low, high = (float(v) for v in model.jnt_range[joint_id])
                value = min(max(value, low), high)
            data.qpos[channel.first.qpos_adr] = value
            data.qvel[channel.first.dof_adr] = 0.0
            self._write_model(index, channel, value, clamp=True)

    def apply_pd(self) -> None:
        """Update the torque of PD-driven channels; call before every physics step (CHN-9)."""
        data = self.data
        for index, member, kp, kd, gain in self._pd_rows:
            torque = kp * (self._pd_targets[index] - data.qpos[member.qpos_adr]) - kd * data.qvel[member.dof_adr]
            data.ctrl[member.actuator_id] = self._clip_ctrl(member, torque / gain)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_model(self, channel: Channel) -> float:
        member, data = channel.first, self.data
        if member.kind == "raw" or (member.kind == "torque" and self.torque_mode == "raw"):
            return float(data.ctrl[member.actuator_id])
        if member.kind == "velocity":
            return _per_gear(member, float(data.actuator_velocity[member.actuator_id]))
        if member.qpos_adr is not None:
            return float(data.qpos[member.qpos_adr])
        return _per_gear(member, float(data.actuator_length[member.actuator_id]))

    def _velocity(self, channel: Channel) -> float:
        member, data = channel.first, self.data
        if member.dof_adr is not None:
            value = float(data.qvel[member.dof_adr])
        else:
            value = _per_gear(member, float(data.actuator_velocity[member.actuator_id]))
        if channel.velocity_unit == "model":
            return value
        if channel.unit == "normalized":
            lower, upper = channel.range  # type: ignore[misc]
            public_low, public_high = channel.public_range
            value *= (public_high - public_low) / (upper - lower)
        elif channel.unit == "degrees" and channel.hinge:
            value = degrees(value)
        return value if channel.identity else channel.scale * value

    def _target(self, index: int, channel: Channel) -> float:
        member = channel.first
        if member.kind in {"position", "velocity"}:
            factor = member.ctrl_per_unit * member.gear
            return float(self.data.ctrl[member.actuator_id]) / factor
        if member.kind == "torque" and self.torque_mode == "pd":
            return float(self._pd_targets[index])
        return float(self.data.ctrl[member.actuator_id])

    def _write_model(self, index: int, channel: Channel, value: float, *, clamp: bool = False) -> None:
        """Set *channel*'s target to a model value; group members get their scaled share (PRF-8)."""
        model, data = self.model, self.data
        for position, member in enumerate(channel.members):
            target = (
                value if position == 0 or not channel.member_scales else value * channel.member_scales[position - 1]
            )
            if member.kind == "position" and member.joint_id is not None and bool(model.jnt_limited[member.joint_id]):
                low, high = (float(v) for v in model.jnt_range[member.joint_id])
                target = min(max(target, low), high)
            if member.kind in {"position", "velocity"}:
                # MuJoCo clamps ctrl to ctrlrange itself; storing the unclamped target keeps it
                # readable at full precision (model_targets, the virtual leader). Homing clamps,
                # as it always has.
                ctrl = member.ctrl_per_unit * member.gear * target
                data.ctrl[member.actuator_id] = self._clip_ctrl(member, ctrl) if clamp else ctrl
            elif member.kind == "torque" and self.torque_mode == "pd":
                if position == 0:
                    self._pd_targets[index] = target
            else:
                data.ctrl[member.actuator_id] = self._clip_ctrl(member, target)

    def _clip_ctrl(self, member: DerivedChannel, ctrl: float) -> float:
        if bool(self.model.actuator_ctrllimited[member.actuator_id]):
            low, high = (float(v) for v in self.model.actuator_ctrlrange[member.actuator_id])
            return min(max(ctrl, low), high)
        return ctrl

    def _pd_gains(self) -> tuple[tuple[int, DerivedChannel, float, float, float], ...]:
        """Critically damped gains from the joint-space inertia at the current pose (CHN-9).

        Returns:
            One ``(channel index, member, kp, kd, actuator gain)`` row per PD-driven channel.

        Raises:
            ValueError: If a profile gain is negative or not finite.
        """
        rows = [
            (index, channel.first)
            for index, channel in enumerate(self.channels)
            if channel.first.kind == "torque" and channel.first.dof_adr is not None
        ]
        if not rows:
            return ()
        import mujoco  # noqa: PLC0415

        inertia = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, self.data, inertia)
        override = self.profile.pd
        bandwidth = override.bandwidth_hz if override is not None else 20.0
        omega = min(2.0 * np.pi * bandwidth, 0.3 / float(self.model.opt.timestep))
        gains = []
        for index, member in rows:
            mass = float(inertia[member.dof_adr, member.dof_adr])
            kp = omega * omega * mass
            kd = 2.0 * sqrt(kp * mass)
            name = self.channels[index].name
            if override is not None and override.kp is not None:
                kp = float(override.kp.get(name, kp))
            if override is not None and override.kd is not None:
                kd = float(override.kd.get(name, kd))
            if not (isfinite(kp) and isfinite(kd) and kp >= 0 and kd >= 0):
                msg = f"Profile {self.profile.name!r} has invalid PD gains for {name!r}"
                raise ValueError(msg)
            gains.append((index, member, kp, kd, float(self.model.actuator_gainprm[member.actuator_id, 0])))
        return tuple(gains)


def _affine(channel: Channel, value: float) -> float:
    if channel.identity:
        return value  # keeps -0.0 and every bit of the converted value
    return channel.offset + channel.scale * value


def _per_gear(member: DerivedChannel, value: float) -> float:
    return value / member.gear if member.gear else value


def _bind(
    layout: DerivedLayout,
    profile: RobotProfile,
    prefix: str,
    default_unit: DefaultUnit,
    torque_mode: TorqueMode,
) -> tuple[Channel, ...]:
    """Resolve the profile's channel overrides (or one channel per actuator) against *layout*.

    An override's actuator name is looked up among the actuators, then among the joints of joint
    actuators, so a custom model whose actuators are named differently still binds by joint name.

    Returns:
        The bound channels in public order.

    Raises:
        ValueError: If an actuator is missing or ambiguous, a group mixes kinds, or a normalized
            channel has no finite range.
    """
    by_actuator = {channel.actuator: channel for channel in layout.channels}
    by_joint: dict[str, list[DerivedChannel]] = {}
    for channel in layout.channels:
        if channel.joint is not None:
            by_joint.setdefault(channel.joint, []).append(channel)

    def resolve(name: str, override: ChannelOverride) -> DerivedChannel:
        member = by_actuator.get(f"{prefix}{name}")
        if member is None and len(by_joint.get(f"{prefix}{name}", ())) == 1:
            member = by_joint[f"{prefix}{name}"][0]
        if member is None:
            msg = f"Profile {profile.name!r} channel {override.name!r}: no actuator or joint {prefix}{name!r}"
            raise ValueError(msg)
        return member

    if profile.channels:
        overrides = profile.channels
        named = [(f"{prefix}{o.name}", o, tuple(resolve(a, o) for a in o.actuators)) for o in overrides]
    else:
        named = [(c.name, ChannelOverride(name=c.name, actuators=(c.actuator,)), (c,)) for c in layout.channels]

    channels = []
    for name, override, members in named:
        if len(members) > 1 and {member.kind for member in members} != {"position"}:
            msg = f"Profile {profile.name!r} group {override.name!r} must contain only position actuators"
            raise ValueError(msg)
        first = members[0]
        if override.unit is not None:
            unit = override.unit
        elif first.kind == "raw" or first.unit == "raw" or (first.kind == "torque" and torque_mode == "raw"):
            unit = "raw"
        elif default_unit == "normalized":
            unit = "normalized"
        else:
            unit = "degrees" if first.unit == "degrees" else "metres"
        limits = override.range or first.range
        if unit == "normalized" and limits is None:
            msg = f"Channel {name!r} has no finite joint range, so it cannot use normalized units"
            raise ValueError(msg)
        channels.append(
            Channel(
                name=name,
                members=members,
                unit=unit,
                scale=override.scale,
                offset=override.offset,
                member_scales=override.member_scales,
                range=limits,
                public_range=(0.0, 100.0) if override.gripper or _is_gripper(name) else (-100.0, 100.0),
                velocity_unit=override.velocity_unit,
            ),
        )
    names = [channel.name for channel in channels]
    if len(set(names)) != len(names):
        msg = f"Profile {profile.name!r} has duplicate channel names: {names}"
        raise ValueError(msg)
    return tuple(channels)


def _is_gripper(name: str) -> bool:
    return name == "gripper" or name.endswith("_gripper")


__all__ = ["ArmChannels", "Channel", "TorqueMode"]
