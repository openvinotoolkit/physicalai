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

NORMALIZED_ROUNDING = 1e-9
"""Fraction of a ``normalized`` range a replayed value may round past its ends, on top of two float32
steps of the public value (then clamped back)."""
REPLAY_MODEL_LIMIT = 1e6
"""Largest joint position (radians or metres) a replay places; recordings never come near it, and it
keeps every member joint finite under any profile scale."""
VELOCITY_TRACKING_GAIN = 10.0
"""Most velocity a velocity actuator is commanded per unit of position error, in 1/s (a ~0.1 s time constant).

A weak actuator reaches a commanded velocity with a time constant ``M_eff / kv`` (the inertia felt
along the actuator at the home pose); its gain is capped at ``kv / (4·M_eff)``, so the position loop
stays critically damped instead of overshooting.
"""


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
    """Bind a profile's public channels to one attached robot of a compiled model.

    Actions are positions in the channel's unit, like the observations, for position, velocity and
    (with ``torque_mode="pd"``) torque channels. Velocity actuators are driven toward their position
    target: every physics step commands ``VELOCITY_TRACKING_GAIN`` times the position error as the
    velocity, so echoing ``joint_positions`` back holds the joint. Raw channels take ``ctrl``.
    """

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
            ValueError: If an override names a missing actuator, a group mixes control kinds, a
                ``normalized`` channel has no finite range, or ``torque_mode="pd"`` meets a torque
                actuator without a hinge or slide joint.
        """
        if torque_mode not in {"pd", "raw"}:
            msg = f"Unsupported torque_mode {torque_mode!r}; expected 'pd' or 'raw'"
            raise ValueError(msg)
        self.model = model
        self.data = data
        self.profile = profile
        self.torque_mode: TorqueMode = torque_mode
        self._prefix = prefix
        self.channels = _bind(layout, profile, prefix, unit or profile.default_unit, torque_mode)
        self.names = tuple(channel.name for channel in self.channels)
        self.unpositionable = tuple(
            channel.name for channel in self.channels if any(member.qpos_adr is None for member in channel.members)
        )
        """Channels with an actuator that drives no hinge or slide joint directly (tendon, site), which
        :meth:`set_positions` cannot place: their position has no unique joint configuration."""
        # Position targets of the channels driven here (PD torque, velocity), in model units.
        self._tracked_targets = np.array([self._read_model(channel) for channel in self.channels], dtype=np.float64)
        self._pd_rows = self._pd_gains() if torque_mode == "pd" else ()
        self._velocity_gains_by_index = dict(self._velocity_gains())

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
        """Return the current positions in public units, in channel order (``float64``).

        Every channel reports a position, also velocity, torque and raw channels; a raw channel's
        position is in model units (radians or metres), while its action stays ``ctrl``.
        """
        return np.array([self.to_public(channel, self._read_model(channel)) for channel in self.channels])

    def read_velocities(self) -> np.ndarray:
        """Return the current velocities in the positions' units per second (CHN-8), as ``float64``."""
        return np.array([self._velocity(channel) for channel in self.channels])

    def model_targets(self) -> np.ndarray:
        """Return each channel's current target in model units (radians, metres, ``ctrl`` for raw).

        Position targets are read back from ``ctrl``, so targets written by scene automation (the
        conveyor autopilot) are included. Velocity and PD torque channels return their position target.
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
        """Teleport each channel's joints to ``qpos[joint]`` at rest, and hold them there.

        Every member of a group is placed (both fingers of a gripper); the group's target comes
        from its first member. Positions are clipped to the joint ranges. Joints not in *qpos*
        keep their position, and channels whose first joint is not in it keep their target.
        """
        data = self.data
        for index, channel in enumerate(self.channels):
            for member in channel.members:
                if member.joint is None or member.joint_id is None or member.joint not in qpos:
                    continue
                data.qpos[member.qpos_adr] = self._clip_to_joint(member, float(qpos[member.joint]))
                data.qvel[member.dof_adr] = 0.0
            first = channel.first
            if first.joint is not None and first.joint in qpos:
                self._write_model(index, channel, float(data.qpos[first.qpos_adr]), clamp=True)

    def hold_current(self) -> None:
        """Make the channels driven here (PD torque, velocity) hold their current positions, after a direct pose set."""
        self._tracked_targets = np.array([self._read_model(channel) for channel in self.channels], dtype=np.float64)

    def replay_placements(self, positions: np.ndarray) -> list[tuple[DerivedChannel, np.ndarray]]:
        """Convert public replay frames to the joint position of every channel member (replay).

        A frame is accepted only if the robot can show it unchanged: a ``normalized`` value inside
        its public range (outside it would be clamped), and every member joint finite and within
        :data:`REPLAY_MODEL_LIMIT` model units. Nothing is clipped otherwise, so recorded positions
        that pressed past a joint limit replay as recorded. :meth:`set_positions` writes exactly
        these values, so a frame accepted here is placed without error.

        Args:
            positions: Public positions, shape ``(frames, len(self))``.

        Returns:
            One ``(member, qpos per frame)`` pair per member of every channel, in channel order.

        Raises:
            ValueError: If a channel is in :attr:`unpositionable`, or naming the first frame and
                channel that cannot be shown unchanged.
        """
        if self.unpositionable:
            msg = f"Channels {list(self.unpositionable)} drive no hinge or slide joint directly; they cannot be placed"
            raise ValueError(msg)
        frames = np.asarray(positions, dtype=np.float64)
        if frames.ndim != 2 or frames.shape[1] != len(self.channels):  # noqa: PLR2004
            msg = f"Expected replay positions of shape (frames, {len(self.channels)}), got {frames.shape}"
            raise ValueError(msg)
        entries: list[tuple[int, int, DerivedChannel, np.ndarray]] = []
        for index, channel in enumerate(self.channels):
            model = _replay_model(frames, index, channel)
            for position, member in enumerate(channel.members):
                scale = 1.0 if position == 0 or not channel.member_scales else channel.member_scales[position - 1]
                qpos = model * scale
                beyond = ~(np.abs(qpos) <= REPLAY_MODEL_LIMIT)  # also catches NaN
                _refuse(
                    frames, index, beyond, channel, f"puts {member.joint} beyond {REPLAY_MODEL_LIMIT:g} model units"
                )
                entries.append((index, position, member, qpos))
        return self._reconcile_shared_joints(frames, entries)

    def _reconcile_shared_joints(
        self, frames: np.ndarray, entries: list[tuple[int, int, DerivedChannel, np.ndarray]]
    ) -> list[tuple[DerivedChannel, np.ndarray]]:
        """Pick one position per joint that every channel reading it reproduces, frame by frame.

        Channels sharing a joint (two actuators on one hinge) each report it in their own unit, so a
        recorded frame holds the same position rounded differently. A candidate position is accepted
        when each channel that reads the joint (its first member) shows it as its requested value,
        within two float32 steps; group members after the first are written but never read.

        Returns:
            One ``(member, qpos per frame)`` pair per member, every member of a joint at the same value.

        Raises:
            ValueError: Naming the first frame and joint no position reproduces.
        """
        by_joint: dict[int, list[tuple[int, int, DerivedChannel, np.ndarray]]] = {}
        for entry in entries:
            by_joint.setdefault(entry[2].qpos_adr, []).append(entry)  # type: ignore[arg-type]
        chosen: dict[int, np.ndarray] = {}
        for address, shared in by_joint.items():
            if len(shared) == 1:
                continue
            readers = [(index, member) for index, position, member, _ in shared if position == 0]
            pick, found = shared[0][3].copy(), np.zeros(len(frames), dtype=bool)
            for *_, candidate in shared:
                shows = np.ones(len(frames), dtype=bool)
                for index, _member in readers:
                    shows &= _shows(self.channels[index], candidate, frames[:, index])
                pick = np.where(shows & ~found, candidate, pick)
                found |= shows
            rows = np.flatnonzero(~found)
            if len(rows):
                names = " and ".join(self.channels[index].name for index, *_ in shared)
                msg = f"frame {int(rows[0])}: {names} place joint {shared[0][2].joint} at different positions"
                raise ValueError(msg)
            chosen[address] = pick
        return [(member, chosen.get(member.qpos_adr, qpos)) for _, _, member, qpos in entries]  # type: ignore[arg-type]

    def set_positions(self, positions: np.ndarray) -> None:
        """Teleport the channels' joints to one frame of public positions at rest, keeping their targets.

        A frame :meth:`replay_placements` refuses raises its ``ValueError``, and nothing is moved.
        """
        placements = self.replay_placements(np.asarray(positions, dtype=np.float64)[None, :])
        data = self.data
        for member, qpos in placements:
            data.qpos[member.qpos_adr] = qpos[0]
            data.qvel[member.dof_adr] = 0.0

    def apply_pd(self) -> None:
        """Update the ``ctrl`` of the channels driven here; call before every physics step.

        PD torque channels get a torque toward their target (CHN-9), velocity channels a velocity.
        """
        data = self.data
        for index, member, kp, kd, gain in self._pd_rows:
            torque = kp * (self._tracked_targets[index] - data.qpos[member.qpos_adr]) - kd * data.qvel[member.dof_adr]
            data.ctrl[member.actuator_id] = self._clip_ctrl(member, torque / gain)
        for index in self._velocity_gains_by_index:
            channel = self.channels[index]
            data.ctrl[channel.first.actuator_id] = self._velocity_ctrl(index, channel)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_model(self, channel: Channel) -> float:
        """Return *channel*'s position in model units, whatever its control kind.

        Observations carry positions (the ``RobotObservation`` contract): velocity, torque and raw
        channels report their joint (or actuator length), not the velocity or ``ctrl`` they take.
        """
        member, data = channel.first, self.data
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
        if member.kind == "position":
            factor = member.ctrl_per_unit * member.gear
            return float(self.data.ctrl[member.actuator_id]) / factor
        if member.kind == "velocity" or (member.kind == "torque" and self.torque_mode == "pd"):
            return float(self._tracked_targets[index])
        return float(self.data.ctrl[member.actuator_id])

    def _write_model(self, index: int, channel: Channel, value: float, *, clamp: bool = False) -> None:
        """Set *channel*'s target to a model value; group members get their scaled share (PRF-8)."""
        data = self.data
        for position, member in enumerate(channel.members):
            target = (
                value if position == 0 or not channel.member_scales else value * channel.member_scales[position - 1]
            )
            if member.kind in {"position", "velocity"} or (member.kind == "torque" and self.torque_mode == "pd"):
                # Position targets past a joint limit only push the joint into its stop.
                target = self._clip_to_joint(member, target)
            if member.kind == "position":
                # MuJoCo clamps ctrl to ctrlrange itself; storing the unclamped target keeps it
                # readable at full precision (model_targets, the virtual leader). Homing clamps,
                # as it always has.
                ctrl = member.ctrl_per_unit * member.gear * target
                data.ctrl[member.actuator_id] = self._clip_ctrl(member, ctrl) if clamp else ctrl
            elif member.kind == "velocity":
                # Velocity channels are never grouped (_bind), so this is the channel's only member.
                self._tracked_targets[index] = target
                data.ctrl[member.actuator_id] = self._velocity_ctrl(index, channel)
            elif member.kind == "torque" and self.torque_mode == "pd":
                if position == 0:
                    self._tracked_targets[index] = target
            else:
                data.ctrl[member.actuator_id] = self._clip_ctrl(member, target)

    def _velocity_ctrl(self, index: int, channel: Channel) -> float:
        """Return the ``ctrl`` that moves a velocity channel toward its position target, within ``ctrlrange``."""
        member = channel.first
        velocity = self._velocity_gains_by_index[index] * (self._tracked_targets[index] - self._read_model(channel))
        return self._clip_ctrl(member, member.ctrl_per_unit * member.gear * velocity)

    def _clip_to_joint(self, member: DerivedChannel, value: float) -> float:
        joint_id = member.joint_id
        if joint_id is None or not bool(self.model.jnt_limited[joint_id]):
            return value
        low, high = (float(v) for v in self.model.jnt_range[joint_id])
        return min(max(value, low), high)

    def _clip_ctrl(self, member: DerivedChannel, ctrl: float) -> float:
        if bool(self.model.actuator_ctrllimited[member.actuator_id]):
            low, high = (float(v) for v in self.model.actuator_ctrlrange[member.actuator_id])
            return min(max(ctrl, low), high)
        return ctrl

    def _velocity_gains(self) -> tuple[tuple[int, float], ...]:
        """Position-tracking gain of each velocity channel, capped by its actuator's response.

        The servo reaches a commanded velocity with time constant ``M_eff / kv``, where ``M_eff`` is
        the inertia felt along the actuator, ``1 / (m · M⁻¹ · mᵀ)`` for its moment row ``m`` (joint,
        tendon or site transmission, gear included). Capping the gain at ``kv / (4 · M_eff)`` keeps
        the position loop critically damped.

        Returns:
            One ``(channel index, gain in 1/s)`` row per velocity channel.
        """
        rows = [index for index, channel in enumerate(self.channels) if channel.first.kind == "velocity"]
        if not rows:
            return ()
        import mujoco  # noqa: PLC0415

        model, data = self.model, self.data
        moments = np.zeros((model.nu, model.nv))
        mujoco.mju_sparse2dense(
            moments, data.actuator_moment, data.moment_rownnz, data.moment_rowadr, data.moment_colind
        )
        gains = []
        for index in rows:
            actuator_id = self.channels[index].first.actuator_id
            gain = VELOCITY_TRACKING_GAIN
            kv = -float(model.actuator_biasprm[actuator_id, 2])
            moment = moments[actuator_id]
            if kv > 0.0 and moment.any():
                solved = np.zeros(model.nv)
                mujoco.mj_solveM(model, data, solved.reshape(1, -1), moment.reshape(1, -1))
                inverse_inertia = float(moment @ solved)  # 1 / M_eff
                if inverse_inertia > 0.0:
                    gain = min(gain, kv * inverse_inertia / 4.0)
            gains.append((index, gain))
        return tuple(gains)

    def _pd_gains(self) -> tuple[tuple[int, DerivedChannel, float, float, float], ...]:
        """Critically damped gains from the joint-space inertia at the current pose (CHN-9).

        Returns:
            One ``(channel index, member, kp, kd, ctrl per joint torque)`` row per PD-driven channel.

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
            # Profile overrides use unprefixed names; the public (prefixed) name also matches.
            name = self.channels[index].name
            short = name.removeprefix(self._prefix)
            if override is not None and override.kp is not None:
                kp = float(override.kp.get(short, override.kp.get(name, kp)))
            if override is not None and override.kd is not None:
                kd = float(override.kd.get(short, override.kd.get(name, kd)))
            if not (isfinite(kp) and isfinite(kd) and kp >= 0 and kd >= 0):
                msg = f"Profile {self.profile.name!r} has invalid PD gains for {name!r}"
                raise ValueError(msg)
            # The joint receives gear * gain * ctrl, so a joint torque needs ctrl = torque / (gear * gain).
            gain = float(self.model.actuator_gainprm[member.actuator_id, 0]) * member.gear
            if gain == 0.0:  # noqa: RUF069 - exactly zero means the actuator cannot move the joint
                continue
            gains.append((index, member, kp, kd, gain))
        return tuple(gains)


def _replay_model(frames: np.ndarray, index: int, channel: Channel) -> np.ndarray:
    """Convert column *index* of public replay frames to the channel's model positions.

    A ``normalized`` value outside its range by more than rounding raises ``ValueError``.

    Returns:
        The model position per frame.
    """
    public = frames[:, index]
    value = public if channel.identity else (public - channel.offset) / channel.scale
    if channel.unit == "normalized":
        lower, upper = channel.range  # type: ignore[misc]
        public_low, public_high = channel.public_range
        fraction = (value - public_low) / (public_high - public_low)
        # A recorded end of the range may lie a hair outside it: observations are float32,
        # and scale and offset round too. Allow two float32 steps at the public ends.
        ends = [_affine(channel, end) for end in channel.public_range]
        step = 2.0 * float(np.spacing(np.float32(max(abs(end) for end in ends))))
        slack = NORMALIZED_ROUNDING + step / abs(ends[1] - ends[0])
        inside = (fraction >= -slack) & (fraction <= 1.0 + slack)
        _refuse(frames, index, ~inside, channel, "is outside its normalized range")
        model = lower + np.clip(fraction, 0.0, 1.0) * (upper - lower)
    elif channel.unit == "degrees" and channel.hinge:
        model = np.radians(value)
    else:
        model = value
    return model


def _shows(channel: Channel, qpos: np.ndarray, requested: np.ndarray) -> np.ndarray:
    """Return, per frame, whether *channel* reads *qpos* (its first member's) as *requested* in float32.

    Returns:
        A boolean per frame: within two float32 steps of the requested value.
    """
    value = qpos
    if channel.unit == "normalized":
        lower, upper = channel.range  # type: ignore[misc]
        public_low, public_high = channel.public_range
        value = public_low + np.clip((qpos - lower) / (upper - lower), 0.0, 1.0) * (public_high - public_low)
    elif channel.unit == "degrees" and channel.hinge:
        value = np.degrees(qpos)
    if not channel.identity:
        value = channel.offset + channel.scale * value
    shown, wanted = value.astype(np.float32), requested.astype(np.float32)
    step = np.spacing(np.maximum(np.abs(shown), np.abs(wanted)))
    return np.abs(shown.astype(np.float64) - wanted.astype(np.float64)) <= 2.0 * step


def _refuse(frames: np.ndarray, index: int, bad: np.ndarray, channel: Channel, reason: str) -> None:
    """Raise for the first frame where *bad* is set, naming its value and *reason*.

    Raises:
        ValueError: If any entry of *bad* is set.
    """
    rows = np.flatnonzero(bad)
    if len(rows):
        row = int(rows[0])
        msg = f"frame {row}: {channel.name} = {frames[row, index]} {reason} ({len(rows)} frames)"
        raise ValueError(msg)


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
        ValueError: If an actuator is missing or ambiguous, a group mixes kinds, a normalized
            channel has no finite range, or a PD torque channel has no joint to drive.
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
        _check_pd_joint(name, first, torque_mode)
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


def _check_pd_joint(name: str, member: DerivedChannel, torque_mode: TorqueMode) -> None:
    """Reject a torque channel the software PD cannot drive: it needs a hinge or slide joint.

    Raises:
        ValueError: If *member* is a torque actuator without a joint and ``torque_mode`` is ``pd``.
    """
    if member.kind == "torque" and torque_mode == "pd" and member.dof_adr is None:
        msg = (
            f"Channel {name!r} is a torque actuator without a hinge or slide joint ({member.transmission} "
            "transmission), so the software PD cannot drive it; use torque_mode='raw' to send its ctrl"
        )
        raise ValueError(msg)


def _is_gripper(name: str) -> bool:
    return name == "gripper" or name.endswith("_gripper")


__all__ = ["ArmChannels", "Channel", "TorqueMode"]
