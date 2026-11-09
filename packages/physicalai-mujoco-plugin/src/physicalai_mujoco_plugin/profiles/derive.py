# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Derive a robot's channels, home pose, sensors and cameras from its compiled robot-only model.

Derivation never looks at a scene, so ``joint_names`` are known before ``connect()``.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Literal

from physicalai_mujoco_plugin.profiles._types import CameraSpec

if TYPE_CHECKING:
    import mujoco
    import numpy as np

ControlKind = Literal["position", "velocity", "torque", "raw"]
TransmissionKind = Literal["joint", "tendon", "other"]
ModelUnit = Literal["degrees", "metres", "raw"]
"""``degrees`` marks hinge joints: radians in the model, degrees in public ``degrees`` units."""


@dataclass(frozen=True)
class DerivedChannel:
    """One actuator, as a candidate public channel. Ids refer to the model it was derived from."""

    name: str
    """Public name: the actuator's, else its joint's, else ``actuator_<i>``."""
    actuator: str
    """Actuator name in the model (``name`` before any prefix)."""
    actuator_id: int
    joint: str | None
    """Hinge or slide joint of a joint transmission."""
    joint_id: int | None
    qpos_adr: int | None
    dof_adr: int | None
    kind: ControlKind
    transmission: TransmissionKind
    unit: ModelUnit
    gear: float
    ctrl_per_unit: float
    """``ctrl`` per model-unit target: ``kp / gain`` (position) or ``kv / gain`` (velocity)."""
    range: tuple[float, float] | None
    """Joint range in model units, when the joint is limited."""


@dataclass(frozen=True)
class SensorLayout:
    """A named slice of ``MjData.sensordata``."""

    name: str
    address: int
    dimension: int


@dataclass(frozen=True)
class FloatingBase:
    """The root body of a floating-base robot and its free joint. Ids refer to the model it was derived from."""

    body: str
    body_id: int
    joint: str
    """Free joint name; empty when the model leaves it unnamed (Go2)."""
    qpos_adr: int
    dof_adr: int
    home: tuple[float, ...]
    """Home pose: world position (3) and orientation quaternion (4, wxyz)."""


@dataclass(frozen=True)
class DerivedLayout:
    """Everything derivable from a compiled robot-only model."""

    channels: tuple[DerivedChannel, ...]
    base: FloatingBase | None
    """The floating base, or ``None`` for a fixed base."""
    home_qpos: dict[str, tuple[float, ...]]
    """Home ``qpos`` of every joint but the floating base's, by joint name: the ``home`` keyframe,
    else the first, else ``qpos0``."""
    home_ctrl: dict[str, float]
    """Home ``ctrl`` by actuator name: the keyframe's, else each channel's home target."""
    sensors: tuple[SensorLayout, ...]
    cameras: tuple[CameraSpec, ...]
    """Every model camera from :func:`derive_profile`; the robot's resolved cameras (CAM-2) from ``robot_layout``."""

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Derived channel names in actuator order."""
        return tuple(channel.name for channel in self.channels)


def control_kind(model: mujoco.MjModel, actuator_id: int) -> tuple[ControlKind, float]:
    """Classify an actuator by its gain and bias (CHN-2).

    Returns:
        The control kind and ``ctrl`` per model-unit target (1 for torque and raw).
    """
    import mujoco  # noqa: PLC0415

    filters = {mujoco.mjtDyn.mjDYN_NONE, mujoco.mjtDyn.mjDYN_FILTER, mujoco.mjtDyn.mjDYN_FILTEREXACT}
    if (
        int(model.actuator_gaintype[actuator_id]) != mujoco.mjtGain.mjGAIN_FIXED
        or int(model.actuator_dyntype[actuator_id]) not in filters
    ):
        return "raw", 1.0
    gain = float(model.actuator_gainprm[actuator_id, 0])
    bias_type = int(model.actuator_biastype[actuator_id])
    if bias_type == mujoco.mjtBias.mjBIAS_NONE:
        return ("torque" if isfinite(gain) and abs(gain) > 0.0 else "raw"), 1.0
    if bias_type == mujoco.mjtBias.mjBIAS_AFFINE and gain > 0.0:
        kp = -float(model.actuator_biasprm[actuator_id, 1])
        kv = -float(model.actuator_biasprm[actuator_id, 2])
        if kp > 0.0:
            return "position", kp / gain
        if kv > 0.0:
            return "velocity", kv / gain
    return "raw", 1.0


def derive_profile(model: mujoco.MjModel) -> DerivedLayout:
    """Derive the layout of a compiled robot-only model (PRF-3).

    Returns:
        Channels in actuator order, the floating base if any, the home pose, sensors and cameras.
    """
    channels = tuple(_derive_channel(model, actuator_id) for actuator_id in range(model.nu))
    base_joint = _find_floating_base(model)
    home_qpos, home_ctrl, qpos = _derive_home(model, channels, base_joint)
    base = None
    if base_joint is not None:
        body_id, address = int(model.jnt_bodyid[base_joint]), int(model.jnt_qposadr[base_joint])
        base = FloatingBase(
            body=model.body(body_id).name,
            body_id=body_id,
            joint=model.joint(base_joint).name,
            qpos_adr=address,
            dof_adr=int(model.jnt_dofadr[base_joint]),
            home=tuple(float(v) for v in qpos[address : address + 7]),
        )
    sensors = tuple(
        SensorLayout(model.sensor(i).name or f"sensor_{i}", int(model.sensor_adr[i]), int(model.sensor_dim[i]))
        for i in range(model.nsensor)
    )
    cameras = tuple(
        CameraSpec(
            name=model.camera(i).name or f"camera_{i}",
            body=model.body(int(model.cam_bodyid[i])).name,
            pos=tuple(float(v) for v in model.cam_pos[i]),  # type: ignore[arg-type]
            quat=tuple(float(v) for v in model.cam_quat[i]),  # type: ignore[arg-type]
            fovy=float(model.cam_fovy[i]),
            source="model",
        )
        for i in range(model.ncam)
    )
    return DerivedLayout(channels, base, home_qpos, home_ctrl, sensors, cameras)


def _derive_channel(model: mujoco.MjModel, actuator_id: int) -> DerivedChannel:  # noqa: PLR0914
    import mujoco  # noqa: PLC0415

    transmission_type = int(model.actuator_trntype[actuator_id])
    target_id = int(model.actuator_trnid[actuator_id, 0])
    joint: str | None = None
    joint_id = qpos_adr = dof_adr = None
    limits: tuple[float, float] | None = None
    unit: ModelUnit = "raw"
    transmission: TransmissionKind = "other"
    if transmission_type in {mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT}:
        transmission = "joint"
        joint_type = int(model.jnt_type[target_id])
        if joint_type in {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}:
            joint, joint_id = model.joint(target_id).name, target_id
            qpos_adr, dof_adr = int(model.jnt_qposadr[target_id]), int(model.jnt_dofadr[target_id])
            unit = "degrees" if joint_type == mujoco.mjtJoint.mjJNT_HINGE else "metres"
            if bool(model.jnt_limited[target_id]):
                lower, upper = (float(v) for v in model.jnt_range[target_id])
                if isfinite(lower) and isfinite(upper) and lower < upper:
                    limits = (lower, upper)
    elif transmission_type == mujoco.mjtTrn.mjTRN_TENDON:
        transmission, unit = "tendon", "metres"

    kind, ctrl_per_unit = control_kind(model, actuator_id)
    actuator = model.actuator(actuator_id).name
    name = actuator or joint or f"actuator_{actuator_id}"
    return DerivedChannel(
        name=name,
        actuator=actuator or name,
        actuator_id=actuator_id,
        joint=joint,
        joint_id=joint_id,
        qpos_adr=qpos_adr,
        dof_adr=dof_adr,
        kind=kind,
        transmission=transmission,
        unit=unit,
        gear=float(model.actuator_gear[actuator_id, 0]),
        ctrl_per_unit=ctrl_per_unit,
        range=limits,
    )


def _find_floating_base(model: mujoco.MjModel) -> int | None:
    """Find the root free joint whose subtree holds a body that an actuator acts on (PoC finding 4).

    Joint, site and body transmissions count, so a drone (site thrust) has a floating base too.

    Returns:
        The free joint's id, or ``None`` for a fixed base.
    """
    import mujoco  # noqa: PLC0415

    actuated = {body for actuator_id in range(model.nu) if (body := _actuator_body(model, actuator_id)) >= 0}
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) != mujoco.mjtJoint.mjJNT_FREE:
            continue
        body_id = int(model.jnt_bodyid[joint_id])
        if int(model.body_parentid[body_id]) == 0 and any(_in_subtree(model, b, body_id) for b in actuated):
            return joint_id
    return None


def _actuator_body(model: mujoco.MjModel, actuator_id: int) -> int:
    import mujoco  # noqa: PLC0415

    transmission_type = int(model.actuator_trntype[actuator_id])
    target_id = int(model.actuator_trnid[actuator_id, 0])
    if target_id < 0:
        return -1
    if transmission_type in {mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT}:
        return int(model.jnt_bodyid[target_id])
    if transmission_type == mujoco.mjtTrn.mjTRN_SITE:
        return int(model.site_bodyid[target_id])
    if transmission_type == mujoco.mjtTrn.mjTRN_BODY:
        return target_id
    return -1


def _in_subtree(model: mujoco.MjModel, body_id: int, root_id: int) -> bool:
    while body_id > 0:
        if body_id == root_id:
            return True
        body_id = int(model.body_parentid[body_id])
    return False


def _derive_home(
    model: mujoco.MjModel,
    channels: tuple[DerivedChannel, ...],
    base_joint: int | None,
) -> tuple[dict[str, tuple[float, ...]], dict[str, float], np.ndarray]:
    """Read the home keyframe, or build a home pose from ``qpos0`` with matching position targets.

    Returns:
        Home ``qpos`` by joint name (without the floating base), home ``ctrl`` by actuator name,
        and the whole home ``qpos`` vector.
    """
    import mujoco  # noqa: PLC0415

    key_names = [model.key(i).name for i in range(model.nkey)]
    key = key_names.index("home") if "home" in key_names else (0 if key_names else None)
    qpos = model.key_qpos[key].copy() if key is not None else model.qpos0.copy()
    widths = {int(mujoco.mjtJoint.mjJNT_FREE): 7, int(mujoco.mjtJoint.mjJNT_BALL): 4}
    home_qpos = {}
    for joint_id in range(model.njnt):
        if joint_id == base_joint:
            continue
        address = int(model.jnt_qposadr[joint_id])
        width = widths.get(int(model.jnt_type[joint_id]), 1)
        home_qpos[model.joint(joint_id).name] = tuple(float(v) for v in qpos[address : address + width])

    if key is not None:
        ctrl = model.key_ctrl[key]
        return home_qpos, {channel.actuator: float(ctrl[channel.actuator_id]) for channel in channels}, qpos

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    home_ctrl = {}
    for channel in channels:
        value = 0.0
        if channel.kind == "position":
            if channel.qpos_adr is not None:
                target = float(qpos[channel.qpos_adr])
            else:
                length = float(data.actuator_length[channel.actuator_id])
                target = length / channel.gear if channel.gear else length
            value = channel.ctrl_per_unit * channel.gear * target
        home_ctrl[channel.actuator] = value
    return home_qpos, home_ctrl, qpos


__all__ = [
    "ControlKind",
    "DerivedChannel",
    "DerivedLayout",
    "FloatingBase",
    "ModelUnit",
    "SensorLayout",
    "TransmissionKind",
    "control_kind",
    "derive_profile",
]
