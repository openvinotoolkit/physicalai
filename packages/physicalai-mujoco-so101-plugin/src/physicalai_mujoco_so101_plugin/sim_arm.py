# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""One simulated arm: a robot profile attached at a mount prefix, read and driven through the compiled model.

A `SimArm` never steps the simulation. The robot that owns the model binds each arm to the model
after every compile (connect and scene switch), then reads joint state from and writes actuator
targets into the shared `MjData`, converting between radians and the arm's joint unit.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from physicalai_mujoco_so101_plugin.robot_profile import RobotProfile

JointUnit = Literal["normalized", "degrees"]
"""Units of ``get_observation`` joint positions and ``send_action`` targets."""


def validate_joint_unit(unit: str) -> JointUnit:
    """Return `unit` if the simulation supports it.

    Raises:
        ValueError: If ``unit`` is not supported.
    """
    if unit not in get_args(JointUnit):
        msg = f"Unsupported unit {unit!r}; expected one of {get_args(JointUnit)}"
        raise ValueError(msg)
    return unit  # type: ignore[return-value]


def _normalized_span(grippers: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
    """Lower bound and width of each joint's normalized range, as the SO101 driver uses them.

    Returns:
        ``(lower, width)``: body joints span ``[-100, 100]``, grippers ``[0, 100]``.
    """
    gripper = np.asarray(grippers, dtype=bool)
    return np.where(gripper, 0.0, -100.0), np.where(gripper, 100.0, 200.0)


def radians_to_normalized(radians: np.ndarray, joint_limits: np.ndarray, grippers: Sequence[bool]) -> np.ndarray:
    """Map joint angles to the SO101 driver's calibrated normalized units.

    The real driver maps each joint's calibrated tick range linearly onto
    ``[-100, 100]`` (grippers onto ``[0, 100]``) and clamps. The simulation
    uses the model's joint range as the calibrated range.

    Returns:
        Normalized positions, clamped to each joint's normalized range.
    """
    low, high = joint_limits[:, 0], joint_limits[:, 1]
    lower, width = _normalized_span(grippers)
    fraction = np.clip((radians - low) / (high - low), 0.0, 1.0)
    return lower + fraction * width


def normalized_to_radians(normalized: np.ndarray, joint_limits: np.ndarray, grippers: Sequence[bool]) -> np.ndarray:
    """Map SO101 normalized units back to joint angles within the model's joint range.

    Returns:
        Joint angles in radians, clamped to each joint's range.
    """
    low, high = joint_limits[:, 0], joint_limits[:, 1]
    lower, width = _normalized_span(grippers)
    fraction = np.clip((normalized - lower) / width, 0.0, 1.0)
    return low + fraction * (high - low)


@dataclass(frozen=True)
class ArmBinding:
    """Addresses of one arm's joints and actuators in a compiled model."""

    joint_ids: tuple[int, ...]
    qpos_adr: tuple[int, ...]
    dof_adr: tuple[int, ...]
    ctrl_indices: tuple[int, ...]
    limits: np.ndarray
    """``(num_joints, 2)`` joint ranges in radians, in joint order."""


class SimArm:
    """One arm of a robot profile, attached under a name prefix, viewed through the compiled model."""

    def __init__(self, profile: RobotProfile, prefix: str = "", unit: JointUnit = "normalized") -> None:
        """Describe an unbound arm; call `bind` with a compiled model before reading or writing it."""
        self.profile = profile
        self.prefix = prefix
        self.unit = validate_joint_unit(unit)
        self.joint_names: tuple[str, ...] = profile.joint_names(prefix)
        self.grippers: tuple[bool, ...] = tuple(name == profile.gripper for name in profile.joint_order)
        self._binding: ArmBinding | None = None

    @property
    def num_joints(self) -> int:
        """Number of public joints, which is also the arm's action size."""
        return len(self.joint_names)

    @property
    def wrist_camera(self) -> str:
        """Name of this arm's wrist camera in the composed model."""
        return f"{self.prefix}{self.profile.wrist_camera}"

    @property
    def is_bound(self) -> bool:
        """Whether the arm is bound to a model."""
        return self._binding is not None

    @property
    def ctrl_indices(self) -> tuple[int, ...]:
        """Actuator index of each public joint, in joint order; empty while unbound."""
        return self._binding.ctrl_indices if self._binding is not None else ()

    @property
    def joint_limits(self) -> np.ndarray | None:
        """``(num_joints, 2)`` joint ranges in radians, or ``None`` while unbound."""
        return self._binding.limits if self._binding is not None else None

    def resolve(self, model: object) -> ArmBinding | None:
        """Look up this arm's joints and actuators in `model` without binding to it.

        Every public joint needs a range (normalized units span it) and
        exactly one direct joint actuator.

        Returns:
            The binding, or ``None`` if `model` cannot drive this arm.
        """
        import mujoco  # noqa: PLC0415

        joint_ids: list[int] = []
        ctrl_indices: list[int] = []
        limits = np.empty((self.num_joints, 2), dtype=np.float64)
        for i, name in enumerate(self.joint_names):
            joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
            if joint_id < 0 or joint_id in joint_ids:
                return None
            joint_ids.append(joint_id)

            matches = [
                actuator_id
                for actuator_id in range(int(model.nu))
                if int(model.actuator_trntype[actuator_id]) == mujoco.mjtTrn.mjTRN_JOINT
                and int(model.actuator_trnid[actuator_id, 0]) == joint_id
            ]
            if len(matches) != 1:
                return None
            ctrl_indices.append(matches[0])

            low, high = (float(value) for value in model.jnt_range[joint_id])
            if not high > low:
                logger.error("Joint {!r} has no range; normalized units need one", name)
                return None
            limits[i] = (low, high)

        if len(set(ctrl_indices)) != self.num_joints:
            return None
        return ArmBinding(
            joint_ids=tuple(joint_ids),
            qpos_adr=tuple(int(model.jnt_qposadr[j]) for j in joint_ids),
            dof_adr=tuple(int(model.jnt_dofadr[j]) for j in joint_ids),
            ctrl_indices=tuple(ctrl_indices),
            limits=limits,
        )

    def bind(self, binding: ArmBinding | None) -> None:
        """Adopt a binding from `resolve`, or ``None`` to unbind."""
        self._binding = binding

    def _require_binding(self) -> ArmBinding:
        if self._binding is None:
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)
        return self._binding

    def to_units(self, radians: np.ndarray) -> np.ndarray:
        """Convert joint angles to this arm's unit.

        Returns:
            Normalized positions (clamped to each joint's range), or degrees.
        """
        if self.unit == "normalized":
            return radians_to_normalized(radians, self._require_binding().limits, self.grippers)
        return np.degrees(radians)

    def from_units(self, values: np.ndarray) -> np.ndarray:
        """Convert targets in this arm's unit to joint angles.

        Returns:
            Joint angles in radians; normalized targets are clamped to each joint's range.
        """
        if self.unit == "normalized":
            return normalized_to_radians(values, self._require_binding().limits, self.grippers)
        return np.radians(values)

    def read(self, data: object) -> tuple[np.ndarray, np.ndarray]:
        """Read joint positions and velocities in this arm's unit.

        Returns:
            ``(positions, velocities)``, one entry per public joint.
        """
        binding = self._require_binding()
        radians = np.asarray(data.qpos[list(binding.qpos_adr)], dtype=np.float64)
        velocities = np.asarray(data.qvel[list(binding.dof_adr)], dtype=np.float64)
        if self.unit == "normalized":
            _, width = _normalized_span(self.grippers)
            velocities = velocities * width / (binding.limits[:, 1] - binding.limits[:, 0])
        else:
            velocities = np.degrees(velocities)
        return self.to_units(radians), velocities

    def write(self, data: object, action: np.ndarray) -> None:
        """Set the actuator targets from `action`, given in this arm's unit."""
        targets = self.from_units(np.asarray(action, dtype=np.float64))
        for index, target in zip(self._require_binding().ctrl_indices, targets, strict=True):
            data.ctrl[index] = float(target)

    def targets(self, data: object) -> np.ndarray:
        """Current actuator targets in radians, in joint order.

        Returns:
            One target per public joint.
        """
        return np.asarray(data.ctrl[list(self._require_binding().ctrl_indices)], dtype=np.float64)

    def go_home(self, model: object, data: object, home: Mapping[str, float]) -> None:
        """Place the joints at `home` (radians, by prefixed joint name) at rest, and target that pose.

        Joints without a home value use the model default (``qpos0``).
        Positions and targets are clipped to the joint and control ranges.
        """
        binding = self._require_binding()
        for name, joint_id, qpos_addr, dof_addr, actuator_id in zip(
            self.joint_names, binding.joint_ids, binding.qpos_adr, binding.dof_adr, binding.ctrl_indices, strict=True
        ):
            value = float(home.get(name, model.qpos0[qpos_addr]))
            if bool(model.jnt_limited[joint_id]):
                low, high = (float(v) for v in model.jnt_range[joint_id])
                value = min(max(value, low), high)
            data.qpos[qpos_addr] = value
            data.qvel[dof_addr] = 0.0

            target = value
            if bool(model.actuator_ctrllimited[actuator_id]):
                low, high = (float(v) for v in model.actuator_ctrlrange[actuator_id])
                target = min(max(target, low), high)
            data.ctrl[actuator_id] = target
