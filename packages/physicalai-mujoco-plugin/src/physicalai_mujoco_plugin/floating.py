# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Floating-base robots of one simulation: base observations, the base-hold welds and fall detection.

A spawned robot's base is welded to its start pose until the first action (SCN-8), so an idle
humanoid does not fall over while nothing drives it. Its observation adds the base pose and
velocities (OBS-2, OBS-3), and its status reports the base height, tilt and falls (DRV-9).
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from physicalai_mujoco_plugin.compose import RobotBinding
    from physicalai_mujoco_plugin.profiles.derive import FloatingBase

FALL_TILT_DEG = 60.0
"""Base tilt from upright beyond which a robot is falling (DRV-9)."""
FALL_DURATION_S = 1.5
"""Simulated time the tilt must stay beyond ``FALL_TILT_DEG`` before the robot counts as fallen."""


@dataclass(frozen=True)
class BaseStatus:
    """Readout of one floating base for ``/health`` and the viewer panel."""

    prefix: str
    height: float
    """Base height above the world origin, in metres."""
    tilt_deg: float
    """Angle between the base's z axis and the world's z axis."""
    held: bool
    """Whether the base-hold weld is active."""
    fallen: bool

    def as_dict(self) -> dict[str, object]:
        """Return the status as JSON-friendly values.

        Returns:
            The fields by name.
        """
        return {
            "prefix": self.prefix,
            "height": self.height,
            "tilt_deg": self.tilt_deg,
            "held": self.held,
            "fallen": self.fallen,
        }


def tilt_degrees(quat: np.ndarray) -> float:
    """Return the angle between a body's z axis and the world's z axis.

    Args:
        quat: The body orientation (``wxyz``, unit length).

    Returns:
        The tilt in degrees, 0 when upright and 180 upside down.
    """
    _, x, y, _ = (float(v) for v in quat)
    # The world z component of the body's z axis: R[2, 2] of the rotation matrix.
    return degrees(acos(min(max(1.0 - 2.0 * (x * x + y * y), -1.0), 1.0)))


def body_frame(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into the frame of a body with orientation *quat*.

    Returns:
        The vector in body coordinates.
    """
    w, x, y, z = (float(v) for v in quat)
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return rotation.T @ np.asarray(vector, dtype=np.float64)


class _Fall:
    """Track how long a base has been tilted past ``FALL_TILT_DEG``, in simulated time."""

    def __init__(self) -> None:
        self.since: float | None = None
        self.fallen = False

    def update(self, tilt_deg: float, sim_time: float) -> bool:
        """Record one tick.

        Returns:
            Whether the base has just been tilted for ``FALL_DURATION_S``.
        """
        if tilt_deg <= FALL_TILT_DEG:
            self.since, self.fallen = None, False
            return False
        if self.since is None or sim_time < self.since:
            self.since = sim_time
        was_fallen = self.fallen
        self.fallen = sim_time - self.since >= FALL_DURATION_S
        return self.fallen and not was_fallen


class FloatingBases:
    """The floating-base robots of one composed model, in anchor order."""

    def __init__(self, data: object, bindings: tuple[RobotBinding, ...]) -> None:
        """Track every binding that has a floating base."""
        self.data = data
        self._robots = tuple((binding, _Fall()) for binding in bindings if binding.layout.base is not None)

    def __bool__(self) -> bool:
        """Whether any robot has a floating base.

        Returns:
            ``True`` with at least one floating base.
        """
        return bool(self._robots)

    def observe(self) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
        """Read the floating bases.

        Returns:
            ``base_*`` entries for ``sensor_data`` (OBS-3) and the values appended to ``state`` after the joint
            positions (OBS-2): per floating base ``base_quat ‖ base_angvel ‖ base_linvel ‖ model
            sensors``, or ``None`` without floating bases.
        """
        data = self.data
        entries: dict[str, np.ndarray] = {}
        state: list[np.ndarray] = []
        for binding, _ in self._robots:
            base: FloatingBase = binding.layout.base  # type: ignore[assignment]
            prefix = binding.prefix
            qpos = data.qpos[base.qpos_adr : base.qpos_adr + 7]
            qvel = data.qvel[base.dof_adr : base.dof_adr + 6]
            quat = np.array(qpos[3:7], dtype=np.float64)
            # The free joint's linear velocity is in the world frame, its angular velocity in the body frame.
            values = {
                "base_pos": np.array(qpos[:3]),
                "base_quat": quat,
                "base_angvel": np.array(qvel[3:6]),
                "base_linvel": body_frame(quat, qvel[:3]),
            }
            for key, value in values.items():
                entries[f"{prefix}{key}"] = value.astype(np.float32)
            sensors = (
                np.asarray(data.sensordata[sensor.address : sensor.address + sensor.dimension], dtype=np.float32)
                for sensor in binding.layout.sensors
            )
            state.extend((
                entries[f"{prefix}base_quat"],
                entries[f"{prefix}base_angvel"],
                entries[f"{prefix}base_linvel"],
                *sensors,
            ))
        return entries, (np.concatenate(state) if state else None)

    def release_holds(self) -> None:
        """Release every active base-hold weld (the first action after a start or reset)."""
        for binding, _ in self._robots:
            if binding.base_hold is not None and self.data.eq_active[binding.base_hold]:
                self.data.eq_active[binding.base_hold] = 0
                logger.info("Released the {}base hold", binding.prefix)

    def arm_holds(self) -> None:
        """Activate every base-hold weld again and forget past falls (``reset()``)."""
        for binding, fall in self._robots:
            if binding.base_hold is not None:
                self.data.eq_active[binding.base_hold] = 1
            fall.since, fall.fallen = None, False

    def update_falls(self, sim_time: float) -> bool:
        """Update fall detection after a control tick (DRV-9).

        Returns:
            Whether a robot has just fallen.
        """
        fell = False
        for binding, fall in self._robots:
            base: FloatingBase = binding.layout.base  # type: ignore[assignment]
            tilt = tilt_degrees(self.data.qpos[base.qpos_adr + 3 : base.qpos_adr + 7])
            if fall.update(tilt, sim_time):
                logger.warning("The {}base has been tilted past {:.0f} degrees; fallen", binding.prefix, FALL_TILT_DEG)
                fell = True
        return fell

    def status(self) -> tuple[BaseStatus, ...]:
        """Snapshot every floating base.

        Returns:
            One status per floating base, in anchor order.
        """
        result = []
        for binding, fall in self._robots:
            base: FloatingBase = binding.layout.base  # type: ignore[assignment]
            qpos = self.data.qpos[base.qpos_adr : base.qpos_adr + 7]
            held = binding.base_hold is not None and bool(self.data.eq_active[binding.base_hold])
            result.append(BaseStatus(binding.prefix, float(qpos[2]), tilt_degrees(qpos[3:7]), held, fall.fallen))
        return tuple(result)


__all__ = ["FALL_DURATION_S", "FALL_TILT_DEG", "BaseStatus", "FloatingBases", "body_frame", "tilt_degrees"]
