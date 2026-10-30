# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Gravity torques of the reBot B601 arm, computed from a bundled URDF.

The reBot B601 motors run a plain PD law in MIT mode, so without help the arm sags by
``gravity torque / kp`` and a limp (``kp = 0``) arm falls. :class:`GravityModel` computes the
torque each arm joint needs to hold the links below it against gravity, so the driver can send it
as the MIT feedforward torque: the PD gains then only correct tracking errors.

The model reads link masses, centres of mass and joint frames from a URDF and needs nothing but
NumPy. Joint positions are in the driver's public frame (radians), which is the frame of the
bundled preview URDFs: joint1 turns about +z, joint2 about -z, joint3 to joint6 about +z.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET  # noqa: S405 - parses the URDFs bundled with this package
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

STANDARD_GRAVITY = 9.80665


def _floats(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(value) for value in text.split()], dtype=np.float64)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll, pitch, yaw: ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

    Returns:
        The 3x3 rotation matrix.
    """
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
    )


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
    )


@dataclass(frozen=True)
class _Joint:
    name: str
    parent: str
    child: str
    origin_pos: np.ndarray
    origin_rot: np.ndarray
    axis: np.ndarray
    revolute: bool


@dataclass(frozen=True)
class _ActiveJoint:
    """An actuated joint above a link during forward kinematics: its index, world axis and anchor."""

    index: int
    axis: np.ndarray
    anchor: np.ndarray


def _parse_links(root: ET.Element) -> dict[str, tuple[float, np.ndarray]]:
    """Read each link's mass and centre of mass (in the link frame) from a URDF.

    Returns:
        Link name to ``(mass, centre of mass)`` for every link with an inertial mass.
    """
    links: dict[str, tuple[float, np.ndarray]] = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        mass = None if inertial is None else inertial.find("mass")
        if inertial is None or mass is None:
            continue
        origin = inertial.find("origin")
        com = _floats(None if origin is None else origin.get("xyz"), (0.0, 0.0, 0.0))
        links[link.attrib["name"]] = (float(mass.attrib["value"]), com)
    return links


def _parse_joint(element: ET.Element) -> _Joint | None:
    parent = element.find("parent")
    child = element.find("child")
    if parent is None or child is None:
        return None
    origin = element.find("origin")
    axis_element = element.find("axis")
    axis = _floats(None if axis_element is None else axis_element.get("xyz"), (1.0, 0.0, 0.0))
    norm = float(np.linalg.norm(axis))
    return _Joint(
        name=element.attrib["name"],
        parent=parent.attrib["link"],
        child=child.attrib["link"],
        origin_pos=_floats(None if origin is None else origin.get("xyz"), (0.0, 0.0, 0.0)),
        origin_rot=_rpy_matrix(_floats(None if origin is None else origin.get("rpy"), (0.0, 0.0, 0.0))),
        axis=axis / norm if norm > 0.0 else axis,
        revolute=element.attrib.get("type") in {"revolute", "continuous"},
    )


class GravityModel:
    """Holding torques ``g(q)`` of a fixed-base serial arm described by a URDF.

    Args:
        urdf_path: URDF whose link inertials and joint frames describe the arm.
        joint_names: URDF names of the actuated joints, in driver order. Every other joint
            (fixed joints, gripper fingers) is held at zero; their links still count as load.
        gravity: Gravity vector in the URDF's root frame, in m/s^2.

    Raises:
        ValueError: If a joint is missing, not revolute, or the URDF has no root link.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        joint_names: Sequence[str],
        *,
        gravity: Sequence[float] = (0.0, 0.0, -STANDARD_GRAVITY),
    ) -> None:
        """Parse the URDF and precompute the kinematic tree.

        Raises:
            ValueError: If a joint is missing, not revolute, or the URDF has no single root.
        """
        root = ET.parse(urdf_path).getroot()  # noqa: S314 - trusted, package-bundled file
        joints = [joint for element in root.findall("joint") if (joint := _parse_joint(element)) is not None]
        children: dict[str, list[_Joint]] = {}
        for joint in joints:
            children.setdefault(joint.parent, []).append(joint)

        roots = sorted({link.attrib["name"] for link in root.findall("link")} - {joint.child for joint in joints})
        if len(roots) != 1:
            msg = f"Expected one root link in {urdf_path}, found {roots}"
            raise ValueError(msg)
        revolute = {joint.name for joint in joints if joint.revolute}
        for name in joint_names:
            if name not in revolute:
                msg = f"{urdf_path} has no revolute joint named {name!r}"
                raise ValueError(msg)

        self._root = roots[0]
        self._children = children
        self._links = _parse_links(root)
        self._joint_index = {name: i for i, name in enumerate(joint_names)}
        self._gravity = np.asarray(gravity, dtype=np.float64)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """URDF names of the actuated joints, in the order :meth:`torques` uses."""
        return tuple(self._joint_index)

    @property
    def total_mass(self) -> float:
        """Sum of every link mass in the URDF, in kg."""
        return float(sum(mass for mass, _ in self._links.values()))

    def torques(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return the joint torques that hold the arm still at ``q`` against gravity.

        Args:
            q: Positions of :attr:`joint_names` in radians, in the URDF's joint frame.

        Returns:
            Holding torques in N·m, one per joint. Sending them as feedforward cancels gravity.

        Raises:
            ValueError: If ``q`` does not have one entry per joint.
        """
        angles = np.asarray(q, dtype=np.float64).reshape(-1)
        if angles.shape != (len(self._joint_index),):
            msg = f"Expected {len(self._joint_index)} joint positions, got shape {angles.shape}"
            raise ValueError(msg)

        torques = np.zeros(len(self._joint_index))
        # Depth-first forward kinematics: (link, world rotation, world position, actuated joints above).
        pending: list[tuple[str, np.ndarray, np.ndarray, tuple[_ActiveJoint, ...]]] = [
            (self._root, np.eye(3), np.zeros(3), ()),
        ]
        while pending:
            link, rotation, position, above = pending.pop()
            if link in self._links and above:
                self._add_holding_torque(torques, link, position + rotation @ self._links[link][1], above)
            for child in self._children.get(link, ()):
                child_rotation = rotation @ child.origin_rot
                child_position = position + rotation @ child.origin_pos
                child_above = above
                index = self._joint_index.get(child.name)
                if index is not None:
                    child_above = (*above, _ActiveJoint(index, child_rotation @ child.axis, child_position))
                    child_rotation @= _axis_angle_matrix(child.axis, angles[index])
                pending.append((child.child, child_rotation, child_position, child_above))
        return torques

    def _add_holding_torque(
        self,
        torques: np.ndarray,
        link: str,
        com_world: np.ndarray,
        above: tuple[_ActiveJoint, ...],
    ) -> None:
        """Add the torque each joint in ``above`` needs to hold ``link``'s weight at ``com_world``."""
        weight = self._links[link][0] * self._gravity
        for joint in above:
            # Gravity's generalised force on the joint; holding it takes the opposite torque.
            torques[joint.index] -= float(np.dot(np.cross(joint.axis, com_world - joint.anchor), weight))


__all__ = ["STANDARD_GRAVITY", "GravityModel"]
