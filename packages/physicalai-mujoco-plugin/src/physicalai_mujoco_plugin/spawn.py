# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared rejection sampling and freejoint writes used by scene resets.

Every scene reset places free objects the same way: sample a point in a polar
arc in front of the robot, reject it when it lands on the target, on another
object or under an arm, and write the result into the object's freejoint.
Keeping one implementation here stops the per-scene copies from drifting apart.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from physicalai_mujoco_plugin.scene_registry import SceneConfig

SPAWN_ATTEMPTS = 200
FREEJOINT_SPAWN_Z = 0.02
SPAWN_CLEAR_OF_ARMS = True
"""Whether scene resets keep spawned objects out of the arms' footprint (:class:`ArmFootprint`).

Not an option: the SO-101 fingerprint script turns it off to show that nothing else changed.
"""
ARM_CLEARANCE = 0.01
"""Gap between a spawned object's horizontal radius and the arms' footprint, in metres."""
ARM_FOOTPRINT_HEIGHT = 0.10
"""Height above the table (``z = 0``) below which a robot geom counts toward the footprint, for the
SO-101; a reach-scaled scene scales it with the layout. Parts higher up (a forearm over the spawn
area) neither block a grasp nor hide much of the object."""

_BOX_CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)


@dataclass(frozen=True)
class ArmFootprint:
    """Where the arms hang low over the table: one convex polygon per low robot geom, seen from above.

    Each polygon is the convex hull of a geom's bounding-box corners projected onto the table
    plane, at the arms' pose when the footprint was taken; only geoms whose lowest corner is at
    most a given height above the table count. Polygons are stored as their edges,
    counter-clockwise, padded to the box's eight corners by repeating the last vertex.
    """

    starts: np.ndarray
    """``(geoms, 8, 2)`` start point of each polygon edge."""
    ends: np.ndarray
    """``(geoms, 8, 2)`` end point of each polygon edge."""

    @classmethod
    def of_robots(cls, model: object, data: object, robot_roots: tuple[int, ...], height: float) -> ArmFootprint:
        """Take the footprint of the geoms of the bodies under *robot_roots* that reach down to *height*.

        The arms are taken at their current pose; *height* is above the table plane ``z = 0``.

        Returns:
            The footprint; it has no polygons when the robots have no geoms.
        """
        import mujoco  # noqa: PLC0415

        mujoco.mj_kinematics(model, data)
        roots = np.asarray(robot_roots, dtype=np.int64)
        geoms = np.flatnonzero(np.isin(np.asarray(model.body_rootid)[np.asarray(model.geom_bodyid)], roots))
        polygons = []
        for geom in geoms:
            center, half = model.geom_aabb[geom, :3], model.geom_aabb[geom, 3:]
            corners = data.geom_xpos[geom] + (center + _BOX_CORNERS * half) @ data.geom_xmat[geom].reshape(3, 3).T
            if float(corners[:, 2].min()) > height:
                continue
            hull = _convex_hull(corners[:, :2])
            polygons.append(np.concatenate([hull, np.repeat(hull[-1:], len(_BOX_CORNERS) - len(hull), axis=0)]))
        starts = np.asarray(polygons, dtype=np.float64).reshape(-1, len(_BOX_CORNERS), 2)
        return cls(starts=starts, ends=np.roll(starts, -1, axis=1))

    def distance(self, xy: tuple[float, float]) -> float:
        """Return how far *xy* is from the footprint: 0 inside a polygon, ``inf`` without polygons.

        Returns:
            The distance in metres.
        """
        if not len(self.starts):
            return float("inf")
        edges = self.ends - self.starts
        offsets = np.asarray(xy, dtype=np.float64) - self.starts
        cross = edges[..., 0] * offsets[..., 1] - edges[..., 1] * offsets[..., 0]
        # A degenerate polygon (a line or a point) has edges both ways, so nothing is inside it.
        if bool(np.any(np.all(cross >= -1e-12, axis=1) & np.any(cross > 1e-12, axis=1))):  # noqa: PLR2004
            return 0.0
        lengths = np.maximum(np.einsum("gei,gei->ge", edges, edges), 1e-18)
        along = np.clip(np.einsum("gei,gei->ge", offsets, edges) / lengths, 0.0, 1.0)
        return float(np.min(np.linalg.norm(offsets - along[..., None] * edges, axis=-1)))


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Return the counter-clockwise convex hull of 2-D *points* (Andrew's monotone chain), without repeats.

    Returns:
        The hull's vertices; one or two points when *points* coincide or lie on a line.
    """
    unique = np.unique(np.round(points, 12), axis=0)
    if len(unique) <= 2:  # noqa: PLR2004 - a point or a segment is its own hull
        return unique

    def chain(ordered: np.ndarray) -> list[np.ndarray]:
        kept: list[np.ndarray] = []
        for point in ordered:
            while len(kept) >= 2:  # noqa: PLR2004 - a turn needs two kept points
                first, second = kept[-1] - kept[-2], point - kept[-2]
                if first[0] * second[1] - first[1] * second[0] > 0:
                    break
                kept.pop()
            kept.append(point)
        return kept

    return np.asarray(chain(unique)[:-1] + chain(unique[::-1])[:-1])


def arm_footprint(
    model: object, data: object, robot_roots: tuple[int, ...], height: float = ARM_FOOTPRINT_HEIGHT
) -> ArmFootprint | None:
    """Return the arms' footprint for a scene reset, or ``None`` when resets ignore the arms.

    Args:
        model: The compiled scene.
        data: Its state, with the arms where the reset finds them.
        robot_roots: Root body of every robot.
        height: Geoms whose lowest point is higher above the table do not count.

    Returns:
        ``None`` without *robot_roots* or with :data:`SPAWN_CLEAR_OF_ARMS` off.
    """
    if not SPAWN_CLEAR_OF_ARMS or not robot_roots:
        return None
    return ArmFootprint.of_robots(model, data, robot_roots, height)


def object_radius(model: object, joint_names: tuple[str, ...]) -> float:
    """Return the largest horizontal radius of the bodies that the named freejoints move.

    A body's radius is the farthest bounding-box corner of its geoms from the body's vertical axis,
    so it holds for any yaw of an upright object.

    Returns:
        The radius in metres; 0 when none of the joints is in the model.
    """
    import mujoco  # noqa: PLC0415

    radius = 0.0
    rotation = np.zeros(9)
    for joint_name in joint_names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint < 0:
            continue
        body = int(model.jnt_bodyid[joint])
        for geom in np.flatnonzero(np.asarray(model.geom_bodyid) == body):
            mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
            center, half = model.geom_aabb[geom, :3], model.geom_aabb[geom, 3:]
            corners = model.geom_pos[geom] + (center + _BOX_CORNERS * half) @ rotation.reshape(3, 3).T
            radius = max(radius, float(np.hypot(corners[:, 0], corners[:, 1]).max()))
    return radius


def sample_clear_position(
    draw: Callable[[], tuple[float, float]],
    *,
    clear_of_others: Callable[[tuple[float, float]], bool] | None = None,
    footprint: ArmFootprint | None = None,
    padding: float = 0.0,
    attempts: int = SPAWN_ATTEMPTS,
) -> tuple[float, float]:
    """Draw positions until one is clear of the other objects and at least *padding* off the arms.

    When no draw within *attempts* is clear of both, the draw clear of the other objects that is
    farthest from the arms is kept (with a debug log), else the last draw.

    Args:
        draw: Draws one candidate ``(x, y)``.
        clear_of_others: Whether a candidate keeps clear of the target and the objects placed so far;
            ``None`` accepts every candidate.
        footprint: The arms' footprint; ``None`` ignores the arms.
        padding: Smallest distance from the footprint: the object's radius and a clearance.
        attempts: Most draws; at least one is made.

    Returns:
        The kept ``(x, y)``.
    """
    last: tuple[float, float] | None = None
    farthest: tuple[float, float] | None = None
    farthest_distance = -1.0
    for _ in range(max(1, attempts)):
        last = draw()
        if clear_of_others is not None and not clear_of_others(last):
            continue
        if footprint is None:
            return last
        distance = footprint.distance(last)
        if distance >= padding:
            return last
        if distance > farthest_distance:
            farthest, farthest_distance = last, distance
    if farthest is not None:
        logger.debug(
            "No spawn position clear of the arms in {} draws; using ({:.3f}, {:.3f}), {:.3f} m from them",
            attempts,
            farthest[0],
            farthest[1],
            farthest_distance,
        )
        return farthest
    assert last is not None  # noqa: S101 - the loop draws at least once
    return last


def sample_spawn_xy(
    rng: np.random.Generator,
    *,
    center: tuple[float, float],
    min_r: float,
    max_r: float,
    angle_half_deg: float,
) -> tuple[float, float]:
    """Sample one point from the polar arc described by the spawn parameters.

    Returns:
        An ``(x, y)`` position in world coordinates.
    """
    r = float(rng.uniform(min_r, max_r))
    half_angle = np.radians(angle_half_deg)
    theta = float(rng.uniform(-half_angle, half_angle))
    return (center[0] + r * float(np.cos(theta)), center[1] + r * float(np.sin(theta)))


def sample_object_positions(
    count: int,
    *,
    rng: np.random.Generator,
    center: tuple[float, float],
    min_r: float,
    max_r: float,
    angle_half_deg: float,
    target_xy: tuple[float, float],
    target_min_sep: float,
    object_min_sep: float,
    attempts: int = SPAWN_ATTEMPTS,
    footprint: ArmFootprint | None = None,
    padding: float = 0.0,
) -> list[tuple[float, float]]:
    """Sample *count* positions kept clear of the target, of each other and of the arms' *footprint*.

    Each position comes from :func:`sample_clear_position`, so the caller always gets exactly
    *count* positions.

    Returns:
        A list of *count* ``(x, y)`` positions.
    """
    positions: list[tuple[float, float]] = []

    def draw() -> tuple[float, float]:
        return sample_spawn_xy(rng, center=center, min_r=min_r, max_r=max_r, angle_half_deg=angle_half_deg)

    def clear_of_others(xy: tuple[float, float]) -> bool:
        x, y = xy
        if np.hypot(x - target_xy[0], y - target_xy[1]) < target_min_sep:
            return False
        return not any(np.hypot(x - px, y - py) < object_min_sep for px, py in positions)

    for _ in range(count):
        # One at a time: each draw keeps clear of the positions placed before it.
        position = sample_clear_position(
            draw, clear_of_others=clear_of_others, footprint=footprint, padding=padding, attempts=attempts
        )
        positions.append(position)
    return positions


def sample_scene_positions(
    scene: SceneConfig,
    count: int,
    *,
    rng: np.random.Generator,
    target_xy: tuple[float, float],
    footprint: ArmFootprint | None = None,
    padding: float = 0.0,
) -> list[tuple[float, float]]:
    """Sample *count* object positions using a scene's spawn parameters, at least *padding* off *footprint*.

    Returns:
        A list of *count* ``(x, y)`` positions.
    """
    return sample_object_positions(
        count,
        rng=rng,
        center=scene.spawn_center,
        min_r=scene.spawn_min_r,
        max_r=scene.spawn_max_r,
        angle_half_deg=scene.spawn_angle_half_deg,
        target_xy=target_xy,
        target_min_sep=scene.target_min_sep,
        object_min_sep=scene.block_min_sep,
        footprint=footprint,
        padding=padding,
    )


def read_body_xy(
    model: object,
    data: object,
    body_name: str,
    default: tuple[float, float],
) -> tuple[float, float]:
    """Return a body's world ``(x, y)``, or *default* when it is not in the model.

    World ``data.xpos`` is used rather than ``model.body_pos`` so that bodies
    nested under another frame report where they actually are.

    Returns:
        The body's world ``(x, y)``, or *default*.
    """
    import mujoco  # noqa: PLC0415

    if not body_name:
        return default
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return default
    return (float(data.xpos[body_id][0]), float(data.xpos[body_id][1]))


def write_freejoint_qpos(
    data: object,
    qpos_adr: int,
    dof_adr: int,
    xy: tuple[float, float],
    *,
    yaw: float,
    z: float = FREEJOINT_SPAWN_Z,
) -> None:
    """Place a freejoint at *xy* with a yaw-only orientation and zero velocity."""
    data.qpos[qpos_adr : qpos_adr + 3] = [xy[0], xy[1], z]
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
    data.qvel[dof_adr : dof_adr + 6] = 0.0


def place_freejoint(
    model: object,
    data: object,
    joint_name: str,
    xy: tuple[float, float],
    rng: np.random.Generator,
    *,
    z: float = FREEJOINT_SPAWN_Z,
) -> bool:
    """Place the named freejoint at *xy* with a random yaw.

    Returns:
        ``False`` when the model has no such joint, ``True`` otherwise.
    """
    import mujoco  # noqa: PLC0415

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return False
    write_freejoint_qpos(
        data,
        int(model.jnt_qposadr[jid]),
        int(model.jnt_dofadr[jid]),
        xy,
        yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
        z=z,
    )
    return True
