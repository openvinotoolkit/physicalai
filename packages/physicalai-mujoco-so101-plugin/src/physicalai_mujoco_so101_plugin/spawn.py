# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared rejection sampling and freejoint writes used by scene resets.

Every scene reset places free objects the same way: sample a point in a polar
arc in front of the robot, reject it when it lands on the target or on another
object, and write the result into the object's freejoint. Keeping one
implementation here stops the per-scene copies from drifting apart.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from physicalai_mujoco_so101_plugin.scene_registry import SceneConfig

SPAWN_ATTEMPTS = 200
FREEJOINT_SPAWN_Z = 0.02


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
) -> list[tuple[float, float]]:
    """Sample *count* positions kept clear of the target and of each other.

    When no candidate satisfies both separations within *attempts* tries, the
    last candidate is used, so the caller always gets exactly *count* positions.

    Returns:
        A list of *count* ``(x, y)`` positions.
    """
    positions: list[tuple[float, float]] = []
    for _ in range(count):
        best: tuple[float, float] | None = None
        # At least one draw, so the caller always gets `count` positions back.
        for _ in range(max(1, attempts)):
            x, y = sample_spawn_xy(
                rng,
                center=center,
                min_r=min_r,
                max_r=max_r,
                angle_half_deg=angle_half_deg,
            )
            best = (x, y)
            if np.hypot(x - target_xy[0], y - target_xy[1]) < target_min_sep:
                continue
            if any(np.hypot(x - px, y - py) < object_min_sep for px, py in positions):
                continue
            positions.append(best)
            break
        else:
            if best is not None:
                positions.append(best)
    return positions


def sample_scene_positions(
    scene: SceneConfig,
    count: int,
    *,
    rng: np.random.Generator,
    target_xy: tuple[float, float],
) -> list[tuple[float, float]]:
    """Sample *count* object positions using a scene's spawn parameters.

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
