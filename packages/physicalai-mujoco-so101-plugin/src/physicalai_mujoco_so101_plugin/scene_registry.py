# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Scene definitions and reset behavior for the MuJoCo SO-101 simulation."""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from physicalai_mujoco_so101_plugin._urdf import get_urdf_path
from physicalai_mujoco_so101_plugin.spawn import (
    place_freejoint,
    read_body_xy,
    sample_scene_positions,
)

if TYPE_CHECKING:
    from pathlib import Path

ResetFn = Callable[[object, object, np.random.Generator], None]


@dataclass(frozen=True)
class SceneConfig:
    """Metadata and reset configuration for a simulation scene."""

    scene_id: str
    display_name: str
    description: str
    scene_xml_relpath: str
    free_joints: tuple[str, ...] = ()
    target_bodies: tuple[str, ...] = ()
    spawn_center: tuple[float, float] = (0.22, 0.0)
    spawn_min_r: float = 0.05
    spawn_max_r: float = 0.14
    spawn_angle_half_deg: float = 50.0
    block_min_sep: float = 0.09
    target_min_sep: float = 0.11
    num_arms: int = 1
    """Number of SO-101 arms the scene model provides."""
    home_qpos: tuple[tuple[str, float], ...] = ()
    """Home joint positions in radians; unlisted arm joints use the model default."""

    @property
    def scene_xml_path(self) -> Path:
        """Absolute path to this scene's XML model."""
        return get_urdf_path() / self.scene_xml_relpath


_GARMENT_FOLD_HOME: tuple[tuple[str, float], ...] = (
    ("left_shoulder_pan", -1.1),
    ("left_shoulder_lift", 0.3),
    ("left_elbow_flex", 0.8),
    ("left_wrist_flex", 0.3),
    ("right_shoulder_pan", 1.1),
    ("right_shoulder_lift", 0.3),
    ("right_elbow_flex", 0.8),
    ("right_wrist_flex", 0.3),
)


# ---------------------------------------------------------------------------
# Scene reset functions
# ---------------------------------------------------------------------------


def _freejoint_spawn_reset(scene_id: str) -> ResetFn:
    """Build a reset that respawns a scene's free objects clear of its target.

    The target body itself is left where it is; only the freejoints listed in
    the scene's ``free_joints`` are randomized, using that scene's spawn arc.

    Returns:
        A reset callback for `scene_id`.
    """

    def reset(model: object, data: object, rng: np.random.Generator) -> None:
        import mujoco  # noqa: PLC0415

        scene = get_scene(scene_id)
        target_body = scene.target_bodies[0] if scene.target_bodies else ""
        target_xy = read_body_xy(model, data, target_body, scene.spawn_center)
        positions = sample_scene_positions(scene, len(scene.free_joints), rng=rng, target_xy=target_xy)

        for joint_name, xy in zip(scene.free_joints, positions, strict=True):
            place_freejoint(model, data, joint_name, xy, rng)

        mujoco.mj_forward(model, data)

    return reset


def _garment_fold_reset(model: object, data: object, rng: np.random.Generator) -> None:  # noqa: ARG001
    import mujoco  # noqa: PLC0415

    for joint_name, val in _GARMENT_FOLD_HOME:
        # pyrefly: ignore [missing-attribute]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            continue
        # pyrefly: ignore [missing-attribute]
        qpos_adr = int(model.jnt_qposadr[jid])
        # pyrefly: ignore [missing-attribute]
        dof_adr = int(model.jnt_dofadr[jid])
        # pyrefly: ignore [missing-attribute]
        data.qpos[qpos_adr] = val
        # pyrefly: ignore [missing-attribute]
        data.qvel[dof_adr] = 0.0
        # pyrefly: ignore [missing-attribute]
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        if aid >= 0:
            # pyrefly: ignore [missing-attribute]
            data.ctrl[aid] = val

    # pyrefly: ignore [missing-attribute]
    if model.nflex > 0:
        # pyrefly: ignore [missing-attribute]
        first_vertex_body = int(model.flex_vertbodyid[0])
        vertex_joints = [j for j in range(model.njnt) if int(model.jnt_bodyid[j]) == first_vertex_body]
        if not vertex_joints:
            # A pinned first vertex has no DOFs, so there is no flex state to restore.
            logger.warning("Flex body {} has no joints; skipping garment reset", first_vertex_body)
        else:
            flex_qpos_adr = min(int(model.jnt_qposadr[j]) for j in vertex_joints)
            flex_dof_adr = min(int(model.jnt_dofadr[j]) for j in vertex_joints)
            data.qpos[flex_qpos_adr : flex_qpos_adr + 3 * model.nflexvert] = model.flex_vert.ravel()
            data.qvel[flex_dof_adr : flex_dof_adr + 3 * model.nflexvert] = 0.0

    # pyrefly: ignore [missing-attribute]
    mujoco.mj_forward(model, data)


def _yahtzee_reset(model: object, data: object, rng: np.random.Generator) -> None:  # noqa: PLR0914
    import mujoco  # noqa: PLC0415

    die_joints = [f"die{i}:joint" for i in range(1, 7)]

    cx, cy = 0.30, 0.0
    cup_jitter = 0.005

    for joint_name in die_joints:
        # pyrefly: ignore [missing-attribute]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            continue
        # pyrefly: ignore [missing-attribute]
        qpos_adr = int(model.jnt_qposadr[jid])
        # pyrefly: ignore [missing-attribute]
        dof_adr = int(model.jnt_dofadr[jid])

        r = float(rng.uniform(0.06, 0.18))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        x = cx + float(rng.uniform(-cup_jitter, cup_jitter)) + r * float(np.cos(theta))
        y = cy + float(rng.uniform(-cup_jitter, cup_jitter)) + r * float(np.sin(theta))

        yaw = float(rng.uniform(0.0, 2.0 * np.pi))
        tilt = float(rng.uniform(-0.3, 0.3))
        c = np.cos(yaw / 2.0)
        s = np.sin(yaw / 2.0)
        drop_z = float(rng.uniform(0.12, 0.18))
        # pyrefly: ignore [missing-attribute]
        data.qpos[qpos_adr : qpos_adr + 3] = [x, y, drop_z]
        # pyrefly: ignore [missing-attribute]
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = [
            c * np.sin(tilt / 2.0),
            s * np.sin(tilt / 2.0),
            s * np.cos(tilt / 2.0),
            c * np.cos(tilt / 2.0),
        ]

        vx = float(rng.uniform(-0.5, 0.5))
        vy = float(rng.uniform(-0.5, 0.5))
        vz = float(rng.uniform(-0.4, -0.05))
        wx = float(rng.uniform(-15.0, 15.0))
        wy = float(rng.uniform(-15.0, 15.0))
        wz = float(rng.uniform(-8.0, 8.0))
        # pyrefly: ignore [missing-attribute]
        data.qvel[dof_adr : dof_adr + 6] = [vx, vy, vz, wx, wy, wz]

    # pyrefly: ignore [missing-attribute]
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# Scene configurations
# ---------------------------------------------------------------------------

_SCENES: dict[str, SceneConfig] = {
    "single_pick_place": SceneConfig(
        scene_id="single_pick_place",
        display_name="Single Pick & Place",
        description="One block and a target disc",
        scene_xml_relpath="scenes/single_pick_place/scene.xml",
        free_joints=("block1:joint",),
        target_bodies=("target",),
        # Keep spawns inside comfortable SO-101 teleop reach (was up to ~0.5 m).
        spawn_center=(0.22, 0.0),
        spawn_min_r=0.05,
        spawn_max_r=0.14,
        spawn_angle_half_deg=50.0,
        target_min_sep=0.11,
    ),
    "yahtzee": SceneConfig(
        scene_id="yahtzee",
        display_name="Yahtzee",
        description="Pick up 6 dice and place them in the cup",
        scene_xml_relpath="scenes/yahtzee/scene.xml",
        free_joints=("die1:joint", "die2:joint", "die3:joint", "die4:joint", "die5:joint", "die6:joint"),
        target_bodies=(),
        spawn_center=(0.30, 0.0),
        spawn_min_r=0.06,
        spawn_max_r=0.18,
        spawn_angle_half_deg=160.0,
        block_min_sep=0.018,
        target_min_sep=0.02,
    ),
    "garment_fold": SceneConfig(
        scene_id="garment_fold",
        display_name="Garment Fold",
        description="Fold a flexible garment lying flat on a table",
        scene_xml_relpath="scenes/garment_fold/scene.xml",
        num_arms=2,
        home_qpos=_GARMENT_FOLD_HOME,
    ),
}

_RESET_FUNCTIONS: dict[str, ResetFn] = {
    "single_pick_place": _freejoint_spawn_reset("single_pick_place"),
    "yahtzee": _yahtzee_reset,
    "garment_fold": _garment_fold_reset,
}


def get_scene(scene_id: str) -> SceneConfig:
    """Return the scene configuration identified by `scene_id`.

    Raises:
        KeyError: If `scene_id` is not registered.
    """
    if scene_id not in _SCENES:
        msg = f"Unknown scene {scene_id!r}. Available: {list(_SCENES)}"
        raise KeyError(msg)
    return _SCENES[scene_id]


def list_scenes() -> dict[str, SceneConfig]:
    """Return all scene configurations by ID."""
    return dict(_SCENES)


def list_scenes_for_arms(num_arms: int) -> dict[str, SceneConfig]:
    """Return the scenes whose model provides exactly `num_arms` SO-101 arms."""
    return {scene_id: scene for scene_id, scene in _SCENES.items() if scene.num_arms == num_arms}


def get_reset_fn(scene_id: str) -> ResetFn | None:
    """Return the reset callback for `scene_id`, if one is registered."""
    return _RESET_FUNCTIONS.get(scene_id)
