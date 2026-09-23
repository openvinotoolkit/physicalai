# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Auto-reset a pick-place episode when the cube stays on the green plate."""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from loguru import logger

from physicalai_mujoco_so101_plugin.spawn import sample_object_positions, write_freejoint_qpos

Phase = Literal["idle", "success_hold"]


@dataclass
class EpisodeAutoResetConfig:
    """Parameters for cube-on-plate success detection and respawn."""

    cube_joint_name: str = "block1:joint"
    cube_body_name: str = "block1"
    target_body_name: str = "target"
    target_radius: float = 0.05
    cube_half_size: float = 0.02
    success_dwell_s: float = 5.0
    cube_static_speed: float = 0.08
    spawn_center: tuple[float, float] = (0.22, 0.0)
    spawn_min_r: float = 0.05
    spawn_max_r: float = 0.14
    spawn_angle_half_deg: float = 50.0
    target_min_sep: float = 0.11
    spawn_attempts: int = 200
    cube_spawn_z: float = 0.02


@dataclass
class _Runtime:
    cube_body_id: int
    target_body_id: int
    cube_qpos_address: int
    cube_dof_address: int
    phase: Phase = "idle"
    success_since: float | None = None
    episode_count: int = 0


class EpisodeAutoReset:
    """Watch for cube-on-plate success, then respawn the cube after a dwell.

    The green plate stays fixed. Only the cube freejoint is randomized.
    """

    def __init__(self, config: EpisodeAutoResetConfig, rng: np.random.Generator | None = None) -> None:
        """Initialize with the given config and an optional RNG for cube respawn placement."""
        self._config = config
        self._rng = rng if rng is not None else np.random.default_rng()
        self._runtime: _Runtime | None = None
        self._last_countdown_s: float | None = None

    @classmethod
    def maybe_create(
        cls,
        model: object,
        *,
        free_joints: tuple[str, ...],
        target_body_name: str,
        spawn_center: tuple[float, float],
        spawn_min_r: float,
        spawn_max_r: float,
        spawn_angle_half_deg: float,
        target_min_sep: float,
        rng: np.random.Generator,
        success_dwell_s: float = 5.0,
    ) -> EpisodeAutoReset | None:
        """Return an auto-reset helper when the scene has one cube and a target plate."""
        import mujoco  # noqa: PLC0415

        if len(free_joints) != 1 or not target_body_name:
            return None

        cube_joint_name = free_joints[0]
        cube_body_name = cube_joint_name.split(":", 1)[0]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, cube_joint_name)
        cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube_body_name)
        target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
        if jid < 0 or cube_body_id < 0 or target_body_id < 0:
            return None

        target_radius = 0.05
        for geom_id in range(int(model.ngeom)):
            if int(model.geom_bodyid[geom_id]) != int(target_body_id):
                continue
            if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                target_radius = float(model.geom_size[geom_id][0])
                break

        config = EpisodeAutoResetConfig(
            cube_joint_name=cube_joint_name,
            cube_body_name=cube_body_name,
            target_body_name=target_body_name,
            target_radius=target_radius,
            success_dwell_s=success_dwell_s,
            spawn_center=spawn_center,
            spawn_min_r=spawn_min_r,
            spawn_max_r=spawn_max_r,
            spawn_angle_half_deg=spawn_angle_half_deg,
            target_min_sep=target_min_sep,
        )
        helper = cls(config, rng=rng)
        helper._runtime = _Runtime(
            cube_body_id=int(cube_body_id),
            target_body_id=int(target_body_id),
            cube_qpos_address=int(model.jnt_qposadr[jid]),
            cube_dof_address=int(model.jnt_dofadr[jid]),
        )
        logger.info(
            "Episode auto-reset enabled (cube on '{}' for {:.1f}s -> respawn cube only)",
            target_body_name,
            success_dwell_s,
        )
        return helper

    def status(self) -> dict[str, object]:
        """Return a JSON-friendly snapshot for the HTTP status endpoint."""
        runtime = self._runtime
        if runtime is None:
            return {"enabled": False}

        countdown_s: float | None = None
        if runtime.phase == "success_hold" and runtime.success_since is not None:
            # Countdown is filled in by update() via last known sim time; callers
            # that only read status between ticks get the last computed value.
            countdown_s = self._last_countdown_s

        return {
            "enabled": True,
            "phase": runtime.phase,
            "episode_count": runtime.episode_count,
            "success_dwell_s": self._config.success_dwell_s,
            "countdown_s": countdown_s,
            "cube": self._config.cube_body_name,
            "target": self._config.target_body_name,
        }

    def notify_manual_reset(self) -> None:
        """Clear success timing after an explicit HTTP/viewer reset."""
        if self._runtime is None:
            return
        self._runtime.phase = "idle"
        self._runtime.success_since = None
        self._last_countdown_s = None

    def update(self, model: object, data: object) -> None:
        """Advance success detection / countdown / cube respawn for one tick."""
        import mujoco  # noqa: PLC0415

        runtime = self._runtime
        if runtime is None:
            return

        now = float(data.time)
        success = self._is_success(model, data, runtime)
        if success:
            if runtime.success_since is None:
                runtime.success_since = now
                runtime.phase = "success_hold"
                logger.info(
                    "Cube on green plate — resetting in {:.1f}s",
                    self._config.success_dwell_s,
                )
            elapsed = now - runtime.success_since
            remaining = max(0.0, self._config.success_dwell_s - elapsed)
            self._last_countdown_s = remaining
            if remaining <= 0.0:
                self._respawn_cube(data, runtime)
                runtime.episode_count += 1
                runtime.phase = "idle"
                runtime.success_since = None
                self._last_countdown_s = None
                logger.info("Episode auto-reset #{} (cube respawned, plate fixed)", runtime.episode_count)
                mujoco.mj_forward(model, data)
            return

        if runtime.phase == "success_hold":
            logger.info("Cube left the green plate — countdown cancelled")
        runtime.phase = "idle"
        runtime.success_since = None
        self._last_countdown_s = None

    def _is_success(self, model: object, data: object, runtime: _Runtime) -> bool:
        _ = model
        config = self._config
        cube_pos = np.asarray(data.xpos[runtime.cube_body_id], dtype=np.float64)
        target_pos = np.asarray(data.xpos[runtime.target_body_id], dtype=np.float64)

        planar = float(np.hypot(cube_pos[0] - target_pos[0], cube_pos[1] - target_pos[1]))
        # Require the cube center to sit inside the disc (with a small margin for half-size).
        if planar > max(0.01, config.target_radius - 0.005):
            return False

        # Cube resting on the table / disc, not being carried high above it.
        if float(cube_pos[2]) > config.cube_half_size + 0.04:
            return False

        speed = float(np.linalg.norm(data.qvel[runtime.cube_dof_address : runtime.cube_dof_address + 3]))
        return speed <= config.cube_static_speed

    def _sample_spawn_xy(self, target_xy: tuple[float, float]) -> tuple[float, float]:
        """Sample a cube position at least ``target_min_sep`` from the plate.

        Returns:
            An ``(x, y)`` position in world coordinates.
        """
        config = self._config
        # Only the cube moves, so there is nothing else to stay clear of.
        positions = sample_object_positions(
            1,
            rng=self._rng,
            center=config.spawn_center,
            min_r=config.spawn_min_r,
            max_r=config.spawn_max_r,
            angle_half_deg=config.spawn_angle_half_deg,
            target_xy=target_xy,
            target_min_sep=config.target_min_sep,
            object_min_sep=0.0,
            attempts=config.spawn_attempts,
        )
        return positions[0]

    def _respawn_cube(self, data: object, runtime: _Runtime) -> None:
        target_xy = (
            float(data.xpos[runtime.target_body_id][0]),
            float(data.xpos[runtime.target_body_id][1]),
        )
        x, y = self._sample_spawn_xy(target_xy)
        write_freejoint_qpos(
            data,
            runtime.cube_qpos_address,
            runtime.cube_dof_address,
            (x, y),
            yaw=float(self._rng.uniform(0.0, 2.0 * np.pi)),
            z=self._config.cube_spawn_z,
        )
        logger.info("Respawned cube at x={:.3f}, y={:.3f}", x, y)
