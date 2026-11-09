# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Scene objects: the free objects of the current scene, their spawn area, poses and viewer drags."""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from physicalai_mujoco_plugin.spawn import sample_object_positions, write_freejoint_qpos

if TYPE_CHECKING:
    import threading
    from collections.abc import Iterable, Mapping

    from physicalai_mujoco_plugin.scene_registry import SceneConfig
    from physicalai_mujoco_plugin.viser_controls import ObjectPose


@dataclass(frozen=True)
class SpawnArea:
    """A scene's free objects, its target body and the arc where objects respawn."""

    free_joints: tuple[str, ...] = ("block1:joint", "block2:joint", "block3:joint")
    target_body_name: str = "target"
    spawn_center: tuple[float, float] = (0.22, 0.0)
    spawn_min_r: float = 0.05
    spawn_max_r: float = 0.14
    spawn_angle_half_deg: float = 50.0
    block_min_sep: float = 0.09
    target_min_sep: float = 0.11

    @classmethod
    def from_scene(cls, scene: SceneConfig) -> SpawnArea:
        """Read a registered scene's spawn area.

        Returns:
            The scene's spawn area.
        """
        return cls.from_scene_config(asdict(scene))

    @classmethod
    def from_scene_config(cls, scene_config: Mapping[str, object]) -> SpawnArea:
        """Read a ``SceneConfig`` as a dict (``dataclasses.asdict``), as the legacy constructor takes it.

        Returns:
            The scene's spawn area.
        """
        targets = tuple(scene_config.get("target_bodies", ()))  # type: ignore[arg-type]
        return cls(
            free_joints=tuple(scene_config["free_joints"]),  # type: ignore[arg-type]
            target_body_name=targets[0] if targets else "",
            spawn_center=tuple(scene_config["spawn_center"]),  # type: ignore[arg-type]
            spawn_min_r=float(scene_config["spawn_min_r"]),  # type: ignore[arg-type]
            spawn_max_r=float(scene_config["spawn_max_r"]),  # type: ignore[arg-type]
            spawn_angle_half_deg=float(scene_config["spawn_angle_half_deg"]),  # type: ignore[arg-type]
            block_min_sep=float(scene_config["block_min_sep"]),  # type: ignore[arg-type]
            target_min_sep=float(scene_config["target_min_sep"]),  # type: ignore[arg-type]
        )


class SceneObjects:
    """Free objects of the current model: addresses, poses for the panel, drags and respawns."""

    def __init__(self, area: SpawnArea, lock: threading.RLock) -> None:
        """Start unbound.

        Args:
            area: The scene's objects and spawn area.
            lock: Guards the state other threads read (poses, follow bodies).
        """
        self.area = area
        self._lock = lock
        self.block_addrs: list[tuple[int, int]] = []
        self.free_addrs: dict[str, tuple[int, int]] = {}
        self.target_body_id: int | None = None
        self.follow_body_ids: dict[str, int] = {}
        self.joint_bodies: dict[str, str] = {}
        self.held: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        """Poses re-applied after every step while a viewer drags an object."""
        self.poses: dict[str, ObjectPose] = {}
        """Latest free-object poses, published by the sim thread under the lock."""

    def bind(self, model: object, follow_bodies: Iterable[str] = ()) -> None:
        """Resolve the area's objects in *model*; *follow_bodies* are extra bodies the viewer may follow."""
        import mujoco  # noqa: PLC0415

        self.block_addrs.clear()
        self.free_addrs.clear()
        self.held.clear()
        for joint_name in self.area.free_joints:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                continue
            addrs = (int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id]))
            self.block_addrs.append(addrs)
            if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                self.free_addrs[joint_name] = addrs

        target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.area.target_body_name)
        self.target_body_id = int(target_id) if target_id >= 0 else None

        follow: dict[str, int] = {}
        joint_bodies: dict[str, str] = {}
        for joint_name in self.free_addrs:
            joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
            body_id = int(model.jnt_bodyid[joint_id])
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
            follow[body_name] = body_id
            joint_bodies[joint_name] = body_name
        candidates = [self.area.target_body_name] if self.area.target_body_name else []
        for body_name in (*candidates, *follow_bodies):
            body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
            if body_id > 0 and body_name not in follow:
                follow[body_name] = body_id
        with self._lock:
            self.follow_body_ids = follow
            self.joint_bodies = joint_bodies

    def clear(self) -> None:
        """Forget the bound model."""
        self.block_addrs.clear()
        self.free_addrs.clear()
        self.held.clear()
        self.target_body_id = None
        with self._lock:
            self.follow_body_ids = {}
            self.joint_bodies = {}
            self.poses = {}

    def follow_targets(self, data: object | None) -> dict[str, tuple[float, float, float]]:
        """Return the current world position of each body the viewer can follow."""
        if data is None:
            return {}
        return {name: tuple(float(v) for v in data.xpos[body_id]) for name, body_id in self.follow_body_ids.items()}  # type: ignore[misc]

    def set_pose(
        self,
        model: object,
        data: object,
        joint: str,
        position: tuple[float, float, float],
        wxyz: tuple[float, float, float, float] | None,
        *,
        hold: bool,
    ) -> None:
        """Teleport a free object, optionally holding it there every step."""
        import mujoco  # noqa: PLC0415

        if joint not in self.free_addrs:
            logger.warning("Unknown free object joint {!r}", joint)
            return
        qpos_addr, _ = self.free_addrs[joint]
        pos = np.asarray(position, dtype=np.float64)
        quat = np.asarray(wxyz if wxyz is not None else data.qpos[qpos_addr + 3 : qpos_addr + 7], dtype=np.float64)
        norm = float(np.linalg.norm(quat))
        if pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(pos).all() or not np.isfinite(norm) or norm == 0:
            logger.warning("Ignoring invalid pose for {!r}", joint)
            return
        quat /= norm
        self._write(data, joint, pos, quat)
        if hold:
            self.held[joint] = (pos, quat)
        else:
            self.held.pop(joint, None)
        mujoco.mj_forward(model, data)

    def apply_held(self, model: object, data: object) -> None:
        """Pin objects a viewer is dragging back to the dragged pose."""
        if not self.held:
            return
        import mujoco  # noqa: PLC0415

        for joint, (position, wxyz) in self.held.items():
            self._write(data, joint, position, wxyz)
        mujoco.mj_forward(model, data)

    def publish(self, data: object | None) -> None:
        """Snapshot free-object poses for the HTTP and viewer threads."""
        from physicalai_mujoco_plugin.viser_controls import ObjectPose  # noqa: PLC0415

        poses: dict[str, ObjectPose] = {}
        if data is not None:
            for joint, (qpos_addr, _) in self.free_addrs.items():
                values = [float(v) for v in data.qpos[qpos_addr : qpos_addr + 7]]
                poses[joint] = ObjectPose(position=tuple(values[:3]), wxyz=tuple(values[3:]))  # type: ignore[arg-type]
        with self._lock:
            self.poses = poses

    def randomize(self, model: object, data: object, rng: np.random.Generator) -> None:
        """Respawn the free objects in the spawn area, clear of the target and of each other."""
        import mujoco  # noqa: PLC0415

        if not self.block_addrs:
            return
        area = self.area
        if self.target_body_id is not None:
            target_xy = (float(data.xpos[self.target_body_id][0]), float(data.xpos[self.target_body_id][1]))
        else:
            target_xy = area.spawn_center
        positions = sample_object_positions(
            len(self.block_addrs),
            rng=rng,
            center=area.spawn_center,
            min_r=area.spawn_min_r,
            max_r=area.spawn_max_r,
            angle_half_deg=area.spawn_angle_half_deg,
            target_xy=target_xy,
            target_min_sep=area.target_min_sep,
            object_min_sep=area.block_min_sep,
        )
        # Keep the target fixed; only respawn free objects.
        for (qpos_addr, dof_addr), xy in zip(self.block_addrs, positions, strict=True):
            write_freejoint_qpos(data, qpos_addr, dof_addr, xy, yaw=float(rng.uniform(0.0, 2.0 * np.pi)))
        mujoco.mj_forward(model, data)

    def _write(self, data: object, joint: str, position: np.ndarray, wxyz: np.ndarray) -> None:
        qpos_addr, dof_addr = self.free_addrs[joint]
        data.qpos[qpos_addr : qpos_addr + 3] = position
        data.qpos[qpos_addr + 3 : qpos_addr + 7] = wxyz
        data.qvel[dof_addr : dof_addr + 6] = 0.0


__all__ = ["SceneObjects", "SpawnArea"]
