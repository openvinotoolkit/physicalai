# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""A see-through copy of every robot at the pose its servos are told to reach, for the viser viewer.

Each position servo that drives a hinge or slide joint directly holds the joint at ``ctrl / gear``;
the ghost shows every body those joints move at that pose, so a target the arm has not reached yet,
or cannot reach (a collision, a weak joint), shows as a gap between the arm and its ghost. Joints
without such a servo (coupled fingers, tendons, free bodies) keep their current position.
"""

from __future__ import annotations

from math import isclose
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    import mujoco

GHOST_RGB = (60, 210, 255)
GHOST_OPACITY = 0.2
_VISIBLE_GROUPS = (0, 1, 2)
"""The geom groups the viewer shows by default; group 3 holds collision hulls."""


def servo_targets(model: mujoco.MjModel) -> list[tuple[int, int, float, int]]:
    """Position servos that drive a hinge or slide joint directly.

    Returns:
        One ``(actuator id, joint qpos address, 1 / gear, joint id)`` per servo.
    """
    import mujoco  # noqa: PLC0415

    rows = []
    for actuator in range(model.nu):
        if int(model.actuator_trntype[actuator]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        if int(model.actuator_biastype[actuator]) != int(mujoco.mjtBias.mjBIAS_AFFINE):
            continue
        kp = float(model.actuator_gainprm[actuator, 0])
        if kp <= 0.0 or not isclose(float(model.actuator_biasprm[actuator, 1]), -kp):
            continue  # not a position servo (kp * (ctrl - length))
        joint = int(model.actuator_trnid[actuator, 0])
        if int(model.jnt_type[joint]) not in {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}:
            continue
        gear = float(model.actuator_gear[actuator, 0])
        if gear == 0.0:  # noqa: RUF069 - exactly zero means the servo cannot move the joint
            continue
        rows.append((actuator, int(model.jnt_qposadr[joint]), 1.0 / gear, joint))
    return rows


def moved_bodies(model: mujoco.MjModel, joints: set[int]) -> list[int]:
    """Bodies that move with any of *joints*: the joints' bodies and everything below them.

    Returns:
        Body ids, in model order.
    """
    driven = {int(model.jnt_bodyid[joint]) for joint in joints}
    bodies = []
    for body in range(1, model.nbody):
        ancestor = body
        while ancestor > 0 and ancestor not in driven:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor > 0:
            bodies.append(body)
    return bodies


class TargetGhost:
    """Translucent meshes of the servo-driven bodies, posed at their servos' targets."""

    def __init__(self, server: object, model: mujoco.MjModel, driven_joints: Iterable[int] = ()) -> None:
        """Add the ghost's meshes to *server*'s scene, at the world origin until the first update.

        Args:
            server: The viser server.
            model: The simulation's model.
            driven_joints: Joints driven otherwise than by a position servo (software PD, velocity
                actuators), whose targets :meth:`update` receives.
        """
        import mujoco  # noqa: PLC0415
        from mjviser.conversions import merge_geoms  # noqa: PLC0415

        self._server = server
        self._model = model
        self._data = mujoco.MjData(model)
        self._servos = servo_targets(model)
        self._handles: list[tuple[int, object]] = []
        for body in moved_bodies(model, {row[3] for row in self._servos} | set(driven_joints)):
            geoms = [
                geom
                for geom in range(model.ngeom)
                if model.geom_bodyid[geom] == body
                and model.geom_group[geom] in _VISIBLE_GROUPS
                and model.geom_rgba[geom, 3] > 0.0
            ]
            if not geoms:
                continue
            mesh = merge_geoms(model, geoms)
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body{body}"
            handle = server.scene.add_mesh_simple(  # type: ignore[attr-defined]
                f"/target_ghost/{name}",
                mesh.vertices,
                mesh.faces,
                color=GHOST_RGB,
                opacity=GHOST_OPACITY,
                cast_shadow=False,
                receive_shadow=False,
            )
            self._handles.append((body, handle))
        self.visible = True
        self._shown = True

    @property
    def bodies(self) -> list[int]:
        """The bodies the ghost shows."""
        return [body for body, _ in self._handles]

    def pose(self, data: mujoco.MjData, driven: Iterable[tuple[int, float]] = ()) -> mujoco.MjData:
        """Pose the ghost's own data at the servo targets in *data*, and at *driven* joint targets.

        Args:
            data: The simulation's data.
            driven: ``(joint id, target)`` of joints driven otherwise than by a position servo
                (software PD, velocity actuators).

        Returns:
            The ghost's data, with ``xpos`` and ``xquat`` at the targets.
        """
        import mujoco  # noqa: PLC0415

        ghost = self._data
        ghost.qpos[:] = data.qpos
        for actuator, qpos_adr, per_gear, _joint in self._servos:
            ghost.qpos[qpos_adr] = data.ctrl[actuator] * per_gear
        for joint, target in driven:
            ghost.qpos[self._model.jnt_qposadr[joint]] = target
        mujoco.mj_kinematics(self._model, ghost)
        return ghost

    def update(self, data: mujoco.MjData, driven: Iterable[tuple[int, float]] | None = ()) -> None:
        """Move the ghost to the targets in *data* and *driven* (sim thread); ``None`` hides it.

        ``driven`` is ``None`` while the targets mean nothing, during a kinematic replay.
        """
        show = self.visible and driven is not None
        if show != self._shown:
            self._shown = show
            for _body, handle in self._handles:
                handle.visible = show  # type: ignore[attr-defined]
        if not self._handles or not show:
            return
        ghost = self.pose(data, driven or ())
        with self._server.atomic():  # type: ignore[attr-defined]
            for body, handle in self._handles:
                handle.position = np.asarray(ghost.xpos[body], dtype=np.float64)  # type: ignore[attr-defined]
                handle.wxyz = np.asarray(ghost.xquat[body], dtype=np.float64)  # type: ignore[attr-defined]

    def set_visible(self, visible: bool) -> None:  # noqa: FBT001 - a GUI checkbox's value
        """Show or hide the ghost."""
        self.visible = visible
        self._shown = visible
        for _body, handle in self._handles:
            handle.visible = visible  # type: ignore[attr-defined]


__all__ = ["GHOST_OPACITY", "GHOST_RGB", "TargetGhost", "moved_bodies", "servo_targets"]
