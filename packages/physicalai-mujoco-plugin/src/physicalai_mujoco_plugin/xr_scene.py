# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The simulation's 3D scene for the headset: every visible geom as triangles, and the bodies' poses.

The headset's page draws the scene itself. It fetches the geometry once (:meth:`SceneStream.blob`):
each visible geom as an indexed triangle mesh in its body's frame, with its colour, and where the
operator stands (``teleop.OperatorView``). After that, only the bodies' poses travel
(:meth:`SceneStream.poses`, about 1 KB per frame). A new model or view, after a scene switch or
edit, gets a new version number; the poses carry it, so the page knows to fetch the geometry again.

Shown are the geoms of groups 0 to 2 (what the viewer shows by default), and the see-through geoms
of group 3 whose material is a glass (its name says so), like the cell's walls: a browser can draw
them see-through. Robot models keep their collision meshes in group 3 too, often see-through.
Infinite planes (the floor) are left out: the headset shows the real floor.

The blob is a little-endian ``uint32`` header length, the JSON header (padded to 4 bytes), the
vertices (16 bytes each: ``float32`` x, y, z and ``int8`` normal x, y, z, pad), then the ``uint32``
triangle indices. The header lists every geom as ``[body, r, g, b, a, first index, index count]``,
and the view as ``{"eye": [x, y, z], "yaw": radians}``.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import gzip
import json
import threading
from typing import TYPE_CHECKING

import numpy as np

from physicalai_mujoco_plugin.studio_urdf import quat_to_matrix

if TYPE_CHECKING:
    import mujoco

    from physicalai_mujoco_plugin.teleop import OperatorView

SHOWN_GROUPS = (0, 1, 2)
GLASS_GROUP = 3
"""Group 3 geoms are shown only when their material's name has :data:`GLASS` in it."""
GLASS = "glass"
DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)
"""A geom's colour when none is given; a material's colour applies then."""
VERTEX = np.dtype([("position", "<f4", 3), ("normal", "i1", 4)])
ROUND_RINGS = 10
ROUND_SEGMENTS = 24

Mesh = tuple[np.ndarray, np.ndarray, np.ndarray]
"""Positions (N, 3), unit normals (N, 3) and triangles (M, 3) of one geom."""


def _box(half: np.ndarray) -> Mesh:
    positions, normals, triangles = [], [], []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        for sign in (-1.0, 1.0):
            base = len(positions)
            normal = np.zeros(3)
            normal[axis] = sign
            for cu, cv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                corner = np.zeros(3)
                corner[axis], corner[u], corner[v] = sign * half[axis], cu * half[u], cv * half[v]
                positions.append(corner)
                normals.append(normal)
            triangles += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return np.array(positions), np.array(normals), np.array(triangles)


def _grid(rows: int, columns: int) -> np.ndarray:
    # Two triangles per cell of a grid of rows x columns vertices, row by row.
    r, c = np.meshgrid(np.arange(rows - 1), np.arange(columns - 1), indexing="ij")
    a = (r * columns + c).ravel()
    b, d = a + 1, a + columns
    return np.concatenate([np.stack([a, d, b], axis=1), np.stack([b, d, d + 1], axis=1)])


def _round(radii: np.ndarray, half_length: float = 0.0) -> Mesh:
    # A sphere or an ellipsoid, or a capsule when its hemispheres stand half_length off the middle.
    top = np.linspace(0.0, np.pi / 2, ROUND_RINGS + 1)
    polar = np.concatenate([top, top + np.pi / 2])  # the equator twice: the capsule's straight part between
    offset = np.repeat([half_length, -half_length], ROUND_RINGS + 1)
    around = np.linspace(0.0, 2 * np.pi, ROUND_SEGMENTS + 1)
    t, p = np.meshgrid(polar, around, indexing="ij")
    unit = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], axis=-1).reshape(-1, 3)
    positions = unit * radii
    positions[:, 2] += np.repeat(offset, ROUND_SEGMENTS + 1)
    normals = unit / radii
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return positions, normals, _grid(len(polar), len(around))


def _cylinder(radius: float, half_height: float) -> Mesh:
    around = np.linspace(0.0, 2 * np.pi, ROUND_SEGMENTS + 1)
    ring = np.stack([np.cos(around), np.sin(around), np.zeros_like(around)], axis=1)
    side = np.concatenate([ring * radius + (0, 0, half_height), ring * radius - (0, 0, half_height)])
    positions, normals, triangles = [side], [np.concatenate([ring, ring])], [_grid(2, len(around))]
    count = len(side)
    for sign in (1.0, -1.0):
        cap = np.concatenate([[(0.0, 0.0, sign * half_height)], ring * radius + (0, 0, sign * half_height)])
        positions.append(cap)
        normals.append(np.tile((0.0, 0.0, sign), (len(cap), 1)))
        edge = np.arange(1, len(around))
        triangles.append(np.stack([np.zeros_like(edge), edge, edge + 1], axis=1) + count)
        count += len(cap)
    return np.concatenate(positions), np.concatenate(normals), np.concatenate(triangles)


def _plane(half: np.ndarray) -> Mesh:
    corners = np.array([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], dtype=float) * (half[0], half[1], 0)
    return corners, np.tile((0.0, 0.0, 1.0), (4, 1)), np.array([(0, 1, 2), (0, 2, 3)])


def _mesh(model: mujoco.MjModel, mesh: int) -> Mesh:
    # The mesh's triangles with its own normals: a vertex is split where its faces' normals differ.
    vertices = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
    faces = model.mesh_face[model.mesh_faceadr[mesh] : model.mesh_faceadr[mesh] + model.mesh_facenum[mesh]]
    count = int(model.mesh_normalnum[mesh])
    if count == 0:
        corners = vertices[faces]
        normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        return corners.reshape(-1, 3), np.repeat(normal, 3, axis=0), np.arange(3 * len(faces)).reshape(-1, 3)
    normals = model.mesh_normal[model.mesh_normaladr[mesh] : model.mesh_normaladr[mesh] + count]
    face_normals = model.mesh_facenormal[model.mesh_faceadr[mesh] : model.mesh_faceadr[mesh] + model.mesh_facenum[mesh]]
    pairs, triangles = np.unique(faces.astype(np.int64) * count + face_normals, return_inverse=True)
    return vertices[pairs // count], normals[pairs % count], triangles.reshape(-1, 3)


def geom_color(model: mujoco.MjModel, geom: int) -> np.ndarray:
    """Return the geom's colour: its own, or its material's when it has none of its own (as MuJoCo draws it).

    Returns:
        RGBA in 0 to 1.
    """
    rgba = model.geom_rgba[geom]
    material = model.geom_matid[geom]
    if material >= 0 and np.allclose(rgba, DEFAULT_RGBA):
        return model.mat_rgba[material]
    return rgba


def _geom_mesh(model: mujoco.MjModel, geom: int) -> Mesh | None:
    import mujoco  # noqa: PLC0415

    kind, size = int(model.geom_type[geom]), model.geom_size[geom]
    shapes = {
        mujoco.mjtGeom.mjGEOM_BOX: lambda: _box(size),
        mujoco.mjtGeom.mjGEOM_SPHERE: lambda: _round(np.full(3, size[0])),
        mujoco.mjtGeom.mjGEOM_ELLIPSOID: lambda: _round(size),
        mujoco.mjtGeom.mjGEOM_CAPSULE: lambda: _round(np.full(3, size[0]), size[1]),
        mujoco.mjtGeom.mjGEOM_CYLINDER: lambda: _cylinder(size[0], size[1]),
        mujoco.mjtGeom.mjGEOM_MESH: lambda: _mesh(model, int(model.geom_dataid[geom])),
    }
    if kind == mujoco.mjtGeom.mjGEOM_PLANE:
        return _plane(size) if size[0] > 0 and size[1] > 0 else None  # an infinite plane is the floor
    build = shapes.get(kind)  # height fields and SDFs are not drawn
    return build() if build is not None else None


def shown(model: mujoco.MjModel, geom: int) -> bool:
    """Return whether the headset draws the geom."""
    alpha = geom_color(model, geom)[3]
    group = int(model.geom_group[geom])
    if alpha <= 0:
        return False
    if group in SHOWN_GROUPS:
        return True
    material = int(model.geom_matid[geom])
    return group == GLASS_GROUP and material >= 0 and GLASS in model.material(material).name.lower()


def scene_blob(model: mujoco.MjModel, view: OperatorView, version: int) -> bytes:
    """Return the scene's geometry in the page's format (see the module's description).

    Returns:
        The uncompressed blob.
    """
    vertices, indices, geoms = [], [], []
    vertex_count = index_count = 0
    for geom in range(model.ngeom):
        if not shown(model, geom):
            continue
        mesh = _geom_mesh(model, geom)
        if mesh is None:
            continue
        positions, normals, triangles = mesh
        rotation = quat_to_matrix(model.geom_quat[geom])
        block = np.zeros(len(positions), dtype=VERTEX)
        block["position"] = positions @ rotation.T + model.geom_pos[geom]
        block["normal"][:, :3] = np.round(normals @ rotation.T * 127)
        vertices.append(block)
        indices.append(triangles.astype(np.uint32).ravel() + vertex_count)
        color = [round(float(c), 4) for c in geom_color(model, geom)]
        geoms.append([int(model.geom_bodyid[geom]), *color, index_count, int(triangles.size)])
        vertex_count += len(positions)
        index_count += triangles.size
    header = json.dumps({
        "version": version,
        "bodies": int(model.nbody),
        "vertices": vertex_count,
        "indices": index_count,
        "geoms": geoms,
        "view": {"eye": list(view.eye), "yaw": view.yaw},
    }).encode()
    header += b" " * (-len(header) % 4)
    parts = [np.uint32(len(header)).tobytes(), header]
    parts += [block.tobytes() for block in vertices] + [block.tobytes() for block in indices]
    return b"".join(parts)


class SceneStream:
    """The newest model's geometry and the newest body poses, from the simulation thread to the headset's."""

    def __init__(self) -> None:
        """Start without a model: the first :meth:`update` brings it."""
        self._lock = threading.Lock()
        self._model: mujoco.MjModel | None = None
        self._view: OperatorView | None = None
        self._version = 0
        self._blob: tuple[int, bytes] | None = None
        self._poses: bytes | None = None

    def update(self, model: mujoco.MjModel, data: mujoco.MjData, view: OperatorView) -> None:
        """Keep the bodies' poses after a step; a new model or view gets a new version."""
        if model is not self._model or view != self._view:
            with self._lock:
                self._model, self._view = model, view
                self._version += 1
        poses = np.empty((model.nbody, 7), dtype=np.float32)
        poses[:, :3] = data.xpos
        poses[:, 3:] = data.xquat
        self._poses = np.float32(self._version).tobytes() + poses.tobytes()

    def poses(self) -> bytes | None:
        """Return the newest poses: ``float32`` version, then x, y, z, qw, qx, qy, qz per body."""
        return self._poses

    def blob(self) -> bytes | None:
        """Return the newest model's geometry, gzip-compressed (built on first request), or ``None`` before a model."""
        with self._lock:
            model, view, version = self._model, self._view, self._version
            if model is None or view is None:
                return None
            if self._blob is not None and self._blob[0] == version:
                return self._blob[1]
        blob = gzip.compress(scene_blob(model, view, version), compresslevel=5)
        with self._lock:
            if self._version == version:
                self._blob = (version, blob)
        return blob


__all__ = ["SceneStream", "geom_color", "scene_blob", "shown"]
