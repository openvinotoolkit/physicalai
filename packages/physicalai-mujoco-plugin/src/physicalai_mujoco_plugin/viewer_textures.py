# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Show 2-D textures on box geoms in the browser viewer.

mjviser (0.0.14) maps textures only onto meshes and heightfields: a textured box, such as the
bucket_sort table top, shows its material's flat colour in the viewer while MuJoCo renders it
textured. This gives textured boxes per-face UVs in mjviser's meshes, tiled like MuJoCo tiles
them (``texrepeat``, per metre with ``texuniform``). It changes only what the viewer draws:
physics and MuJoCo's own renders, such as the robot cameras a policy sees, stay as they are.

Drop this once mjviser maps box textures itself. If mjviser's internals change, the patch is
skipped and boxes show their flat colour again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

_INSTALLED = False


def _box_texture_id(conv: Any, model: Any, geom_id: int) -> int:  # noqa: ANN401
    """Return the 2-D texture a box geom's material carries, or ``-1``."""
    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        return -1
    matid = int(model.geom_matid[geom_id])
    if matid < 0:
        return -1
    texid = int(conv._get_texture_id(model, matid))  # noqa: SLF001
    if texid < 0 or int(model.tex_type[texid]) != int(mujoco.mjtTexture.mjTEXTURE_2D):
        return -1
    return texid


def box_uvs(
    vertices: np.ndarray, normals: np.ndarray, half_size: np.ndarray, repeat: np.ndarray, *, uniform: bool
) -> np.ndarray:
    """Return per-vertex UVs that project each face of a box onto the plane it lies in.

    Args:
        vertices: ``(n, 3)`` vertex positions in the geom frame, three per face (unmerged).
        normals: ``(n, 3)`` the normal of each vertex's face.
        half_size: The box's half extents in metres.
        repeat: The material's ``texrepeat``.
        uniform: The material's ``texuniform``: repeat per metre instead of once per face.

    Returns:
        ``(n, 2)`` texture coordinates.
    """
    uv = np.zeros((len(vertices), 2))
    axes = np.argmax(np.abs(normals), axis=1)
    for axis, pair in ((0, [1, 2]), (1, [0, 2]), (2, [0, 1])):
        rows = axes == axis
        coords = vertices[rows][:, pair]
        if uniform:
            uv[rows] = coords * repeat
        else:
            uv[rows] = (coords / (2.0 * np.maximum(half_size[pair], 1e-9)) + 0.5) * repeat
    return uv


def install_box_textures() -> bool:
    """Patch mjviser once so textured boxes keep their texture in the viewer.

    Returns:
        Whether the patch is in place.
    """
    global _INSTALLED  # noqa: PLW0603
    if _INSTALLED:
        return True
    try:
        import mjviser.conversions as conv  # noqa: PLC0415
        import trimesh  # noqa: PLC0415

        create_primitive_mesh: Callable[[Any, int], Any] = conv.create_primitive_mesh
        get_geom_texture_id: Callable[[Any, int], int] = conv.get_geom_texture_id
        _ = conv._get_texture_id, conv._extract_texture_image  # noqa: SLF001
    except (ImportError, AttributeError) as exc:
        logger.debug("Box textures stay flat in the viewer: {}", exc)
        return False

    def texture_id(model: Any, geom_id: int) -> int:  # noqa: ANN401
        texid = _box_texture_id(conv, model, geom_id)
        return texid if texid >= 0 else get_geom_texture_id(model, geom_id)

    def primitive(model: Any, geom_id: int) -> Any:  # noqa: ANN401
        mesh = create_primitive_mesh(model, geom_id)
        texid = _box_texture_id(conv, model, geom_id)
        image = conv._extract_texture_image(model, texid) if texid >= 0 else None  # noqa: SLF001
        if image is None:
            return mesh
        matid = int(model.geom_matid[geom_id])
        mesh.unmerge_vertices()  # three vertices per face, so each face gets its own projection
        normals = np.repeat(np.asarray(mesh.face_normals), 3, axis=0)
        uv = box_uvs(
            np.asarray(mesh.vertices),
            normals,
            np.asarray(model.geom_size[geom_id], dtype=np.float64),
            np.asarray(model.mat_texrepeat[matid], dtype=np.float64),
            uniform=bool(model.mat_texuniform[matid]),
        )
        material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=np.asarray(model.mat_rgba[matid]),
            baseColorTexture=image,
            metallicFactor=0.0,
            roughnessFactor=0.95,
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        return mesh

    conv.create_primitive_mesh = primitive
    conv.get_geom_texture_id = texture_id
    _INSTALLED = True
    return True


__all__ = ["box_uvs", "install_box_textures"]
