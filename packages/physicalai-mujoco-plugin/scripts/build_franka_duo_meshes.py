# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build the meshes of the Franka Duo's stand from Franka's ``franka_description``.

The ``franka_duo`` profile stands its two arms on Franka's own parts (``profiles/franka.py``):

- the FR3 Duo mount, ``accessories/fr3_duo_mount_v0_3``: the bracket (``fr3_duo_mount``) and its
  cover (``fr3_duo_cover``, 68 mm above the bracket's frame), in the frame of the mount's mounting
  point, 50 mm below and 37 mm behind the arm bases;
- the column, ``accessories/franka_spine_v0_1``, the mobile FR3 Duo's spine, in its own frame on the
  floor; its carriage holds the mounting point 266.711 mm in front of it, 100 mm up plus its travel.

This script writes them to ``urdf/franka_duo/assets/`` in MuJoCo's binary MSH format, like the
bucket_sort meshes (``build_bucket_sort_meshes.py``, whose writer and decimation it uses): one
visual mesh per source colour, decimated, in metres (``<part>_<black|grey|white>.msh``), and the
collision mesh of each part (``<part>_collision.msh``), vertices only, as MuJoCo collides the
convex hull of a mesh anyway. The visual meshes keep their frame; the column's comes in millimetres.

It needs trimesh, pycollada and pymeshlab, which the plugin does not depend on. Run it from the
package directory on a checkout of ``franka_description`` (Apache-2.0) at the commit below:

    git clone https://github.com/frankarobotics/franka_description ~/franka_description
    git -C ~/franka_description checkout 7aeeddc449edf8d62b594f9e36a81da53e7796f9
    uv run --no-project --with trimesh==4.8.3 --with pymeshlab==2025.7 --with pycollada --with numpy \
        --with scipy --with lxml --with networkx python scripts/build_franka_duo_meshes.py ~/franka_description
"""  # noqa: INP001

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import build_bucket_sort_meshes
import trimesh
from build_bucket_sort_meshes import _decimate, _write_msh

# The stand's parts are ten times the objects' size: let decimation move their bounds by up to 1 mm.
build_bucket_sort_meshes._BOUNDS_TOLERANCE = 1e-3  # noqa: SLF001

OUT = Path(__file__).resolve().parent.parent / "urdf" / "franka_duo" / "assets"
SOURCE_COMMIT = "7aeeddc449edf8d62b594f9e36a81da53e7796f9"


@dataclass(frozen=True)
class StandPart:
    """One part of the stand: its visual COLLADA file, its collision STL and the face budget per material."""

    name: str
    visual: str
    collision: str
    faces: int
    scale: float = 1.0
    """Visual source units to metres; the collision STLs are in metres."""


PARTS = (
    StandPart(
        "duo_mount",
        "meshes/accessories/fr3_duo_mount_v0_3/visual/fr3_duo_mount.dae",
        "meshes/accessories/fr3_duo_mount_v0_3/collision/fr3_duo_mount.stl",
        6000,
    ),
    StandPart(
        "duo_cover",
        "meshes/accessories/fr3_duo_mount_v0_3/visual/fr3_duo_cover.dae",
        "meshes/accessories/fr3_duo_mount_v0_3/collision/fr3_duo_cover.stl",
        3000,
    ),
    StandPart(
        "spine",
        "meshes/accessories/franka_spine_v0_1/visual/franka_spine.dae",
        "meshes/accessories/franka_spine_v0_1/collision/franka_spine.stl",
        2000,
        scale=0.001,
    ),
)

def _material_name(geometry: trimesh.Trimesh) -> str:
    """Name a geometry's material by its grey level: the sources are black, grey (the mount's metal) and white.

    Returns:
        ``black``, ``grey`` or ``white``.
    """
    material = getattr(geometry.visual, "material", None)
    colour = np.asarray(getattr(material, "main_color", (255, 255, 255, 255)), dtype=np.float64)[:3] / 255.0
    level = float(colour.mean())
    return "black" if level < 0.2 else "grey" if level < 0.7 else "white"  # noqa: PLR2004


def build_part(source: Path, part: StandPart) -> None:
    """Write a part's visual meshes, one per material, and its collision mesh."""
    scene = trimesh.load(source / part.visual, force="scene")
    by_material: dict[str, list[trimesh.Trimesh]] = {}
    for geometry in scene.dump():  # each node's geometry, placed by the scene graph
        mesh = trimesh.Trimesh(np.asarray(geometry.vertices) * part.scale, geometry.faces, process=True)
        by_material.setdefault(_material_name(geometry), []).append(mesh)
    total = sum(len(mesh.faces) for meshes in by_material.values() for mesh in meshes)
    for material, meshes in sorted(by_material.items()):
        mesh = trimesh.util.concatenate(meshes)
        budget = max(200, round(part.faces * len(mesh.faces) / total))
        small = _decimate(mesh, budget)
        vertices, faces = _write_msh(small, OUT / f"{part.name}_{material}.msh")
        print(f"{part.name}_{material}: {len(mesh.faces)} -> {faces} faces, {vertices} vertices")  # noqa: T201
    collision = trimesh.load(source / part.collision, force="mesh", process=True)
    vertices, _ = _write_msh(collision, OUT / f"{part.name}_collision.msh", faces=False)
    print(f"{part.name}_collision: {vertices} vertices, bounds {np.round(collision.bounds, 4).tolist()}")  # noqa: T201


def main() -> None:
    """Build every part of the stand from the franka_description checkout given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help=f"franka_description checkout at {SOURCE_COMMIT}")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for part in PARTS:
        build_part(args.source.expanduser(), part)


if __name__ == "__main__":
    main()
