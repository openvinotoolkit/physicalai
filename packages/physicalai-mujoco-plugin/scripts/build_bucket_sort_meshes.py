# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build the bucket_sort scene meshes from the source CAD exports.

The sources are far too heavy for MuJoCo (14k to 55k faces each), mix millimetres and
metres, and keep their colours in ``.mtl`` files, which MuJoCo ignores. This script writes
light OBJ meshes to ``urdf/scenes/bucket_sort/assets/``:

- one visual mesh per source material (``<object>_<part>.obj``), decimated, in metres, with
  the object's origin at its bounding-box centre; ``scene.xml`` gives each part the colour of
  its material (printed below as ``rgba``);
- one convex collision hull per object (``<object>_hull.obj``), from the decimated parts;
- the bucket's visual mesh (``bucket.obj``), turned so its open side faces +x and its origin
  is the centre of its footprint on the floor. Its collision is box primitives in scene.xml.

It needs trimesh and pymeshlab, which the plugin does not depend on. Run it from
the package directory with the sources in one folder:

    uv run --no-project --with trimesh==4.8.3 --with pymeshlab==2025.7 --with numpy --with scipy \
        python scripts/build_bucket_sort_meshes.py ~/Downloads/bucket_demo

The source folder holds ``BUCKET-150p.stl``, ``9V-battery/9V-battery.obj``,
``cpu/cpu - Part 1.obj``, ``cpu/cpu - Part 2.obj`` and
``resistor parts - resistor body/resistor parts - resistor body.obj``, each OBJ with its MTL.
"""  # noqa: INP001

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymeshlab
import trimesh

OUT = Path(__file__).resolve().parent.parent / "urdf" / "scenes" / "bucket_sort" / "assets"


@dataclass(frozen=True)
class Part:
    """One visual part of an object: the faces of one source material."""

    name: str
    """Suffix of the output mesh, ``<object>_<name>.obj``."""
    source: str
    """Source file, relative to the source folder."""
    material: str | None
    """Source material whose faces this part takes; ``None`` takes the whole file."""
    faces: int
    """Target face count after decimation; at least the source's count keeps it as it is."""
    lift: float = 0.0
    """Metres to move the part up, to keep a decal off a coplanar surface below it."""


@dataclass(frozen=True)
class SmallObject:
    """A graspable object: its parts share the source frame and become one body."""

    name: str
    parts: tuple[Part, ...]
    scale: float = 1.0
    """Source units to metres."""


_GREY = "0.647059_0.647059_0.647059_0.000000_0.000000"
_DARK = "0.262745_0.282353_0.301961_0.000000_0.000000"
_GOLD = "1.000000_0.756863_0.054902_0.000000_0.000000"
_WHITE = "1.000000_1.000000_1.000000_0.000000_0.000000"

OBJECTS = (
    SmallObject(
        "battery",
        (
            Part("metal", "9V-battery/9V-battery.obj", _GREY, 2500),
            Part("label", "9V-battery/9V-battery.obj", _DARK, 1200),
            Part("band", "9V-battery/9V-battery.obj", _GOLD, 800),
        ),
    ),
    SmallObject(
        "cpu",
        (
            Part("body", "cpu/cpu - Part 1.obj", _DARK, 2500),
            # Printed marking on the lid, coplanar with the lid's top: lifted 0.2 mm against z-fighting.
            Part("marking", "cpu/cpu - Part 1.obj", _WHITE, 1800, lift=0.0002),
            Part("lever", "cpu/cpu - Part 2.obj", None, 400),
        ),
    ),
    SmallObject(
        "resistor",
        (Part("body", "resistor parts - resistor body/resistor parts - resistor body.obj", None, 3000),),
    ),
)

BUCKET_SOURCE = "BUCKET-150p.stl"
BUCKET_FACES = 5000
BUCKET_SCALE = 0.001
"""The bucket STL is in millimetres."""


def _load_parts(source_dir: Path, obj: SmallObject) -> list[tuple[Part, trimesh.Trimesh, np.ndarray]]:
    """Load each part's faces in the shared source frame, in metres.

    Returns:
        ``(part, mesh, rgba)`` per part, rgba from the material's diffuse colour.
    """
    loaded = []
    for part in obj.parts:
        scene = trimesh.load(source_dir / part.source, group_material=True, process=False)
        if isinstance(scene, trimesh.Trimesh):
            meshes = {"": scene}
        else:
            meshes = dict(scene.geometry)
        if part.material is None:
            if len(meshes) != 1:
                msg = f"{part.source} has {len(meshes)} materials; name the one {obj.name}_{part.name} takes"
                raise ValueError(msg)
            source = next(iter(meshes.values()))
        else:
            source = meshes[part.material]
        material = getattr(source.visual, "material", None)
        diffuse = np.asarray(getattr(material, "diffuse", [255, 255, 255, 255]), dtype=np.float64) / 255.0
        mesh = trimesh.Trimesh(source.vertices * obj.scale, source.faces, process=True)
        loaded.append((part, mesh, diffuse))
    return loaded


def _decimate(mesh: trimesh.Trimesh, faces: int) -> trimesh.Trimesh:
    """Reduce *mesh* to about *faces* faces with MeshLab's quadric edge collapse, keeping its open boundaries.

    The mesh is decimated in millimetres, where the default quality thresholds suit these parts.

    Returns:
        The decimated mesh; *mesh* itself when it already has no more than *faces*.
    """
    if len(mesh.faces) <= faces:
        return mesh
    meshes = pymeshlab.MeshSet()
    meshes.add_mesh(pymeshlab.Mesh(np.asarray(mesh.vertices) * 1000.0, np.asarray(mesh.faces)))
    meshes.meshing_decimation_quadric_edge_collapse(
        targetfacenum=faces, preserveboundary=True, preservenormal=True, planarquadric=True
    )
    out = meshes.current_mesh()
    small = trimesh.Trimesh(out.vertex_matrix() / 1000.0, out.face_matrix(), process=True)
    small.remove_unreferenced_vertices()
    return small


def _write_obj(mesh: trimesh.Trimesh, path: Path, comment: str) -> None:
    """Write vertices and faces only, at 0.01 mm precision; MuJoCo computes the normals."""
    lines = [f"# {line}" for line in comment.splitlines()]
    lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in mesh.vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in mesh.faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_object(source_dir: Path, obj: SmallObject) -> None:
    """Write an object's visual parts and its convex collision hull, centred on its bounding box."""
    loaded = _load_parts(source_dir, obj)
    combined = trimesh.util.concatenate([mesh for _, mesh, _ in loaded])
    lo, hi = combined.bounds
    centre = (lo + hi) / 2.0
    decimated = []
    for part, mesh, rgba in loaded:
        small = _decimate(mesh, part.faces)
        small.apply_translation(-centre)
        decimated.append(small.copy())  # the hull takes the part where it is, without the lift
        small.apply_translation([0.0, 0.0, part.lift])
        name = f"{obj.name}_{part.name}"
        _write_obj(small, OUT / f"{name}.obj", f"{name}: {part.source}, generated by scripts/build_bucket_sort_meshes.py")
        print(  # noqa: T201
            f"{name}: {len(mesh.faces)} -> {len(small.faces)} faces, rgba {' '.join(f'{c:.3f}' for c in rgba)}"
        )
    hull = trimesh.util.concatenate(decimated).convex_hull
    _write_obj(hull, OUT / f"{obj.name}_hull.obj", f"{obj.name}_hull: convex hull of the {obj.name} visual parts")
    size = (hi - lo) * 1000.0
    print(  # noqa: T201
        f"{obj.name}_hull: {len(hull.faces)} faces, size {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm, "
        f"volume {hull.volume * 1e6:.1f} cm^3"
    )


def build_bucket(source_dir: Path) -> None:
    """Write the bucket: open side toward +x, origin at the centre of its footprint on the floor."""
    mesh = trimesh.load(source_dir / BUCKET_SOURCE, process=True)
    mesh = trimesh.Trimesh(mesh.vertices * BUCKET_SCALE, mesh.faces, process=True)
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]])
    # The source's open side faces +y; turn -90 degrees about z so it faces +x.
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2.0, [0.0, 0.0, 1.0]))
    small = _decimate(mesh, BUCKET_FACES)
    _write_obj(small, OUT / "bucket.obj", f"bucket: {BUCKET_SOURCE}, generated by scripts/build_bucket_sort_meshes.py")
    size = (small.bounds[1] - small.bounds[0]) * 1000.0
    print(  # noqa: T201
        f"bucket: {len(mesh.faces)} -> {len(small.faces)} faces, size {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm"
    )


def main() -> None:
    """Build every mesh of the scene from the source folder given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_dir", type=Path, help="Folder with the bucket STL and the object OBJ folders")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    build_bucket(args.source_dir.expanduser())
    for obj in OBJECTS:
        build_object(args.source_dir.expanduser(), obj)


if __name__ == "__main__":
    main()
