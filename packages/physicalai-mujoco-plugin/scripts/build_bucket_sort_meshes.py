# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build the bucket_sort scene meshes from the source CAD exports.

The sources are far too heavy for MuJoCo (14k to 55k faces each), mix millimetres and
metres, and keep their colours in ``.mtl`` files, which MuJoCo ignores. This script writes
light OBJ meshes to ``urdf/scenes/bucket_sort/assets/``:

- one visual mesh per source material (``<object>_<part>.obj``), decimated, in metres, with
  the object's origin at its bounding-box centre; ``scene.xml`` gives each part the colour of
  its material (printed below as ``rgba``);
- the CPU's six legs: the export holds one leg (``cpu - Part 2.obj``, its two pegs in the first of
  the body's six leg sockets), which is copied into the other five sockets;
- the resistor's four colour bands, cut out of its body mesh as separate meshes
  (``resistor_band<n>.obj``) that ``scene.xml`` colours;
- one convex collision hull per object (``<object>_hull.obj``), from the decimated parts;
- the bucket's visual mesh (``bucket.obj``), laid on its long side with its opening up (see
  :func:`build_bucket`), origin at the centre of its footprint on the table. Its collision is
  box primitives in scene.xml.

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
    copies: tuple[tuple[float, bool], ...] = ((0.0, False),)
    """Where the part appears, in the source frame: a shift along x in metres, and whether it is
    mirrored across the plane ``y = mirror_y``. One entry without either keeps it where it is."""
    mirror_y: float = 0.0


@dataclass(frozen=True)
class SmallObject:
    """A graspable object: its parts share the source frame and become one body."""

    name: str
    parts: tuple[Part, ...]
    scale: float = 1.0
    """Source units to metres."""
    bands: tuple[tuple[float, float], ...] = ()
    """Colour bands cut out of the first part: ``(y_low, y_high)`` in metres, in the centred frame."""


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
            # One leg; the body has sockets at x = 2, 23.5 and 45 mm on both long sides (y = -27.37
            # and +4.63 mm, mirror plane between them).
            Part(
                "legs",
                "cpu/cpu - Part 2.obj",
                None,
                400,
                copies=tuple((dx, mirror) for mirror in (False, True) for dx in (0.0, 0.0215, 0.043)),
                mirror_y=-0.011370,
            ),
        ),
    ),
    SmallObject(
        "resistor",
        (Part("body", "resistor parts - resistor body/resistor parts - resistor body.obj", None, 3000),),
        # Four 5.5 mm bands placed as in the reference photo of a carbon-film resistor: the first on
        # one end cap's shoulder, two on the neck, the tolerance band on the other cap (the caps
        # span |y| > 17 mm and are widest at |y| = 23 mm).
        bands=((-0.0234, -0.0179), (-0.0104, -0.0049), (0.0027, 0.0082), (0.0205, 0.0260)),
    ),
)

_BOUNDS_TOLERANCE = 5e-5
"""Largest change of a mesh's bounding box under decimation, in metres."""

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


def _place_copies(mesh: trimesh.Trimesh, part: Part) -> trimesh.Trimesh:
    """Return the part's copies (:attr:`Part.copies`) as one mesh.

    Returns:
        *mesh* itself for a part with one unmoved copy.
    """
    if part.copies == ((0.0, False),):
        return mesh
    copies = []
    for dx, mirror in part.copies:
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces)
        vertices[:, 0] += dx
        if mirror:
            vertices[:, 1] = 2.0 * part.mirror_y - vertices[:, 1]
            faces = faces[:, ::-1]  # a mirror turns the faces inside out
        copies.append(trimesh.Trimesh(vertices, faces, process=False))
    return trimesh.util.concatenate(copies)


def _cut_bands(mesh: trimesh.Trimesh, bands: tuple[tuple[float, float], ...]) -> tuple[trimesh.Trimesh, list]:
    """Cut *mesh* at planes across y into the parts outside the bands and one mesh per band.

    Both sides of each cut share the new vertices on the plane, so the pieces meet without gaps.

    Returns:
        ``(rest, bands)``: the mesh outside every band, and each band's mesh in order.
    """
    from trimesh.intersections import slice_mesh_plane  # noqa: PLC0415

    def between(lo: float | None, hi: float | None) -> trimesh.Trimesh:
        piece = mesh
        if lo is not None:
            piece = slice_mesh_plane(piece, plane_normal=[0.0, 1.0, 0.0], plane_origin=[0.0, lo, 0.0])
        if hi is not None:
            piece = slice_mesh_plane(piece, plane_normal=[0.0, -1.0, 0.0], plane_origin=[0.0, hi, 0.0])
        return piece

    edges = [None, *(edge for band in bands for edge in band), None]
    rest = trimesh.util.concatenate([between(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)])
    return rest, [between(lo, hi) for lo, hi in bands]


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
    grown = np.abs(small.bounds - mesh.bounds).max()
    if grown > _BOUNDS_TOLERANCE:
        msg = f"Decimation moved the mesh's bounds by {grown * 1000:.2f} mm"
        raise ValueError(msg)
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
    combined = trimesh.util.concatenate([_place_copies(mesh, part) for part, mesh, _ in loaded])
    lo, hi = combined.bounds
    centre = (lo + hi) / 2.0
    decimated = []
    for index, (part, mesh, rgba) in enumerate(loaded):
        small = _place_copies(_decimate(mesh, part.faces), part)
        small.apply_translation(-centre)
        decimated.append(small.copy())  # the hull takes the part where it is, without the lift
        small.apply_translation([0.0, 0.0, part.lift])
        name = f"{obj.name}_{part.name}"
        comment = f"{name}: {part.source}, generated by scripts/build_bucket_sort_meshes.py"
        if index == 0 and obj.bands:
            small, bands = _cut_bands(small, obj.bands)
            for number, band in enumerate(bands, start=1):
                _write_obj(band, OUT / f"{obj.name}_band{number}.obj", f"{obj.name}_band{number}: cut from {comment}")
            print(f"{obj.name}: cut {len(bands)} bands, {sum(len(b.faces) for b in bands)} faces")  # noqa: T201
        _write_obj(small, OUT / f"{name}.obj", comment)
        print(  # noqa: T201
            f"{name}: {len(mesh.faces)} x {len(part.copies)} -> {len(small.faces)} faces, "
            f"rgba {' '.join(f'{c:.3f}' for c in rgba)}"
        )
    hull = trimesh.util.concatenate(decimated).convex_hull
    _write_obj(hull, OUT / f"{obj.name}_hull.obj", f"{obj.name}_hull: convex hull of the {obj.name} visual parts")
    size = (hi - lo) * 1000.0
    print(  # noqa: T201
        f"{obj.name}_hull: {len(hull.faces)} faces, size {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm, "
        f"volume {hull.volume * 1e6:.1f} cm^3"
    )


def build_bucket(source_dir: Path) -> None:
    """Write the bucket lying on its long side with its opening up, origin at the centre of its footprint.

    The STL stands 202.5 mm tall on a 150 x 120 mm floor, with its large opening on the +y side
    (it wraps over the top, behind a sloped lip at the back). Lying on its back wall (the STL's
    y = 0 face, 150 x 202.5 mm) puts that opening up: 150 mm across (x), 202.5 mm long (y, the
    STL's z, the STL's floor at -y) and 120 mm deep (z, the STL's y).
    """
    mesh = trimesh.load(source_dir / BUCKET_SOURCE, process=True)
    mesh = trimesh.Trimesh(mesh.vertices * BUCKET_SCALE, mesh.faces, process=True)
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2.0, -lo[1], -(lo[2] + hi[2]) / 2.0])
    # (x, y, z) -> (-x, z, y): a proper rotation (determinant +1) that puts the STL's +y up.
    turn = np.eye(4)
    turn[:3, :3] = [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    mesh.apply_transform(turn)
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
