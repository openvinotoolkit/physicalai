"""Generate the Yahtzee scene texture assets."""  # noqa: INP001

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "urdf" / "scenes" / "yahtzee" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

# Wood texture from OpenGameArt.org: Tiny Texture Pack 2, wood_01-512x512.png
# by Screaming Brain Studios, CC0 license.
# Re-download from:
#   https://opengameart.org/sites/default/files/oga-textures/134697/wood_01-512x512.png


# ---------------------------------------------------------------------------
# Minimal PNG writer (no external dependencies)
# ---------------------------------------------------------------------------


def _write_png(path: Path, data: np.ndarray) -> None:
    h, w, _channels = data.shape
    raw = b""
    for y in range(h):
        raw += b"\x00"
        raw += data[y, :, :].tobytes()
    compressed = zlib.compress(raw)

    def chunk(ctype: bytes, cdata: bytes) -> bytes:
        c = ctype + cdata
        return struct.pack(">I", len(cdata)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    iend = b""
    path.write_bytes(sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", iend))


# ---------------------------------------------------------------------------
# Dot drawing helpers
# ---------------------------------------------------------------------------


def _draw_dot(arr: np.ndarray, cx: float, cy: float, r: float, color: tuple[int, ...]) -> None:
    h, w = arr.shape[:2]
    ys, xs = np.ogrid[:h, :w]
    cx_px, cy_px = int(cx * w), int(cy * h)
    r_px = max(1, int(r * min(w, h)))
    mask = (xs - cx_px) ** 2 + (ys - cy_px) ** 2 <= r_px**2
    arr[mask] = color


_DIE_FACES: list[list[tuple[float, float]]] = [
    [(0.5, 0.5)],  # 1: centre
    [(0.25, 0.75), (0.75, 0.25)],  # 2: diagonal
    [(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)],  # 3: diagonal
    [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)],  # 4: corners
    [(0.25, 0.25), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.75, 0.75)],  # 5: corners+centre
    [
        (0.25, 0.1667),
        (0.25, 0.5),
        (0.25, 0.8333),
        (0.75, 0.1667),
        (0.75, 0.5),
        (0.75, 0.8333),
    ],  # 6: two columns of three
]

CELL = 128
DOT_RADIUS = 0.12


def _make_face(dots: list[tuple[float, float]]) -> np.ndarray:
    img = np.full((CELL, CELL, 3), 255, dtype=np.uint8)
    for cx, cy in dots:
        _draw_dot(img, cx, cy, DOT_RADIUS, (0, 0, 0))
    return img


# ---------------------------------------------------------------------------
# Texture atlas — cross layout
#
#          [top:2]
#   [left:4] [front:1] [right:3] [back:6]
#         [bottom:5]
# ---------------------------------------------------------------------------

ATLAS_W, ATLAS_H = CELL * 4, CELL * 3
atlas = np.full((ATLAS_H, ATLAS_W, 3), 255, dtype=np.uint8)

# cell positions (col, row) in 4x3 grid
_cells = {
    1: (1, 1),  # front
    2: (1, 0),  # top
    3: (2, 1),  # right
    4: (0, 1),  # left
    5: (1, 2),  # bottom
    6: (3, 1),  # back
}

for face_id, dots in enumerate(_DIE_FACES, 1):
    face_img = _make_face(dots)
    col, row = _cells[face_id]
    y0, x0 = row * CELL, col * CELL
    atlas[y0 : y0 + CELL, x0 : x0 + CELL] = face_img

_write_png(OUT / "die_atlas.png", atlas)

# Also save individual face PNGs (useful for reference)
for face_id, dots in enumerate(_DIE_FACES, 1):
    _write_png(OUT / f"die_face_{face_id}.png", _make_face(dots))


# ---------------------------------------------------------------------------
# Cube mesh with UV coordinates
#
# Half-size = 0.01, each face maps to its atlas cell.
# UV coordinates are projected per-face (24 unshared vertices, 12 triangles).
# ---------------------------------------------------------------------------


S = 0.01  # half-size

# Each vertex has a position and a UV.  We use 4 unique vertices per face
# (24 total) so that each face gets the correct UV region of the atlas.
# Vertex order per face: CCW when viewed from outside (outward-facing normals).

_face_verts: dict[int, list[tuple[float, float, float, float, float]]] = {
    # face_id -> [(x, y, z, u, vt), ...]  4 corners
    # front (+z): face 1
    1: [
        (-S, -S, S, 0.25, 0.33),
        (S, -S, S, 0.50, 0.33),
        (S, S, S, 0.50, 0.67),
        (-S, S, S, 0.25, 0.67),
    ],
    # top (+y): face 2
    2: [
        (S, S, -S, 0.50, 0.00),
        (-S, S, -S, 0.25, 0.00),
        (-S, S, S, 0.25, 0.33),
        (S, S, S, 0.50, 0.33),
    ],
    # right (+x): face 3
    3: [
        (S, -S, -S, 0.50, 0.33),
        (S, S, -S, 0.75, 0.33),
        (S, S, S, 0.75, 0.67),
        (S, -S, S, 0.50, 0.67),
    ],
    # left (-x): face 4
    4: [
        (-S, S, -S, 0.25, 0.33),
        (-S, -S, -S, 0.00, 0.33),
        (-S, -S, S, 0.00, 0.67),
        (-S, S, S, 0.25, 0.67),
    ],
    # bottom (-y): face 5
    5: [
        (-S, -S, -S, 0.25, 0.67),
        (S, -S, -S, 0.50, 0.67),
        (S, -S, S, 0.50, 1.00),
        (-S, -S, S, 0.25, 1.00),
    ],
    # back (-z): face 6
    6: [
        (S, -S, -S, 0.75, 0.33),
        (-S, -S, -S, 1.00, 0.33),
        (-S, S, -S, 1.00, 0.67),
        (S, S, -S, 0.75, 0.67),
    ],
}

# Write vertices and texcoords, then faces.
# Each quad → 2 triangles: (0,1,2) and (2,3,0) — same winding for both.
lines: list[str] = ["# Cube mesh — UV mapped to die_atlas.png\n", "o die_cube\n"]
idx = 1
for face_id in range(1, 7):
    corners = _face_verts[face_id]
    for tri in [(0, 1, 2), (2, 3, 0)]:
        for ci in tri:
            x, y, z, u, vt = corners[ci]
            lines.extend((f"v {x} {y} {z}\n", f"vt {u} {1.0 - vt}\n"))
            idx += 2  # we just wrote 2 lines
lines.append("usemtl die_visual\ns off\n")

for fi in range(12):
    base = 1 + fi * 3
    lines.append(f"f {base}/{base} {base + 1}/{base + 1} {base + 2}/{base + 2}\n")

(OUT / "die_cube.obj").write_text("".join(lines))
