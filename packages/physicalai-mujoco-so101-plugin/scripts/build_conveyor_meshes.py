# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build and export the conveyor_sort visual meshes with Blender.

Run headless from the package directory:

    blender --background --factory-startup --python scripts/build_conveyor_meshes.py

or send this file to an open Blender session (for example through the
BlenderMCP add-on) to watch the build. Either way the OBJ files are written to
urdf/scenes/conveyor_sort/assets/.

World coordinates match MuJoCo (Z up, metres). The belt runs along Y, centred on
x = BELT_X, flowing from +Y (entry hood) to -Y (drive end). The numbers here must
stay in sync with the collision primitives in conveyor_sort/scene.xml.
"""  # noqa: INP001

import os
from pathlib import Path

import bpy
import bmesh
from math import radians

BELT_X = 0.25
BELT_HALF_LEN = 0.40
BELT_HALF_W = 0.04
BELT_TOP = 0.060
RAIL_W = 0.012
RAIL_TOP = 0.070
RAIL_BOTTOM = 0.035
ROLLER_R = 0.012

# Fresh collection so re-runs replace the previous build.
name = "ConveyorSort"
if name in bpy.data.collections:
    col = bpy.data.collections[name]
    for obj in list(col.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
else:
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
for obj_name in ("Cube",):
    if obj_name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)


def material(mat_name, rgba, metallic=0.0, roughness=0.5):
    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    mat.diffuse_color = rgba
    return mat


ALU = material("alu", (0.72, 0.74, 0.77, 1), metallic=0.8, roughness=0.35)
STEEL = material("steel", (0.55, 0.56, 0.58, 1), metallic=0.9, roughness=0.25)
PAINT = material("hood_paint", (0.20, 0.33, 0.52, 1), roughness=0.45)
DARK = material("motor", (0.12, 0.12, 0.13, 1), roughness=0.6)
BELT = material("belt_rubber", (0.15, 0.16, 0.17, 1), roughness=0.9)
BIN = material("bin_plastic", (0.85, 0.85, 0.85, 1), roughness=0.55)


def box(obj_name, center, size, mat, bevel=0.0015, part=None):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
    obj = bpy.context.active_object
    obj.name = obj_name
    obj.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel > 0:
        mod = obj.modifiers.new("bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 2
    obj.data.materials.append(mat)
    obj["part"] = part or obj_name
    _link(obj)
    return obj


def cylinder_x(obj_name, center, radius, length, mat, verts=24, part=None):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=length, location=center,
                                        rotation=(0.0, radians(90), 0.0))
    obj = bpy.context.active_object
    obj.name = obj_name
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    mod = obj.modifiers.new("bevel", "BEVEL")
    mod.width = 0.001
    mod.segments = 1
    obj.data.materials.append(mat)
    obj["part"] = part or obj_name
    _link(obj)
    return obj


def _link(obj):
    for c in obj.users_collection:
        c.objects.unlink(obj)
    col.objects.link(obj)


# --- Frame: two extruded aluminium side rails with a slot, feet and cross bars ---
rail_len = 2 * BELT_HALF_LEN + 0.03
rail_h = RAIL_TOP - RAIL_BOTTOM
for side, sx in (("L", 1), ("R", -1)):
    x = BELT_X + sx * (BELT_HALF_W + 0.002 + RAIL_W / 2)
    box(f"rail_{side}", (x, 0.0, RAIL_BOTTOM + rail_h / 2), (RAIL_W, rail_len, rail_h), ALU, part="frame")
    # T-slot groove on the outer face, dark so it reads as a groove.
    box(f"rail_slot_{side}", (x + sx * RAIL_W / 2, 0.0, RAIL_BOTTOM + rail_h / 2),
        (0.002, rail_len - 0.01, 0.006), DARK, bevel=0.0, part="frame_dark")
    for fy in (-0.36, 0.0, 0.36):
        box(f"leg_{side}_{fy:+.2f}", (x, fy, RAIL_BOTTOM / 2), (0.012, 0.02, RAIL_BOTTOM), ALU, part="frame")
        box(f"foot_{side}_{fy:+.2f}", (x, fy, 0.002), (0.03, 0.03, 0.004), ALU, part="frame")
for fy in (-0.36, 0.0, 0.36):
    box(f"crossbar_{fy:+.2f}", (BELT_X, fy, RAIL_BOTTOM + 0.004), (2 * BELT_HALF_W + 0.004, 0.02, 0.008), ALU,
        part="frame")

# --- Rollers at both ends; the belt wraps around them ---
for tag, y in (("drive", -BELT_HALF_LEN - 0.005), ("idle", BELT_HALF_LEN + 0.005)):
    cylinder_x(f"roller_{tag}", (BELT_X, y, BELT_TOP - ROLLER_R), ROLLER_R, 2 * BELT_HALF_W, STEEL, part="rollers")
    cylinder_x(f"axle_{tag}", (BELT_X, y, BELT_TOP - ROLLER_R), 0.004, 2 * BELT_HALF_W + 2 * RAIL_W + 0.012, STEEL,
               verts=12, part="rollers")

# --- Drive motor + gearbox on the outer (+X) side of the drive end ---
motor_x = BELT_X + BELT_HALF_W + RAIL_W + 0.03
cylinder_x("motor_can", (motor_x + 0.012, -BELT_HALF_LEN - 0.005, 0.05), 0.02, 0.05, DARK, part="motor")
box("gearbox", (motor_x - 0.02, -BELT_HALF_LEN - 0.005, 0.05), (0.022, 0.04, 0.04), DARK, bevel=0.003, part="motor")

# --- Entry hood: sheet-metal tunnel over the first 12 cm, hides item spawning ---
hood_y0, hood_y1 = BELT_HALF_LEN - 0.10, BELT_HALF_LEN + 0.025
hood_len = hood_y1 - hood_y0
hood_cy = (hood_y0 + hood_y1) / 2
hood_top = 0.140
wall_t = 0.004
inner = BELT_HALF_W + 0.002 + RAIL_W
for side, sx in (("L", 1), ("R", -1)):
    box(f"hood_wall_{side}", (BELT_X + sx * (inner + wall_t / 2), hood_cy, (RAIL_BOTTOM + hood_top) / 2),
        (wall_t, hood_len, hood_top - RAIL_BOTTOM), PAINT, bevel=0.001, part="hood")
box("hood_roof", (BELT_X, hood_cy, hood_top + wall_t / 2), (2 * inner + 2 * wall_t + 0.006, hood_len, wall_t),
    PAINT, bevel=0.001, part="hood")
box("hood_back", (BELT_X, hood_y1 - wall_t / 2, (RAIL_TOP + hood_top) / 2),
    (2 * inner, wall_t, hood_top - RAIL_TOP), PAINT, bevel=0.001, part="hood")
# Rubber strip curtain at the tunnel mouth.
for i in range(6):
    w = 2 * BELT_HALF_W / 6
    box(f"curtain_{i}", (BELT_X - BELT_HALF_W + w * (i + 0.5), hood_y0 + 0.003, hood_top - 0.02),
        (w - 0.002, 0.002, 0.04), BELT, bevel=0.0, part="hood_dark")

# --- One bin shell, modelled at the origin; MuJoCo places four copies ---
BIN_HALF = 0.05
BIN_H = 0.045
BIN_T = 0.004
bin_loc = (0.0, -0.75, 0.0)  # off to the side in Blender so it does not overlap the belt
bx, by, _ = bin_loc
box("bin_floor", (bx, by, BIN_T / 2), (2 * BIN_HALF, 2 * BIN_HALF, BIN_T), BIN, bevel=0.001, part="bin")
for i, (dx, dy, sx_, sy_) in enumerate(((BIN_HALF - BIN_T / 2, 0, BIN_T, 2 * BIN_HALF),
                                        (-(BIN_HALF - BIN_T / 2), 0, BIN_T, 2 * BIN_HALF),
                                        (0, BIN_HALF - BIN_T / 2, 2 * BIN_HALF, BIN_T),
                                        (0, -(BIN_HALF - BIN_T / 2), 2 * BIN_HALF, BIN_T))):
    box(f"bin_wall_{i}", (bx + dx, by + dy, BIN_H / 2), (sx_, sy_, BIN_H), BIN, bevel=0.0012, part="bin")
    # Rolled rim along the top edge.
    rim_len = sy_ if sx_ == BIN_T else sx_
    rim = box(f"bin_rim_{i}", (bx + dx, by + dy, BIN_H), (sx_ + 0.003, sy_ + 0.003, 0.004), BIN, bevel=0.0015,
              part="bin")

# Viewport: solid shading with material colours so the build is readable live.
for area in ([] if bpy.app.background else bpy.context.screen.areas):
    if area.type == "VIEW_3D":
        space = area.spaces.active
        space.shading.type = "SOLID"
        space.shading.color_type = "MATERIAL"
        space.clip_start = 0.001
        region = next(r for r in area.regions if r.type == "WINDOW")
        with bpy.context.temp_override(area=area, region=region):
            bpy.ops.view3d.view_all()
print("built", len(col.objects), "objects")


# --- Export: one single-object OBJ per material group (MuJoCo keeps only the first object) ---
import bmesh

_here = Path(__file__).resolve().parent if "__file__" in globals() else None
out = os.environ.get("CONVEYOR_ASSET_DIR") or str(
    (_here.parent if _here else Path.cwd()) / "urdf" / "scenes" / "conveyor_sort" / "assets"
)
parts = sorted({o["part"] for o in col.objects})
bin_offset = bin_loc
# Parts with a 2D texture in scene.xml get box-projected UVs (the reject bin's stripes).
UV_PARTS = {"bin": 0.05}  # part -> metres per texture repeat
report = {}


def box_project_uvs(bm, tile):
    """UV each face from the two world axes most perpendicular to its normal."""
    uv_layer = bm.loops.layers.uv.new("UVMap")
    for face in bm.faces:
        n = face.normal
        axis = max(range(3), key=lambda i: abs(n[i]))
        u_i, v_i = [(1, 2), (0, 2), (0, 1)][axis]
        for loop in face.loops:
            co = loop.vert.co
            loop[uv_layer].uv = (co[u_i] / tile, co[v_i] / tile)


dg = bpy.context.evaluated_depsgraph_get()
for part in parts:
    objs = [o for o in col.objects if o.get("part") == part]
    # Bake each object's modifiers into one temporary mesh so the OBJ has a single object.
    bm = bmesh.new()
    for o in objs:
        me = bpy.data.meshes.new_from_object(o.evaluated_get(dg))
        me.transform(o.matrix_world)
        bm.from_mesh(me)
        bpy.data.meshes.remove(me)
    if part == "bin":
        bmesh.ops.translate(bm, verts=bm.verts, vec=(-bin_offset[0], -bin_offset[1], -bin_offset[2]))
    bm.normal_update()
    if part in UV_PARTS:
        box_project_uvs(bm, UV_PARTS[part])
    me = bpy.data.meshes.new(f"export_{part}")
    bm.to_mesh(me); bm.free()
    tmp = bpy.data.objects.new(f"export_{part}", me)
    bpy.context.scene.collection.objects.link(tmp)
    bpy.ops.object.select_all(action="DESELECT")
    tmp.select_set(True); bpy.context.view_layer.objects.active = tmp
    path = os.path.join(out, f"conveyor_{part}.obj" if part != "bin" else "bin.obj")
    bpy.ops.wm.obj_export(filepath=path, export_selected_objects=True, forward_axis="Y", up_axis="Z",
                          apply_modifiers=True, export_materials=False, export_uv=part in UV_PARTS, export_normals=True,
                          export_triangulated_mesh=True)
    bpy.data.objects.remove(tmp, do_unlink=True); bpy.data.meshes.remove(me)
    report[os.path.basename(path)] = (len(objs), os.path.getsize(path))
print(report)
