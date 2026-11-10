# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Both selectable bucket-sort overview poses use nominal D435 RGB optics."""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import itertools
import json
from pathlib import Path
from xml.etree import ElementTree

import mujoco
import numpy as np
import pytest
from physicalai_mujoco_plugin.bucket_sort import BUCKET_COLORS, OBJECTS, TABLE_X, TABLE_Y
from physicalai_mujoco_plugin.compose import CameraAim, OverviewStyle, reaim_overview_rig
from physicalai_mujoco_plugin.http_server import SetOverviewCommand
from physicalai_mujoco_plugin.profiles import get_profile
from physicalai_mujoco_plugin.robot import MuJoCoRobot
from physicalai_mujoco_plugin.scene_registry import get_scene
from physicalai_mujoco_plugin.sim import Sim, load_sim

POSES: dict[OverviewStyle, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "shoulder": ((-0.17, 0.68, 0.90), (0.45, 0.0, 0.05)),
    "front": ((0.83, -0.80, 0.90), (0.45, -0.20, 0.05)),
}


def _load(style: OverviewStyle) -> Sim:
    scene = get_scene("bucket_sort")
    return load_sim(
        scene.scene_xml_path,
        scene,
        get_profile("franka_fr3"),
        unit="degrees",
        torque_mode="pd",
        rng=np.random.default_rng(0),
        reseed=lambda: None,
        arms=1,
        overview=style,
    )


def _visible_geometry(model: mujoco.MjModel, data: mujoco.MjData, *, task_only: bool = False) -> np.ndarray:
    """World vertices/bounds; the short-pole views promise task coverage, not whole-arm coverage."""
    corners = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    bodies = {model.body(name).id for name in (*OBJECTS, *(f"bucket_{color}" for color in BUCKET_COLORS))}
    points = []
    for geom in range(model.ngeom):
        if task_only and model.geom_bodyid[geom] not in bodies:
            continue
        if model.geom_bodyid[geom] == 0 or model.geom_rgba[geom, 3] == 0:
            continue
        rotation = data.geom_xmat[geom].reshape(3, 3)
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            local = model.mesh_vert[start : start + count]
        else:
            local = model.geom_aabb[geom, :3] + corners * model.geom_aabb[geom, 3:]
        points.append(local @ rotation.T + data.geom_xpos[geom])
    return np.concatenate(points)


@pytest.mark.parametrize("style", POSES)
def test_both_styles_match_the_documented_mount_and_rgb_optics(style: OverviewStyle) -> None:
    sim = _load(style)
    model, data = sim.model, sim.data
    camera = model.camera("overview").id
    eye, target = map(np.asarray, POSES[style])
    forward = (target - eye) / np.linalg.norm(target - eye)
    rotation = data.cam_xmat[camera].reshape(3, 3)

    np.testing.assert_allclose(data.cam_xpos[camera], eye, atol=1e-9)
    np.testing.assert_allclose(-rotation[:, 2], forward, atol=1e-9)
    assert abs(rotation[2, 0]) < 1e-9, "both images are level"
    assert model.cam_fovy[camera] == pytest.approx(42.0), "D435 RGB, not depth FOV"
    assert get_scene("bucket_sort").overview_styles == ("shoulder", "front")
    # Off-axis poles still use a right-handed, unmirrored camera: right = forward cross world-up.
    right = np.cross(forward, (0.0, 0.0, 1.0))
    np.testing.assert_allclose(rotation[:, 0], right / np.linalg.norm(right), atol=1e-9)


@pytest.mark.parametrize("style", POSES)
def test_mounts_use_table_edges_and_realistic_pole_heights(style: OverviewStyle) -> None:
    eye, _ = POSES[style]
    assert eye[0] == (TABLE_X[0] if style == "shoulder" else TABLE_X[1])
    assert TABLE_Y[0] < eye[1] < TABLE_Y[1]
    assert 0.8 <= eye[2] <= 1.0


@pytest.mark.parametrize("style", POSES)
def test_schematic_metadata_matches_the_running_camera(style: OverviewStyle) -> None:
    diagram = Path(__file__).parents[1] / "docs" / "bucket-sort-camera-setup.svg"
    metadata = ElementTree.parse(diagram).getroot().find("{http://www.w3.org/2000/svg}metadata")
    assert metadata is not None and metadata.text is not None
    setup = json.loads(metadata.text)
    np.testing.assert_allclose(setup["cameras"][style]["eye_m"], POSES[style][0])
    np.testing.assert_allclose(setup["cameras"][style]["aim_m"], POSES[style][1])
    assert setup["table_x_m"] == list(TABLE_X)
    assert setup["table_y_m"] == list(TABLE_Y)


@pytest.mark.parametrize("style", POSES)
@pytest.mark.parametrize("width", [640, 848])
@pytest.mark.parametrize("pole_height", [0.8, 0.9, 1.0])
def test_both_styles_frame_complete_task_geometry_across_reset_layouts(
    style: OverviewStyle, width: int, pole_height: float
) -> None:
    """Project complete bucket/object meshes, not just centres; upper-arm cropping is intentional."""
    sim = _load(style)
    model, data = sim.model, sim.data
    eye, aim = POSES[style]
    reaim_overview_rig(model, data, CameraAim(eye=(*eye[:2], pole_height), target=aim))
    mujoco.mj_forward(model, data)
    camera = model.camera("overview").id
    focal = 240 / np.tan(np.radians(model.cam_fovy[camera]) / 2)
    for seed in (0, 1, 7, 42, 99):
        assert sim.on_reset is not None
        sim.on_reset(model, data, np.random.default_rng(seed))
        local = (_visible_geometry(model, data, task_only=True) - data.cam_xpos[camera]) @ data.cam_xmat[camera].reshape(3, 3)
        assert np.all(local[:, 2] < 0)
        u = width / 2 + focal * local[:, 0] / -local[:, 2]
        v = 240 - focal * local[:, 1] / -local[:, 2]
        assert 8 <= u.min() and u.max() <= width - 8, (style, width, seed, u.min(), u.max())
        assert 8 <= v.min() and v.max() <= 480 - 8, (style, seed, v.min(), v.max())


def test_user_can_switch_both_ways_without_resetting_the_bucket_sort_task() -> None:
    robot = MuJoCoRobot("franka_fr3", scene="bucket_sort", substeps=1, cameras=[], seed=7)
    robot.connect()
    try:
        robot.get_observation()
        task = robot._sim.on_reset  # noqa: SLF001
        instruction, layout, count = task.instruction, task.layout, task.reset_count
        joints = robot._sim.data.qpos.copy()  # noqa: SLF001
        for style in ("front", "shoulder", "front"):
            robot._commands.put(SetOverviewCommand(style=style))  # noqa: SLF001
            robot.get_observation()
            sim = robot._sim  # noqa: SLF001
            camera = sim.model.camera("overview").id
            np.testing.assert_allclose(sim.data.cam_xpos[camera], POSES[style][0], atol=1e-9)
            assert sim.model.cam_fovy[camera] == pytest.approx(42.0)
            assert robot._panel_state().overview_style == style  # noqa: SLF001
            assert robot._http_status()["overview_styles"] == ["shoulder", "front"]  # noqa: SLF001
            assert (task.instruction, task.layout, task.reset_count) == (instruction, layout, count)
        # Only normal physics steps elapsed; no reset or pose change to implement a POV switch.
        np.testing.assert_allclose(robot._sim.data.qpos, joints, atol=1e-3)  # noqa: SLF001
    finally:
        robot.disconnect()
