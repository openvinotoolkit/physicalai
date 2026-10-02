"""Conveyor belt, item feed and scoring in the bundled conveyor_sort model (real MuJoCo, no display)."""

from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import pytest

from physicalai_mujoco_so101_plugin import http_server
from physicalai_mujoco_so101_plugin.conveyor import (
    BELT_TILE,
    MAX_BELT_SPEED,
    ConveyorSort,
    pool_item_names,
)
from physicalai_mujoco_so101_plugin.http_server import SetAutoResetCommand, SetBeltSpeedCommand
from physicalai_mujoco_so101_plugin.mujoco_robot import MuJoCoSO101
from physicalai_mujoco_so101_plugin.scene_registry import get_scene
from physicalai_mujoco_so101_plugin.viser_controls import _episode_markdown

SUBSTEPS = 10


@pytest.fixture
def sim() -> tuple[object, object, ConveyorSort]:
    model = get_scene("conveyor_sort").load_model()
    data = mujoco.MjData(model)
    conveyor = ConveyorSort.maybe_create(model, rng=np.random.default_rng(0), belt_speed=0.05)
    assert conveyor is not None
    conveyor.reset_items(model, data)
    return model, data, conveyor


def run(model: object, data: object, conveyor: ConveyorSort, seconds: float) -> None:
    ticks = round(seconds / (SUBSTEPS * model.opt.timestep))
    for _ in range(ticks):
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        conveyor.update(model, data)


def body_id(model: object, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def test_scene_items_match_the_pool_and_generated_xml(sim: tuple) -> None:
    model, _, conveyor = sim
    names = pool_item_names()
    assert len(names) == 24
    assert {item.name for item in conveyor._items} == set(names)
    xml = (Path(get_scene("conveyor_sort").scene_xml_path).parent / "conveyor_items.xml").read_text()
    assert [line.split('"')[1] for line in xml.splitlines() if "<body name=" in line] == list(names)
    assert all(body_id(model, name) > 0 for name in names)


def test_textured_geoms_are_uv_meshes_so_the_browser_viewer_draws_them(sim: tuple) -> None:
    # mjviser draws primitives with a flat color and reduces cube maps to one color per face,
    # so every textured surface must be a UV-mapped mesh with a 2D texture. The floor plane
    # is exempt: the viewer draws its own ground grid.
    model, _, _ = sim
    role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    for geom in range(model.ngeom):
        mat = int(model.geom_matid[geom])
        if mat < 0 or int(model.mat_texid[mat, role]) < 0:
            continue
        if int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat)
        assert int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_MESH), name
        assert model.mesh_texcoordnum[int(model.geom_dataid[geom])] > 0, name
        assert int(model.tex_type[int(model.mat_texid[mat, role])]) == int(mujoco.mjtTexture.mjTEXTURE_2D), name


def test_http_speed_limit_matches_conveyor() -> None:
    assert http_server.MAX_BELT_SPEED == MAX_BELT_SPEED


def test_belt_carries_items_at_belt_speed_without_tipping(sim: tuple) -> None:
    model, data, conveyor = sim
    run(model, data, conveyor, 4.0)
    assert conveyor._on_belt, "the feed should have placed an item"
    for item in conveyor._on_belt.values():
        assert data.qvel[item.dof_adr + 1] == pytest.approx(-0.05, abs=0.002)
        up_z = data.xmat[item.body_id].reshape(3, 3)[2, 2]
        assert up_z > 0.99


def test_belt_joint_wraps_within_one_tile(sim: tuple) -> None:
    model, data, conveyor = sim
    run(model, data, conveyor, 3.0)
    q = data.qpos[conveyor._belt_qpos_adr]
    assert -BELT_TILE < q <= 0.0
    assert conveyor._travel == pytest.approx(0.05 * 3.0, rel=0.05)


def test_unsorted_items_ride_off_the_end_as_misses(sim: tuple) -> None:
    model, data, conveyor = sim
    conveyor._config.items_per_episode = 2
    conveyor.set_belt_speed(MAX_BELT_SPEED)
    run(model, data, conveyor, 14.0)
    status = conveyor.status()
    assert status["episode_count"] == 1
    assert status["last_episode"] == {"correct": 0, "wrong": 0, "missed": 2}


@pytest.mark.parametrize(
    ("item", "bin_name", "outcome"),
    [
        ("item_cube_red", "red", "correct"),
        ("item_hex_blue", "blue", "correct"),
        ("item_cylinder_green", "green", "correct"),
        ("item_cube_purple", "reject", "correct"),
        ("item_hex_red_cracked", "reject", "correct"),
        ("item_cube_blue", "red", "wrong"),
        ("item_cylinder_green_cracked", "green", "wrong"),
        ("item_cylinder_red", None, "missed"),
    ],
)
def test_items_resting_in_bins_are_scored_by_the_rule(
    sim: tuple, item: str, bin_name: str | None, outcome: str
) -> None:
    model, data, conveyor = sim
    conveyor.set_active(False)
    target = next(i for i in conveyor._items if i.name == item)
    xy = data.xpos[body_id(model, f"bin_{bin_name}")][:2] if bin_name else (0.40, 0.30)
    data.qpos[target.qpos_adr : target.qpos_adr + 3] = (*xy, 0.08)
    conveyor._on_belt[item] = target
    mujoco.mj_forward(model, data)
    run(model, data, conveyor, 2.0)
    score = conveyor.status()["score"]
    assert score[outcome] == 1
    assert sum(score.values()) == 1
    assert item not in conveyor._on_belt
    # Parked back in the supply crate (it settles ~0.5 mm onto the crate floor).
    assert tuple(data.xpos[target.body_id]) == pytest.approx(target.park_xyz, abs=2e-3)


def test_paused_belt_stops_and_feeds_nothing(sim: tuple) -> None:
    model, data, conveyor = sim
    conveyor.set_active(False)
    run(model, data, conveyor, 1.0)
    assert conveyor.status()["spawned"] == 0
    assert data.ctrl[conveyor._belt_actuator] == 0.0


def test_stack_light_shows_running_item_soon_and_paused(sim: tuple) -> None:
    model, data, conveyor = sim

    def lit(light: str) -> bool:
        mocap, on_pos = conveyor._lights[light]
        return bool(np.allclose(data.mocap_pos[mocap], on_pos))

    assert set(conveyor._lights) == {"green", "amber", "red"}
    warn = conveyor._config.warn_s

    def run_until(soon: bool) -> None:
        for _ in range(500):
            run(model, data, conveyor, 0.02)
            if (conveyor.status()["next_item_s"] <= warn) == soon:
                return
        pytest.fail("light state never reached")

    # The first item spawns at once, so amber starts lit; wait for a gap, then the next item.
    run_until(soon=False)
    assert conveyor.status()["lights"] == {"green": True, "amber": False, "red": False}
    assert lit("green") and not lit("amber") and not lit("red")
    run_until(soon=True)
    assert conveyor.status()["lights"] == {"green": True, "amber": True, "red": False}
    assert lit("green") and lit("amber") and not lit("red")
    conveyor.set_active(False)
    conveyor.update(model, data)
    assert conveyor.status()["lights"] == {"green": False, "amber": False, "red": True}
    assert conveyor.status()["next_item_s"] is None
    assert lit("red") and not lit("green") and not lit("amber")


def test_unlit_lamps_hide_inside_the_motor_can(sim: tuple) -> None:
    # The browser viewer draws the floor as a see-through grid, so a lamp parked under it shows.
    model, data, conveyor = sim
    conveyor.set_active(False)  # red on, green and amber off
    run(model, data, conveyor, 0.02)
    obj = Path(get_scene("conveyor_sort").scene_xml_path).parent / "assets" / "conveyor_motor.obj"
    verts = np.array([[float(v) for v in line.split()[1:4]] for line in obj.read_text().splitlines() if line.startswith("v ")])
    can = verts[verts[:, 0] > verts[:, 0].max() - 0.05 + 1e-6]  # the can, not the gearbox beside it
    centre = (can.min(axis=0) + can.max(axis=0)) / 2
    radius = (can.max(axis=0)[2] - can.min(axis=0)[2]) / 2
    for light in ("green", "amber"):
        mocap, _ = conveyor._lights[light]
        body = int(np.flatnonzero(model.body_mocapid == mocap)[0])
        geom = int(model.body_geomadr[body])
        lamp_r, lamp_half = model.geom_size[geom][:2]
        pos = data.mocap_pos[mocap]
        assert pos[2] - lamp_half > 0.0, "above the floor"
        assert can.min(axis=0)[0] < pos[0] - lamp_r and pos[0] + lamp_r < can.max(axis=0)[0]
        # The lamp's cross-section corner must stay inside the can's circular section.
        assert np.hypot(abs(pos[1] - centre[1]) + lamp_r, abs(pos[2] - centre[2]) + lamp_half) < radius


def test_held_item_is_not_scored_as_a_miss(sim: tuple) -> None:
    model, data, conveyor = sim
    conveyor.set_active(False)
    target = conveyor._items[0]
    conveyor._on_belt[target.name] = target
    for _ in range(100):
        # Pin the item in the air off the belt, as a gripper holding it still would.
        data.qpos[target.qpos_adr : target.qpos_adr + 3] = (0.10, 0.0, 0.15)
        data.qvel[target.dof_adr : target.dof_adr + 6] = 0.0
        mujoco.mj_step(model, data)
        conveyor.update(model, data)
    assert target.name in conveyor._on_belt
    assert sum(conveyor.status()["score"].values()) == 0


def test_robot_runs_conveyor_and_keeps_belt_speed_across_switches() -> None:
    scene = get_scene("conveyor_sort")
    robot = MuJoCoSO101(model_path=str(scene.scene_xml_path), scene_config=asdict(scene))
    robot.connect()
    try:
        episode = robot._http_status()["episode"]
        assert episode["kind"] == "conveyor"
        robot._commands.put(SetBeltSpeedCommand(speed=0.02))
        robot._commands.put(SetAutoResetCommand(enabled=True, dwell_s=5.0))
        robot._drain_commands()
        assert robot._episode_auto_reset.dwell_s == 0.5, "the pick-place dwell must not reach the conveyor"
        assert robot._switch_to_scene("single_pick_place")
        assert "kind" not in robot._http_status()["episode"]
        assert robot._switch_to_scene("conveyor_sort")
        assert robot._http_status()["episode"]["belt_speed"] == 0.02
    finally:
        robot.disconnect()


def test_conveyor_markdown_shows_speed_and_score() -> None:
    text = _episode_markdown({
        "kind": "conveyor",
        "active": True,
        "belt_speed": 0.03,
        "episode_count": 1,
        "spawned": 4,
        "items_per_episode": 10,
        "score": {"correct": 2, "wrong": 1, "missed": 0},
        "last_episode": {"correct": 7, "wrong": 2, "missed": 1},
        "rule": "r",
    })
    assert "running, 3.0 cm/s" in text
    assert "**Episode 2:** 4/10 items fed, 2 correct, 1 wrong, 0 missed" in text
    assert "**Last episode:** 7 correct, 2 wrong, 1 missed" in text
