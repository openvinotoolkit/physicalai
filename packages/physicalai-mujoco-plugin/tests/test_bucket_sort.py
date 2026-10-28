# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The bucket_sort scene: layout, instructions and success, on the real model with the Franka FR3 (no display)."""

from __future__ import annotations

import itertools
import re
from collections import Counter

import mujoco
import numpy as np
import pytest

from physicalai_mujoco_plugin import bucket_sort as bs
from physicalai_mujoco_plugin.bucket_sort import (
    ALL_PAIRS,
    BUCKET_COLORS,
    BUCKET_JOINTS,
    OBJECTS,
    BucketSortEpisode,
    BucketSortTask,
    Instruction,
    check_pairs,
    object_extent,
    sample_instruction,
)
from physicalai_mujoco_plugin.http_server import (
    ResetCommand,
    SetAutoResetCommand,
    SetObjectPoseCommand,
    SetSeedCommand,
)
from physicalai_mujoco_plugin.profiles import SO101_PROFILE, get_profile
from physicalai_mujoco_plugin.robot import MuJoCoRobot
from physicalai_mujoco_plugin.scene_registry import get_scene, list_scenes_for
from physicalai_mujoco_plugin.sim import load_sim

FR3 = get_profile("franka_fr3")


@pytest.fixture(scope="module")
def sim():
    scene = get_scene("bucket_sort")
    return load_sim(
        scene.scene_xml_path,
        scene,
        FR3,
        unit="degrees",
        torque_mode="pd",
        rng=np.random.default_rng(0),
        reseed=lambda: None,
        arms=1,
    )


def _reset(sim, seed: int) -> BucketSortTask:
    task = sim.on_reset
    task(sim.model, sim.data, np.random.default_rng(seed))
    return task


def _step(sim, seconds: float) -> None:
    for _ in range(int(seconds / sim.model.opt.timestep)):
        mujoco.mj_step(sim.model, sim.data)


def _drop_into(sim, name: str, color: str) -> None:
    """Teleport object *name* just above the floor of the *color* bucket, at rest."""
    model, data = sim.model, sim.data
    bucket = model.body(f"bucket_{color}").id
    centre = data.xpos[bucket] + data.xmat[bucket].reshape(3, 3) @ np.array([0.003, 0.0, 0.0])
    joint = model.joint(f"{name}:joint")
    adr, dof = int(joint.qposadr[0]), int(joint.dofadr[0])
    data.qpos[adr : adr + 3] = (centre[0], centre[1], 0.006 + object_extent(model, name)[1] + 0.005)
    data.qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[dof : dof + 6] = 0.0
    mujoco.mj_forward(model, data)


class TestScene:
    def test_registered_for_one_franka_arm(self) -> None:
        scene = get_scene("bucket_sort")
        assert scene.robots == ("franka_fr3", "franka_panda", "franka_duo")
        assert scene.arm_counts == (1,)
        assert scene.free_joints == (*(f"{name}:joint" for name in OBJECTS), *BUCKET_JOINTS)
        assert BUCKET_JOINTS == ("bucket_black:joint", "bucket_blue:joint", "bucket_red:joint")
        assert "bucket_sort" in list_scenes_for(FR3, 1)
        assert "bucket_sort" not in list_scenes_for(SO101_PROFILE)
        assert scene.overview_styles == ("shoulder", "front")

    def test_compiles_with_three_buckets_and_three_objects(self, sim) -> None:
        model = sim.model
        assert [binding.prefix for binding in sim.bindings] == [""]
        assert len(sim.joint_names) == 8
        assert model.camera("overview").id >= 0
        colors = {}
        for color in BUCKET_COLORS:
            body = model.body(f"bucket_{color}")
            joint = model.joint(f"bucket_{color}:joint")
            assert joint.type[0] == mujoco.mjtJoint.mjJNT_FREE
            assert joint.bodyid[0] == body.id
            assert 0.15 <= model.body_mass[body.id] <= 0.25  # about a printed bucket
            geoms = np.flatnonzero(model.geom_bodyid == body.id)
            visual = [g for g in geoms if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
            assert len(visual) == 1
            assert model.mesh(model.geom_dataid[visual[0]]).name == "bucket"
            colors[color] = tuple(model.mat_rgba[model.geom_matid[visual[0]]][:3])
            # Bottom, three walls and the lip collide; the visual mesh does not.
            assert sum(model.geom_contype[g] > 0 for g in geoms) == 5
        assert len(set(colors.values())) == 3
        for name in OBJECTS:
            joint = model.joint(f"{name}:joint")
            assert joint.type[0] == mujoco.mjtJoint.mjJNT_FREE
            body = model.body(name).id
            collision = [g for g in np.flatnonzero(model.geom_bodyid == body) if model.geom_contype[g] > 0]
            assert len(collision) == 1
            assert model.mesh(model.geom_dataid[collision[0]]).name == f"{name}_hull"
        masses = {name: float(model.body_mass[model.body(name).id]) for name in OBJECTS}
        assert masses == pytest.approx({"resistor": 0.040, "battery": 0.045, "cpu": 0.030})

    def test_objects_are_real_size_and_fit_the_franka_hand(self, sim) -> None:
        # Bounding boxes of the source CAD, in mm; the CPU includes its six legs.
        sizes = {"resistor": (34.5, 67.7, 29.9), "battery": (26.0, 47.5, 17.1), "cpu": (70.0, 38.0, 20.9)}
        model = sim.model
        rotation = np.zeros(9)
        for name, size in sizes.items():
            geom = next(g for g in np.flatnonzero(model.geom_bodyid == model.body(name).id) if model.geom_contype[g])
            mesh = model.geom_dataid[geom]
            vertices = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
            # In the body frame: MuJoCo moves a mesh geom to the mesh's principal axes.
            mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
            points = model.geom_pos[geom] + vertices @ rotation.reshape(3, 3).T
            np.testing.assert_allclose(1000 * np.ptp(points, axis=0), size, atol=0.1)
            np.testing.assert_allclose(points.min(axis=0) + points.max(axis=0), 0.0, atol=1e-4)  # origin at the centre
            assert min(size[:2]) < 1000 * 0.08  # narrower than the hand's full opening


def _mesh_points(model: mujoco.MjModel, geom: int) -> np.ndarray:
    """Return a mesh geom's vertices in its body's frame."""
    mesh = model.geom_dataid[geom]
    vertices = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
    return model.geom_pos[geom] + vertices @ rotation.reshape(3, 3).T


def _visual_geom(model: mujoco.MjModel, mesh_name: str) -> int:
    return next(g for g in range(model.ngeom) if model.geom_dataid[g] >= 0 and model.mesh(model.geom_dataid[g]).name == mesh_name)


class TestObjectLooks:
    def test_cpu_has_three_legs_on_each_long_side_reaching_the_table(self, sim) -> None:
        model = sim.model
        legs = _mesh_points(model, _visual_geom(model, "cpu_legs"))
        body = _mesh_points(model, _visual_geom(model, "cpu_body"))
        hull = _mesh_points(model, _visual_geom(model, "cpu_hull"))
        # Legs stand outside both long sides of the body and reach the bottom of the collision hull.
        assert legs[:, 1].min() < body[:, 1].min() - 0.002
        assert legs[:, 1].max() > body[:, 1].max() + 0.002
        assert legs[:, 2].min() == pytest.approx(hull[:, 2].min(), abs=1e-4)
        assert body[:, 2].min() > hull[:, 2].min() + 0.003
        for side in (legs[legs[:, 1] < body[:, 1].min()], legs[legs[:, 1] > body[:, 1].max()]):
            # Three 10 mm legs with 11.5 mm between them: three runs of x without a gap over 8 mm.
            x = np.sort(side[:, 0])
            assert np.count_nonzero(np.diff(x) > 0.008) == 2
            assert np.ptp(x) == pytest.approx(0.053, abs=5e-4)

    def test_cpu_rests_level_on_its_legs(self, sim) -> None:
        task = _reset(sim, 2)
        _step(sim, 1.0)
        body = sim.model.body("cpu").id
        assert sim.data.xmat[body][8] == pytest.approx(1.0, abs=1e-4)  # upright: less than 1 degree of tilt
        assert sim.data.xpos[body][2] == pytest.approx(object_extent(sim.model, "cpu")[1], abs=0.002)
        assert not task.evaluate(sim.model, sim.data).success

    def test_resistor_has_four_colour_bands_on_its_body(self, sim) -> None:
        model = sim.model
        resistor = model.body("resistor").id
        geoms = [g for g in np.flatnonzero(model.geom_bodyid == resistor) if model.geom_dataid[g] >= 0]
        bands = {model.mesh(model.geom_dataid[g]).name: g for g in geoms if "band" in model.mesh(model.geom_dataid[g]).name}
        assert sorted(bands) == [f"resistor_band_{color}" for color in ("brown", "red", "yellow")]
        rings = []
        for name, geom in bands.items():
            assert model.geom_contype[geom] == 0
            assert model.mat(model.geom_matid[geom]).name == name.replace("resistor_", "")
            y = np.sort(_mesh_points(model, geom)[:, 1])
            # One mesh per colour: split it into its rings where y jumps (they are over 7 mm apart).
            for ring in np.split(y, np.flatnonzero(np.diff(y) > 0.006) + 1):
                assert 0.005 < np.ptp(ring) <= 0.0056  # a 5.5 mm ring around the body
                assert np.abs(ring).max() < 0.030  # clear of the ends
                rings.append((ring.mean(), name.removeprefix("resistor_band_")))
        # Brown, red, brown and yellow along the body, as in the reference photo.
        assert [color for _, color in sorted(rings)] == ["brown", "red", "brown", "yellow"]


class TestInstructions:
    def test_text_names_the_object_and_the_bucket(self) -> None:
        assert Instruction("cpu", "red").text == "put the CPU in the red bucket"
        assert Instruction("cpu", "red").canonical_text == "put the CPU in the red bucket"
        assert Instruction("battery", "black").as_dict() == {
            "object": "battery",
            "bucket": "black",
            "text": "put the 9 V battery in the black bucket",
            "canonical_text": "put the 9 V battery in the black bucket",
        }
        worded = Instruction("battery", "black", wording="Drop the battery in the black bin.")
        assert worded.text == "Drop the battery in the black bin."
        assert worded.canonical_text == "put the 9 V battery in the black bucket"
        assert worded == Instruction("battery", "black")  # the same instruction, worded differently

    def test_all_nine_pairs_by_default(self) -> None:
        assert check_pairs(None) == ALL_PAIRS
        assert len(ALL_PAIRS) == 9
        drawn = Counter(sample_instruction(np.random.default_rng(seed)) for seed in range(900))
        assert {(i.object, i.bucket) for i in drawn} == set(ALL_PAIRS)
        assert min(drawn.values()) > 60  # about 100 each

    def test_sampling_respects_the_allowed_pairs(self) -> None:
        held_out = {("cpu", "red"), ("battery", "blue"), ("resistor", "black")}
        train = [pair for pair in ALL_PAIRS if pair not in held_out]
        rng = np.random.default_rng(1)
        drawn = {(i.object, i.bucket) for i in (sample_instruction(rng, train) for _ in range(500))}
        assert drawn == set(train)
        rng = np.random.default_rng(1)
        assert {(i.object, i.bucket) for i in (sample_instruction(rng, held_out) for _ in range(100))} == held_out

    def test_check_pairs_orders_and_rejects(self) -> None:
        assert check_pairs([("cpu", "red"), ("battery", "black"), ("cpu", "red")]) == (
            ("battery", "black"),
            ("cpu", "red"),
        )
        with pytest.raises(ValueError, match="Unknown"):
            check_pairs([("cpu", "green")])
        with pytest.raises(ValueError, match="At least one"):
            check_pairs([])

    def test_task_draws_from_its_allowed_pairs(self, sim) -> None:
        task = sim.on_reset
        try:
            task.set_allowed_pairs([("resistor", "blue")])
            for seed in range(5):
                _reset(sim, seed)
                assert task.instruction == Instruction("resistor", "blue")
            assert task.describe()["allowed_pairs"] == [["resistor", "blue"]]
        finally:
            task.set_allowed_pairs(None)


class TestLayout:
    def test_layout_repeats_per_seed_and_varies_across_seeds(self, sim) -> None:
        def layout(seed: int) -> tuple[tuple[str, ...], Instruction, str, np.ndarray]:
            task = _reset(sim, seed)
            bodies = [f"bucket_{color}" for color in BUCKET_COLORS] + list(OBJECTS)
            poses = np.array([sim.data.xpos[sim.model.body(name).id] for name in bodies])
            return task.slot_colors, task.instruction, task.instruction.text, poses.ravel()

        first, again = layout(7), layout(7)
        assert first[:3] == again[:3]
        np.testing.assert_array_equal(first[3], again[3])
        orders = {layout(seed)[0] for seed in range(60)}
        assert orders == set(itertools.permutations(BUCKET_COLORS))

    def test_reset_writes_the_sampled_layout_at_rest(self, sim) -> None:
        model, data = sim.model, sim.data
        for seed in range(30):
            data.qvel[:] = 1.0  # a reset stills the buckets whatever they were doing
            task = _reset(sim, seed)
            layout = task.layout
            assert not layout.fallback
            for color, pose in layout.buckets.items():
                body = model.body(f"bucket_{color}").id
                np.testing.assert_allclose(data.xpos[body], (pose.x, pose.y, 0.0), atol=1e-9)
                assert _yaw(data, body) == pytest.approx(_wrap(pose.yaw), abs=1e-9)
                dof = int(model.joint(f"bucket_{color}:joint").dofadr[0])
                assert not data.qvel[dof : dof + 6].any()
            for name, pose in layout.objects.items():
                np.testing.assert_allclose(data.xpos[model.body(name).id][:2], (pose.x, pose.y), atol=1e-9)
            # The colours are listed left to right: by decreasing y.
            ys = [layout.buckets[color].y for color in task.slot_colors]
            assert ys == sorted(ys, reverse=True)
            slots = task.describe()["slots"]
            assert [slot["slot"] for slot in slots] == ["left", "centre", "right"]
            assert slots[0] == {"slot": "left", "color": task.slot_colors[0], **layout.buckets[task.slot_colors[0]].as_dict()}

    def test_objects_never_overlap_each_other_or_the_buckets(self, sim) -> None:
        model, data = sim.model, sim.data
        radii = {name: object_extent(model, name)[0] for name in OBJECTS}
        for seed in range(200):
            _reset(sim, seed)
            xy = {name: data.xpos[model.body(name).id][:2].copy() for name in OBJECTS}
            for x, y in xy.values():
                assert bs.OBJECT_X[0] <= x <= bs.OBJECT_X[1]
                assert bs.OBJECT_Y[0] <= y <= bs.OBJECT_Y[1]
                assert bs.REACH_RADIUS[0] <= np.hypot(x, y) <= bs.REACH_RADIUS[1]
            for a, b in itertools.combinations(OBJECTS, 2):
                assert np.linalg.norm(xy[a] - xy[b]) >= radii[a] + radii[b] + bs.OBJECT_GAP, seed
            for name, centre in xy.items():
                for color in BUCKET_COLORS:
                    # Brute force, apart from the module's geometry: the footprint's boundary every 2 mm.
                    gap = np.linalg.norm(_footprint_outline(sim, color) - centre, axis=1).min() - radii[name]
                    assert gap >= bs.OBJECT_BUCKET_GAP - 0.002, (seed, name, color)
                    assert bs.bucket_containing(model, data, np.array([*centre, 0.01])) is None
            # Objects are released just above the table: only the buckets' bottoms touch it at reset.
            touching = {(model.geom(data.contact.geom1[i]).name, model.geom(data.contact.geom2[i]).name) for i in range(data.ncon)}
            assert touching <= {("table_top", f"bucket_{color}_bottom") for color in BUCKET_COLORS}, seed

    def test_reset_puts_the_arm_at_the_scene_home_with_the_hand_open(self, sim) -> None:
        model, data = sim.model, sim.data
        data.qpos[model.joint("fr3_joint1").qposadr[0]] = 1.0
        data.ctrl[model.actuator("gripper").id] = 0.0
        _reset(sim, 0)
        for joint, value in get_scene("bucket_sort").home_pose():
            if joint == "gripper":
                assert data.ctrl[model.actuator("gripper").id] == pytest.approx(value)
            else:
                assert data.qpos[model.joint(joint).qposadr[0]] == pytest.approx(value)

    def test_objects_settle_on_the_table(self, sim) -> None:
        model, data = sim.model, sim.data
        for seed in range(3):
            _reset(sim, seed)
            start = {name: data.xpos[model.body(name).id].copy() for name in OBJECTS}
            _step(sim, 1.0)
            for name in OBJECTS:
                pos = data.xpos[model.body(name).id]
                assert np.linalg.norm(pos[:2] - start[name][:2]) < 0.005
                assert 0.0 < pos[2] < 0.02
            assert min(data.contact.dist[: data.ncon]) > -0.001


_TABLE_FAR_EDGE = 0.83
_TABLE_HALF_WIDTH = 0.9


class TestBucketPose:
    def test_table_spans_the_robot_and_the_work_area(self, sim) -> None:
        table = sim.model.geom("table_top")
        np.testing.assert_allclose(table.pos - table.size, (-0.17, -_TABLE_HALF_WIDTH, -0.04))
        np.testing.assert_allclose(table.pos + table.size, (_TABLE_FAR_EDGE, _TABLE_HALF_WIDTH, 0.0))
        for corner in range(1, 5):
            leg = sim.model.geom(f"table_leg_{corner}")
            assert abs(leg.pos[0] - table.pos[0]) == pytest.approx(table.size[0] - 0.04, abs=0.005)
            assert abs(leg.pos[1]) == pytest.approx(_TABLE_HALF_WIDTH - 0.04)


    def test_bucket_lies_on_its_long_side_with_the_opening_up(self, sim) -> None:
        model = sim.model
        points = _mesh_points(model, _visual_geom(model, "bucket"))
        np.testing.assert_allclose(1000 * np.ptp(points, axis=0), (150.0, 202.5, 120.0), atol=0.1)
        assert points[:, 2].min() == pytest.approx(0.0, abs=1e-4)  # on the table, not in it or above it
        np.testing.assert_allclose(points[:, :2].min(axis=0) + points[:, :2].max(axis=0), 0.0, atol=1e-4)
        assert bs.BUCKET_SIZE == pytest.approx((0.150, 0.2025, 0.120))
        # The collision bottom's top is the mesh's inner bottom (6 mm), under the whole opening but the lip;
        # it stands on the table and carries the bucket.
        bottom = model.geom("bucket_blue_bottom")
        assert bottom.pos[2] + bottom.size[2] == pytest.approx(0.006)
        assert bottom.pos[2] - bottom.size[2] == pytest.approx(0.0)
        assert bottom.pos[0] + bottom.size[0] == pytest.approx(0.075)
        assert bottom.pos[1] - bottom.size[1] == pytest.approx(-0.10125)
        for wall in ("wall_left", "wall_right", "wall_end"):
            geom = model.geom(f"bucket_blue_{wall}")
            assert geom.pos[2] + geom.size[2] == pytest.approx(bs.BUCKET_SIZE[2])  # walls up to the rim
            assert geom.pos[2] - geom.size[2] == pytest.approx(0.006)  # standing on the bottom

    def test_buckets_face_the_robot_and_stay_apart_on_the_table(self, sim) -> None:
        assert bs.BUCKET_YAW == pytest.approx(np.pi / 2)  # radians
        model, data = sim.model, sim.data
        for seed in range(100):
            task = _reset(sim, seed)
            outlines = {}
            for color in task.slot_colors:
                body = model.body(f"bucket_{color}").id
                rotation = data.xmat[body].reshape(3, 3)
                open_end = rotation[:2, 1]  # the open end, above the lip, is at +y in the bucket's frame
                turn = np.degrees(np.arccos(np.clip(open_end @ (-1.0, 0.0), -1, 1)))
                assert turn <= np.degrees(bs.BUCKET_YAW_RANGE) + 1e-6  # toward the robot
                assert rotation[2, 2] == pytest.approx(1.0)
                outline = outlines[color] = _footprint_outline(sim, color)
                assert outline[:, 0].min() >= -0.17 + bs.TABLE_MARGIN - 1e-9
                assert outline[:, 0].max() <= _TABLE_FAR_EDGE - bs.TABLE_MARGIN + 1e-9
                assert np.abs(outline[:, 1]).max() <= _TABLE_HALF_WIDTH - bs.TABLE_MARGIN
                for local in bs.BUCKET_REACH_POINTS:
                    point = data.xpos[body][:2] + rotation[:2, :2] @ local
                    assert bs.REACH_RADIUS[0] <= np.hypot(*point) <= bs.REACH_RADIUS[1]
            for a, b in itertools.combinations(outlines, 2):
                gap = np.linalg.norm(outlines[a][:, None] - outlines[b][None], axis=2).min()
                assert gap >= bs.BUCKET_GAP - 0.002, seed


def _yaw(data, body: int) -> float:
    rotation = data.xmat[body].reshape(3, 3)
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def _wrap(angle: float) -> float:
    return float(np.angle(np.exp(1j * angle)))


def _footprint_outline(sim, color: str, step: float = 0.002) -> np.ndarray:
    """Return points every *step* along the outer rectangle of the *color* bucket as it stands now."""
    body = sim.model.body(f"bucket_{color}").id
    hx, hy = bs.BUCKET_SIZE[0] / 2, bs.BUCKET_SIZE[1] / 2
    xs, ys = np.arange(-hx, hx + 1e-9, step), np.arange(-hy, hy + 1e-9, step)
    local = np.concatenate([
        np.column_stack([xs, np.full_like(xs, -hy)]),
        np.column_stack([xs, np.full_like(xs, hy)]),
        np.column_stack([np.full_like(ys, -hx), ys]),
        np.column_stack([np.full_like(ys, hx), ys]),
    ])
    rotation = sim.data.xmat[body].reshape(3, 3)[:2, :2]
    return sim.data.xpos[body][:2] + local @ rotation.T


def _bucket_pose(sim, color: str) -> tuple[np.ndarray, np.ndarray]:
    body = sim.model.body(f"bucket_{color}").id
    return sim.data.xpos[body].copy(), sim.data.xmat[body].reshape(3, 3).copy()


def _move_bucket(sim, color: str, xy: tuple[float, float], yaw: float) -> None:
    joint = sim.model.joint(f"bucket_{color}:joint")
    adr, dof = int(joint.qposadr[0]), int(joint.dofadr[0])
    sim.data.qpos[adr : adr + 3] = (*xy, 0.0)
    sim.data.qpos[adr + 3 : adr + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
    sim.data.qvel[dof : dof + 6] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


class TestDynamicBuckets:
    def test_buckets_rest_still_on_the_table(self, sim) -> None:
        for seed in range(3):
            _reset(sim, seed)
            start = {color: _bucket_pose(sim, color) for color in BUCKET_COLORS}
            _step(sim, 2.0)
            for color, (pos, _) in start.items():
                now, rotation = _bucket_pose(sim, color)
                assert np.linalg.norm(now[:2] - pos[:2]) < 1e-4  # no creep
                assert abs(now[2]) < 5e-4
                assert rotation[2, 2] == pytest.approx(1.0, abs=1e-6)  # level
            assert min(sim.data.contact.dist[: sim.data.ncon]) > -0.001

    @pytest.mark.parametrize("name", OBJECTS)
    def test_an_object_dropped_from_the_hand_lands_inside_and_leaves_the_bucket_in_place(self, sim, name) -> None:
        # Released 0.26 m above the table, about where a hand over the bucket lets go: it must not
        # pass through the 6 mm bottom or shove the 0.2 kg bucket.
        model, data = sim.model, sim.data
        task = _reset(sim, 9)
        color = BUCKET_COLORS[OBJECTS.index(name)]
        pos, rotation = _bucket_pose(sim, color)
        joint = model.joint(f"{name}:joint")
        adr, dof = int(joint.qposadr[0]), int(joint.dofadr[0])
        data.qpos[adr : adr + 3] = (*(pos + rotation @ (0.02, -0.01, 0.0))[:2], 0.26)
        data.qpos[adr + 3 : adr + 7] = (np.cos(0.3), np.sin(0.3), 0.0, 0.0)  # tilted, to land on an edge
        data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(model, data)
        deepest = 0.0
        for _ in range(int(1.5 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            deepest = min(deepest, *data.contact.dist[: data.ncon], 0.0)
        assert task.evaluate(model, data).in_bucket[name] == color
        assert data.xpos[model.body(name).id][2] > 0.006  # on the bucket's bottom, not under it
        assert deepest > -0.006
        now, now_rotation = _bucket_pose(sim, color)
        assert np.linalg.norm(now[:2] - pos[:2]) < 0.002
        assert now_rotation[2, 2] == pytest.approx(1.0, abs=1e-4)

    def test_success_follows_a_moved_bucket(self, sim) -> None:
        task = _reset(sim, 3)
        instruction = task.instruction
        old, _ = _bucket_pose(sim, instruction.bucket)
        # Away from where every layout puts a bucket, turned the other way: the interior moves with it.
        _move_bucket(sim, instruction.bucket, (0.55, 0.65), -0.5)
        _drop_into(sim, instruction.object, instruction.bucket)
        _step(sim, 0.5)
        state = task.evaluate(sim.model, sim.data)
        assert state.success
        assert state.buckets[instruction.bucket]["position"][:2] == pytest.approx([0.55, 0.65], abs=1e-3)
        assert bs.bucket_containing(sim.model, sim.data, old + (0.0, 0.0, 0.05)) is None


class TestSuccess:
    def test_false_at_reset(self, sim) -> None:
        for seed in range(10):
            task = _reset(sim, seed)
            state = task.evaluate(sim.model, sim.data)
            assert not state.success
            assert state.in_bucket == dict.fromkeys(OBJECTS)

    def test_named_object_in_named_bucket_succeeds_and_other_buckets_do_not(self, sim) -> None:
        task = _reset(sim, 3)
        instruction = task.instruction
        _drop_into(sim, instruction.object, instruction.bucket)
        _step(sim, 0.5)
        state = task.evaluate(sim.model, sim.data)
        assert state.in_bucket[instruction.object] == instruction.bucket
        assert state.resting[instruction.object]
        assert state.success

        wrong = next(color for color in BUCKET_COLORS if color != instruction.bucket)
        _drop_into(sim, instruction.object, wrong)
        _step(sim, 0.5)
        state = task.evaluate(sim.model, sim.data)
        assert state.in_bucket[instruction.object] == wrong
        assert not state.success

    def test_another_object_in_a_bucket_spoils_the_success(self, sim) -> None:
        task = _reset(sim, 4)
        instruction = task.instruction
        other = next(name for name in OBJECTS if name != instruction.object)
        _drop_into(sim, instruction.object, instruction.bucket)
        _drop_into(sim, other, next(color for color in BUCKET_COLORS if color != instruction.bucket))
        _step(sim, 0.5)
        state = task.evaluate(sim.model, sim.data)
        assert state.in_bucket[instruction.object] == instruction.bucket
        assert not state.success

    def test_a_moving_object_has_not_settled(self, sim) -> None:
        task = _reset(sim, 5)
        instruction = task.instruction
        _drop_into(sim, instruction.object, instruction.bucket)
        dof = int(sim.model.joint(f"{instruction.object}:joint").dofadr[0])
        sim.data.qvel[dof + 2] = -0.5
        assert not task.evaluate(sim.model, sim.data).success


class TestEpisode:
    def test_status_reports_the_instruction_and_restarts_after_the_dwell(self, sim) -> None:
        task = _reset(sim, 6)
        resets = []

        def reset_scene() -> None:
            resets.append(task.instruction)
            task(sim.model, sim.data, np.random.default_rng(0))

        episode = BucketSortEpisode(task, reset_scene, success_dwell_s=0.2)
        episode.update(sim.model, sim.data)
        status = episode.status()
        assert status["kind"] == "bucket_sort"
        assert status["enabled"] is True
        assert status["instruction"] == task.instruction.as_dict()
        assert [slot["color"] for slot in status["slots"]] == list(task.slot_colors)
        assert status["success"] is False
        assert status["phase"] == "idle"
        for color in BUCKET_COLORS:
            body = sim.model.body(f"bucket_{color}").id
            assert status["buckets"][color]["position"] == pytest.approx(sim.data.xpos[body])
            assert status["buckets"][color]["wxyz"] == pytest.approx(sim.data.xquat[body])

        done = task.instruction
        _drop_into(sim, done.object, done.bucket)
        _step(sim, 0.3)
        episode.update(sim.model, sim.data)
        assert episode.status()["phase"] == "success_hold"
        assert episode.status()["success"] is True
        _step(sim, 0.25)
        episode.update(sim.model, sim.data)
        status = episode.status()
        assert resets == [done]
        assert status["episode_count"] == 1
        assert status["phase"] == "idle"
        assert status["success"] is False
        assert status["in_bucket"] == dict.fromkeys(OBJECTS)

    def test_paused_episode_reports_but_does_not_restart(self, sim) -> None:
        task = _reset(sim, 8)
        episode = BucketSortEpisode(task, pytest.fail, success_dwell_s=0.0, active=False)
        _drop_into(sim, task.instruction.object, task.instruction.bucket)
        _step(sim, 0.3)
        episode.update(sim.model, sim.data)
        assert episode.status()["success"] is True
        assert episode.status()["episode_count"] == 0


def test_robot_serves_the_task_and_reset_draws_a_new_one() -> None:
    robot = MuJoCoRobot("franka_fr3", scene="bucket_sort", substeps=1, cameras=[])
    robot.connect()
    try:
        assert isinstance(robot._episode_auto_reset, BucketSortEpisode)  # noqa: SLF001
        robot.get_observation()
        episode = robot._http_status()["episode"]  # noqa: SLF001
        assert episode["kind"] == "bucket_sort"
        assert episode["instruction"]["canonical_text"].startswith("put the ")
        assert episode["instruction"]["text"]
        assert episode["vary_text"] is True

        def layout() -> tuple[object, ...]:
            robot._commands.put(ResetCommand())  # noqa: SLF001
            robot.get_observation()
            episode = robot._http_status()["episode"]  # noqa: SLF001
            return episode["instruction"]["text"], tuple((slot["color"], slot["x"], slot["y"]) for slot in episode["slots"])

        # The buckets are free objects: the pose route moves them, and /health reports where they are.
        assert {obj["joint"] for obj in robot._http_status()["objects"]} >= set(BUCKET_JOINTS)  # noqa: SLF001
        robot._commands.put(SetObjectPoseCommand(joint="bucket_red:joint", position=(0.6, 0.65, 0.0), wxyz=None))  # noqa: SLF001
        robot.get_observation()
        robot.get_observation()
        red = robot._http_status()["episode"]["buckets"]["red"]  # noqa: SLF001
        assert red["position"][:2] == pytest.approx([0.6, 0.65], abs=1e-3)

        robot._commands.put(SetSeedCommand(seed=11))  # noqa: SLF001
        assert layout() == layout()
        robot._commands.put(SetSeedCommand(seed=None))  # noqa: SLF001
        assert len({layout() for _ in range(8)}) > 1
    finally:
        robot.disconnect()


def test_automatic_reset_repeats_a_fixed_seed_and_drops_drags() -> None:
    robot = MuJoCoRobot("franka_fr3", scene="bucket_sort", substeps=1, cameras=[])
    robot.connect()
    try:

        def run(*commands: object) -> dict[str, object]:
            for command in commands:
                robot._commands.put(command)  # noqa: SLF001
            robot.get_observation()
            return robot._http_status()["episode"]  # noqa: SLF001

        def layout(episode: dict[str, object]) -> tuple[object, ...]:
            return episode["instruction"]["text"], tuple((slot["color"], slot["x"], slot["y"]) for slot in episode["slots"])

        seeded = run(SetAutoResetCommand(enabled=True, dwell_s=0.0), SetSeedCommand(seed=11), ResetCommand())
        task = robot._sim.on_reset  # noqa: SLF001
        target = task.instruction
        other = next(name for name in OBJECTS if name != target.object)
        dragged = (0.45, -0.5, 0.3)
        run(SetObjectPoseCommand(joint=f"{other}:joint", position=dragged, wxyz=None, hold=True))
        _drop_into(robot._sim, target.object, target.bucket)  # noqa: SLF001
        for _ in range(2000):
            episode = run()
            if episode["episode_count"] == 1:
                break
        else:
            pytest.fail("the success never started a new episode")

        # The automatic reset reseeds as a manual one does, and a drag no longer pins its object.
        assert layout(episode) == layout(seeded)
        assert robot._objects.held == {}  # noqa: SLF001
        run()
        position = robot._sim.data.xpos[robot._sim.model.body(other).id]  # noqa: SLF001
        assert position[2] < 0.1
        assert position[:2] != pytest.approx(dragged[:2], abs=1e-3)
    finally:
        robot.disconnect()


@pytest.fixture(scope="module")
def footprints(sim) -> dict[str, bs.ObjectFootprint]:
    return {name: bs.object_footprint(sim.model, name) for name in OBJECTS}


class TestLayoutSampler:
    def test_geometry_helpers(self) -> None:
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        assert bs.polygon_gap(square, square + (1.5, 0.0)) == pytest.approx(0.5)
        assert bs.polygon_gap(square, square + (1.3, 1.4)) == pytest.approx(0.5)  # corner to corner
        assert bs.polygon_gap(square, square + (0.5, 0.5)) < 0
        assert bs.circle_gap((2.0, 0.5), 0.25, square) == pytest.approx(0.75)
        assert bs.circle_gap((0.5, 0.5), 0.1, square) == pytest.approx(-0.6)
        corners = bs.bucket_footprint(bs.Pose2D(0.6, 0.1, bs.BUCKET_YAW))
        np.testing.assert_allclose(corners.min(axis=0), (0.6 - 0.10125, 0.1 - 0.075))
        np.testing.assert_allclose(corners.max(axis=0), (0.6 + 0.10125, 0.1 + 0.075))

    def test_object_footprints_close_across_the_short_side(self, sim, footprints) -> None:
        # The resistor and the battery are narrow along their body x, the CPU along its body y.
        assert footprints["resistor"].closing_yaw == 0.0
        assert footprints["battery"].closing_yaw == 0.0
        assert footprints["cpu"].closing_yaw == pytest.approx(np.pi / 2)
        for name, shape in footprints.items():
            assert (shape.radius, shape.half_height) == pytest.approx(object_extent(sim.model, name))

    def test_every_layout_passes_every_check(self, footprints) -> None:
        draws = []
        for seed in range(300):
            layout = bs.sample_layout(np.random.default_rng(seed), footprints)
            assert not layout.fallback, seed
            draws.append(layout.draws)
            buckets = list(layout.buckets.values())
            for i, pose in enumerate(buckets):
                assert not bs.bucket_problems(pose, buckets[:i] + buckets[i + 1 :]), seed
                assert bs.BUCKET_X[0] <= pose.x <= bs.BUCKET_X[1]
                assert bs.BUCKET_Y[0] <= pose.y <= bs.BUCKET_Y[1]
                assert abs(pose.yaw - bs.BUCKET_YAW) <= bs.BUCKET_YAW_RANGE
            for name, pose in layout.objects.items():
                others = {other: p for other, p in layout.objects.items() if other != name}
                assert not bs.object_problems(name, pose, footprints, buckets, others), seed
        assert np.mean(draws) < 60  # about 40 draws for six placements

    def test_layouts_cover_the_areas(self, footprints) -> None:
        layouts = [bs.sample_layout(np.random.default_rng(seed), footprints) for seed in range(300)]
        buckets = np.array([[p.x, p.y, p.yaw - bs.BUCKET_YAW] for layout in layouts for p in layout.buckets.values()])
        objects = np.array([[p.x, p.y] for layout in layouts for p in layout.objects.values()])
        assert np.ptp(buckets[:, 0]) > 0.17
        assert np.ptp(buckets[:, 1]) > 0.75
        assert buckets[:, 2].min() < -np.radians(22) and buckets[:, 2].max() > np.radians(22)
        assert np.ptp(objects[:, 0]) > 0.15
        assert np.ptp(objects[:, 1]) > 0.6
        # Not a row: in most layouts the buckets stand at clearly different depths.
        stagger = np.array([np.ptp([p.x for p in layout.buckets.values()]) for layout in layouts])
        assert np.mean(stagger > 0.05) > 0.6
        # Every colour stands left, centre and right, about equally often.
        ranks = Counter((color, i) for layout in layouts for i, color in enumerate(layout.colors_left_to_right))
        assert len(ranks) == 9
        assert min(ranks.values()) > 70  # about 100 each

    def test_fallback_layout_passes_the_checks_and_is_used_when_nothing_fits(self, footprints, monkeypatch) -> None:
        fallback = bs.FALLBACK_LAYOUT
        buckets = list(fallback.buckets.values())
        for i, pose in enumerate(buckets):
            assert not bs.bucket_problems(pose, buckets[:i])
        for name, pose in fallback.objects.items():
            others = {other: p for other, p in fallback.objects.items() if other != name}
            assert not bs.object_problems(name, pose, footprints, buckets, others)
        monkeypatch.setattr(bs, "BUCKET_X", (0.95, 0.99))  # off the table
        monkeypatch.setattr(bs, "LAYOUT_RESTARTS", 2)
        layout = bs.sample_layout(np.random.default_rng(0), footprints)
        assert layout.fallback
        assert layout.buckets == fallback.buckets


class _ReachIK:
    """Damped least-squares IK of the FR3's TCP, fingers down, the null space pushing away from joint limits."""

    TCP = np.array([0.0, 0.0, 0.1034])
    HOME = np.radians([0.0, -45.0, 0.0, -135.0, 0.0, 90.0, 45.0])

    def __init__(self, model) -> None:
        self.model, self.data = model, mujoco.MjData(model)
        joints = [model.joint(f"fr3_joint{i}").id for i in range(1, 8)]
        self.qadr = np.array([model.jnt_qposadr[j] for j in joints])
        self.dadr = np.array([model.jnt_dofadr[j] for j in joints])
        self.lo = np.array([model.jnt_range[j][0] for j in joints])
        self.hi = np.array([model.jnt_range[j][1] for j in joints])
        self.hand = model.body("hand").id

    def margin(self, pos: np.ndarray, yaw: float) -> float:
        """Return the smallest joint margin (deg) of a solution, or -1 when the pose is not reached."""
        model, data = self.model, self.data
        c, s = np.cos(yaw), np.sin(yaw)
        goal = np.array([[-s, c, 0.0], [c, s, 0.0], [0.0, 0.0, -1.0]])
        lo, hi = self.lo + np.radians(7.0), self.hi - np.radians(7.0)
        centre, half = (self.lo + self.hi) / 2, (self.hi - self.lo) / 2
        q = self.HOME.copy()
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        quat, e_rot = np.zeros(4), np.zeros(3)
        for it in range(110):
            data.qpos[self.qadr] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            tcp = data.xpos[self.hand] + data.xmat[self.hand].reshape(3, 3) @ self.TCP
            e_pos = pos - tcp
            mujoco.mju_mat2Quat(quat, (goal @ data.xmat[self.hand].reshape(3, 3).T).ravel())
            mujoco.mju_quat2Vel(e_rot, quat, 1.0)
            if it == 109:
                break
            pull = np.zeros(7)
            if it >= 30:  # first reach the pose, then move away from the limits
                x = (q - centre) / half
                pull = 0.02 * (-6.0 * x**5 / half)
                pull[1] += 2.0 * q[1] / 0.09 * np.exp(-(q[1] ** 2) / 0.09)
                if np.abs(pull).max() > 0.03:
                    pull *= 0.03 / np.abs(pull).max()
            mujoco.mj_jac(model, data, jacp, jacr, tcp, self.hand)
            jac = np.vstack([jacp[:, self.dadr], jacr[:, self.dadr]])
            free = np.ones(7, dtype=bool)
            for _ in range(7):
                jf = jac * free
                pinv = jf.T @ np.linalg.inv(jf @ jf.T + 0.03**2 * np.eye(6))
                dq = (pinv @ np.concatenate([e_pos, e_rot]) + (np.eye(7) - pinv @ jf) @ (pull * free)) * free
                blocked = free & (((q >= hi - 1e-9) & (dq > 0)) | ((q <= lo + 1e-9) & (dq < 0)))
                if not blocked.any():
                    break
                free &= ~blocked
            if np.abs(dq).max() > 0.1:
                dq *= 0.1 / np.abs(dq).max()
            q = np.clip(q + dq, lo, hi)
        if np.linalg.norm(e_pos) > 0.003 or np.linalg.norm(e_rot) > np.radians(2.0):
            return -1.0
        return float(np.degrees(min((q - self.lo).min(), (self.hi - q).min())))


class TestReach:
    @pytest.mark.parametrize(
        ("radius", "heights"),
        [(bs.REACH_RADIUS[0], (0.013, 0.10, 0.16)), (bs.REACH_RADIUS[1], (0.12, 0.20, 0.27))],
        ids=["nearest", "farthest"],
    )
    def test_hand_reaches_the_reach_limits_fingers_down_with_a_joint_margin(self, sim, radius, heights) -> None:
        # Four hand headings relative to the base's direction, each with its half-turn twin (the
        # hand is symmetric): the better of the two keeps every joint 8 deg inside its limits.
        ik = _ReachIK(sim.model)
        for azimuth in np.radians([0.0, 35.0, -35.0]):
            for z in heights:
                pos = np.array([radius * np.cos(azimuth), radius * np.sin(azimuth), z])
                for heading in np.radians([0.0, 45.0, 90.0, 135.0]):
                    best = max(ik.margin(pos, azimuth + heading + flip) for flip in (0.0, np.pi))
                    assert best >= 8.0, (radius, np.degrees(azimuth), z, np.degrees(heading), best)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


_OBJECT_WORDS = {
    name: {word for label, _ in bs.PHRASES.object_names[name] for word in _words(label)} | set(_words(bs.OBJECT_LABELS[name]))
    for name in OBJECTS
}
_CONTAINER_WORDS = {word for word, _ in bs.PHRASES.containers} | set(bs.PHRASES.plurals.values())


class TestPhrasing:
    def test_canonical_when_variation_is_off(self, monkeypatch) -> None:
        rng = np.random.default_rng(0)
        for _ in range(50):
            instruction = sample_instruction(rng, vary_text=False)
            assert instruction.text == instruction.canonical_text
        monkeypatch.setattr(bs, "VARY_INSTRUCTION_TEXT", False)
        assert sample_instruction(np.random.default_rng(1)).text == sample_instruction(np.random.default_rng(1)).canonical_text

    def test_object_words_name_one_object(self) -> None:
        for a, b in itertools.combinations(OBJECTS, 2):
            assert not _OBJECT_WORDS[a] & _OBJECT_WORDS[b], (a, b)
        for template, _, _ in bs.PHRASES.templates:
            words = set(_words(template))
            assert not words & set(BUCKET_COLORS), template
            assert not words & set().union(*_OBJECT_WORDS.values()), template
            assert template.count("{object}") == 1
            assert ("{bucket}" in template) != ("{color} one" in template), template

    @pytest.mark.parametrize(("obj", "color"), ALL_PAIRS)
    def test_every_sentence_names_the_right_object_and_colour_and_nothing_else(self, obj, color) -> None:
        rng = np.random.default_rng(ALL_PAIRS.index((obj, color)))
        names = [label for label, _ in bs.PHRASES.object_names[obj]] + [bs.OBJECT_LABELS[obj]]
        for _ in range(2000):
            text = bs.word_instruction(rng, obj, color)
            words = _words(text)
            assert [w for w in words if w in BUCKET_COLORS] == [color], text
            for other in OBJECTS:
                if other != obj:
                    assert not set(words) & _OBJECT_WORDS[other], text
            assert any(re.search(rf"\bthe {re.escape(name)}\b", text, re.IGNORECASE) for name in names), text
            containers = [i for i, w in enumerate(words) if w in _CONTAINER_WORDS]
            assert len(containers) == 1, text
            if "one" in words:
                assert containers[0] < words.index("one"), text  # "the red one" only after a container word
            else:
                assert words[containers[0] - 1] == color, text  # "the red bin"
            assert text[0].isupper() or text == bs.canonical_text(obj, color)
            assert text.count("please") <= 1

    def test_wording_repeats_per_seed(self) -> None:
        first = [sample_instruction(np.random.default_rng(seed)).text for seed in range(200)]
        assert first == [sample_instruction(np.random.default_rng(seed)).text for seed in range(200)]
        assert len(set(first)) > 100

    def test_distribution_keeps_the_canonical_sentence_common_and_no_phrase_rare(self) -> None:
        rng = np.random.default_rng(5)
        n = 8000
        texts = [bs.word_instruction(rng, "battery", "red") for _ in range(n)]
        canonical = sum(text == "put the 9 V battery in the red bucket" for text in texts) / n
        assert canonical == pytest.approx(bs.PHRASES.canonical_fraction, abs=0.02)
        varied = [text for text in texts if text != "put the 9 V battery in the red bucket"]
        lowered = [text.lower() for text in varied]
        for label, _ in bs.PHRASES.object_names["battery"]:
            share = sum(bool(re.search(rf"\bthe {re.escape(label.lower())}\b", t)) for t in lowered) / len(varied)
            assert share > 0.08, label
        for word, _ in bs.PHRASES.containers:
            share = sum(bool(re.search(rf"\b({word}|{bs.PHRASES.plurals[word]})\b", t)) for t in lowered) / len(varied)
            assert share > 0.08, word
        starts = Counter(" ".join(_words(t)[:2]) for t in varied if not t.lower().startswith("please"))
        assert len(starts) >= 12
        assert max(starts.values()) / len(varied) < 0.25
        assert 0.1 < sum("please" in t for t in lowered) / len(varied) < 0.3
        assert 0.1 < sum(t.endswith("?") for t in varied) / len(varied) < 0.2

    def test_task_words_its_instruction_and_can_turn_it_off(self, sim) -> None:
        task = sim.on_reset
        try:
            texts = {_reset(sim, seed).instruction.text for seed in range(40)}
            assert len(texts) > 20
            assert task.describe()["instruction"]["canonical_text"] == task.canonical_text
            task.set_vary_text(False)
            for seed in range(10):
                _reset(sim, seed)
                assert task.instruction.text == task.canonical_text
            assert task.describe()["vary_text"] is False
            # The layout and the pair do not depend on the wording.
            task.set_vary_text(True)
            worded = (_reset(sim, 3).instruction, task.layout)
            task.set_vary_text(False)
            assert (_reset(sim, 3).instruction, task.layout) == worded
        finally:
            task.set_vary_text(None)
