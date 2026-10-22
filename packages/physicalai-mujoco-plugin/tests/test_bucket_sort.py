# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The bucket_sort scene: layout, instructions and success, on the real model with the Franka FR3 (no display)."""

from __future__ import annotations

import itertools
from collections import Counter

import mujoco
import numpy as np
import pytest

from physicalai_mujoco_plugin import bucket_sort as bs
from physicalai_mujoco_plugin.bucket_sort import (
    ALL_PAIRS,
    BUCKET_COLORS,
    OBJECTS,
    BucketSortEpisode,
    BucketSortTask,
    Instruction,
    check_pairs,
    object_extent,
    sample_instruction,
)
from physicalai_mujoco_plugin.http_server import ResetCommand, SetSeedCommand
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
    def test_registered_for_one_franka_fr3(self) -> None:
        scene = get_scene("bucket_sort")
        assert scene.robots == ("franka_fr3",)
        assert scene.arm_counts == (1,)
        assert scene.free_joints == tuple(f"{name}:joint" for name in OBJECTS)
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
            assert model.body_mocapid[body.id] >= 0
            geoms = np.flatnonzero(model.geom_bodyid == body.id)
            visual = [g for g in geoms if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
            assert len(visual) == 1
            assert model.mesh(model.geom_dataid[visual[0]]).name == "bucket"
            colors[color] = tuple(model.mat_rgba[model.geom_matid[visual[0]]][:3])
            # Floor and three walls collide; the visual mesh does not.
            assert sum(model.geom_contype[g] > 0 for g in geoms) == 4
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
        assert sorted(bands) == [f"resistor_band{i}" for i in range(1, 5)]
        colors = [model.mat(model.geom_matid[bands[f"resistor_band{i}"]]).name for i in range(1, 5)]
        assert colors == ["band_brown", "band_black", "band_orange", "band_gold"]
        for geom in bands.values():
            assert model.geom_contype[geom] == 0
            y = _mesh_points(model, geom)[:, 1]
            assert 0.003 < np.ptp(y) <= 0.0041  # a 4 mm ring across the body
            assert np.abs(y).max() < 0.017  # between the end caps


class TestInstructions:
    def test_text_names_the_object_and_the_bucket(self) -> None:
        assert Instruction("cpu", "red").text == "put the CPU in the red bucket"
        assert Instruction("battery", "black").as_dict() == {
            "object": "battery",
            "bucket": "black",
            "text": "put the 9 V battery in the black bucket",
        }

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
    def test_colour_permutation_repeats_per_seed_and_varies_across_seeds(self, sim) -> None:
        def layout(seed: int) -> tuple[tuple[str, ...], Instruction, np.ndarray]:
            task = _reset(sim, seed)
            mocap = sim.data.mocap_pos.copy()
            objects = np.array([sim.data.xpos[sim.model.body(name).id] for name in OBJECTS])
            return task.slot_colors, task.instruction, np.concatenate([mocap.ravel(), objects.ravel()])

        first, again = layout(7), layout(7)
        assert first[:2] == again[:2]
        np.testing.assert_array_equal(first[2], again[2])
        permutations = {layout(seed)[0] for seed in range(60)}
        assert permutations == set(itertools.permutations(BUCKET_COLORS))

    def test_buckets_stand_in_their_slots_with_small_jitter(self, sim) -> None:
        for seed in range(30):
            task = _reset(sim, seed)
            for color, slot_y in zip(task.slot_colors, bs.SLOT_Y, strict=True):
                pos = sim.data.mocap_pos[sim.model.body_mocapid[sim.model.body(f"bucket_{color}").id]]
                assert abs(pos[0] - bs.slot_x(task.open_side)) <= bs.BUCKET_JITTER_XY
                assert abs(pos[1] - slot_y) <= bs.BUCKET_JITTER_XY
                assert pos[2] == 0.0
            assert task.describe()["slots"][0] == {"slot": "left", "y": 0.25, "color": task.slot_colors[0]}

    def test_objects_never_overlap_each_other_or_the_buckets(self, sim) -> None:
        model, data = sim.model, sim.data
        (x_lo, x_hi), (y_lo, y_hi) = bs.OBJECT_ZONE
        radii = {name: object_extent(model, name)[0] for name in OBJECTS}
        for seed in range(200):
            _reset(sim, seed)
            xy = {name: data.xpos[model.body(name).id][:2].copy() for name in OBJECTS}
            for name, (x, y) in xy.items():
                assert x_lo <= x <= x_hi
                assert y_lo <= y <= y_hi
            for a, b in itertools.combinations(OBJECTS, 2):
                assert np.linalg.norm(xy[a] - xy[b]) >= radii[a] + radii[b] + bs.OBJECT_GAP, seed
            # Nothing touches at reset: objects are released just above the table.
            assert data.ncon == 0, seed

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


class TestOpenSide:
    def test_default_faces_away_from_the_robot(self) -> None:
        assert bs.BUCKET_OPEN_SIDE == "away"
        assert bs.SLOT_X == pytest.approx(0.60)

    @pytest.mark.parametrize("side", ["away", "robot", "+y", "-y"])
    def test_buckets_turn_and_the_success_volume_follows(self, sim, side: str) -> None:
        task = sim.on_reset
        try:
            task.open_side = side
            _reset(sim, 9)
            model, data = sim.model, sim.data
            for color in BUCKET_COLORS:
                body = model.body(f"bucket_{color}").id
                opening = data.xmat[body].reshape(3, 3)[:, 0]  # the mesh's open side faces its +x
                expected = {"away": (1, 0), "robot": (-1, 0), "+y": (0, 1), "-y": (0, -1)}[side]
                assert np.degrees(np.arccos(np.clip(opening[:2] @ expected, -1, 1))) <= 5.0 + 1e-6
                near = data.xpos[body][0] - (0.06 if side in {"away", "robot"} else 0.075)
                assert abs(near - bs.BUCKET_NEAR_EDGE) <= bs.BUCKET_JITTER_XY
            instruction = task.instruction
            _drop_into(sim, instruction.object, instruction.bucket)
            _step(sim, 0.5)
            assert task.evaluate(model, data).success
            assert task.describe()["open_side"] == side
        finally:
            task.open_side = bs.BUCKET_OPEN_SIDE

    def test_rejects_an_unknown_side(self) -> None:
        with pytest.raises(ValueError, match="open side"):
            BucketSortTask(open_side="up")  # type: ignore[arg-type]


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
        episode = BucketSortEpisode(task, np.random.default_rng(0), success_dwell_s=0.2)
        episode.update(sim.model, sim.data)
        status = episode.status()
        assert status["kind"] == "bucket_sort"
        assert status["enabled"] is True
        assert status["instruction"] == task.instruction.as_dict()
        assert [slot["color"] for slot in status["slots"]] == list(task.slot_colors)
        assert status["success"] is False
        assert status["phase"] == "idle"

        done = task.instruction
        _drop_into(sim, done.object, done.bucket)
        _step(sim, 0.3)
        episode.update(sim.model, sim.data)
        assert episode.status()["phase"] == "success_hold"
        assert episode.status()["success"] is True
        _step(sim, 0.25)
        episode.update(sim.model, sim.data)
        status = episode.status()
        assert status["episode_count"] == 1
        assert status["phase"] == "idle"
        assert status["success"] is False
        assert status["in_bucket"] == dict.fromkeys(OBJECTS)

    def test_paused_episode_reports_but_does_not_restart(self, sim) -> None:
        task = _reset(sim, 8)
        episode = BucketSortEpisode(task, np.random.default_rng(0), success_dwell_s=0.0, active=False)
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
        assert episode["instruction"]["text"].startswith("put the ")

        def layout() -> tuple[object, ...]:
            robot._commands.put(ResetCommand())  # noqa: SLF001
            robot.get_observation()
            episode = robot._http_status()["episode"]  # noqa: SLF001
            return episode["instruction"]["text"], tuple(slot["color"] for slot in episode["slots"])

        robot._commands.put(SetSeedCommand(seed=11))  # noqa: SLF001
        assert layout() == layout()
        robot._commands.put(SetSeedCommand(seed=None))  # noqa: SLF001
        assert len({layout() for _ in range(8)}) > 1
    finally:
        robot.disconnect()
