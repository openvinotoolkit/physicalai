# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Scene resets keep spawned objects out of the arms' footprint (spawn.ArmFootprint).

The synthetic scenes have one "arm": a base at the origin, a low hand hanging over the spawn area
and a forearm high above it. Only the hand reaches into the footprint's height band.
"""

# MuJoCo's bindings are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from loguru import logger

from physicalai_mujoco_plugin import scene_registry, spawn
from physicalai_mujoco_plugin.profiles import get_profile
from physicalai_mujoco_plugin.scene_registry import SceneConfig, get_scene
from physicalai_mujoco_plugin.sim import load_sim
from physicalai_mujoco_plugin.spawn import (
    ARM_CLEARANCE,
    ARM_FOOTPRINT_HEIGHT,
    ArmFootprint,
    arm_footprint,
    object_radius,
    sample_clear_position,
)

_SCENE = """<mujoco><compiler angle="radian"/><worldbody>
  <geom name="floor" type="plane" size="1 1 0.01"/>
  <body name="base"><joint name="pan" axis="0 0 1"/><geom type="box" size="0.02 0.02 0.02" pos="0 0 0.02"/>
    <body name="hand" pos="{hand_x} 0 0.06"><geom name="hand" type="box" size="0.03 0.03 0.02"/></body>
    <body name="forearm" pos="0.22 0 0.30"><geom name="forearm" type="box" size="0.10 0.10 0.02"/></body>
  </body>
  <body name="target" pos="0.6 0.4 0"><geom type="cylinder" size="0.05 0.001" contype="0" conaffinity="0"/></body>
  {objects}
</worldbody><actuator><position name="pan" joint="pan"/></actuator></mujoco>"""
_BLOCK = '<body name="block1" pos="0.5 0.5 0.02"><freejoint name="block1:joint"/><geom type="box" size="0.02 0.02 0.02"/></body>'
_DICE = "".join(
    f'<body name="die{i}" pos="0.5 {0.1 * i} 0.01"><freejoint name="die{i}:joint"/><geom type="box" size="0.008 0.008 0.008"/></body>'
    for i in range(1, 7)
)
_DISC = SceneConfig(
    scene_id="footprint_test",
    display_name="Footprint test",
    description="One block on a disc around the arm's low hand",
    scene_xml_relpath="unused.xml",
    free_joints=("block1:joint",),
    target_bodies=("target",),
    spawn_center=(0.22, 0.0),
    spawn_min_r=0.0,
    spawn_max_r=0.12,
    spawn_angle_half_deg=180.0,
)
_SEEDS = range(60)


def _scene(objects: str, hand_x: float = 0.22) -> tuple[mujoco.MjModel, mujoco.MjData, tuple[int, ...]]:
    model = mujoco.MjModel.from_xml_string(_SCENE.format(objects=objects, hand_x=hand_x))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, (model.body("base").id,)


def _square(half: float, center: tuple[float, float] = (0.0, 0.0)) -> ArmFootprint:
    corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]]) + center
    starts = np.concatenate([corners, np.repeat(corners[-1:], 4, axis=0)])[None]
    return ArmFootprint(starts=starts, ends=np.roll(starts, -1, axis=1))


class TestArmFootprint:
    def test_only_robot_geoms_low_over_the_table_count(self) -> None:
        model, data, roots = _scene(_BLOCK)

        footprint = ArmFootprint.of_robots(model, data, roots, ARM_FOOTPRINT_HEIGHT)

        assert len(footprint.starts) == 2  # base and hand; the forearm is 28 cm up, the block is no robot
        assert footprint.distance((0.22, 0.0)) == 0.0
        assert footprint.distance((0.22, 0.10)) == pytest.approx(0.07)  # 7 cm past the hand's +y face
        assert len(ArmFootprint.of_robots(model, data, roots, 0.5).starts) == 3

    def test_a_turned_geom_covers_its_own_outline_not_its_bounding_square(self) -> None:
        model, data, roots = _scene("")
        model.body("hand").quat = [math.cos(math.pi / 8), 0, 0, math.sin(math.pi / 8)]  # 45 degrees about z
        mujoco.mj_forward(model, data)

        footprint = ArmFootprint.of_robots(model, data, roots, ARM_FOOTPRINT_HEIGHT)

        diagonal = 0.03 * math.sqrt(2)
        assert footprint.distance((0.22 + diagonal - 1e-4, 0.0)) == 0.0
        # The corner of the axis-aligned bounding square lies outside the diamond.
        assert footprint.distance((0.22 + 0.03, 0.03)) > 0.01

    def test_the_rule_can_be_switched_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model, data, roots = _scene(_BLOCK)

        assert arm_footprint(model, data, ()) is None
        assert arm_footprint(model, data, roots) is not None
        monkeypatch.setattr(spawn, "SPAWN_CLEAR_OF_ARMS", False)
        assert arm_footprint(model, data, roots) is None

    def test_object_radius_holds_for_any_yaw(self) -> None:
        model, _, _ = _scene(_BLOCK)

        assert object_radius(model, ("block1:joint",)) == pytest.approx(0.02 * math.sqrt(2))
        assert object_radius(model, ("missing:joint",)) == 0.0


class TestSampleClearPosition:
    def _draws(self, *points: tuple[float, float]) -> object:
        sequence = iter(points)
        return lambda: next(sequence)

    def test_without_a_footprint_the_first_draw_clear_of_the_others_wins(self) -> None:
        draw = self._draws((0.0, 0.0), (0.5, 0.0), (0.6, 0.0))

        assert sample_clear_position(draw, clear_of_others=lambda xy: xy[0] > 0.1) == (0.5, 0.0)

    def test_a_draw_too_close_to_the_arm_is_drawn_again(self) -> None:
        draw = self._draws((0.0, 0.0), (0.03, 0.0), (0.2, 0.0))

        assert sample_clear_position(draw, footprint=_square(0.01), padding=0.05) == (0.2, 0.0)

    def test_without_a_clear_draw_the_one_farthest_from_the_arm_is_kept(self) -> None:
        messages: list[str] = []
        sink = logger.add(messages.append, level="DEBUG", format="{message}")
        try:
            draws = ((0.0, 0.0), (0.05, 0.0), (0.03, 0.0), (0.07, 0.0))
            kept = sample_clear_position(
                self._draws(*draws),
                clear_of_others=lambda xy: xy != (0.07, 0.0),  # the farthest sits on the target
                footprint=_square(0.01),
                padding=1.0,
                attempts=len(draws),
            )
        finally:
            logger.remove(sink)

        assert kept == (0.05, 0.0)
        assert any("No spawn position clear of the arms in 4 draws" in message for message in messages)

    def test_without_any_draw_clear_of_the_others_the_last_one_is_kept(self) -> None:
        draw = self._draws((0.0, 0.0), (0.05, 0.0))

        kept = sample_clear_position(
            draw, clear_of_others=lambda _xy: False, footprint=_square(0.01), padding=1.0, attempts=2
        )

        assert kept == (0.05, 0.0)


class TestResets:
    """The single_pick_place and yahtzee resets keep their objects off the arm at reset time."""

    def test_the_block_never_spawns_under_the_low_hand(self) -> None:
        model, data, roots = _scene(_BLOCK)
        footprint = ArmFootprint.of_robots(model, data, roots, ARM_FOOTPRINT_HEIGHT)
        padding = object_radius(model, ("block1:joint",)) + ARM_CLEARANCE
        reset = scene_registry._freejoint_spawn_reset(_DISC, roots)  # noqa: SLF001
        ignoring_arm = scene_registry._freejoint_spawn_reset(_DISC)  # noqa: SLF001

        distances = []
        for seed in _SEEDS:
            reset(model, data, np.random.default_rng(seed))
            distances.append(footprint.distance(tuple(data.joint("block1:joint").qpos[:2])))
        unguarded = []
        for seed in _SEEDS:
            ignoring_arm(model, data, np.random.default_rng(seed))
            unguarded.append(footprint.distance(tuple(data.joint("block1:joint").qpos[:2])))

        assert min(distances) >= padding
        assert min(unguarded) < padding  # the disc does reach under the hand

    def test_the_hand_counts_where_the_reset_finds_it(self) -> None:
        model, data, roots = _scene(_BLOCK, hand_x=0.0)
        reset = scene_registry._freejoint_spawn_reset(_DISC, roots)  # noqa: SLF001
        model.body("hand").pos = [0.22, 0.0, 0.06]  # the arm moved over the disc after the reset was built
        footprint = ArmFootprint.of_robots(model, data, roots, ARM_FOOTPRINT_HEIGHT)
        padding = object_radius(model, ("block1:joint",)) + ARM_CLEARANCE

        for seed in _SEEDS:
            reset(model, data, np.random.default_rng(seed))
            assert footprint.distance(tuple(data.joint("block1:joint").qpos[:2])) >= padding

    def test_the_dice_never_drop_onto_the_low_hand(self) -> None:
        model, data, roots = _scene(_DICE, hand_x=0.42)
        footprint = ArmFootprint.of_robots(model, data, roots, ARM_FOOTPRINT_HEIGHT)
        dice = tuple(f"die{i}:joint" for i in range(1, 7))
        padding = object_radius(model, dice) + ARM_CLEARANCE
        reset = scene_registry._yahtzee_reset(roots)  # noqa: SLF001

        distances = []
        for seed in _SEEDS:
            reset(model, data, np.random.default_rng(seed))
            distances += [footprint.distance(tuple(data.joint(die).qpos[:2])) for die in dice]
        unguarded = []
        for seed in _SEEDS:
            scene_registry._yahtzee_reset()(model, data, np.random.default_rng(seed))  # noqa: SLF001
            unguarded += [footprint.distance(tuple(data.joint(die).qpos[:2])) for die in dice]

        assert min(distances) >= padding
        assert min(unguarded) < padding

    def test_with_the_rule_off_the_reset_ignores_the_arm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model, data, roots = _scene(_BLOCK)
        reset = scene_registry._freejoint_spawn_reset(_DISC, roots)  # noqa: SLF001
        ignoring_arm = scene_registry._freejoint_spawn_reset(_DISC)  # noqa: SLF001
        monkeypatch.setattr(spawn, "SPAWN_CLEAR_OF_ARMS", False)

        for seed in _SEEDS:
            reset(model, data, np.random.default_rng(seed))
            switched_off = data.qpos.copy()
            ignoring_arm(model, data, np.random.default_rng(seed))
            np.testing.assert_array_equal(data.qpos, switched_off)


@pytest.mark.requires_download
@pytest.mark.parametrize("arms", [1, 2])
def test_so_arm100_blocks_spawn_clear_of_its_jaws(arms: int) -> None:
    """The SO-ARM100's home pose hangs its jaws 7 cm over the middle of the spawn arc."""
    profile = get_profile("so_arm100")
    scene = get_scene("single_pick_place")
    sim = load_sim(
        scene.scene_xml_path,
        scene,
        profile,
        unit="degrees",
        torque_mode="pd",
        rng=np.random.default_rng(0),
        reseed=lambda: None,
        arms=arms,
    )
    height = ARM_FOOTPRINT_HEIGHT * scene.layout_scale(profile)
    footprint = ArmFootprint.of_robots(sim.model, sim.data, sim.robot_roots, height)
    padding = object_radius(sim.model, scene.free_joints) + ARM_CLEARANCE
    assert sim.on_reset is not None

    for seed in _SEEDS:
        sim.on_reset(sim.model, sim.data, np.random.default_rng(seed))
        assert footprint.distance(tuple(sim.data.joint("block1:joint").qpos[:2])) >= padding
