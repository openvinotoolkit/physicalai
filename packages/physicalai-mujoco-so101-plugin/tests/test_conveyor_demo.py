"""Scripted conveyor_sort demonstrator on the bundled model (real MuJoCo, no display)."""

import mujoco
import numpy as np
import pytest
from loguru import logger

from physicalai_mujoco_so101_plugin.conveyor import ConveyorSort
from physicalai_mujoco_so101_plugin.conveyor_demo import ArmIK, ConveyorDemonstrator, DemoConfig, _yaw_distance
from physicalai_mujoco_so101_plugin.scene_registry import get_reset_fn, get_scene

SUBSTEPS = 10


@pytest.fixture
def sim() -> tuple[object, object, ConveyorSort]:
    model = get_scene("conveyor_sort").load_model()
    data = mujoco.MjData(model)
    get_reset_fn("conveyor_sort")(model, data, np.random.default_rng(0))
    conveyor = ConveyorSort.maybe_create(model, rng=np.random.default_rng(1), belt_speed=0.03)
    assert conveyor is not None
    return model, data, conveyor


@pytest.mark.parametrize("y", [-0.15, -0.05, 0.05, 0.14])
@pytest.mark.parametrize("yaw", [np.pi / 2, np.pi / 4])
def test_ik_reaches_grasps_across_the_pick_window(sim: tuple, y: float, yaw: float) -> None:
    model, data, _ = sim
    ik = ArmIK(model)
    q0 = data.qpos[ik.qpos_adr].copy()
    tilts = []
    # The jaw works either way round; the demonstrator picks the more upright variant.
    for flip in (0.0, np.pi):
        q, err = ik.solve(q0, np.array([0.25, y, 0.08]), yaw + flip, iters=80)
        assert err < 0.001
        assert np.all(q >= ik.lower) and np.all(q <= ik.upper)
        ik.data.qpos[ik.qpos_adr] = q
        mujoco.mj_kinematics(model, ik.data)
        if _yaw_distance(ik.gripper_yaw(ik.data), yaw) < np.radians(2):
            tilts.append(ik.tilt(ik.data))
    assert tilts
    assert min(tilts) < np.radians(8)


def test_ik_reaches_every_bin(sim: tuple) -> None:
    model, data, conveyor = sim
    ik = ArmIK(model)
    q0 = data.qpos[ik.qpos_adr].copy()
    for name, body in conveyor.bin_body_ids.items():
        target = data.xpos[body] + np.array([0.0, 0.0, DemoConfig().drop_z])
        for yaw in (np.pi / 2, -np.pi / 2, 0.0):
            _, err = ik.solve(q0, target, yaw, iters=80)
            assert err < 0.001, (name, yaw)


def test_grasp_yaw_closes_along_the_belt_and_matches_the_faces(sim: tuple) -> None:
    model, data, conveyor = sim
    demo = ConveyorDemonstrator(model, conveyor)
    cube = next(i for i in conveyor._items if i.shape == "cube")
    for item_yaw in (0.0, 0.3, 1.2, -2.0):
        adr = cube.qpos_adr
        data.qpos[adr + 3 : adr + 7] = (np.cos(item_yaw / 2), 0.0, 0.0, np.sin(item_yaw / 2))
        mujoco.mj_kinematics(model, data)
        yaw = demo._grasp_yaw(data, cube, 0.0)
        # Within 45 deg of the belt direction, and on a face normal (multiple of 90 deg off the cube).
        assert _yaw_distance(yaw, np.pi / 2) <= np.pi / 4 + 1e-9
        assert np.isclose(((yaw - item_yaw) / (np.pi / 2)) % 1.0, 0.0, atol=1e-9) or np.isclose(
            ((yaw - item_yaw) / (np.pi / 2)) % 1.0, 1.0, atol=1e-9
        )
    cylinder = next(i for i in conveyor._items if i.shape == "cylinder")
    assert demo._grasp_yaw(data, cylinder, np.pi) == pytest.approx(1.5 * np.pi)


@pytest.mark.slow
def test_demonstrator_sorts_a_short_episode(sim: tuple) -> None:
    model, data, conveyor = sim
    logger.disable("physicalai_mujoco_so101_plugin")
    try:
        conveyor._config.items_per_episode = 3
        demo = ConveyorDemonstrator(model, conveyor)
        dt = SUBSTEPS * model.opt.timestep
        released_at: list[float] = []  # sim time of every tick spent releasing an item
        while conveyor.status()["episode_count"] < 1 and data.time < 60.0:
            demo.step(data, dt)
            if demo.phase == "release":
                released_at.append(float(data.time))
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)
            conveyor.update(model, data)
    finally:
        logger.enable("physicalai_mujoco_so101_plugin")
    status = conveyor.status()
    assert status["episode_count"] == 1
    assert status["last_episode"]["correct"] >= 2
    assert demo.stats.attempts >= 3
    # The jaws stay open for the whole release time at drop height, not what is left after lowering.
    releases = np.split(np.asarray(released_at), np.flatnonzero(np.diff(released_at) > 2 * dt) + 1)
    assert len(releases) >= 2
    assert all(ticks[-1] - ticks[0] >= demo.config.release_s - 2 * dt for ticks in releases)
