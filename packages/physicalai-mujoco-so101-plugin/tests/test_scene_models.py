"""Exercise bundled models with real MuJoCo, without a renderer or display."""

from dataclasses import asdict

import mujoco
import numpy as np
import pytest

from physicalai.config import Config, instantiate
from physicalai.robot import Robot
from physicalai_mujoco_so101_plugin.mujoco_robot import BiMuJoCoSO101, MuJoCoSO101
from physicalai_mujoco_so101_plugin.scene_registry import get_scene, list_scenes


def make_robot(scene_id: str) -> MuJoCoSO101:
    scene = get_scene(scene_id)
    cls = BiMuJoCoSO101 if scene_id == "garment_fold" else MuJoCoSO101
    return cls(model_path=str(scene.scene_xml_path), scene_config=asdict(scene))


@pytest.mark.parametrize("scene_id", list(list_scenes()))
def test_scene_connect_reset_and_reconnect(scene_id: str) -> None:
    robot = instantiate(Config.from_instance(make_robot(scene_id)))
    assert isinstance(robot, Robot)
    try:
        for _ in range(2):
            robot.connect()
            assert robot._current_scene_id == scene_id
            assert robot._scene_on_reset is not None
            robot._scene_on_reset(robot._model, robot._data, np.random.default_rng(42))
            observation = robot.get_observation()
            assert observation.joint_positions.shape == (len(robot.joint_names),)
            assert np.isfinite(robot._data.qpos).all()
            robot.send_action(observation.joint_positions)
            robot.disconnect()
            assert not robot.is_connected()
    finally:
        robot.disconnect()


def test_switch_initializes_cube_and_keeps_target_fixed() -> None:
    robot = make_robot("pick_lift")
    robot.connect()
    try:
        scene = get_scene("single_pick_place")
        defaults = mujoco.MjModel.from_xml_path(str(scene.scene_xml_path))
        data = mujoco.MjData(defaults)
        mujoco.mj_forward(defaults, data)
        target = defaults.body("target").id
        target_position = data.xpos[target].copy()
        cube_default = data.joint("block1:joint").qpos.copy()

        assert robot._switch_to_scene("single_pick_place")
        cube = robot._data.joint("block1:joint").qpos
        assert not np.array_equal(cube, cube_default)
        np.testing.assert_array_equal(robot._data.xpos[robot._model.body("target").id], target_position)
        assert np.linalg.norm(cube[:2] - target_position[:2]) >= scene.target_min_sep
        assert robot._episode_auto_reset.status()["phase"] == "idle"
    finally:
        robot.disconnect()


def test_failed_switch_reset_preserves_live_scene(monkeypatch: pytest.MonkeyPatch) -> None:
    robot = make_robot("single_pick_place")
    robot.connect()
    model = robot._model

    def fail(*args: object) -> None:
        raise ValueError("reset failed")

    monkeypatch.setattr("physicalai_mujoco_so101_plugin.scene_registry.get_reset_fn", lambda _: fail)
    try:
        with pytest.raises(ValueError, match="reset failed"):
            robot._switch_to_scene("pick_lift")
        assert robot._model is model
        assert robot._current_scene_id == "single_pick_place"
        assert np.isfinite(robot.get_observation().joint_positions).all()
    finally:
        robot.disconnect()


def test_incompatible_initial_model_leaves_robot_disconnected() -> None:
    robot = BiMuJoCoSO101(model_path=str(get_scene("single_pick_place").scene_xml_path))
    with pytest.raises(ValueError, match="joints/actuators"):
        robot.connect()
    assert not robot.is_connected()


def test_only_v4l2_outputs_claim_devices(tmp_path) -> None:
    sink = tmp_path / "video60"
    alias = tmp_path / "camera"
    alias.symlink_to(sink)
    robot = MuJoCoSO101(
        model_path="scene.xml",
        cameras=[{"name": "wrist", "device": str(alias)}, {"name": "overview"}],
    )
    assert robot.device_ids == (f"v4l2:{sink}",)
    assert make_robot("single_pick_place").device_ids == make_robot("garment_fold").device_ids == ()
