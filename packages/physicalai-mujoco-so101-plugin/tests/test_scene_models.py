"""Exercise bundled models with real MuJoCo, without a renderer or display."""

from dataclasses import asdict
from unittest.mock import MagicMock

import mujoco
import numpy as np
import pytest

from physicalai.config import Config, instantiate
from physicalai.robot import Robot
from physicalai_mujoco_so101_plugin._urdf import get_urdf_path
from physicalai_mujoco_so101_plugin.http_server import (
    HomeCommand,
    SetAutoResetCommand,
    SetObjectPoseCommand,
    SetSeedCommand,
)
from physicalai_mujoco_so101_plugin.mujoco_robot import BiMuJoCoSO101, MuJoCoSO101
from physicalai_mujoco_so101_plugin.scene_registry import get_scene, list_scenes, list_scenes_for_arms
from physicalai_mujoco_so101_plugin.viser_controls import ObjectPose


def make_robot(scene_id: str) -> MuJoCoSO101:
    scene = get_scene(scene_id)
    cls = BiMuJoCoSO101 if scene.num_arms == 2 else MuJoCoSO101
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
    robot = make_robot("yahtzee")
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
            robot._switch_to_scene("yahtzee")
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


@pytest.mark.parametrize("scene_id", ["single_pick_place", "garment_fold"])
def test_home_places_arm_joints_and_targets(scene_id: str) -> None:
    robot = make_robot(scene_id)
    robot.connect()
    try:
        robot.send_action(np.full(len(robot.joint_names), 25.0, dtype=np.float32))
        for _ in range(30):
            robot.get_observation()

        robot._commands.put(HomeCommand())
        robot._drain_commands()

        home = dict(get_scene(scene_id).home_qpos)
        model, data = robot._model, robot._data
        for index, name in enumerate(robot.joint_names):
            joint = model.joint(name)
            expected = home.get(name, float(model.qpos0[joint.qposadr[0]]))
            if model.jnt_limited[joint.id]:
                expected = float(np.clip(expected, *model.jnt_range[joint.id]))
            assert data.joint(name).qpos[0] == pytest.approx(expected)
            assert data.joint(name).qvel[0] == 0.0
            assert data.ctrl[robot._ctrl_indices[index]] == pytest.approx(expected)
    finally:
        robot.disconnect()


def test_object_pose_holds_while_dragged_and_falls_when_released() -> None:
    robot = make_robot("single_pick_place")
    robot.connect()
    try:
        lifted = (0.2, 0.05, 0.15)
        robot._commands.put(SetObjectPoseCommand("block1:joint", lifted, (2.0, 0.0, 0.0, 0.0), hold=True))
        for _ in range(20):
            robot.get_observation()
        cube = robot._data.joint("block1:joint")
        np.testing.assert_allclose(cube.qpos[:3], lifted)
        np.testing.assert_allclose(cube.qpos[3:], [1.0, 0.0, 0.0, 0.0])  # normalized
        assert robot._http_status()["objects"][0]["position"] == pytest.approx(list(lifted))

        robot._commands.put(SetObjectPoseCommand("block1:joint", lifted, hold=False))
        for _ in range(60):
            robot.get_observation()
        assert robot._data.joint("block1:joint").qpos[2] < lifted[2]
        assert robot._held_objects == {}
    finally:
        robot.disconnect()


def test_http_status_lists_every_free_object() -> None:
    robot = make_robot("yahtzee")
    robot.connect()
    try:
        joints = [obj["joint"] for obj in robot._http_status()["objects"]]
        assert joints == list(get_scene("yahtzee").free_joints)
    finally:
        robot.disconnect()


def test_fixed_seed_repeats_scene_switch_layouts() -> None:
    robot = make_robot("single_pick_place")
    robot.connect()
    try:
        robot._commands.put(SetSeedCommand(seed=2024))
        robot._drain_commands()
        layouts = []
        for _ in range(2):
            assert robot._switch_to_scene("yahtzee")
            layouts.append(np.concatenate([robot._data.joint(j).qpos[:3] for j in get_scene("yahtzee").free_joints]))
        np.testing.assert_allclose(layouts[0], layouts[1])
    finally:
        robot.disconnect()


def test_auto_reset_settings_survive_scene_switches() -> None:
    robot = make_robot("single_pick_place")
    robot.connect()
    try:
        robot._commands.put(SetAutoResetCommand(enabled=False, dwell_s=2.5))
        robot._drain_commands()
        assert robot._switch_to_scene("yahtzee")
        assert robot._http_status()["episode"] == {"enabled": False}
        assert robot._switch_to_scene("single_pick_place")

        episode = robot._http_status()["episode"]
        assert episode["active"] is False
        assert episode["success_dwell_s"] == 2.5
    finally:
        robot.disconnect()


def test_compatible_scene_lists_match_real_models() -> None:
    for scene_id, scene in list_scenes().items():
        robot_cls = BiMuJoCoSO101 if scene.num_arms == 2 else MuJoCoSO101
        model = mujoco.MjModel.from_xml_path(str(scene.scene_xml_path))
        assert robot_cls(model_path="unused")._actuator_indices_for_joint_order(model) is not None, scene_id
        assert scene_id in list_scenes_for_arms(scene.num_arms)


@pytest.mark.parametrize(
    ("scene_id", "expected"),
    [
        ("single_pick_place", ("block1", "target", "gripper")),
        ("yahtzee", ("die1", "die2", "die3", "die4", "die5", "die6", "gripper")),
        ("garment_fold", ("left_gripper", "right_gripper")),
    ],
)
def test_viewer_follow_targets(scene_id: str, expected: tuple[str, ...]) -> None:
    robot = make_robot(scene_id)
    robot.connect()
    try:
        assert tuple(robot._follow_body_ids) == expected
        for name, body_id in robot._follow_body_ids.items():
            assert robot._model.body(body_id).name == name
    finally:
        robot.disconnect()


def test_viewer_follow_targets_track_bodies_and_the_world_stays_put() -> None:
    robot = make_robot("yahtzee")
    robot.connect()
    try:
        state = robot._panel_state()
        dice = tuple(f"die{i}" for i in range(1, 7))
        assert tuple(state.follow_targets) == (*dice, "gripper")
        assert state.object_bodies == {f"{die}:joint": die for die in dice}
        np.testing.assert_allclose(state.view_center, robot._model.stat.center)

        pose = ObjectPose(position=(0.3, 0.1, 0.1), wxyz=(1.0, 0.0, 0.0, 0.0))
        robot._set_object_pose("die2:joint", pose.position, pose.wxyz, hold=False)
        np.testing.assert_allclose(robot._panel_state().follow_targets["die2"], pose.position)

        scene = MagicMock()
        robot._viser_scene = scene
        robot._sync_viser()
        scene.update_from_mjdata.assert_called_once_with(robot._data)
    finally:
        robot._viser_scene = None
        robot.disconnect()


def _wrist_camera_pose_in_gripper(model_path: str, camera: str, gripper: str) -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gripper_rot = data.xmat[model.body(gripper).id].reshape(3, 3)
    cam = model.camera(camera).id
    position = gripper_rot.T @ (data.cam_xpos[cam] - data.xpos[model.body(gripper).id])
    rotation = gripper_rot.T @ data.cam_xmat[cam].reshape(3, 3)
    return position, rotation


@pytest.mark.parametrize(
    ("model_path", "camera", "gripper"),
    [
        (str(get_scene("single_pick_place").scene_xml_path), "wrist", "gripper"),
        (str(get_scene("garment_fold").scene_xml_path), "left_wrist", "left_gripper"),
        (str(get_scene("garment_fold").scene_xml_path), "right_wrist", "right_gripper"),
    ],
)
def test_wrist_cameras_match_the_reference_model(model_path: str, camera: str, gripper: str) -> None:
    """The arm is defined in three XML files; their wrist cameras must stay identical."""
    reference = str(get_urdf_path() / "so101/so101.xml")
    ref_position, ref_rotation = _wrist_camera_pose_in_gripper(reference, "wrist", "gripper")
    position, rotation = _wrist_camera_pose_in_gripper(model_path, camera, gripper)

    np.testing.assert_allclose(position, ref_position, atol=1e-6)
    np.testing.assert_allclose(rotation, ref_rotation, atol=1e-6)
    # As on the physical SO-101, the wrist camera is on the gripper's -y side.
    assert position[1] < 0


@pytest.mark.parametrize(
    ("scene_id", "camera", "tip_site"),
    [
        ("single_pick_place", "wrist", "gripperframe"),
        ("garment_fold", "left_wrist", "left_gripperframe"),
        ("garment_fold", "right_wrist", "right_gripperframe"),
    ],
)
def test_wrist_cameras_see_the_jaws_in_the_lower_half(scene_id: str, camera: str, tip_site: str) -> None:
    """Frames are streamed as MuJoCo renders them, so camera-frame -y is the bottom of the image."""
    model = mujoco.MjModel.from_xml_path(str(get_scene(scene_id).scene_xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = model.camera(camera).id
    tip = data.cam_xmat[cam].reshape(3, 3).T @ (data.site_xpos[model.site(tip_site).id] - data.cam_xpos[cam])
    assert tip[2] < 0  # in front of the camera (MuJoCo cameras look along -z)
    assert tip[1] < 0  # below the image centre


@pytest.mark.parametrize("scene_id", list(list_scenes()))
def test_scene_xml_camera_reload_keeps_the_compiled_poses(scene_id: str) -> None:
    """Live XML camera edits are re-applied by hand; unchanged XML must give the compiled poses."""
    robot = make_robot(scene_id)
    robot.connect()
    try:
        model, data = robot._model, robot._data
        compiled = data.cam_xmat.copy()
        robot._update_camera_from_xml()
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.cam_xmat, compiled, atol=1e-6)
    finally:
        robot.disconnect()


def _urdf_joint_limits(urdf_name: str) -> dict[str, tuple[float, float]]:
    from defusedxml import ElementTree

    root = ElementTree.parse(get_urdf_path() / urdf_name).getroot()
    return {
        joint.get("name"): (float(limit.get("lower")), float(limit.get("upper")))
        for joint in root.iter("joint")
        if (limit := joint.find("limit")) is not None
    }


@pytest.mark.parametrize(
    ("model_path", "urdf_name"),
    [
        *[
            (str(get_scene(scene_id).scene_xml_path), "so101/so101_new_calib.urdf")
            for scene_id in list_scenes_for_arms(1)
        ],
        *[(str(get_scene(scene_id).scene_xml_path), "so101/so101_dual.urdf") for scene_id in list_scenes_for_arms(2)],
        (str(get_urdf_path() / "so101/so101.xml"), "so101/so101_new_calib.urdf"),
    ],
)
def test_joint_and_control_ranges_match_the_urdf(model_path: str, urdf_name: str) -> None:
    """The URDF is the source of truth; normalized units span these ranges."""
    model = mujoco.MjModel.from_xml_path(model_path)
    limits = _urdf_joint_limits(urdf_name)

    for name, (lower, upper) in limits.items():
        np.testing.assert_allclose(model.jnt_range[model.joint(name).id], (lower, upper), atol=1e-5, err_msg=name)
        np.testing.assert_allclose(model.actuator(name).ctrlrange, (lower, upper), atol=1e-4, err_msg=name)
