"""Exercise bundled models with real MuJoCo, without a renderer or display."""

import time
from dataclasses import asdict
from unittest.mock import MagicMock, patch

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
        defaults = scene.load_model()
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


def test_simulation_claims_no_devices() -> None:
    robot = MuJoCoSO101(model_path="scene.xml", cameras=[{"name": "wrist"}, {"name": "overview"}])
    assert robot.device_ids == ()
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
        model = scene.load_model()
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


def _wrist_camera_pose_in_gripper(model: mujoco.MjModel, camera: str, gripper: str) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gripper_rot = data.xmat[model.body(gripper).id].reshape(3, 3)
    cam = model.camera(camera).id
    position = gripper_rot.T @ (data.cam_xpos[cam] - data.xpos[model.body(gripper).id])
    rotation = gripper_rot.T @ data.cam_xmat[cam].reshape(3, 3)
    return position, rotation


@pytest.mark.parametrize(
    ("scene_id", "camera", "gripper"),
    [
        *[(scene_id, "wrist", "gripper") for scene_id in list_scenes_for_arms(1)],
        ("garment_fold", "left_wrist", "left_gripper"),
        ("garment_fold", "right_wrist", "right_gripper"),
    ],
)
def test_wrist_cameras_keep_their_pose_on_the_gripper(scene_id: str, camera: str, gripper: str) -> None:
    """Every arm's wrist camera keeps the pose and field of view that trained policies saw."""
    model = get_scene(scene_id).load_model()
    position, rotation = _wrist_camera_pose_in_gripper(model, camera, gripper)
    expected_rotation = np.zeros(9)
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array([0.57, 0.0, np.pi]), "xyz")
    mujoco.mju_quat2Mat(expected_rotation, quat)

    np.testing.assert_allclose(position, [0.0, -0.055, -0.045], atol=1e-6)
    np.testing.assert_allclose(rotation, expected_rotation.reshape(3, 3), atol=1e-6)
    assert model.cam_fovy[model.camera(camera).id] == pytest.approx(75.0)
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
    model = get_scene(scene_id).load_model()
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
    ("scene_id", "urdf_name"),
    [
        *[(scene_id, "so101/so101_new_calib.urdf") for scene_id in list_scenes_for_arms(1)],
        *[(scene_id, "so101/so101_dual.urdf") for scene_id in list_scenes_for_arms(2)],
    ],
)
def test_joint_and_control_ranges_match_the_urdf(scene_id: str, urdf_name: str) -> None:
    """The URDF is the source of truth; normalized units span these ranges."""
    model = get_scene(scene_id).load_model()
    limits = _urdf_joint_limits(urdf_name)

    for name, (lower, upper) in limits.items():
        np.testing.assert_allclose(model.jnt_range[model.joint(name).id], (lower, upper), atol=1e-5, err_msg=name)
        np.testing.assert_allclose(model.actuator(name).ctrlrange, (lower, upper), atol=1e-4, err_msg=name)


def test_cameras_render_on_their_own_thread_from_pose_snapshots() -> None:
    scene = get_scene("conveyor_sort")
    robot = MuJoCoSO101(
        model_path=str(scene.scene_xml_path), scene_config=asdict(scene), cameras=[{"name": "overview", "fps": 100}]
    )
    rendered = np.full((4, 6, 3), 7, dtype=np.uint8)
    renderer = MagicMock()
    renderer.render.return_value = rendered
    with patch("mujoco.Renderer", return_value=renderer):
        robot.connect()
        try:
            assert robot._camera_thread is not None
            deadline = time.monotonic() + 5.0
            while robot._frame_buffers["overview"].snapshot() is None and time.monotonic() < deadline:
                robot._step_and_sync()  # publishes a pose snapshot every tick
                time.sleep(0.01)
            snapshot = robot._frame_buffers["overview"].snapshot()
            assert snapshot is not None
            np.testing.assert_array_equal(snapshot.frame, rendered)
            # The renderer got the camera thread's own MjData, not the simulation's.
            rendered_from = renderer.update_scene.call_args.args[0]
            assert rendered_from is not robot._data
            np.testing.assert_allclose(rendered_from.qpos, robot._data.qpos)
            for _ in range(5):
                robot._step_and_sync()
                time.sleep(0.01)
            timing = robot._http_status()["timing"]
            assert timing["cameras_on_thread"] is True
            assert timing["control_hz"] is not None
        finally:
            robot.disconnect()
    assert robot._camera_thread is None
    renderer.close.assert_called()


def test_a_camera_thread_that_does_not_stop_keeps_its_own_renderers() -> None:
    """A render stuck past stop() must not see, or close, the next scene's renderers."""
    import threading

    from physicalai_mujoco_so101_plugin import camera_thread

    scene = get_scene("conveyor_sort")
    robot = MuJoCoSO101(
        model_path=str(scene.scene_xml_path), scene_config=asdict(scene), cameras=[{"name": "overview", "fps": 100}]
    )
    rendering, release = threading.Event(), threading.Event()
    stuck = MagicMock()

    def hang() -> np.ndarray:
        rendering.set()
        release.wait(10.0)
        return np.full((4, 6, 3), 99, dtype=np.uint8)  # an old-scene frame

    stuck.render.side_effect = hang
    fresh = MagicMock()
    fresh.render.return_value = np.zeros((4, 6, 3), dtype=np.uint8)
    original_stop = camera_thread.CameraThread.stop
    renderers = iter([stuck, fresh])
    built_for: list[object] = []

    def build(model: object, _height: int, _width: int) -> MagicMock:
        built_for.append(model)
        return next(renderers)

    with (
        patch("mujoco.Renderer", side_effect=build),
        patch.object(camera_thread.CameraThread, "stop", lambda self, timeout_s=0.2: original_stop(self, timeout_s)),
    ):
        robot.connect()
        try:
            deadline = time.monotonic() + 5.0
            while not rendering.is_set() and time.monotonic() < deadline:
                robot._step_and_sync()
                time.sleep(0.01)
            assert rendering.is_set()
            old_model = robot._model
            assert robot._switch_to_scene("single_pick_place")  # the old thread is still inside render()
            stuck.close.assert_not_called()  # left to the thread that still uses it
            deadline = time.monotonic() + 5.0
            while fresh.update_scene.call_count == 0 and time.monotonic() < deadline:
                robot._step_and_sync()
                time.sleep(0.01)
            assert fresh.update_scene.call_count > 0
            release.set()
            deadline = time.monotonic() + 5.0
            while not stuck.close.called and time.monotonic() < deadline:
                time.sleep(0.01)
            stuck.close.assert_called_once()  # by its own thread, once the render returned
            assert built_for == [old_model, robot._model]  # each thread renders its own model
            latest = robot._frame_buffers["overview"].snapshot()
            assert latest is not None
            assert not (latest.frame == 99).all()  # the late old-scene frame was dropped
            fresh.close.assert_not_called()
            assert stuck.update_scene.call_count == 1  # never used again after the switch
        finally:
            release.set()
            robot.disconnect()
    fresh.close.assert_called_once()


def test_the_camera_thread_renders_nothing_before_the_first_pose() -> None:
    """A zeroed snapshot puts every body at the origin with zero quaternions; never render it."""
    from physicalai_mujoco_so101_plugin.camera_thread import CameraThread

    model = mujoco.MjModel.from_xml_path(str(get_scene("conveyor_sort").scene_xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    seen: list[np.ndarray] = []
    thread = CameraThread(model, setup=lambda _model: None, render=lambda d: seen.append(d.qpos.copy()), teardown=lambda: None)
    thread.start()
    try:
        time.sleep(0.05)
        assert seen == []
        thread.publish(data)
        deadline = time.monotonic() + 2.0
        while not seen and time.monotonic() < deadline:
            time.sleep(0.005)
        np.testing.assert_array_equal(seen[0], data.qpos)
    finally:
        assert thread.stop()
