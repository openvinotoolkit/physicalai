from __future__ import annotations

import sys
import threading
import types
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import pytest
from physicalai.config import Config

from physicalai_mujoco_plugin.constants import BIMANUAL_SO101_JOINT_ORDER, SO101_JOINT_ORDER
from physicalai_mujoco_plugin.http_server import (
    HomeCommand,
    ResetCommand,
    SetAutoResetCommand,
    SetObjectPoseCommand,
    SetSeedCommand,
    ShutdownCommand,
    SwitchSceneCommand,
)
from physicalai_mujoco_plugin.profiles.so101 import SO101_JOINT_RANGES
from physicalai_mujoco_plugin.robot import MuJoCoObservation, MuJoCoRobot
from physicalai_mujoco_plugin.sim import default_cameras

RANGES = np.array([(low, high) for _name, low, high in SO101_JOINT_RANGES])
"""The SO-101 profile's pinned joint ranges (radians), in SO101_JOINT_ORDER."""
GRIPPER = np.array([name.endswith("gripper") for name in SO101_JOINT_ORDER])


def radians_to_normalized(radians: np.ndarray) -> np.ndarray:
    """The SO101 driver's mapping: body joints span [-100, 100], the gripper [0, 100]."""
    fraction = np.clip((radians - RANGES[:, 0]) / (RANGES[:, 1] - RANGES[:, 0]), 0.0, 1.0)
    return np.where(GRIPPER, fraction * 100.0, fraction * 200.0 - 100.0)


def normalized_to_radians(normalized: np.ndarray) -> np.ndarray:
    fraction = np.clip(np.where(GRIPPER, normalized / 100.0, (normalized + 100.0) / 200.0), 0.0, 1.0)
    return RANGES[:, 0] + fraction * (RANGES[:, 1] - RANGES[:, 0])


def so101(model_path: str | None = None, **kwargs: object) -> MuJoCoRobot:
    """The SO-101 on a test model: one physics step per tick and no camera streams, unless overridden."""
    return MuJoCoRobot("so101", model_path=model_path, **{"substeps": 1, "cameras": [], **kwargs})  # type: ignore[arg-type]


def _arm_xml(prefix: str = "", x: float = 0.0, *, ranges: bool = True, reverse_actuators: bool = False) -> tuple[str, str]:
    """A 6-joint chain with the SO-101's joint names, and its position actuators."""
    bodies, actuators = "", []
    for index, (name, low, high) in enumerate(SO101_JOINT_RANGES):
        limit = f'range="{low} {high}"' if ranges else ""
        bodies += f'<body name="{prefix}link{index}" pos="{x if index == 0 else 0} 0 0.05">'
        bodies += f'<joint name="{prefix}{name}" axis="0 1 0" {limit}/>'
        bodies += '<geom type="capsule" fromto="0 0 0 0 0 0.05" size="0.01" mass="0.05"/>'
        actuators.append(f'<position name="{prefix}{name}" joint="{prefix}{name}" kp="20"/>')
    bodies += "</body>" * len(SO101_JOINT_RANGES)
    if reverse_actuators:
        actuators.reverse()
    return bodies, "".join(actuators)


def _write_model(tmp_path, *, bimanual: bool = False, **arm) -> str:
    """Write a robot-complete scene (no ``robot_mount`` frames), so no Menagerie download is needed."""
    if bimanual:
        left, left_act = _arm_xml("left_", -0.2, **arm)
        right, right_act = _arm_xml("right_", 0.2, **arm)
        bodies, actuators = left + right, left_act + right_act
    else:
        bodies, actuators = _arm_xml(**arm)
    path = tmp_path / ("bimanual.xml" if bimanual else "arm.xml")
    path.write_text(
        f"""<mujoco><compiler angle="radian"/><option timestep="0.002"/><worldbody>
        <camera name="overview" pos="0 -1 0.5" xyaxes="1 0 0 0 0.5 1"/>
        {bodies}
        <body name="block1" pos="0.2 0 0.02"><freejoint name="block1:joint"/><geom type="box" size="0.01 0.01 0.01"/></body>
        </worldbody><actuator>{actuators}</actuator></mujoco>""",
    )
    return str(path)


@pytest.fixture
def model_path(tmp_path) -> str:
    return _write_model(tmp_path)


@pytest.fixture
def bimanual_path(tmp_path) -> str:
    return _write_model(tmp_path, bimanual=True)


@pytest.fixture
def robot(model_path):
    robot = so101(model_path)
    robot.connect()
    yield robot
    robot.disconnect()


class TestConstruction:
    def test_defaults(self) -> None:
        robot = MuJoCoRobot()
        assert robot.profile.name == "so101"
        assert robot._current_scene_id == "single_pick_place"  # noqa: SLF001
        assert robot._unit == "normalized"  # noqa: SLF001
        assert robot._model is None  # noqa: SLF001
        assert robot._viser_host == "127.0.0.1"  # noqa: SLF001
        assert robot.joint_names == list(SO101_JOINT_ORDER)  # before connect, without a download
        assert robot.device_ids == ()

    def test_a_two_arm_scene_prefixes_the_names(self) -> None:
        assert MuJoCoRobot(scene="garment_fold").joint_names == list(BIMANUAL_SO101_JOINT_ORDER)

    def test_exports_only_the_supplied_arguments(self) -> None:
        robot = MuJoCoRobot("so101", scene="yahtzee", substeps=3, enable_viewer=True)

        assert Config.from_instance(robot) == {
            "class_path": "physicalai_mujoco_plugin.robot.MuJoCoRobot",
            "init_args": {"profile": "so101", "scene": "yahtzee", "substeps": 3, "enable_viewer": True},
        }

    def test_round_trips_through_its_config(self) -> None:
        robot = MuJoCoRobot("so101", scene="yahtzee", unit="degrees", seed=3)
        restored = Config.from_instance(robot).instantiate()

        assert isinstance(restored, MuJoCoRobot)
        assert (restored.profile.name, restored._current_scene_id, restored._unit, restored._seed) == (  # noqa: SLF001
            "so101",
            "yahtzee",
            "degrees",
            3,
        )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"unit": "ticks"}, "Unsupported unit"),
            ({"torque_mode": "fast"}, "Unsupported torque_mode"),
            ({"rate_hz": 0.0}, "rate_hz must be positive"),
            ({"substeps": 0}, "substeps at least 1"),
            ({"scene": "conveyor_sort", "profile": "ur5e"}, "does not support"),
        ],
    )
    def test_rejects_invalid_arguments(self, kwargs: dict[str, object], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            MuJoCoRobot(**kwargs)  # type: ignore[arg-type]

    def test_pickles_as_its_recipe(self) -> None:
        robot = MuJoCoRobot(scene="yahtzee", substeps=3, http_port=9000, viser_host="0.0.0.0")  # noqa: S104
        state = robot.__getstate__()
        assert (state["scene"], state["substeps"]) == ("yahtzee", 3)

        restored = MuJoCoRobot.__new__(MuJoCoRobot)
        restored.__setstate__(state)
        assert not restored.is_connected()
        assert (restored._substeps, restored._http_port, restored._viser_host) == (3, 9000, "0.0.0.0")  # noqa: SLF001, S104


class TestConnect:
    def test_connect_is_idempotent_and_disconnect_too(self, model_path) -> None:
        robot = so101(model_path)
        robot.disconnect()
        robot.connect()
        model = robot._model  # noqa: SLF001
        robot.connect()
        assert robot.is_connected()
        assert robot._model is model  # noqa: SLF001
        robot.disconnect()
        robot.disconnect()
        assert not robot.is_connected()

    def test_actuators_are_bound_by_name_not_order(self, tmp_path) -> None:
        robot = so101(_write_model(tmp_path, reverse_actuators=True))
        robot.connect()
        action = np.linspace(-50, 50, 6)
        robot.send_action(action)

        np.testing.assert_allclose(robot._data.ctrl[::-1], normalized_to_radians(action))  # noqa: SLF001
        robot.disconnect()

    def test_connect_rejects_a_model_without_the_robots_joints(self, tmp_path) -> None:
        path = tmp_path / "empty.xml"
        path.write_text("<mujoco><worldbody/></mujoco>")
        robot = so101(str(path))

        with pytest.raises(ValueError, match="no actuator or joint 'shoulder_pan'"):
            robot.connect()
        assert not robot.is_connected()

    def test_a_robot_complete_model_has_one_robot_per_name_prefix(self, bimanual_path) -> None:
        robot = so101(bimanual_path)
        assert robot.joint_names == list(BIMANUAL_SO101_JOINT_ORDER)  # before connect, as after
        robot.connect()
        assert robot.joint_names == list(BIMANUAL_SO101_JOINT_ORDER)
        assert robot.get_observation().joint_positions.shape == (12,)
        robot.send_action(np.arange(12, dtype=np.float32))
        with pytest.raises(ValueError, match="Expected action shape"):
            robot.send_action(np.zeros(6))
        robot.disconnect()

    @pytest.mark.parametrize("key", [ord("n"), ord("N")])
    def test_scene_switch_key(self, key: int) -> None:
        robot = MuJoCoRobot()
        robot._key_callback(key)  # noqa: SLF001
        assert robot._pending_scene_switch is True  # noqa: SLF001

    def test_substeps_default_keeps_real_time(self, model_path) -> None:
        robot = MuJoCoRobot("so101", model_path=model_path, rate_hz=25.0, cameras=[])
        robot.connect()
        assert robot._substeps == 20  # noqa: SLF001 - 1 / (25 Hz * 2 ms)
        robot.disconnect()


class TestObservationsAndActions:
    def test_observation_shapes(self, robot) -> None:
        obs = robot.get_observation()

        assert isinstance(obs, MuJoCoObservation)
        np.testing.assert_array_equal(obs.state, obs.joint_positions)
        assert obs.joint_positions.shape == (6,)
        assert obs.joint_positions.dtype == np.float32
        assert obs.sensor_data["velocities"].shape == (6,)
        assert obs.images is None
        assert obs.timestamp > 0

    def test_requires_connect(self) -> None:
        robot = MuJoCoRobot()
        with pytest.raises(ConnectionError, match="not connected"):
            robot.get_observation()
        with pytest.raises(ConnectionError, match="not connected"):
            robot.send_action(np.zeros(6))

    def test_wrong_shape_is_rejected(self, robot) -> None:
        with pytest.raises(ValueError, match="Expected action shape"):
            robot.send_action(np.array([1.0, 2.0, 3.0]))

    def test_non_finite_action_is_discarded(self, robot) -> None:
        robot.send_action(np.full(6, 10.0))
        ctrl = robot._data.ctrl.copy()  # noqa: SLF001

        robot.send_action(np.array([np.nan, 0, 0, 0, 0, 0]))

        np.testing.assert_array_equal(robot._data.ctrl, ctrl)  # noqa: SLF001

    def test_observation_and_action_use_normalized_units(self, robot) -> None:
        robot._data.qpos[:6] = RANGES[:, 0] + 0.3 * (RANGES[:, 1] - RANGES[:, 0])  # noqa: SLF001
        robot._data.qvel[:6] = 0.0  # noqa: SLF001
        robot._substeps = 0  # noqa: SLF001 - read back the state as set, without stepping

        obs = robot.get_observation()

        np.testing.assert_allclose(obs.joint_positions, radians_to_normalized(robot._data.qpos[:6]), rtol=1e-6)  # noqa: SLF001
        robot.send_action(obs.joint_positions)
        np.testing.assert_allclose(robot._data.ctrl, robot._data.qpos[:6], atol=1e-5)  # noqa: SLF001

    def test_velocities_scale_with_the_normalized_span(self, robot) -> None:
        robot._substeps = 0  # noqa: SLF001
        robot._data.qvel[:6] = 1.0  # noqa: SLF001

        velocities = robot.get_observation().sensor_data["velocities"]

        span = RANGES[:, 1] - RANGES[:, 0]
        np.testing.assert_allclose(velocities, np.where(GRIPPER, 100.0, 200.0) / span, rtol=1e-6)

    def test_degrees_unit_keeps_joint_angles(self, model_path) -> None:
        robot = so101(model_path, unit="degrees")
        robot.connect()
        robot._substeps = 0  # noqa: SLF001
        robot._data.qpos[:6] = [0.1, -0.2, 0.3, -0.4, 0.5, 0.6]  # noqa: SLF001

        obs = robot.get_observation()
        np.testing.assert_allclose(obs.joint_positions, np.degrees(robot._data.qpos[:6]), rtol=1e-6)  # noqa: SLF001

        action = np.array([10.0, 20.0, -5.0, 0.0, 15.0, 30.0])
        robot.send_action(action)
        np.testing.assert_allclose(robot._data.ctrl, np.radians(action))  # noqa: SLF001
        robot.disconnect()

    def test_joints_without_a_model_range_use_the_profiles_range(self, tmp_path) -> None:
        robot = so101(_write_model(tmp_path, ranges=False))
        robot.connect()
        robot.send_action(np.full(6, -100.0))

        np.testing.assert_allclose(robot._data.ctrl[:5], RANGES[:5, 0])  # noqa: SLF001
        robot.disconnect()

    @pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.9, 1.0])
    def test_normalized_matches_the_so101_driver_mapping(self, robot, fraction: float) -> None:
        """Body joints span [-100, 100] across their range, the gripper [0, 100], as on the real SO101."""
        robot._substeps = 0  # noqa: SLF001
        robot._data.qpos[:6] = RANGES[:, 0] + fraction * (RANGES[:, 1] - RANGES[:, 0])  # noqa: SLF001

        positions = robot.get_observation().joint_positions

        expected = np.where(GRIPPER, fraction * 100.0, fraction * 200.0 - 100.0)
        np.testing.assert_allclose(positions, expected, atol=1e-4)

    def test_normalized_values_outside_the_range_are_clamped(self, robot) -> None:
        robot.send_action(np.full(6, -150.0))
        np.testing.assert_allclose(robot._data.ctrl, RANGES[:, 0])  # noqa: SLF001


class TestCommands:
    def test_reset_command_runs_the_scene_reset(self, robot) -> None:
        robot._sim.on_reset = MagicMock()  # noqa: SLF001
        robot._commands.put(ResetCommand())  # noqa: SLF001

        robot.get_observation()

        robot._sim.on_reset.assert_called_once_with(robot._model, robot._data, robot._rng)  # noqa: SLF001

    def test_a_failing_scene_reset_does_not_escape(self, robot) -> None:
        robot._sim.on_reset = MagicMock(side_effect=ValueError("min() arg is an empty sequence"))  # noqa: SLF001
        robot._commands.put(ResetCommand())  # noqa: SLF001

        robot.get_observation()

        robot._sim.on_reset.assert_called_once()  # noqa: SLF001

    def test_viewer_reset_survives_a_failing_scene_reset(self, robot) -> None:
        robot._sim.on_reset = MagicMock(side_effect=ValueError("boom"))  # noqa: SLF001
        robot._last_sim_time = 5.0  # noqa: SLF001

        robot._handle_viewer_reset()  # noqa: SLF001

        robot._sim.on_reset.assert_called_once()  # noqa: SLF001

    def test_reset_without_a_scene_respawns_the_free_objects(self, robot) -> None:
        assert robot._sim.on_reset is None  # noqa: SLF001 - a custom model has no registered scene
        robot._commands.put(ResetCommand())  # noqa: SLF001

        with patch.object(robot._objects, "randomize") as randomize:  # noqa: SLF001
            robot.get_observation()

        randomize.assert_called_once()

    def test_switch_scene_command(self, robot) -> None:
        robot._commands.put(SwitchSceneCommand(scene_id="yahtzee"))  # noqa: SLF001
        with patch.object(robot, "_switch_to_scene") as switch:
            robot.get_observation()
        switch.assert_called_once_with("yahtzee")

    def test_unknown_scene_does_not_escape(self, robot) -> None:
        robot._commands.put(SwitchSceneCommand(scene_id="nope"))  # noqa: SLF001
        assert robot.get_observation().joint_positions.shape == (6,)

    def test_shutdown_command_sets_owner_event(self, robot, monkeypatch: pytest.MonkeyPatch) -> None:
        event = threading.Event()
        fake_worker = types.ModuleType("physicalai.robot.transport._owner_worker")
        fake_worker.shutdown = event
        monkeypatch.setitem(sys.modules, "physicalai.robot.transport._owner_worker", fake_worker)

        robot._commands.put(ShutdownCommand())  # noqa: SLF001
        robot.get_observation()

        assert event.is_set()

    def test_shutdown_without_owner_event_does_not_raise(self, robot) -> None:
        robot._commands.put(ShutdownCommand())  # noqa: SLF001
        robot.get_observation()

    def test_home_places_the_joints_and_holds_them(self, robot) -> None:
        robot.send_action(np.full(6, 50.0))
        for _ in range(5):
            robot.get_observation()
        robot._commands.put(HomeCommand())  # noqa: SLF001
        robot._substeps = 0  # noqa: SLF001

        robot.get_observation()

        np.testing.assert_array_equal(robot._data.qpos[:6], 0.0)  # noqa: SLF001 - qpos0, no keyframe
        np.testing.assert_array_equal(robot._data.ctrl, 0.0)  # noqa: SLF001

    def test_a_failing_command_does_not_stop_the_next(self, robot) -> None:
        robot._commands.put(HomeCommand())  # noqa: SLF001
        robot._commands.put(ResetCommand())  # noqa: SLF001

        with (
            patch.object(robot, "_go_home", side_effect=IndexError("bad model")),
            patch.object(robot, "_run_scene_reset") as scene_reset,
        ):
            obs = robot.get_observation()

        scene_reset.assert_called_once()
        assert obs.joint_positions.shape == (6,)

    def test_fixed_seed_makes_resets_repeat(self, robot) -> None:
        draws: list[float] = []
        robot._sim.on_reset = lambda _m, _d, rng: draws.append(float(rng.random()))  # noqa: SLF001

        robot._commands.put(SetSeedCommand(seed=123))  # noqa: SLF001
        robot._commands.put(ResetCommand())  # noqa: SLF001
        robot._commands.put(ResetCommand())  # noqa: SLF001
        robot.get_observation()
        robot._commands.put(SetSeedCommand(seed=None))  # noqa: SLF001
        robot._commands.put(ResetCommand())  # noqa: SLF001
        robot.get_observation()

        assert draws[0] == draws[1]
        assert draws[2] != draws[1]
        assert robot._http_status()["seed"] is None  # noqa: SLF001

    def test_reseeding_keeps_the_generator_shared_with_auto_reset(self) -> None:
        robot = MuJoCoRobot()
        rng = robot._rng  # noqa: SLF001
        robot._set_seed(5)  # noqa: SLF001
        assert robot._rng is rng  # noqa: SLF001

    def test_auto_reset_settings_apply_and_persist(self, robot) -> None:
        helper = MagicMock()
        robot._episode_auto_reset = helper  # noqa: SLF001

        robot._commands.put(SetAutoResetCommand(enabled=False))  # noqa: SLF001
        robot._commands.put(SetAutoResetCommand(dwell_s=2.0))  # noqa: SLF001
        robot.get_observation()

        helper.set_active.assert_called_with(False)  # noqa: FBT003
        helper.set_dwell.assert_called_with(2.0)
        with patch(
            "physicalai_mujoco_plugin.episode_auto_reset.EpisodeAutoReset.maybe_create",
            return_value=None,
        ) as maybe_create:
            robot._init_episode_auto_reset()  # noqa: SLF001
        assert maybe_create.call_args.kwargs["active"] is False
        assert maybe_create.call_args.kwargs["success_dwell_s"] == 2.0

    def test_auto_reset_command_without_a_helper_is_survivable(self, robot) -> None:
        robot._episode_auto_reset = None  # noqa: SLF001
        robot._commands.put(SetAutoResetCommand(enabled=False))  # noqa: SLF001

        robot.get_observation()

        assert robot._auto_reset_active is False  # noqa: SLF001

    def test_object_pose_holds_and_unknown_objects_are_ignored(self, robot) -> None:
        robot._commands.put(SetObjectPoseCommand(joint="nope", position=(0.0, 0.0, 0.1)))  # noqa: SLF001
        robot.get_observation()
        assert robot._objects.held == {}  # noqa: SLF001

        robot._commands.put(SetObjectPoseCommand(joint="block1:joint", position=(0.1, 0.1, 0.3), hold=True))  # noqa: SLF001
        robot.get_observation()
        np.testing.assert_allclose(robot._objects.poses["block1:joint"].position, (0.1, 0.1, 0.3))  # noqa: SLF001


class TestStatus:
    def test_status_before_connect(self) -> None:
        robot = MuJoCoRobot(cameras=[{"name": "overview"}])
        status = robot._http_status()  # noqa: SLF001

        assert status["connected"] is False
        assert status["scene"] == "single_pick_place"
        assert status["profile"] == "so101"
        assert "garment_fold" in status["scenes"]
        assert "garment_fold" not in status["compatible_scenes"]
        assert status["objects"] == []
        assert status["cameras"] == [{"name": "overview", "width": 640, "height": 480, "fps": 30, "rendering": False}]
        assert MuJoCoRobot(scene="garment_fold")._http_status()["compatible_scenes"] == ["garment_fold"]  # noqa: SLF001

    def test_status_after_connect(self, robot) -> None:
        status = robot._http_status()  # noqa: SLF001

        assert status["connected"] is True
        assert status["joint_names"] == list(SO101_JOINT_ORDER)
        assert status["units"] == ["normalized"] * 6
        assert status["objects"][0]["joint"] == "block1:joint"

    def test_status_reports_the_episode_helper(self, robot) -> None:
        robot._episode_auto_reset = MagicMock()  # noqa: SLF001
        robot._episode_auto_reset.status.return_value = {"enabled": True, "phase": "idle"}  # noqa: SLF001
        assert robot._http_status()["episode"] == {"enabled": True, "phase": "idle"}  # noqa: SLF001

        robot._episode_auto_reset = None  # noqa: SLF001
        assert robot._http_status()["episode"] == {"enabled": False}  # noqa: SLF001


class TestSceneSwitching:
    def test_a_scene_that_changes_the_joint_names_is_rejected(self, tmp_path) -> None:
        """The transport advertised the names on connect: a prefixed one-arm model cannot become an unprefixed one."""
        bodies, actuators = _arm_xml("left_")
        path = tmp_path / "left_arm.xml"
        path.write_text(
            f'''<mujoco><compiler angle="radian"/><worldbody>{bodies}</worldbody><actuator>{actuators}</actuator></mujoco>'''
        )
        robot = so101(str(path))
        robot.connect()
        try:
            assert robot.joint_names[0] == "left_shoulder_pan"
            assert robot._switch_to_scene("yahtzee") is False  # noqa: SLF001
            assert robot.joint_names[0] == "left_shoulder_pan"
        finally:
            robot.disconnect()

    def test_default_cameras_follow_the_new_scene(self, model_path) -> None:
        robot = MuJoCoRobot("so101", model_path=model_path, substeps=1)
        with patch("mujoco.Renderer"):
            robot.connect()
            try:
                assert [config.name for config in robot._cameras.configs] == ["overview"]  # noqa: SLF001
                assert robot._switch_to_scene("single_pick_place")  # noqa: SLF001
                assert [config.name for config in robot._cameras.configs] == ["wrist", "overview"]  # noqa: SLF001
                assert set(robot._cameras.frame_buffers) == {"wrist", "overview"}  # noqa: SLF001
            finally:
                robot.disconnect()

    def test_a_different_arm_count_is_rejected_before_loading(self, robot) -> None:
        model = robot._model  # noqa: SLF001
        with patch("physicalai_mujoco_plugin.compose.compose_scene") as compose:
            assert robot._switch_to_scene("garment_fold") is False  # noqa: SLF001

        compose.assert_not_called()
        assert robot._model is model  # noqa: SLF001
        assert robot.get_observation().joint_positions.shape == (6,)

    def test_a_scene_that_fails_to_bind_keeps_the_running_one(self, robot) -> None:
        model = robot._model  # noqa: SLF001
        with patch.object(robot, "_load", side_effect=ValueError("no actuator")):
            assert robot._switch_to_scene("yahtzee") is False  # noqa: SLF001

        assert robot._model is model  # noqa: SLF001
        assert robot._current_scene_id is None  # noqa: SLF001

    def test_a_live_xml_edit_pauses_the_camera_thread(self, robot) -> None:
        """The camera thread reads the shared model, so the edit must not race it."""
        calls: list[str] = []
        cameras = MagicMock(on_thread=True)
        cameras.stop.side_effect = lambda: calls.append("stop")
        cameras.start.side_effect = lambda *_: calls.append("start")
        watcher = MagicMock()
        watcher.changed.return_value = True
        watcher.apply.side_effect = lambda *_: calls.append("apply")
        robot._cameras, robot._watcher = cameras, watcher  # noqa: SLF001

        robot.get_observation()

        assert calls == ["stop", "apply", "start"]
        cameras.start.assert_called_once_with(robot._model, robot._data)  # noqa: SLF001

    def test_an_unchanged_xml_leaves_the_cameras_running(self, robot) -> None:
        cameras = MagicMock(on_thread=True)
        watcher = MagicMock()
        watcher.changed.return_value = False
        robot._cameras, robot._watcher = cameras, watcher  # noqa: SLF001

        robot.get_observation()

        cameras.stop.assert_not_called()
        watcher.apply.assert_not_called()

    def test_scene_key_cycles_only_compatible_scenes(self, bimanual_path) -> None:
        robot = so101(bimanual_path, scene="garment_fold")
        robot.connect()
        robot._pending_scene_switch = True  # noqa: SLF001

        with patch.object(robot, "_switch_to_scene") as switch:
            robot._check_pending_scene_switch()  # noqa: SLF001

        switch.assert_not_called()  # garment_fold is the only two-arm scene
        robot.disconnect()


class TestHttpServer:
    @staticmethod
    def _free_port() -> int:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def test_connect_starts_server_disconnect_stops_it(self, model_path) -> None:
        import http.client

        port = self._free_port()
        robot = so101(model_path, http_port=port)
        robot.connect()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            assert robot._http_server is not None  # noqa: SLF001
            conn.request("GET", "/health")
            assert conn.getresponse().status == 200
        finally:
            conn.close()
            robot.disconnect()
        assert robot._http_server is None  # noqa: SLF001
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        with pytest.raises(ConnectionRefusedError):
            conn.request("GET", "/health")

    def test_busy_port_continues_without_http(self, model_path) -> None:
        import socket

        port = self._free_port()
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("127.0.0.1", port))
        blocker.listen(1)
        try:
            robot = so101(model_path, http_port=port)
            robot.connect()
            assert robot._http_server is None  # noqa: SLF001
            robot.disconnect()
        finally:
            blocker.close()

    def test_http_disabled_by_default(self, robot) -> None:
        assert robot._http_server is None  # noqa: SLF001


class TestDefaultCameras:
    """Camera selection only: tests never open a real renderer (CI runners have no OpenGL)."""

    def test_robot_cameras_then_overview(self, robot) -> None:
        assert [config.name for config in default_cameras(robot._sim)] == ["overview"]  # noqa: SLF001

    def test_bimanual_wrists_around_the_overview(self) -> None:
        robot = MuJoCoRobot(scene="garment_fold", cameras=[])
        robot.connect()
        try:
            names = [config.name for config in default_cameras(robot._sim)]  # noqa: SLF001
        finally:
            robot.disconnect()
        assert names == ["left_wrist", "overview", "right_wrist"]

    def test_none_streams_the_defaults(self, model_path) -> None:
        robot = MuJoCoRobot("so101", model_path=model_path)
        with patch("mujoco.Renderer"):
            robot.connect()
            try:
                assert [config.name for config in robot._cameras.configs] == ["overview"]  # noqa: SLF001
            finally:
                robot.disconnect()

    def test_explicit_empty_list_streams_nothing(self, model_path) -> None:
        robot = MuJoCoRobot("so101", model_path=model_path, cameras=[])
        robot.connect()
        assert robot._cameras.configs == []  # noqa: SLF001
        robot.disconnect()


def test_generic_driver_derives_channels_of_a_custom_model(tmp_path) -> None:
    path = tmp_path / "one_joint.xml"
    path.write_text(
        """<mujoco><compiler angle="radian"/><worldbody><body name="arm"><joint name="hinge" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02"/></body></worldbody>
        <actuator><position name="hinge_target" joint="hinge" kp="10"/></actuator></mujoco>""",
    )
    robot = MuJoCoRobot("ur5e", model_path=str(path), cameras=[])

    assert robot.joint_names == ["hinge_target"]
    robot.connect()
    robot.send_action(np.array([15.0]))
    assert robot._data.ctrl[0] == pytest.approx(np.radians(15.0))  # noqa: SLF001
    assert robot.get_observation().joint_positions.shape == (1,)
    robot.disconnect()


def test_model_cameras_in_status_use_the_live_renderers(robot) -> None:
    # A real renderer is needed only by the camera tests in test_scene_models.py and test_cameras.py.
    assert robot._http_status()["cameras"] == []  # noqa: SLF001
    assert mujoco.mj_name2id(robot._model, mujoco.mjtObj.mjOBJ_CAMERA, "overview") >= 0  # noqa: SLF001
