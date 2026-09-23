from __future__ import annotations

import sys
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from physicalai.config import Config

from physicalai_mujoco_so101_plugin.http_server import ResetCommand, ShutdownCommand, SwitchSceneCommand
from physicalai_mujoco_so101_plugin.mujoco_robot import BiMuJoCoSO101, MuJoCoSO101, MuJoCoSO101Observation


@pytest.fixture
def mock_mujoco() -> MagicMock:
    """Mock MuJoCo functions at the function-call level.

    Yields:
        MagicMock: The mock context.
    """
    with (
        patch("mujoco.MjModel.from_xml_path") as mock_from_xml,
        patch("mujoco.MjData") as mock_data_cls,
        patch("mujoco.mj_forward"),
        patch("mujoco.mj_step"),
        patch("mujoco.mj_name2id", return_value=0),
        patch("mujoco.mjtObj", create=True),
    ):
        mock_model = MagicMock()
        mock_model.nq = 6
        mock_model.nv = 6
        mock_model.nu = 6
        mock_model.opt.timestep = 0.005
        mock_model.jnt_qposadr = [0, 1, 2, 3, 4, 5]
        mock_model.jnt_dofadr = [0, 1, 2, 3, 4, 5]
        mock_from_xml.return_value = mock_model

        mock_data = MagicMock()
        mock_data.qpos = np.array([0.0, 0.5, -0.3, 0.1, -0.2, 0.8])
        mock_data.qvel = np.array([0.0, 0.1, -0.05, 0.02, -0.01, 0.0])
        mock_data.ctrl = np.zeros(6)
        mock_data_cls.return_value = mock_data

        yield mock_model


@pytest.fixture
def mock_mujoco_bimanual() -> MagicMock:
    """Mock MuJoCo with a 12-DOF bimanual model.

    Yields:
        MagicMock: The mock context.
    """
    with (
        patch("mujoco.MjModel.from_xml_path") as mock_from_xml,
        patch("mujoco.MjData") as mock_data_cls,
        patch("mujoco.mj_forward"),
        patch("mujoco.mj_step"),
        patch("mujoco.mj_name2id", return_value=0),
        patch("mujoco.mjtObj", create=True),
    ):
        mock_model = MagicMock()
        mock_model.nq = 12
        mock_model.nv = 12
        mock_model.nu = 12
        mock_model.opt.timestep = 0.005
        mock_model.jnt_qposadr = list(range(12))
        mock_model.jnt_dofadr = list(range(12))
        mock_from_xml.return_value = mock_model

        mock_data = MagicMock()
        mock_data.qpos = np.array([0.0] * 12)
        mock_data.qvel = np.array([0.0] * 12)
        mock_data.ctrl = np.zeros(12)
        mock_data_cls.return_value = mock_data

        yield mock_model


class TestMuJoCoSO101ObservationRead:
    def test_state_property(self) -> None:
        obs = MuJoCoSO101Observation(
            joint_positions=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
            timestamp=12345.0,
        )
        assert np.allclose(obs.state, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        assert obs.timestamp == 12345.0

    def test_defaults(self) -> None:
        obs = MuJoCoSO101Observation(
            joint_positions=np.array([0.0] * 6, dtype=np.float32),
            timestamp=0.0,
        )
        assert obs.sensor_data is None
        assert obs.images is None


class TestMuJoCoSO101Construction:
    def test_default_construction(self) -> None:
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        assert robot._model_path == "/fake/model.xml"  # noqa: SLF001
        assert robot._substeps == 1  # noqa: SLF001
        assert robot._model is None  # noqa: SLF001
        assert robot._data is None  # noqa: SLF001

    def test_custom_substeps(self) -> None:
        robot = MuJoCoSO101(model_path="/fake/model.xml", substeps=5)
        assert robot._substeps == 5  # noqa: SLF001

    def test_joint_names(self) -> None:
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        expected = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        assert robot.joint_names == expected

    def test_device_ids(self) -> None:
        robot = MuJoCoSO101(model_path="/path/to/so101.xml")
        assert robot.device_ids == ()

    def test_exports_owner_construction_recipe(self) -> None:
        robot = MuJoCoSO101(model_path="/fake/model.xml", substeps=3, enable_viewer=True)

        assert Config.from_instance(robot) == {
            "class_path": "physicalai_mujoco_so101_plugin.mujoco_robot.MuJoCoSO101",
            "init_args": {
                "model_path": "/fake/model.xml",
                "substeps": 3,
                "enable_viewer": True,
            },
        }


class TestMuJoCoSO101Connect:
    def test_connect_success(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        assert robot.is_connected()

    def test_connect_idempotent(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot.connect()
        assert robot.is_connected()

    def test_disconnect(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot.disconnect()
        assert not robot.is_connected()
        assert robot._scene_on_reset is None  # noqa: SLF001

    def test_disconnect_idempotent(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.disconnect()
        robot.disconnect()

    @pytest.mark.parametrize("key", [ord("n"), ord("N")])
    def test_scene_switch_key(self, key: int) -> None:
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot._key_callback(key)  # noqa: SLF001
        assert robot._pending_scene_switch is True  # noqa: SLF001


class TestMuJoCoSO101ObservationReadBack:
    def test_get_observation_returns_degrees(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()

        obs = robot.get_observation()

        assert isinstance(obs, MuJoCoSO101Observation)
        assert obs.joint_positions.shape == (6,)
        assert obs.sensor_data is not None
        assert "velocities" in obs.sensor_data
        assert obs.sensor_data["velocities"].shape == (6,)
        assert obs.timestamp > 0

    def test_get_observation_before_connect(self) -> None:
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        with pytest.raises(ConnectionError, match="not connected"):
            robot.get_observation()


class TestMuJoCoSO101Actions:
    def test_send_action(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()

        action = np.array([10.0, 20.0, -5.0, 0.0, 15.0, 30.0], dtype=np.float32)
        robot.send_action(action, goal_time=0.1)

    def test_send_action_wrong_shape(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()

        with pytest.raises(ValueError, match="Expected action shape"):
            robot.send_action(np.array([1.0, 2.0, 3.0]))

    def test_send_action_before_connect(self) -> None:
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        with pytest.raises(ConnectionError, match="not connected"):
            robot.send_action(np.array([0.0] * 6))


class TestMuJoCoSO101Pickling:
    def test_getstate_before_connect(self) -> None:
        robot = MuJoCoSO101(model_path="/fake/model.xml", substeps=3)
        state = robot.__getstate__()
        assert state == {
            "_model_path": "/fake/model.xml",
            "_substeps": 3,
            "_enable_viewer": False,
            "_cameras": [],
            "_free_joints": ("block1:joint", "block2:joint", "block3:joint"),
            "_target_body_name": "target",
            "_spawn_center": (0.22, 0.0),
            "_spawn_min_r": 0.05,
            "_spawn_max_r": 0.14,
            "_spawn_angle_half_deg": 50.0,
            "_block_min_sep": 0.09,
            "_target_min_sep": 0.11,
            "_current_scene_id": None,
            "_owner_name": "",
            "_http_host": "127.0.0.1",
            "_http_port": 0,
            "_viser_port": 9090,
        }

    def test_getstate_after_connect(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        state = robot.__getstate__()
        assert "_model_path" in state
        assert "_substeps" in state
        assert "_enable_viewer" in state
        assert "_cameras" in state
        assert "_model" not in state

    def test_setstate(self) -> None:
        state = {"_model_path": "/fake/model.xml", "_substeps": 2, "_enable_viewer": True, "_cameras": []}
        robot = MuJoCoSO101.__new__(MuJoCoSO101)
        robot.__setstate__(state)
        assert robot._model_path == "/fake/model.xml"  # noqa: SLF001
        assert robot._substeps == 2  # noqa: SLF001
        assert robot._enable_viewer is True  # noqa: SLF001
        assert robot._model is None  # noqa: SLF001
        assert robot._data is None  # noqa: SLF001
        assert robot._viser_scene is None  # noqa: SLF001
        assert robot._native_viewer is None  # noqa: SLF001
        assert robot._http_host == "127.0.0.1"  # noqa: SLF001
        assert robot._http_port == 0  # noqa: SLF001
        assert robot._viser_port == 9090  # noqa: SLF001

    def test_setstate_restores_http_config(self) -> None:
        state = {
            "_model_path": "/fake/model.xml",
            "_substeps": 1,
            "_enable_viewer": False,
            "_cameras": [],
            "_http_host": "0.0.0.0",  # noqa: S104
            "_http_port": 9000,
        }
        robot = MuJoCoSO101.__new__(MuJoCoSO101)
        robot.__setstate__(state)
        assert robot._http_host == "0.0.0.0"  # noqa: SLF001, S104
        assert robot._http_port == 9000  # noqa: SLF001


class TestBiMuJoCoSO101:
    def test_joint_names(self) -> None:
        robot = BiMuJoCoSO101(model_path="/fake/model.xml")
        assert len(robot.joint_names) == 12
        assert robot.joint_names[:6] == [
            "left_shoulder_pan",
            "left_shoulder_lift",
            "left_elbow_flex",
            "left_wrist_flex",
            "left_wrist_roll",
            "left_gripper",
        ]
        assert robot.joint_names[6:] == [
            "right_shoulder_pan",
            "right_shoulder_lift",
            "right_elbow_flex",
            "right_wrist_flex",
            "right_wrist_roll",
            "right_gripper",
        ]

    def test_num_joints(self) -> None:
        robot = BiMuJoCoSO101(model_path="/fake/model.xml")
        assert robot.NUM_JOINTS == 12

    def test_exports_owner_construction_recipe(self) -> None:
        robot = BiMuJoCoSO101(model_path="/fake/model.xml", substeps=2)

        assert Config.from_instance(robot) == {
            "class_path": "physicalai_mujoco_so101_plugin.mujoco_robot.BiMuJoCoSO101",
            "init_args": {
                "model_path": "/fake/model.xml",
                "substeps": 2,
            },
        }

    def test_connect_and_observation(self, mock_mujoco_bimanual: MagicMock) -> None:
        _ = mock_mujoco_bimanual
        robot = BiMuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()

        obs = robot.get_observation()
        assert isinstance(obs, MuJoCoSO101Observation)
        assert obs.joint_positions.shape == (12,)

    def test_send_action(self, mock_mujoco_bimanual: MagicMock) -> None:
        _ = mock_mujoco_bimanual
        robot = BiMuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()

        action = np.arange(12, dtype=np.float32)
        robot.send_action(action, goal_time=0.1)

    def test_send_action_wrong_shape(self, mock_mujoco_bimanual: MagicMock) -> None:
        _ = mock_mujoco_bimanual
        robot = BiMuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()

        with pytest.raises(ValueError, match="Expected action shape"):
            robot.send_action(np.array([1.0, 2.0, 3.0]))


class TestHttpCommands:
    def test_reset_command_calls_scene_reset(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot._scene_on_reset = MagicMock()  # noqa: SLF001
        robot._commands.put(ResetCommand())  # noqa: SLF001

        robot.get_observation()

        robot._scene_on_reset.assert_called_once_with(robot._model, robot._data, robot._rng)  # noqa: SLF001

    def test_reset_command_survives_a_failing_scene_reset(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot._scene_on_reset = MagicMock(side_effect=ValueError("min() arg is an empty sequence"))  # noqa: SLF001
        robot._commands.put(ResetCommand())  # noqa: SLF001

        robot.get_observation()

        robot._scene_on_reset.assert_called_once()  # noqa: SLF001

    def test_viewer_reset_survives_a_failing_scene_reset(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot._scene_on_reset = MagicMock(side_effect=ValueError("boom"))  # noqa: SLF001
        robot._last_sim_time = 5.0  # noqa: SLF001
        robot._data.time = 0.0  # noqa: SLF001

        robot._handle_viewer_reset()  # noqa: SLF001

        robot._scene_on_reset.assert_called_once()  # noqa: SLF001

    def test_reset_command_falls_back_to_randomize(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot._scene_on_reset = None  # noqa: SLF001
        robot._commands.put(ResetCommand())  # noqa: SLF001

        with patch.object(robot, "_randomize_blocks") as randomize:
            robot.get_observation()

        randomize.assert_called_once()

    def test_switch_scene_command(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot._commands.put(SwitchSceneCommand(scene_id="pick_lift"))  # noqa: SLF001

        with patch.object(robot, "_switch_to_scene") as switch:
            robot.get_observation()

        switch.assert_called_once_with("pick_lift")

    def test_unknown_switch_scene_command_is_handled(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot._commands.put(SwitchSceneCommand(scene_id="nope"))  # noqa: SLF001

        with patch.object(robot, "_switch_to_scene", side_effect=KeyError("nope")):
            robot.get_observation()

    def test_shutdown_command_sets_owner_event(self, mock_mujoco: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()

        event = threading.Event()
        fake_worker = types.ModuleType("physicalai.robot.transport._owner_worker")
        fake_worker.shutdown = event
        monkeypatch.setitem(sys.modules, "physicalai.robot.transport._owner_worker", fake_worker)

        robot._commands.put(ShutdownCommand())  # noqa: SLF001
        robot.get_observation()

        assert event.is_set()

    def test_shutdown_without_owner_event_does_not_raise(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot._commands.put(ShutdownCommand())  # noqa: SLF001
        robot.get_observation()

    def test_http_status_shape(self) -> None:
        robot = MuJoCoSO101(
            model_path="/fake/model.xml",
            cameras=[{"name": "overview", "device": None}],
        )
        status = robot._http_status()  # noqa: SLF001
        assert status["connected"] is False
        assert status["scene"] is None
        assert "single_pick_place" in status["scenes"]
        assert status["cameras"] == [
            {
                "name": "overview",
                "width": 640,
                "height": 480,
                "fps": 30,
                "device": None,
                "rendering": False,
            },
        ]

    def test_http_status_reports_the_episode_helper(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        auto_reset = MagicMock()
        auto_reset.status.return_value = {"enabled": True, "phase": "idle"}
        robot._episode_auto_reset = auto_reset  # noqa: SLF001

        assert robot._http_status()["episode"] == {"enabled": True, "phase": "idle"}  # noqa: SLF001

    def test_http_status_without_an_episode_helper(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot._episode_auto_reset = None  # noqa: SLF001

        assert robot._http_status()["episode"] == {"enabled": False}  # noqa: SLF001


class TestSceneSwitching:
    def test_compatible_scene_is_installed(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()

        with patch("physicalai_mujoco_so101_plugin.scene_registry.get_reset_fn", return_value=MagicMock()):
            assert robot._switch_to_scene("pick_lift") is True  # noqa: SLF001
        assert robot._current_scene_id == "pick_lift"  # noqa: SLF001

    def test_scene_without_this_robots_joints_is_rejected(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        model_before = robot._model  # noqa: SLF001

        with patch("mujoco.mj_name2id", return_value=-1):
            assert robot._switch_to_scene("garment_fold") is False  # noqa: SLF001

        assert robot._model is model_before  # noqa: SLF001
        assert robot._current_scene_id is None  # noqa: SLF001

    def test_scene_with_too_few_actuators_is_rejected(self, mock_mujoco: MagicMock) -> None:
        robot = BiMuJoCoSO101(model_path="/fake/model.xml")
        # A single-arm model cannot drive the bimanual actuator range.
        mock_mujoco.nu = 6
        robot._model = mock_mujoco  # noqa: SLF001

        assert robot._switch_to_scene("single_pick_place") is False  # noqa: SLF001

    def test_observation_still_works_after_a_rejected_switch(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()

        with patch("mujoco.mj_name2id", return_value=-1):
            robot._commands.put(SwitchSceneCommand(scene_id="garment_fold"))  # noqa: SLF001

        obs = robot.get_observation()
        assert obs.joint_positions.shape == (6,)


class TestRecreateViserScene:
    def test_failed_rebuild_clears_the_stale_scene(self, mock_mujoco: MagicMock) -> None:
        """A rebuild failure must not leave the old (now-mismatched) scene wired up."""
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot._viser_server = MagicMock()  # noqa: SLF001
        robot._viser_scene = MagicMock()  # stale scene from before the hot-swap  # noqa: SLF001

        with patch("mjviser.ViserMujocoScene", side_effect=RuntimeError("boom")):
            robot._recreate_viser_scene()  # noqa: SLF001

        assert robot._viser_scene is None  # noqa: SLF001

    def test_successful_rebuild_replaces_the_scene(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot._viser_server = MagicMock()  # noqa: SLF001
        new_scene = MagicMock()

        with patch("mjviser.ViserMujocoScene", return_value=new_scene):
            robot._recreate_viser_scene()  # noqa: SLF001

        assert robot._viser_scene is new_scene  # noqa: SLF001

    def test_no_viser_server_is_a_noop(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot._viser_server = None  # noqa: SLF001

        robot._recreate_viser_scene()  # noqa: SLF001

        assert robot._viser_scene is None  # noqa: SLF001


class TestLaunchViserViewer:
    def test_viser_console_does_not_use_owner_stdout(self, mock_mujoco: MagicMock) -> None:
        """Viser's teardown message must survive the owner closing stdout after READY."""
        _ = mock_mujoco
        import rich

        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        console = rich.get_console()
        original_file = console.file
        try:
            with (
                patch("viser.ViserServer", return_value=MagicMock()),
                patch("mjviser.ViserMujocoScene", return_value=MagicMock()),
            ):
                assert robot._launch_viser_viewer() is True  # noqa: SLF001
            assert console.file is sys.stderr
        finally:
            console.file = original_file

    def test_stops_partially_created_server_on_scene_failure(self, mock_mujoco: MagicMock) -> None:
        """A server created before the scene build fails must not leak the port/thread."""
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        server = MagicMock()

        with (
            patch("viser.ViserServer", return_value=server),
            patch("mjviser.ViserMujocoScene", side_effect=RuntimeError("boom")),
        ):
            launched = robot._launch_viser_viewer()  # noqa: SLF001

        assert launched is False
        server.stop.assert_called_once()
        assert robot._viser_server is None  # noqa: SLF001


class TestViserControlGui:
    @staticmethod
    def _build(mock_mujoco: MagicMock) -> tuple[MuJoCoSO101, MagicMock, MagicMock]:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        reset_button, shutdown_button = MagicMock(), MagicMock()
        server = MagicMock()
        server.gui.add_button.side_effect = [reset_button, shutdown_button]
        viser_module = MagicMock()

        robot._add_viser_control_gui(server, viser_module)  # noqa: SLF001

        return robot, reset_button, shutdown_button

    def test_reset_button_enqueues_reset_command(self, mock_mujoco: MagicMock) -> None:
        robot, reset_button, _ = self._build(mock_mujoco)
        on_reset = reset_button.on_click.call_args.args[0]

        on_reset(MagicMock())

        assert isinstance(robot._commands.get_nowait(), ResetCommand)  # noqa: SLF001

    def test_shutdown_confirm_enqueues_shutdown_command_and_closes_modal(self, mock_mujoco: MagicMock) -> None:
        robot, _, shutdown_button = self._build(mock_mujoco)
        on_shutdown_click = shutdown_button.on_click.call_args.args[0]

        client = MagicMock()
        confirm_button, cancel_button = MagicMock(), MagicMock()
        client.gui.add_button.side_effect = [confirm_button, cancel_button]

        on_shutdown_click(MagicMock(client=client))
        on_confirm = confirm_button.on_click.call_args.args[0]
        on_confirm(MagicMock())

        assert isinstance(robot._commands.get_nowait(), ShutdownCommand)  # noqa: SLF001
        client.gui.add_modal.return_value.__exit__.assert_called_once()

    def test_shutdown_cancel_closes_modal_without_enqueueing(self, mock_mujoco: MagicMock) -> None:
        robot, _, shutdown_button = self._build(mock_mujoco)
        on_shutdown_click = shutdown_button.on_click.call_args.args[0]

        client = MagicMock()
        confirm_button, cancel_button = MagicMock(), MagicMock()
        client.gui.add_button.side_effect = [confirm_button, cancel_button]

        on_shutdown_click(MagicMock(client=client))
        on_cancel = cancel_button.on_click.call_args.args[0]
        on_cancel(MagicMock())

        assert robot._commands.empty()  # noqa: SLF001

    def test_shutdown_click_without_client_is_a_noop(self, mock_mujoco: MagicMock) -> None:
        robot, _, shutdown_button = self._build(mock_mujoco)
        on_shutdown_click = shutdown_button.on_click.call_args.args[0]

        on_shutdown_click(MagicMock(client=None))

        assert robot._commands.empty()  # noqa: SLF001


class TestSceneXmlWatch:
    def test_include_graph_is_walked_once_per_poll(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()

        with patch.object(robot, "_collect_scene_xml_paths", return_value=[]) as collect:
            robot._scene_xml_paths = None  # noqa: SLF001
            for _ in range(5):
                robot._scene_xml_paths_cached()  # noqa: SLF001

        collect.assert_called_once()

    def test_xml_is_not_polled_on_every_tick(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()

        with patch.object(robot, "_snapshot_scene_xml_mtimes") as snapshot:
            for _ in range(10):
                robot.get_observation()

        snapshot.assert_not_called()

    def test_unreadable_include_does_not_escape(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot._scene_xml_paths = [Path("/fake/deleted-include.xml")]  # noqa: SLF001

        robot._update_camera_from_xml()  # noqa: SLF001


class TestHttpServerIntegration:
    @staticmethod
    def _free_port() -> int:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def test_connect_starts_server_disconnect_stops_it(self, mock_mujoco: MagicMock) -> None:
        import http.client

        _ = mock_mujoco
        port = self._free_port()
        robot = MuJoCoSO101(model_path="/fake/model.xml", http_port=port)
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

    def test_connect_with_busy_port_continues_without_http(self, mock_mujoco: MagicMock) -> None:
        import socket

        _ = mock_mujoco
        port = self._free_port()
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("127.0.0.1", port))
        blocker.listen(1)
        try:
            robot = MuJoCoSO101(model_path="/fake/model.xml", http_port=port)
            robot.connect()
            assert robot._http_server is None  # noqa: SLF001
            robot.disconnect()
        finally:
            blocker.close()

    def test_http_disabled_by_default_port_zero(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        try:
            assert robot._http_server is None  # noqa: SLF001
        finally:
            robot.disconnect()
