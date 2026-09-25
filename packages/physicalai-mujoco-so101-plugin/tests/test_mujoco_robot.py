from __future__ import annotations

import queue
import sys
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import mujoco
import pytest
from physicalai.config import Config

from physicalai_mujoco_so101_plugin.constants import BIMANUAL_SO101_JOINT_ORDER, JOINT_LIMITS_DEG, SO101_JOINT_ORDER
from physicalai_mujoco_so101_plugin.http_server import (
    HomeCommand,
    ResetCommand,
    SetAutoResetCommand,
    SetObjectPoseCommand,
    SetSeedCommand,
    ShutdownCommand,
    SwitchSceneCommand,
)
from physicalai_mujoco_so101_plugin.mujoco_robot import (
    BiMuJoCoSO101,
    MuJoCoSO101,
    MuJoCoSO101Observation,
    normalized_to_radians,
    radians_to_normalized,
)

# The SO-101 model's joint ranges (radians), in SO101_JOINT_ORDER.
SO101_JOINT_RANGES = np.radians([JOINT_LIMITS_DEG[name] for name in SO101_JOINT_ORDER])


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
        patch("mujoco.mj_name2id") as mock_name2id,
    ):
        joint_ids = {name: index for index, name in enumerate(SO101_JOINT_ORDER)}
        mock_name2id.side_effect = lambda _model, obj, name: (
            joint_ids.get(name, -1) if obj == mujoco.mjtObj.mjOBJ_JOINT else 0
        )
        mock_model = MagicMock()
        mock_model.nq = 6
        mock_model.nv = 6
        mock_model.nu = 6
        mock_model.actuator_trntype = np.full(6, mujoco.mjtTrn.mjTRN_JOINT, dtype=np.int32)
        mock_model.actuator_trnid = np.column_stack((np.arange(6), np.zeros(6, dtype=np.int32)))
        mock_model.opt.timestep = 0.005
        mock_model.stat.center = np.zeros(3)
        mock_model.stat.extent = 1.0
        mock_model.jnt_qposadr = [0, 1, 2, 3, 4, 5]
        mock_model.jnt_dofadr = [0, 1, 2, 3, 4, 5]
        mock_model.jnt_range = SO101_JOINT_RANGES.copy()
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
        patch("mujoco.mj_name2id") as mock_name2id,
    ):
        joint_ids = {name: index for index, name in enumerate(BIMANUAL_SO101_JOINT_ORDER)}
        mock_name2id.side_effect = lambda _model, obj, name: (
            joint_ids.get(name, -1) if obj == mujoco.mjtObj.mjOBJ_JOINT else 0
        )
        mock_model = MagicMock()
        mock_model.nq = 12
        mock_model.nv = 12
        mock_model.nu = 12
        mock_model.actuator_trntype = np.full(12, mujoco.mjtTrn.mjTRN_JOINT, dtype=np.int32)
        mock_model.actuator_trnid = np.column_stack((np.arange(12), np.zeros(12, dtype=np.int32)))
        mock_model.opt.timestep = 0.005
        mock_model.stat.center = np.zeros(3)
        mock_model.stat.extent = 1.0
        mock_model.jnt_qposadr = list(range(12))
        mock_model.jnt_dofadr = list(range(12))
        mock_model.jnt_range = np.tile(SO101_JOINT_RANGES, (2, 1))
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
        assert robot._viser_host == "127.0.0.1"  # noqa: SLF001

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

    def test_connect_maps_actuators_by_joint_name(self, mock_mujoco: MagicMock) -> None:
        mock_mujoco.actuator_trnid[:, 0] = np.arange(6)[::-1]
        robot = MuJoCoSO101(model_path="/fake/model.xml")

        robot.connect()

        assert robot._ctrl_indices == tuple(reversed(range(6)))  # noqa: SLF001
        action = np.arange(6, dtype=np.float32)
        robot.send_action(action)
        expected = normalized_to_radians(action, SO101_JOINT_RANGES, SO101_JOINT_ORDER)
        np.testing.assert_allclose(robot._data.ctrl[::-1], expected)  # noqa: SLF001

    @pytest.mark.parametrize(
        "actuator_trntype,actuator_trnid",
        [
            (
                np.full(6, mujoco.mjtTrn.mjTRN_JOINT),
                np.column_stack(([0, 1, 2, 3, 4, 1000], np.zeros(6))),
            ),
            (np.full(6, mujoco.mjtTrn.mjTRN_JOINT), np.column_stack(([0, 0, 2, 3, 4, 5], np.zeros(6)))),
            (np.full(6, mujoco.mjtTrn.mjTRN_SITE), np.column_stack((np.arange(6), np.zeros(6)))),
        ],
    )
    def test_connect_rejects_missing_or_ambiguous_joint_actuator_mapping(
        self,
        mock_mujoco: MagicMock,
        actuator_trntype: np.ndarray,
        actuator_trnid: np.ndarray,
    ) -> None:
        mock_mujoco.nu = len(actuator_trntype)
        mock_mujoco.actuator_trntype = actuator_trntype
        mock_mujoco.actuator_trnid = actuator_trnid
        robot = MuJoCoSO101(model_path="/fake/model.xml")

        with pytest.raises(ValueError, match="joints/actuators"):
            robot.connect()

        assert not robot.is_connected()

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
    def test_get_observation_shapes(self, mock_mujoco: MagicMock) -> None:
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
            "_viser_host": "127.0.0.1",
            "_viser_port": 9090,
            "_unit": "normalized",
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
            "_viser_host": "0.0.0.0",  # noqa: S104
        }
        robot = MuJoCoSO101.__new__(MuJoCoSO101)
        robot.__setstate__(state)
        assert robot._http_host == "0.0.0.0"  # noqa: SLF001, S104
        assert robot._http_port == 9000  # noqa: SLF001
        assert robot._viser_host == "0.0.0.0"  # noqa: SLF001, S104


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
        robot._commands.put(SwitchSceneCommand(scene_id="yahtzee"))  # noqa: SLF001

        with patch.object(robot, "_switch_to_scene") as switch:
            robot.get_observation()

        switch.assert_called_once_with("yahtzee")

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
            assert robot._switch_to_scene("yahtzee") is True  # noqa: SLF001
        assert robot._current_scene_id == "yahtzee"  # noqa: SLF001

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
                patch("viser.ViserServer", return_value=MagicMock()) as server_factory,
                patch("mjviser.ViserMujocoScene", return_value=MagicMock()),
            ):
                assert robot._launch_viser_viewer() is True  # noqa: SLF001
            server_factory.assert_called_once_with(host="127.0.0.1", port=9090, verbose=False)
            assert console.file is sys.stderr
        finally:
            console.file = original_file

    def test_configured_viser_host_is_used(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml", viser_host="0.0.0.0")  # noqa: S104
        robot.connect()

        with (
            patch("viser.ViserServer", return_value=MagicMock()) as server_factory,
            patch("mjviser.ViserMujocoScene", return_value=MagicMock()),
        ):
            assert robot._launch_viser_viewer() is True  # noqa: SLF001

        server_factory.assert_called_once_with(host="0.0.0.0", port=9090, verbose=False)  # noqa: S104

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


class TestBuildViserGui:
    @staticmethod
    def _server() -> MagicMock:
        return MagicMock()

    def test_rebuild_clears_the_previous_gui_and_scene(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        server = self._server()
        robot._viser_server = server  # noqa: SLF001

        with patch("mjviser.ViserMujocoScene", return_value=MagicMock()):
            robot._build_viser_gui()  # noqa: SLF001
            robot._build_viser_gui()  # noqa: SLF001
            robot._build_viser_gui()  # noqa: SLF001

        assert server.gui.reset.call_count == 3
        assert server.scene.reset.call_count == 3
        # The panel's one camera hook, registered when it was created.
        assert server.on_client_connect.call_count == 1

    def test_mjviser_body_tracking_and_camera_gui_are_replaced(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot._viser_server = self._server()  # noqa: SLF001
        scene = MagicMock()

        with patch("mjviser.ViserMujocoScene", return_value=scene):
            robot._build_viser_gui()  # noqa: SLF001

        assert scene.camera_tracking_enabled is False
        scene.create_scene_gui.assert_not_called()
        scene.set_refresh_handler.assert_called_once()

    def test_tab_order(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        server = self._server()
        robot._viser_server = server  # noqa: SLF001

        with patch("mjviser.ViserMujocoScene", return_value=MagicMock()):
            robot._build_viser_gui()  # noqa: SLF001

        tabs = server.gui.add_tab_group.return_value
        labels = [call.args[0] for call in tabs.add_tab.call_args_list]
        assert labels == ["Simulation", "Camera", "Visualization", "Groups"]
        assert robot._viser_panel is not None  # noqa: SLF001

    def test_panel_survives_rebuilds_and_close_drops_it(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot._viser_server = self._server()  # noqa: SLF001

        with patch("mjviser.ViserMujocoScene", return_value=MagicMock()):
            robot._build_viser_gui()  # noqa: SLF001
            panel = robot._viser_panel  # noqa: SLF001
            robot._recreate_viser_scene()  # noqa: SLF001

        assert robot._viser_panel is panel  # noqa: SLF001
        robot._close_viewer()  # noqa: SLF001
        assert robot._viser_panel is None  # noqa: SLF001

    def test_panel_submits_to_the_current_command_queue(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot._viser_server = self._server()  # noqa: SLF001
        with patch("mjviser.ViserMujocoScene", return_value=MagicMock()):
            robot._build_viser_gui()  # noqa: SLF001

        robot._commands = queue.Queue()  # noqa: SLF001
        robot._viser_panel._submit(ResetCommand())  # noqa: SLF001

        assert isinstance(robot._commands.get_nowait(), ResetCommand)  # noqa: SLF001

    def test_panel_refresh_failure_does_not_stop_the_loop(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot._viser_scene = MagicMock()  # noqa: SLF001
        robot._viser_panel = MagicMock()  # noqa: SLF001
        robot._viser_panel.refresh.side_effect = RuntimeError("boom")  # noqa: SLF001

        robot.get_observation()
        robot.get_observation()

        assert robot._viser_panel_failed is True  # noqa: SLF001


class TestOperatorCommands:
    def test_home_command(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot._commands.put(HomeCommand())  # noqa: SLF001

        with patch.object(robot, "_go_home") as go_home:
            robot.get_observation()

        go_home.assert_called_once()

    def test_a_failing_command_does_not_escape(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot._commands.put(HomeCommand())  # noqa: SLF001
        robot._commands.put(ResetCommand())  # noqa: SLF001

        with (
            patch.object(robot, "_go_home", side_effect=IndexError("bad model")),
            patch.object(robot, "_run_scene_reset") as scene_reset,
        ):
            obs = robot.get_observation()

        scene_reset.assert_called_once()
        assert obs.joint_positions.shape == (6,)

    def test_fixed_seed_makes_resets_repeat(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        draws: list[float] = []
        robot._scene_on_reset = lambda _m, _d, rng: draws.append(float(rng.random()))  # noqa: SLF001

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

    def test_reseeding_keeps_the_generator_shared_with_auto_reset(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        rng = robot._rng  # noqa: SLF001
        robot._set_seed(5)  # noqa: SLF001
        assert robot._rng is rng  # noqa: SLF001

    def test_auto_reset_settings_apply_and_persist(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        helper = MagicMock()
        robot._episode_auto_reset = helper  # noqa: SLF001

        robot._commands.put(SetAutoResetCommand(enabled=False))  # noqa: SLF001
        robot._commands.put(SetAutoResetCommand(dwell_s=2.0))  # noqa: SLF001
        robot.get_observation()

        helper.set_active.assert_called_with(False)  # noqa: FBT003
        helper.set_dwell.assert_called_with(2.0)
        with patch(
            "physicalai_mujoco_so101_plugin.episode_auto_reset.EpisodeAutoReset.maybe_create",
            return_value=None,
        ) as maybe_create:
            robot._init_episode_auto_reset()  # noqa: SLF001
        assert maybe_create.call_args.kwargs["active"] is False
        assert maybe_create.call_args.kwargs["success_dwell_s"] == 2.0

    def test_auto_reset_command_without_a_helper_is_survivable(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot._episode_auto_reset = None  # noqa: SLF001
        robot._commands.put(SetAutoResetCommand(enabled=False))  # noqa: SLF001

        robot.get_observation()

        assert robot._auto_reset_active is False  # noqa: SLF001

    def test_unknown_object_pose_is_ignored(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot._commands.put(SetObjectPoseCommand(joint="nope", position=(0.0, 0.0, 0.1)))  # noqa: SLF001

        robot.get_observation()

        assert robot._held_objects == {}  # noqa: SLF001

    def test_http_status_reports_compatible_scenes_seed_and_objects(self) -> None:
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        status = robot._http_status()  # noqa: SLF001
        assert "garment_fold" in status["scenes"]
        assert "garment_fold" not in status["compatible_scenes"]
        assert status["seed"] is None
        assert status["objects"] == []

        assert BiMuJoCoSO101(model_path="/fake/model.xml")._http_status()["compatible_scenes"] == ["garment_fold"]  # noqa: SLF001


class TestSceneCompatibility:
    def test_switch_rejects_a_different_arm_count_before_loading(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()

        with patch("mujoco.MjModel.from_xml_path") as load:
            assert robot._switch_to_scene("garment_fold") is False  # noqa: SLF001

        load.assert_not_called()

    def test_scene_key_cycles_only_compatible_scenes(self, mock_mujoco_bimanual: MagicMock) -> None:
        _ = mock_mujoco_bimanual
        robot = BiMuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        robot._current_scene_id = "garment_fold"  # noqa: SLF001
        robot._pending_scene_switch = True  # noqa: SLF001

        with patch.object(robot, "_switch_to_scene") as switch:
            robot._check_pending_scene_switch()  # noqa: SLF001

        switch.assert_not_called()


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


class TestCameraFrames:
    @staticmethod
    def _stream_one_frame(mock_mujoco: MagicMock, rendered: np.ndarray, *, mirror: bool) -> np.ndarray:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml", cameras=[{"name": "wrist", "mirror_horizontal": mirror}])
        renderer = MagicMock()
        renderer.render.return_value = rendered
        with patch("mujoco.Renderer", return_value=renderer):
            robot.connect()
        robot._render_cameras()  # noqa: SLF001
        return robot._frame_buffers["wrist"].snapshot().frame  # noqa: SLF001

    def test_frames_are_streamed_as_rendered(self, mock_mujoco: MagicMock) -> None:
        """mujoco.Renderer already returns upright images; flipping again turns them upside down."""
        rendered = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        np.testing.assert_array_equal(self._stream_one_frame(mock_mujoco, rendered, mirror=False), rendered)

    def test_mirror_flips_left_to_right_only(self, mock_mujoco: MagicMock) -> None:
        rendered = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        np.testing.assert_array_equal(self._stream_one_frame(mock_mujoco, rendered, mirror=True), rendered[:, ::-1])


def _driver_normalized(fraction: float, *, gripper: bool) -> float:
    """What the SO101 driver reports for a joint at ``fraction`` of its calibrated range."""
    return fraction * 100.0 if gripper else fraction * 200.0 - 100.0


class TestJointUnits:
    @pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.9, 1.0])
    def test_normalized_matches_the_so101_driver_mapping(self, fraction: float) -> None:
        low, high = SO101_JOINT_RANGES[:, 0], SO101_JOINT_RANGES[:, 1]
        radians = low + fraction * (high - low)

        normalized = radians_to_normalized(radians, SO101_JOINT_RANGES, SO101_JOINT_ORDER)

        expected = [_driver_normalized(fraction, gripper=name == "gripper") for name in SO101_JOINT_ORDER]
        np.testing.assert_allclose(normalized, expected, atol=1e-9)
        np.testing.assert_allclose(
            normalized_to_radians(normalized, SO101_JOINT_RANGES, SO101_JOINT_ORDER), radians, atol=1e-12
        )

    def test_bimanual_grippers_use_the_gripper_range(self) -> None:
        limits = np.tile(SO101_JOINT_RANGES, (2, 1))
        normalized = radians_to_normalized(limits[:, 0], limits, BIMANUAL_SO101_JOINT_ORDER)
        grippers = [i for i, name in enumerate(BIMANUAL_SO101_JOINT_ORDER) if name.endswith("gripper")]

        assert grippers == [5, 11]
        np.testing.assert_allclose(normalized[grippers], 0.0)
        np.testing.assert_allclose(np.delete(normalized, grippers), -100.0)

    def test_values_outside_the_range_are_clamped(self) -> None:
        radians = SO101_JOINT_RANGES[:, 1] + 0.2
        np.testing.assert_allclose(radians_to_normalized(radians, SO101_JOINT_RANGES, SO101_JOINT_ORDER), 100.0)

        targets = normalized_to_radians(np.full(6, -150.0), SO101_JOINT_RANGES, SO101_JOINT_ORDER)
        np.testing.assert_allclose(targets, SO101_JOINT_RANGES[:, 0])

    def test_observation_and_action_use_normalized_units(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml")
        robot.connect()
        qpos, qvel = robot._data.qpos.copy(), robot._data.qvel.copy()  # noqa: SLF001

        obs = robot.get_observation()

        np.testing.assert_allclose(
            obs.joint_positions, radians_to_normalized(qpos, SO101_JOINT_RANGES, SO101_JOINT_ORDER), rtol=1e-6
        )
        span = SO101_JOINT_RANGES[:, 1] - SO101_JOINT_RANGES[:, 0]
        units = np.array([100.0 if name == "gripper" else 200.0 for name in SO101_JOINT_ORDER])
        np.testing.assert_allclose(obs.sensor_data["velocities"], qvel * units / span, rtol=1e-6)

        robot.send_action(obs.joint_positions)
        np.testing.assert_allclose(robot._data.ctrl, qpos, atol=1e-5)  # noqa: SLF001

    def test_degrees_unit_keeps_joint_angles(self, mock_mujoco: MagicMock) -> None:
        _ = mock_mujoco
        robot = MuJoCoSO101(model_path="/fake/model.xml", unit="degrees")
        robot.connect()

        obs = robot.get_observation()
        np.testing.assert_allclose(obs.joint_positions, np.degrees(robot._data.qpos), rtol=1e-6)  # noqa: SLF001

        action = np.array([10.0, 20.0, -5.0, 0.0, 15.0, 30.0])
        robot.send_action(action)
        np.testing.assert_allclose(robot._data.ctrl, np.radians(action))  # noqa: SLF001

    def test_non_default_unit_is_part_of_the_recipe(self) -> None:
        robot = MuJoCoSO101(model_path="/fake/model.xml", unit="degrees")

        assert Config.from_instance(robot)["init_args"]["unit"] == "degrees"

    def test_rejects_an_unknown_unit(self) -> None:
        with pytest.raises(ValueError, match="Unsupported unit"):
            MuJoCoSO101(model_path="/fake/model.xml", unit="ticks")  # type: ignore[arg-type]

    def test_connect_rejects_a_joint_without_a_range(self, mock_mujoco: MagicMock) -> None:
        mock_mujoco.jnt_range[1] = (0.0, 0.0)
        robot = MuJoCoSO101(model_path="/fake/model.xml")

        with pytest.raises(ValueError, match="joints/actuators"):
            robot.connect()

