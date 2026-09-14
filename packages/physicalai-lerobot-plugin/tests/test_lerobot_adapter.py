# ruff: noqa: SLF001

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from typing import Any, Literal, cast
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
from physicalai.robot.interface import Robot

_MOCK_CONFIG_TYPE = "mock_robot"


def _ensure_optional_sdk_spec() -> None:
    sdk = sys.modules.get("scservo_sdk")
    if sdk is not None and getattr(sdk, "__spec__", None) is None:
        sdk.__spec__ = ModuleSpec("scservo_sdk", None)


def test_adapters_are_config_exportable() -> None:
    from physicalai.config import Config

    from physicalai_lerobot_plugin.lerobot_adapter import (
        LeRobotAdapter,
        LeRobotTeleoperatorAdapter,
    )

    adapter = LeRobotAdapter(_MOCK_CONFIG_TYPE, {}, _robot=MagicMock())
    teleoperator = LeRobotTeleoperatorAdapter(_MOCK_CONFIG_TYPE, {}, _teleoperator=MagicMock())

    assert Config.from_instance(adapter).init_args["config_type"] == _MOCK_CONFIG_TYPE
    assert Config.from_instance(teleoperator).init_args["config_type"] == _MOCK_CONFIG_TYPE
    assert isinstance(adapter, Robot)
    assert isinstance(teleoperator, Robot)


def test_teleoperator_config_lookup_imports_registered_types() -> None:
    from physicalai_lerobot_plugin.lerobot_adapter import _teleoperator_config_class

    _ensure_optional_sdk_spec()
    assert _teleoperator_config_class("so101_leader") is not None


def test_dynamic_import_guard_allows_only_trusted_prefixes(monkeypatch: pytest.MonkeyPatch) -> None:
    from physicalai_lerobot_plugin.lerobot_adapter import _is_allowed_dynamic_import

    assert _is_allowed_dynamic_import("lerobot.robots.so_follower")
    assert _is_allowed_dynamic_import("lerobot.teleoperators.so_leader")

    assert not _is_allowed_dynamic_import("os")
    assert not _is_allowed_dynamic_import("subprocess")
    assert not _is_allowed_dynamic_import("lerobotx.robots.fake")

    monkeypatch.setenv("TRUST_UNVERIFIED_PLUGIN", "yes")

    assert _is_allowed_dynamic_import("lerobot_robot_example")
    assert _is_allowed_dynamic_import("lerobot_teleoperator_example")


def test_config_import_rejects_untrusted_package_root() -> None:
    from physicalai_lerobot_plugin.lerobot_adapter import _import_config_modules

    with pytest.raises(ValueError, match="Unsupported package root"):
        _import_config_modules("subprocess")


def _make_adapter(
    mock_robot: MagicMock,
    role: Literal["follower", "leader"] = "follower",
) -> Any:
    from physicalai_lerobot_plugin.lerobot_adapter import LeRobotAdapter

    return LeRobotAdapter(
        config_type=_MOCK_CONFIG_TYPE,
        config_kwargs={},
        role=role,
        _robot=mock_robot,
    )


SO100_POS_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


@pytest.fixture
def mock_lerobot_robot() -> MagicMock:
    robot = MagicMock()
    robot._is_connected_state = False

    type(robot).is_connected = PropertyMock(side_effect=lambda: robot._is_connected_state)

    def connect(calibrate: bool = True) -> None:  # noqa: FBT001, FBT002
        robot._is_connected_state = True

    robot.connect.side_effect = connect

    def disconnect() -> None:
        robot._is_connected_state = False

    robot.disconnect.side_effect = disconnect

    def get_observation() -> dict:
        return {key: float(i * 10) for i, key in enumerate(SO100_POS_KEYS)}

    robot.get_observation.side_effect = get_observation

    def send_action(action: dict) -> dict:
        return action

    robot.send_action.side_effect = send_action

    return robot


class TestLeRobotAdapterConstruction:
    def test_defaults(self, mock_lerobot_robot: MagicMock) -> None:

        adapter = _make_adapter(mock_lerobot_robot)

        assert not adapter.is_connected()
        with pytest.raises(RuntimeError, match="not yet discovered"):
            _ = adapter.NUM_JOINTS

    def test_invalid_role(self, mock_lerobot_robot: MagicMock) -> None:

        with pytest.raises(ValueError, match="Invalid role"):
            _make_adapter(mock_lerobot_robot, role=cast(Any, "invalid"))

    def test_satisfies_robot_protocol(self, mock_lerobot_robot: MagicMock) -> None:
        assert isinstance(_make_adapter(mock_lerobot_robot), Robot)

    def test_device_ids_include_configured_port(self, mock_lerobot_robot: MagicMock) -> None:
        from physicalai_lerobot_plugin.lerobot_adapter import LeRobotAdapter

        adapter = LeRobotAdapter(_MOCK_CONFIG_TYPE, {"port": "/dev/ttyACM0"}, _robot=mock_lerobot_robot)

        assert adapter.device_ids == ("serial:ttyACM0",)

    def test_device_ids_collect_nested_ports(self, mock_lerobot_robot: MagicMock) -> None:
        from physicalai_lerobot_plugin.lerobot_adapter import LeRobotAdapter

        config_kwargs = {
            "left_arm": {"port": "/dev/ttyACM0"},
            "right_arm": {"port": "/dev/ttyACM1"},
        }
        adapter = LeRobotAdapter(_MOCK_CONFIG_TYPE, config_kwargs, _robot=mock_lerobot_robot)

        assert adapter.device_ids == (
            "serial:ttyACM0",
            "serial:ttyACM1",
        )


class TestLeRobotAdapterLifecycle:
    def test_connect_and_disconnect(self, mock_lerobot_robot: MagicMock) -> None:

        adapter = _make_adapter(mock_lerobot_robot)

        assert not adapter.is_connected()
        adapter.connect()
        assert adapter.is_connected()
        assert mock_lerobot_robot.connect.called

        adapter.disconnect()
        assert not adapter.is_connected()

    def test_connect_is_idempotent(self, mock_lerobot_robot: MagicMock) -> None:

        adapter = _make_adapter(mock_lerobot_robot)

        adapter.connect()
        adapter.connect()
        assert mock_lerobot_robot.connect.call_count == 1

    def test_disconnect_is_idempotent(self, mock_lerobot_robot: MagicMock) -> None:

        adapter = _make_adapter(mock_lerobot_robot)

        adapter.disconnect()
        adapter.disconnect()
        assert mock_lerobot_robot.disconnect.call_count == 0


class TestLeRobotAdapterAutoDetection:
    def test_connect_discovers_joint_order(self, mock_lerobot_robot: MagicMock) -> None:

        adapter = _make_adapter(mock_lerobot_robot)

        adapter.connect()
        assert adapter.joint_names == [
            "elbow_flex",
            "gripper",
            "shoulder_lift",
            "shoulder_pan",
            "wrist_flex",
            "wrist_roll",
        ]
        assert adapter.NUM_JOINTS == 6

    def test_get_observation_auto_discovers(self, mock_lerobot_robot: MagicMock) -> None:

        adapter = _make_adapter(mock_lerobot_robot)

        obs = adapter.get_observation()
        assert adapter.NUM_JOINTS == 6
        np.testing.assert_array_almost_equal(
            obs.joint_positions,
            np.array([20.0, 50.0, 10.0, 0.0, 30.0, 40.0], dtype=np.float32),
        )

    def test_joint_names_autodetected_from_pos_suffix(self, mock_lerobot_robot: MagicMock) -> None:

        custom_obs = {"j1.pos": 1.0, "j2.pos": 2.0, "j3.pos": 3.0}
        mock_lerobot_robot.get_observation.side_effect = None
        mock_lerobot_robot.get_observation.return_value = custom_obs

        adapter = _make_adapter(mock_lerobot_robot)
        obs = adapter.get_observation()
        assert adapter.joint_names == ["j1", "j2", "j3"]
        np.testing.assert_array_almost_equal(
            obs.joint_positions,
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )

    def test_joint_names_strip_fallback_position_suffixes(self, mock_lerobot_robot: MagicMock) -> None:

        custom_obs = {"j1pos": 1.0, "j2pos": 2.0, "shoulder_position": 3.0}
        mock_lerobot_robot.get_observation.side_effect = None
        mock_lerobot_robot.get_observation.return_value = custom_obs

        adapter = _make_adapter(mock_lerobot_robot)
        adapter.get_observation()
        assert adapter.joint_names == ["j1", "j2", "shoulder"]

    def test_properties_raise_before_discovery(self, mock_lerobot_robot: MagicMock) -> None:

        adapter = _make_adapter(mock_lerobot_robot)

        with pytest.raises(RuntimeError, match="not yet discovered"):
            _ = adapter.joint_names

        with pytest.raises(RuntimeError, match="not yet discovered"):
            _ = adapter.NUM_JOINTS


def test_teleoperator_uses_non_position_action_keys() -> None:
    from physicalai_lerobot_plugin.lerobot_adapter import LeRobotTeleoperatorAdapter

    teleoperator = MagicMock()
    teleoperator.action_features = {"delta_x": {}, "delta_y": {}}
    adapter = LeRobotTeleoperatorAdapter(_MOCK_CONFIG_TYPE, {}, _teleoperator=teleoperator)

    assert adapter.joint_names == ["delta_x", "delta_y"]
    assert adapter.NUM_JOINTS == 2


class TestLeRobotAdapterObservation:
    def test_observation_has_correct_structure(self, mock_lerobot_robot: MagicMock) -> None:

        adapter = _make_adapter(mock_lerobot_robot)

        obs = adapter.get_observation()
        assert obs.timestamp > 0
        assert obs.state is obs.joint_positions
        assert isinstance(obs.joint_positions, np.ndarray)
        assert obs.joint_positions.dtype == np.float32

    def test_observation_wraps_images_in_frames(self, mock_lerobot_robot: MagicMock) -> None:
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        mock_lerobot_robot.get_observation.return_value = {"joint.pos": 1.0, "camera": image}
        mock_lerobot_robot.get_observation.side_effect = None
        adapter = _make_adapter(mock_lerobot_robot)

        observation = adapter.get_observation()

        assert observation.images is not None
        frame = observation.images["camera"]
        assert frame.data is image
        assert frame.sequence == 0
        assert frame.timestamp > 0


class TestLeRobotAdapterAction:
    def test_send_action_follower(self, mock_lerobot_robot: MagicMock) -> None:

        adapter = _make_adapter(mock_lerobot_robot)
        adapter.connect()

        sorted_pos_keys = sorted(SO100_POS_KEYS)
        action = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        adapter.send_action(action)

        expected_dict = dict(zip(sorted_pos_keys, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], strict=True))
        mock_lerobot_robot.send_action.assert_called_once_with(expected_dict)

    def test_send_action_leader_raises(self, mock_lerobot_robot: MagicMock) -> None:

        adapter = _make_adapter(mock_lerobot_robot, role="leader")

        action = np.zeros(3, dtype=np.float32)
        with pytest.raises(RuntimeError, match="Cannot send actions to a leader"):
            adapter.send_action(action)

    def test_send_action_wrong_shape(self, mock_lerobot_robot: MagicMock) -> None:

        adapter = _make_adapter(mock_lerobot_robot)
        adapter.connect()

        action = np.zeros(3, dtype=np.float32)
        with pytest.raises(ValueError, match="Expected action shape"):
            adapter.send_action(action)

    def test_send_action_raises_before_discovery(self, mock_lerobot_robot: MagicMock) -> None:

        adapter = _make_adapter(mock_lerobot_robot)
        action = np.zeros(6, dtype=np.float32)

        with pytest.raises(RuntimeError, match="not yet discovered"):
            adapter.send_action(action)

    def test_send_action_raises_after_disconnect(self, mock_lerobot_robot: MagicMock) -> None:
        adapter = _make_adapter(mock_lerobot_robot)
        adapter.connect()
        adapter.disconnect()

        with pytest.raises(ConnectionError, match="not connected"):
            adapter.send_action(np.zeros(6, dtype=np.float32))
