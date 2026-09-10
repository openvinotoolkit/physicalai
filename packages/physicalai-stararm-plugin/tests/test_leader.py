from __future__ import annotations

import sys
from importlib import import_module
from importlib.machinery import ModuleSpec
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from physicalai.config import Config

if TYPE_CHECKING:
    from collections.abc import Generator

_mock_smart_servo = MagicMock()
_mock_smart_servo.__spec__ = ModuleSpec("motorbridge_smart_servo", None)
sys.modules.setdefault("motorbridge_smart_servo", _mock_smart_servo)


class _ServoFactoryFn:
    servo_angles: list[float]


class _StararmPackageModule:
    stararm102hd: object
    stararm102ld: object


def _make_mock_smart_servo() -> MagicMock:
    module = MagicMock()
    bus = MagicMock()
    module.FashionStarServo.return_value = bus
    bus.ping.return_value = True

    servo_angles = [0.0, 10.0, -10.0, 30.0, 40.0, 50.0, 60.0]

    def read_angle_side_effect(servo_id: int, *, multi_turn: bool = True) -> MagicMock:
        _ = multi_turn
        sample = MagicMock()
        sample.raw_deg = cast("_ServoFactoryFn", _make_mock_smart_servo).servo_angles[servo_id]
        sample.filtered_deg = cast("_ServoFactoryFn", _make_mock_smart_servo).servo_angles[servo_id]
        sample.reliable = True
        return sample

    cast("_ServoFactoryFn", _make_mock_smart_servo).servo_angles = servo_angles

    bus.read_angle.side_effect = read_angle_side_effect
    return module


@pytest.fixture
def mock_smart_servo() -> Generator[MagicMock]:
    module = _make_mock_smart_servo()
    sys.modules.pop("physicalai_stararm_plugin.stararm102hd", None)
    sys.modules.pop("physicalai_stararm_plugin.stararm102ld", None)
    sys.modules.pop("physicalai_stararm_plugin", None)
    pkg = sys.modules.get("physicalai_stararm_plugin")
    if pkg is not None and hasattr(pkg, "stararm102hd"):
        del cast("_StararmPackageModule", pkg).stararm102hd
    if pkg is not None and hasattr(pkg, "stararm102ld"):
        del cast("_StararmPackageModule", pkg).stararm102ld
    with patch.dict(sys.modules, {"motorbridge_smart_servo": module}):
        import_module("physicalai_stararm_plugin.stararm102hd")
        import_module("physicalai_stararm_plugin.stararm102ld")
        yield module


def _create_robot(mock_smart_servo: MagicMock, **kwargs: Any) -> Any:
    from physicalai_stararm_plugin import StarArm102HDLeader

    _ = mock_smart_servo
    return StarArm102HDLeader(**kwargs)


class TestStarArm102HDLeaderConstruction:
    def test_defaults(self, mock_smart_servo: MagicMock) -> None:
        robot = _create_robot(mock_smart_servo)

        assert robot.port == "/dev/ttyUSB0"
        assert robot.baudrate == 1_000_000
        assert robot.control_mode == "passive"
        assert robot.joint_names == [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_yaw",
            "wrist_roll",
            "gripper",
        ]

    def test_exports_recipe_and_device_identity(self, mock_smart_servo: MagicMock) -> None:
        robot = _create_robot(mock_smart_servo, port="/dev/ttyUSB1", baudrate=115200)

        assert robot.device_ids == ("stararm102-hd:/dev/ttyUSB1",)
        assert Config.from_instance(robot)["init_args"] == {"port": "/dev/ttyUSB1", "baudrate": 115200}

    def test_invalid_baudrate_raises(self, mock_smart_servo: MagicMock) -> None:
        from physicalai_stararm_plugin import StarArm102HDLeader

        with pytest.raises(ValueError, match="baudrate"):
            StarArm102HDLeader(baudrate=0)


class TestStarArm102HDLeaderLifecycle:
    def test_connect_pings_and_configures(self, mock_smart_servo: MagicMock) -> None:
        robot = _create_robot(mock_smart_servo, port="/dev/ttyUSB1")
        robot.connect()

        mock_smart_servo.FashionStarServo.assert_called_once_with("/dev/ttyUSB1", baudrate=1_000_000)
        bus = mock_smart_servo.FashionStarServo.return_value

        assert bus.ping.call_args_list == [
            call(0),
            call(1),
            call(2),
            call(3),
            call(4),
            call(5),
            call(6),
        ]
        assert bus.unlock.call_count == 7
        assert bus.reset_multi_turn.call_count == 7

    def test_connect_failure_cleans_up(self, mock_smart_servo: MagicMock) -> None:
        bus = mock_smart_servo.FashionStarServo.return_value
        bus.ping.return_value = False
        robot = _create_robot(mock_smart_servo)

        with pytest.raises(ConnectionError, match="did not respond"):
            robot.connect()

        bus.close.assert_called_once()
        assert robot.is_connected() is False

    def test_connect_is_idempotent(self, mock_smart_servo: MagicMock) -> None:
        robot = _create_robot(mock_smart_servo)
        robot.connect()
        robot.connect()

        mock_smart_servo.FashionStarServo.assert_called_once()

    def test_disconnect_closes_bus(self, mock_smart_servo: MagicMock) -> None:
        robot = _create_robot(mock_smart_servo)
        robot.connect()
        bus = mock_smart_servo.FashionStarServo.return_value

        robot.disconnect()

        bus.close.assert_called_once()
        assert robot.is_connected() is False


class TestStarArm102HDLeaderObservation:
    def test_observation_returns_angles(self, mock_smart_servo: MagicMock) -> None:
        robot = _create_robot(mock_smart_servo)
        robot.connect()
        obs = robot.get_observation()

        expected = np.array([0, 10, -10, 30, 40, 50, 60], dtype=np.float32)
        np.testing.assert_allclose(obs.joint_positions, expected)
        assert isinstance(obs.timestamp, float)
        assert obs.sensor_data is not None
        assert "raw_positions" in obs.sensor_data
        assert "reliable" in obs.sensor_data

    def test_send_action_is_noop_in_passive_mode(self, mock_smart_servo: MagicMock) -> None:
        robot = _create_robot(mock_smart_servo)
        robot.connect()
        bus = mock_smart_servo.FashionStarServo.return_value

        robot.send_action(np.zeros(7, dtype=np.float32))
        bus.set_angle.assert_not_called()

    def test_send_action_commands_in_assist_mode(self, mock_smart_servo: MagicMock) -> None:
        robot = _create_robot(mock_smart_servo, control_mode="assist", command_interval_ms=20)
        robot.connect()
        bus = mock_smart_servo.FashionStarServo.return_value

        robot.send_action(np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32), goal_time=0.1)

        assert bus.set_angle.call_count == 7
        assert bus.set_angle.call_args_list[0] == call(0, 1.0, multi_turn=True, interval_ms=100)

    def test_hold_and_release(self, mock_smart_servo: MagicMock) -> None:
        robot = _create_robot(mock_smart_servo, control_mode="assist")
        robot.connect()
        bus = mock_smart_servo.FashionStarServo.return_value

        robot.hold_position(goal_time=0.2)
        assert robot.is_holding is True
        assert bus.set_angle.call_count == 7

        robot.release_hold()
        assert robot.is_holding is False


class TestStarArm102LDLeader:
    def test_ld_device_identity(self, mock_smart_servo: MagicMock) -> None:
        from physicalai_stararm_plugin import StarArm102LDLeader

        robot = StarArm102LDLeader(port="/dev/ttyUSB2")
        assert robot.device_ids == ("stararm102-ld:/dev/ttyUSB2",)

    def test_ld_connect_and_observe(self, mock_smart_servo: MagicMock) -> None:
        from physicalai_stararm_plugin import StarArm102LDLeader

        robot = StarArm102LDLeader()
        robot.connect()
        obs = robot.get_observation()
        assert obs.joint_positions.shape == (7,)
