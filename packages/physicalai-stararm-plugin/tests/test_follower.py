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
    stararm102fl: object


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
    sys.modules.pop("physicalai_stararm_plugin.stararm102fl", None)
    sys.modules.pop("physicalai_stararm_plugin", None)
    pkg = sys.modules.get("physicalai_stararm_plugin")
    if pkg is not None and hasattr(pkg, "stararm102fl"):
        del cast("_StararmPackageModule", pkg).stararm102fl
    with patch.dict(sys.modules, {"motorbridge_smart_servo": module}):
        import_module("physicalai_stararm_plugin.stararm102fl")
        yield module


def _create_robot(mock_smart_servo: MagicMock, **kwargs: Any) -> Any:
    from physicalai_stararm_plugin import StarArm102FLFollower

    _ = mock_smart_servo
    return StarArm102FLFollower(**kwargs)


class TestStarArm102FLFollowerConstruction:
    def test_defaults(self, mock_smart_servo: MagicMock) -> None:
        robot = _create_robot(mock_smart_servo)

        assert robot.port == "/dev/ttyUSB0"
        assert robot.baudrate == 1_000_000
        assert robot.command_interval_ms == 10

    def test_exports_recipe_and_device_identity(self, mock_smart_servo: MagicMock) -> None:
        robot = _create_robot(mock_smart_servo, port="/dev/ttyUSB1", baudrate=115200, command_interval_ms=20)

        assert robot.device_ids == ("stararm102-fl:/dev/ttyUSB1",)
        assert Config.from_instance(robot)["init_args"] == {
            "port": "/dev/ttyUSB1",
            "baudrate": 115200,
            "command_interval_ms": 20,
        }

    def test_invalid_values_raise(self, mock_smart_servo: MagicMock) -> None:
        from physicalai_stararm_plugin import StarArm102FLFollower

        with pytest.raises(ValueError, match="baudrate"):
            StarArm102FLFollower(baudrate=0)
        with pytest.raises(ValueError, match="command_interval_ms"):
            StarArm102FLFollower(command_interval_ms=-1)


class TestStarArm102FLFollowerLifecycle:
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

    def test_disconnect_closes_bus(self, mock_smart_servo: MagicMock) -> None:
        robot = _create_robot(mock_smart_servo)
        robot.connect()
        bus = mock_smart_servo.FashionStarServo.return_value

        robot.disconnect()

        bus.close.assert_called_once()
        assert robot.is_connected() is False


class TestStarArm102FLFollowerActionObservation:
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

    def test_send_action_calls_set_angle(self, mock_smart_servo: MagicMock) -> None:
        robot = _create_robot(mock_smart_servo, command_interval_ms=10)
        robot.connect()
        bus = mock_smart_servo.FashionStarServo.return_value

        action = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
        robot.send_action(action, goal_time=0.2)

        assert bus.set_angle.call_count == 7
        first_call = bus.set_angle.call_args_list[0]
        assert first_call == call(0, 1.0, multi_turn=True, interval_ms=200)

    def test_send_action_validates_shape(self, mock_smart_servo: MagicMock) -> None:
        robot = _create_robot(mock_smart_servo)
        robot.connect()

        with pytest.raises(ValueError, match="Expected action shape"):
            robot.send_action(np.zeros(6, dtype=np.float32))
