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

    def sync_monitor_side_effect(servo_ids: list[int]) -> dict[int, MagicMock]:
        monitors = {}
        for servo_id in servo_ids:
            monitor = MagicMock()
            monitor.angle_deg = cast("_ServoFactoryFn", _make_mock_smart_servo).servo_angles[servo_id]
            monitor.reliable = True
            monitors[servo_id] = monitor
        return monitors

    cast("_ServoFactoryFn", _make_mock_smart_servo).servo_angles = servo_angles

    bus.sync_monitor.side_effect = sync_monitor_side_effect
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

    def test_observation_reads_all_servos_in_one_sync_command(self, mock_smart_servo: MagicMock) -> None:
        robot = _create_robot(mock_smart_servo)
        robot.connect()
        robot.get_observation()

        mock_smart_servo.FashionStarServo.return_value.sync_monitor.assert_called_once_with([0, 1, 2, 3, 4, 5, 6])

    def test_observation_holds_glitch_then_accepts_persistent_jump(self, mock_smart_servo: MagicMock) -> None:
        robot = _create_robot(mock_smart_servo)
        robot.connect()
        angles = cast("_ServoFactoryFn", _make_mock_smart_servo).servo_angles
        robot.get_observation()

        angles[0] = 120.0  # shoulder_pan jumps 120 deg in one sample, past the 90 deg glitch threshold
        held = [robot.get_observation() for _ in range(3)]
        assert all(obs.joint_positions[0] == pytest.approx(0.0) for obs in held)
        assert all(obs.sensor_data["reliable"][0] == 0.0 for obs in held)

        assert robot.get_observation().joint_positions[0] == pytest.approx(120.0)

    def test_disable_torque_unlocks_every_servo(self, mock_smart_servo: MagicMock) -> None:
        robot = _create_robot(mock_smart_servo, unlock_on_connect=False)
        robot.connect()
        bus = mock_smart_servo.FashionStarServo.return_value
        bus.unlock.assert_not_called()

        robot.disable_torque()

        assert bus.unlock.call_args_list == [call(i) for i in range(7)]
        assert robot.is_holding is False

    def test_set_zero_position_sets_origin_and_resets_baseline(self, mock_smart_servo: MagicMock) -> None:
        robot = _create_robot(mock_smart_servo)
        robot.connect()
        robot.get_observation()
        bus = mock_smart_servo.FashionStarServo.return_value
        bus.reset_mock()

        robot.set_zero_position()

        assert bus.set_origin_point.call_args_list == [call(i) for i in range(7)]
        assert bus.reset_multi_turn.call_args_list == [call(i) for i in range(7)]
        # The origin moved, so the next reading must not be held back as a glitch.
        cast("_ServoFactoryFn", _make_mock_smart_servo).servo_angles[0] = 120.0
        assert robot.get_observation().joint_positions[0] == pytest.approx(120.0)

    def test_reconnect_starts_new_glitch_baseline(self, mock_smart_servo: MagicMock) -> None:
        robot = _create_robot(mock_smart_servo)
        robot.connect()
        robot.get_observation()
        robot.disconnect()

        cast("_ServoFactoryFn", _make_mock_smart_servo).servo_angles[0] = 120.0  # moved while disconnected
        robot.connect()
        obs = robot.get_observation()

        assert obs.joint_positions[0] == pytest.approx(120.0)
        assert obs.sensor_data["reliable"][0] == 1.0

    def test_observation_never_responded_servo_raises(self, mock_smart_servo: MagicMock) -> None:
        robot = _create_robot(mock_smart_servo)
        robot.connect()
        mock_smart_servo.FashionStarServo.return_value.sync_monitor.side_effect = lambda ids: dict.fromkeys(ids)

        with pytest.raises(ConnectionError, match="never responded"):
            robot.get_observation()

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
