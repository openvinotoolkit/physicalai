from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from importlib import import_module
from importlib.machinery import ModuleSpec
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from physicalai.config import Config

from physicalai_rebot_b601_plugin.constants import REBOT_B601_RS_JOINT_DIRECTIONS, REBOT_B601_RS_JOINT_ORDER

if TYPE_CHECKING:
    from collections.abc import Generator

_mock_motorbridge = MagicMock()
_mock_motorbridge.__spec__ = ModuleSpec("motorbridge", None)
sys.modules.setdefault("motorbridge", _mock_motorbridge)


class _RebotPackageModule:
    rs: object


@dataclass(frozen=True)
class _MotorState:
    can_id: int = 1
    arbitration_id: int = 0xFD
    status_code: int = 0
    pos: float = 0.0
    vel: float = 0.0
    torq: float = 0.0
    t_mos: float = 25.0
    t_rotor: float = 26.0


def _make_mock_motorbridge() -> MagicMock:
    module = MagicMock()
    module.Mode.MIT = MagicMock(name="Mode.MIT")

    controller = MagicMock()
    motors = [MagicMock(name=f"rs_motor_{i}") for i in range(7)]
    controller.add_robstride_motor.side_effect = motors
    controller.mock_motors = motors
    module.Controller.return_value = controller

    for idx, motor in enumerate(motors, 1):
        motor.get_state.return_value = _MotorState(can_id=idx, pos=math.radians(idx * 10.0))

    return module


@pytest.fixture
def mock_motorbridge() -> Generator[MagicMock]:
    module = _make_mock_motorbridge()
    sys.modules.pop("physicalai_rebot_b601_plugin.rs", None)
    sys.modules.pop("physicalai_rebot_b601_plugin", None)
    pkg = sys.modules.get("physicalai_rebot_b601_plugin")
    if pkg is not None and hasattr(pkg, "rs"):
        del cast("_RebotPackageModule", pkg).rs
    with patch.dict(sys.modules, {"motorbridge": module}):
        import_module("physicalai_rebot_b601_plugin.rs")
        yield module


def _create_robot(mock_motorbridge: MagicMock, **kwargs: Any) -> Any:
    from physicalai_rebot_b601_plugin import ReBotB601RS

    _ = mock_motorbridge
    return ReBotB601RS(**kwargs)


class TestReBotB601RSConstruction:
    def test_defaults(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)

        assert robot.port == "can0"
        assert robot.can_adapter == "socketcan"
        assert robot.role == "follower"
        assert robot.joint_names == [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_yaw",
            "wrist_roll",
            "gripper",
        ]

    def test_invalid_adapter_raises(self, mock_motorbridge: MagicMock) -> None:
        from physicalai_rebot_b601_plugin import ReBotB601RS

        with pytest.raises(ValueError, match="Invalid can_adapter"):
            ReBotB601RS(can_adapter="damiao")  # pyrefly: ignore[bad-argument-type]

    def test_invalid_role_raises(self, mock_motorbridge: MagicMock) -> None:
        from physicalai_rebot_b601_plugin import ReBotB601RS

        with pytest.raises(ValueError, match="Invalid role"):
            ReBotB601RS(role="leader")  # pyrefly: ignore[bad-argument-type]

    @pytest.mark.parametrize(
        "name", ["gripper_mit_kp", "gripper_mit_kd", "gripper_mit_torque_limit", "gripper_mit_hold_torque_limit"]
    )
    @pytest.mark.parametrize("value", [-1.0, math.inf, math.nan])
    def test_invalid_gripper_gain_raises(self, mock_motorbridge: MagicMock, name: str, value: float) -> None:
        with pytest.raises(ValueError, match="gripper MIT"):
            _create_robot(mock_motorbridge, **{name: value})

    def test_exports_recipe_and_device_identity(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, port="can1", mit_kp={"shoulder_lift": 40.0})

        assert robot.device_ids == ("rebot-rs:socketcan:can1",)
        assert Config.from_instance(robot)["init_args"] == {
            "port": "can1",
            "mit_kp": {"shoulder_lift": 40.0},
        }

    @pytest.mark.parametrize("gain_name", ["mit_kp", "mit_kd"])
    @pytest.mark.parametrize("gain", [-1.0, math.inf, math.nan])
    def test_invalid_mit_scalar_gain_raises(self, mock_motorbridge: MagicMock, gain_name: str, gain: float) -> None:
        with pytest.raises(ValueError, match=gain_name):
            _create_robot(mock_motorbridge, **{gain_name: gain})

    @pytest.mark.parametrize("gain_name", ["mit_kp", "mit_kd"])
    def test_invalid_mit_gain_dict_raises(self, mock_motorbridge: MagicMock, gain_name: str) -> None:
        with pytest.raises(ValueError, match=gain_name):
            _create_robot(mock_motorbridge, **{gain_name: {"gripper": 1.0}})
        with pytest.raises(ValueError, match=gain_name):
            _create_robot(mock_motorbridge, **{gain_name: {"shoulder_pan": -1.0}})


class TestReBotB601RSLifecycle:
    def test_connect_socketcan_registers_and_configures_motors(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, port="can1")
        robot.connect()

        mock_motorbridge.Controller.assert_called_once_with(channel="can1")
        controller = mock_motorbridge.Controller.return_value
        assert controller.add_robstride_motor.call_args_list == [
            call(0x01, 0xFD, "rs-06"),
            call(0x02, 0xFD, "rs-06"),
            call(0x03, 0xFD, "rs-06"),
            call(0x04, 0xFD, "rs-00"),
            call(0x05, 0xFD, "rs-00"),
            call(0x06, 0xFD, "rs-00"),
            call(0x07, 0xFD, "rs-00"),
        ]
        controller.disable_all.assert_called_once()
        controller.enable_all.assert_called_once()
        assert [motor.ensure_mode.call_args for motor in controller.mock_motors] == [
            call(mock_motorbridge.Mode.MIT),
        ] * 7

    def test_robstride_adapter_raises_not_implemented(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, can_adapter="robstride")

        with pytest.raises(NotImplementedError, match="robstride"):
            robot.connect()

        mock_motorbridge.Controller.assert_not_called()

    def test_connect_is_idempotent(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        robot.connect()

        mock_motorbridge.Controller.assert_called_once()

    def test_connect_failure_cleans_up(self, mock_motorbridge: MagicMock) -> None:
        controller = mock_motorbridge.Controller.return_value
        controller.add_robstride_motor.side_effect = RuntimeError("hardware error")
        robot = _create_robot(mock_motorbridge)

        with pytest.raises(RuntimeError, match="hardware error"):
            robot.connect()

        controller.close.assert_called_once()
        assert robot.is_connected() is False

    def test_disconnect_disables_and_closes(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        controller = mock_motorbridge.Controller.return_value
        motors = list(controller.mock_motors)

        robot.disconnect()

        assert controller.disable_all.call_count == 2
        for motor in motors:
            motor.clear_error.assert_called_once()
            motor.close.assert_called_once()
        controller.close.assert_called_once()
        assert robot.is_connected() is False


class TestReBotB601RSObservation:
    def test_observation_returns_degrees_and_sensor_data(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        obs = robot.get_observation()

        # Motor states report idx * 10 degrees; observations are in the action frame (raw / direction).
        expected = np.array(
            [
                (idx + 1) * 10.0 / REBOT_B601_RS_JOINT_DIRECTIONS[name]
                for idx, name in enumerate(REBOT_B601_RS_JOINT_ORDER)
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(obs.joint_positions, expected, rtol=1e-5)
        assert obs.joint_positions.dtype == np.float32
        assert isinstance(obs.timestamp, float)
        assert obs.sensor_data is not None
        assert set(obs.sensor_data) == {
            "velocities",
            "torques",
            "mos_temperatures",
            "rotor_temperatures",
            "status_codes",
        }

    def test_missing_feedback_raises(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        controller = mock_motorbridge.Controller.return_value
        controller.mock_motors[0].get_state.return_value = None

        with pytest.raises(ConnectionError, match="No feedback"):
            robot.get_observation()


class TestReBotB601RSAction:
    def test_send_action_maps_clips_and_sends_mit(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        controller = mock_motorbridge.Controller.return_value
        motors = list(controller.mock_motors)
        motors[6].get_state.return_value = _MotorState(pos=0.0, vel=0.0)

        action = np.array([200.0, 200.0, -500.0, -45.0, -100.0, 100.0, 100.0], dtype=np.float32)
        robot.send_action(action)

        motors[0].send_mit.assert_called_once_with(math.radians(145.0), 0.0, 50.0, 3.0, 0.0)
        motors[1].send_mit.assert_called_once_with(math.radians(170.0), 0.0, 150.0, 10.0, 0.0)
        motors[2].send_mit.assert_called_once_with(math.radians(200.0), 0.0, 150.0, 10.0, 0.0)
        motors[3].send_mit.assert_called_once_with(math.radians(45.0), 0.0, 50.0, 5.0, 0.0)
        motors[4].send_mit.assert_called_once_with(math.radians(90.0), 0.0, 50.0, 4.0, 0.0)
        motors[5].send_mit.assert_called_once_with(math.radians(90.0), 0.0, 50.0, 4.0, 0.0)
        # The gripper has not moved yet, so it counts as stalled and uses the hold limit.
        gripper_tau = motors[6].send_mit.call_args.args[4]
        assert gripper_tau == pytest.approx(1.0)

    def test_gripper_torque_clamps_negative(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        controller = mock_motorbridge.Controller.return_value
        motors = list(controller.mock_motors)
        motors[6].get_state.return_value = _MotorState(pos=math.radians(270.0), vel=0.0)

        robot.send_action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))

        gripper_tau = motors[6].send_mit.call_args.args[4]
        assert gripper_tau == pytest.approx(-1.0)

    def test_gripper_uses_moving_torque_limit_while_moving(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        gripper = mock_motorbridge.Controller.return_value.mock_motors[6]
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        gripper.get_state.return_value = _MotorState(pos=math.radians(270.0))
        robot.send_action(action)
        gripper.get_state.return_value = _MotorState(pos=math.radians(260.0))
        robot.send_action(action)

        assert gripper.send_mit.call_args.args[4] == pytest.approx(-3.5)

    def test_gripper_open_stall_cuts_torque(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        gripper = mock_motorbridge.Controller.return_value.mock_motors[6]
        gripper.get_state.return_value = _MotorState(pos=0.0)
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 45.0], dtype=np.float32)

        taus = []
        for _ in range(5):
            robot.send_action(action)
            taus.append(gripper.send_mit.call_args.args[4])

        assert taus[:4] == [pytest.approx(1.0)] * 4
        assert taus[4] == 0.0

    def test_gripper_open_stall_warns_once_while_torque_stays_cut(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        gripper = mock_motorbridge.Controller.return_value.mock_motors[6]
        gripper.get_state.return_value = _MotorState(pos=0.0)
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 45.0], dtype=np.float32)

        with patch("physicalai_rebot_b601_plugin.rs.logger") as logger:
            for _ in range(20):
                robot.send_action(action)

        assert gripper.send_mit.call_args.args[4] == 0.0
        logger.warning.assert_called_once()

    def test_send_action_max_relative_target_clamps(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, max_relative_target=5.0)
        robot.connect()
        motors = list(mock_motorbridge.Controller.return_value.mock_motors)
        for motor in motors:
            motor.get_state.return_value = _MotorState(pos=math.radians(10.0))

        robot.send_action(np.array([-145.0, 170.0, 0.0, -12.0, 0.0, 0.0, 0.0], dtype=np.float32))

        # present=10 everywhere: large jumps are limited to a 5-degree step, small ones pass through.
        motors[0].send_mit.assert_called_once_with(math.radians(5.0), 0.0, 50.0, 3.0, 0.0)
        motors[1].send_mit.assert_called_once_with(math.radians(15.0), 0.0, 150.0, 10.0, 0.0)
        motors[3].send_mit.assert_called_once_with(math.radians(12.0), 0.0, 50.0, 5.0, 0.0)

    def test_send_action_max_relative_target_scales_gripper_step(self, mock_motorbridge: MagicMock) -> None:
        # Uncapped torque so the impedance term exposes the limited gripper target.
        robot = _create_robot(
            mock_motorbridge, max_relative_target=10.0, gripper_mit_torque_limit=100.0, gripper_mit_hold_torque_limit=100.0
        )
        robot.connect()
        motors = list(mock_motorbridge.Controller.return_value.mock_motors)
        for motor in motors:
            motor.get_state.return_value = _MotorState(pos=0.0)

        robot.send_action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0], dtype=np.float32))

        # 30 joint deg maps to 180 motor deg; a 10 joint-deg step on the 6x gripper is 60 motor deg.
        gripper_tau = motors[6].send_mit.call_args.args[4]
        assert gripper_tau == pytest.approx(12.0 * math.radians(60.0))

    @pytest.mark.parametrize("max_relative_target", [0.0, -1.0, math.inf, math.nan])
    def test_invalid_max_relative_target_raises(self, mock_motorbridge: MagicMock, max_relative_target: float) -> None:
        with pytest.raises(ValueError, match="max_relative_target must be a finite positive value"):
            _create_robot(mock_motorbridge, max_relative_target=max_relative_target)

    def test_send_action_wrong_shape_raises(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)
        robot.connect()

        with pytest.raises(ValueError, match="Expected action shape"):
            robot.send_action(np.zeros(6, dtype=np.float32))

    def test_send_action_disconnected_raises(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)

        with pytest.raises(ConnectionError, match="not connected"):
            robot.send_action(np.zeros(7, dtype=np.float32))

    def test_set_zero_position_disables_torque_and_zeroes_every_motor(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        controller = mock_motorbridge.Controller.return_value
        controller.reset_mock()

        robot.set_zero_position()

        controller.disable_all.assert_called_once()
        for motor in controller.mock_motors:
            motor.set_zero_position.assert_called_once()

    def test_set_zero_position_disconnected_raises(self, mock_motorbridge: MagicMock) -> None:
        with pytest.raises(ConnectionError, match="not connected"):
            _create_robot(mock_motorbridge).set_zero_position()

    def test_disable_enable_torque(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        controller = mock_motorbridge.Controller.return_value

        robot.disable_torque()
        controller.disable_all.assert_called()

        robot.enable_torque()
        controller.enable_all.assert_called()

    def test_shoulder_pan_accepts_negative_base_rotation(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        motors = list(mock_motorbridge.Controller.return_value.mock_motors)

        robot.send_action(np.array([-30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))

        motors[0].send_mit.assert_called_once_with(math.radians(-30.0), 0.0, 50.0, 3.0, 0.0)

    def test_echoing_observation_holds_current_pose(self, mock_motorbridge: MagicMock) -> None:
        """Sending the observation back as the action must hold every joint in place."""
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        controller = mock_motorbridge.Controller.return_value
        motors = list(controller.mock_motors)
        raw_deg = [20.0, 30.0, 40.0, 10.0, -20.0, 15.0, 60.0]
        for motor, degrees in zip(motors, raw_deg, strict=True):
            motor.get_state.return_value = _MotorState(pos=math.radians(degrees), vel=0.0)

        robot.send_action(robot.get_observation().joint_positions)

        for motor, degrees in zip(motors[:6], raw_deg[:6], strict=True):
            target = motor.send_mit.call_args.args[0]
            assert target == pytest.approx(math.radians(degrees), abs=1e-5)
        gripper_tau = motors[6].send_mit.call_args.args[4]
        assert gripper_tau == pytest.approx(0.0, abs=1e-5)

    def test_send_action_custom_mit_gains(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(
            mock_motorbridge,
            mit_kp=40.0,
            mit_kd={
                "shoulder_pan": 1.0,
                "shoulder_lift": 2.0,
                "elbow_flex": 3.0,
                "wrist_flex": 4.0,
                "wrist_yaw": 5.0,
                "wrist_roll": 6.0,
            },
        )
        robot.connect()
        controller = mock_motorbridge.Controller.return_value
        motors = list(controller.mock_motors)

        robot.send_action(np.zeros(7, dtype=np.float32))

        for index, motor in enumerate(motors[:6]):
            _target, _vel, kp, kd, _tau = motor.send_mit.call_args.args
            assert (kp, kd) == (40.0, float(index + 1))

    def test_partial_mit_gain_dict_uses_defaults_for_other_joints(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, mit_kp={"shoulder_lift": 40.0, "elbow_flex": 45.0})
        robot.connect()
        controller = mock_motorbridge.Controller.return_value
        motors = list(controller.mock_motors)

        robot.send_action(np.zeros(7, dtype=np.float32))

        kps = [motor.send_mit.call_args.args[2] for motor in motors[:6]]
        assert kps == [50.0, 40.0, 45.0, 50.0, 50.0, 50.0]
