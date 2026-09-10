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

from physicalai_rebot_b601_plugin.constants import (
    REBOT_B601_DM_MIT_KD,
    REBOT_B601_DM_MIT_KP,
    REBOT_B601_DM_POS_VEL_DEG_S,
)

if TYPE_CHECKING:
    from collections.abc import Generator

_mock_motorbridge = MagicMock()
_mock_motorbridge.__spec__ = ModuleSpec("motorbridge", None)
sys.modules.setdefault("motorbridge", _mock_motorbridge)


class _RebotPackageModule:
    dm: object


@dataclass(frozen=True)
class _MotorState:
    can_id: int = 1
    arbitration_id: int = 0x11
    status_code: int = 0
    pos: float = 0.0
    vel: float = 0.0
    torq: float = 0.0
    t_mos: float = 25.0
    t_rotor: float = 26.0


def _make_mock_motorbridge() -> MagicMock:
    module = MagicMock()
    module.Mode.MIT = MagicMock(name="Mode.MIT")
    module.Mode.POS_VEL = MagicMock(name="Mode.POS_VEL")
    module.Mode.FORCE_POS = MagicMock(name="Mode.FORCE_POS")

    controller = MagicMock()
    motors = [MagicMock(name=f"motor_{i}") for i in range(7)]
    controller.add_damiao_motor.side_effect = motors
    controller.mock_motors = motors
    module.Controller.from_dm_serial.return_value = controller
    module.Controller.return_value = controller

    for idx, motor in enumerate(motors, 1):
        motor.get_state.return_value = _MotorState(can_id=idx, arbitration_id=0x10 + idx, pos=math.radians(idx * 10.0))

    return module


@pytest.fixture
def mock_motorbridge() -> Generator[MagicMock]:
    module = _make_mock_motorbridge()
    sys.modules.pop("physicalai_rebot_b601_plugin.dm", None)
    sys.modules.pop("physicalai_rebot_b601_plugin", None)
    pkg = sys.modules.get("physicalai_rebot_b601_plugin")
    if pkg is not None and hasattr(pkg, "dm"):
        del cast("_RebotPackageModule", pkg).dm
    with patch.dict(sys.modules, {"motorbridge": module}):
        import_module("physicalai_rebot_b601_plugin.dm")
        yield module


def _create_robot(mock_motorbridge: MagicMock, **kwargs: Any) -> Any:
    from physicalai_rebot_b601_plugin import ReBotB601DM

    _ = mock_motorbridge
    return ReBotB601DM(**kwargs)


class TestReBotB601DMConstruction:
    def test_defaults(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)

        assert robot.port == "/dev/ttyACM0"
        assert robot.can_adapter == "damiao"
        assert robot.role == "follower"
        assert robot.control_mode == "pos_vel"
        assert robot.gripper_control_mode == "force_pos"
        assert robot.joint_names == [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_yaw",
            "wrist_roll",
            "gripper",
        ]

    def test_exports_recipe_and_device_identity(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, port="/dev/ttyACM1", can_adapter="socketcan")

        assert robot.device_ids == ("rebot-dm:socketcan:/dev/ttyACM1",)
        assert Config.from_instance(robot)["init_args"] == {
            "port": "/dev/ttyACM1",
            "can_adapter": "socketcan",
        }

    def test_invalid_adapter_raises(self, mock_motorbridge: MagicMock) -> None:
        from physicalai_rebot_b601_plugin import ReBotB601DM

        with pytest.raises(ValueError, match="Invalid can_adapter"):
            ReBotB601DM(can_adapter="bad")  # pyrefly: ignore[bad-argument-type]

    def test_invalid_role_raises(self, mock_motorbridge: MagicMock) -> None:
        from physicalai_rebot_b601_plugin import ReBotB601DM

        with pytest.raises(ValueError, match="Invalid role"):
            ReBotB601DM(role="leader")  # pyrefly: ignore[bad-argument-type]

    def test_negative_baud_raises(self, mock_motorbridge: MagicMock) -> None:
        with pytest.raises(ValueError, match="dm_serial_baud"):
            _create_robot(mock_motorbridge, dm_serial_baud=-1)

    def test_invalid_force_ratio_raises(self, mock_motorbridge: MagicMock) -> None:
        with pytest.raises(ValueError, match="force_pos_torque_ratio"):
            _create_robot(mock_motorbridge, force_pos_torque_ratio=1.5)


class TestReBotB601DMLifecycle:
    def test_connect_damiao_serial(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, port="/dev/ttyACM1")
        robot.connect()

        mock_motorbridge.Controller.from_dm_serial.assert_called_once_with(serial_port="/dev/ttyACM1", baud=921600)
        controller = mock_motorbridge.Controller.from_dm_serial.return_value
        assert controller.add_damiao_motor.call_args_list == [
            call(0x01, 0x11, "4340P"),
            call(0x02, 0x12, "4340P"),
            call(0x03, 0x13, "4340P"),
            call(0x04, 0x14, "4310"),
            call(0x05, 0x15, "4310"),
            call(0x06, 0x16, "4310"),
            call(0x07, 0x17, "4310"),
        ]
        controller.disable_all.assert_called_once()
        controller.enable_all.assert_called_once()
        motors = list(controller.mock_motors)
        assert motors[0].ensure_mode.call_args == call(mock_motorbridge.Mode.POS_VEL)
        assert motors[6].ensure_mode.call_args == call(mock_motorbridge.Mode.FORCE_POS)

    def test_connect_mit_modes(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(
            mock_motorbridge,
            control_mode="mit",
            gripper_control_mode="mit",
        )
        robot.connect()

        controller = mock_motorbridge.Controller.from_dm_serial.return_value
        motors = list(controller.mock_motors)
        assert motors[0].ensure_mode.call_args == call(mock_motorbridge.Mode.MIT)
        assert motors[6].ensure_mode.call_args == call(mock_motorbridge.Mode.MIT)

    def test_connect_socketcan(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, can_adapter="socketcan", port="can1")
        robot.connect()

        mock_motorbridge.Controller.assert_called_once_with(channel="can1")

    def test_connect_is_idempotent(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        robot.connect()

        mock_motorbridge.Controller.from_dm_serial.assert_called_once()

    def test_connect_failure_cleans_up(self, mock_motorbridge: MagicMock) -> None:
        controller = mock_motorbridge.Controller.from_dm_serial.return_value
        controller.add_damiao_motor.side_effect = RuntimeError("hardware error")
        robot = _create_robot(mock_motorbridge)

        with pytest.raises(RuntimeError, match="hardware error"):
            robot.connect()

        controller.close.assert_called_once()
        assert robot.is_connected() is False

    def test_disconnect_disables_and_closes(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        controller = mock_motorbridge.Controller.from_dm_serial.return_value
        motors = list(controller.mock_motors)

        robot.disconnect()

        assert controller.disable_all.call_count == 2
        for motor in motors:
            motor.clear_error.assert_called_once()
            motor.close.assert_called_once()
        controller.close.assert_called_once()
        assert robot.is_connected() is False


class TestReBotB601DMObservation:
    def test_observation_returns_degrees_and_sensor_data(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        obs = robot.get_observation()

        np.testing.assert_allclose(obs.joint_positions, np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.float32))
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
        controller = mock_motorbridge.Controller.from_dm_serial.return_value
        controller.mock_motors[0].get_state.return_value = None

        with pytest.raises(ConnectionError, match="No feedback"):
            robot.get_observation()


class TestReBotB601DMAction:
    def test_send_action_mit_maps_clips_and_sends(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, control_mode="mit", gripper_control_mode="mit")
        robot.connect()
        controller = mock_motorbridge.Controller.from_dm_serial.return_value
        motors = list(controller.mock_motors)

        action = np.array([200.0, 170.0, -500.0, -45.0, -100.0, 100.0, 100.0], dtype=np.float32)
        robot.send_action(action)

        expected_targets = (-145.0, -170.0, -200.0, -45.0, -90.0, -90.0, -270.0)
        for i, (motor, target) in enumerate(zip(motors, expected_targets, strict=True)):
            name = robot.joint_names[i]
            motor.send_mit.assert_called_once_with(
                math.radians(target),
                0.0,
                REBOT_B601_DM_MIT_KP[name],
                REBOT_B601_DM_MIT_KD[name],
                0.0,
            )

    def test_send_action_pos_vel_uses_goal_time_for_velocity(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, control_mode="pos_vel", gripper_control_mode="force_pos")
        robot.connect()
        controller = mock_motorbridge.Controller.from_dm_serial.return_value
        motors = list(controller.mock_motors)

        # Set each motor 10 degrees from its mapped target.
        for motor in motors:
            motor.get_state.return_value = _MotorState(pos=0.0)
        action = np.array(
            [-10.0, 10.0, -10.0, 10.0, 10.0, -10.0, 10.0 / 6.0],
            dtype=np.float32,
        )
        robot.send_action(action, goal_time=0.2)

        velocity = math.radians(50.0)
        for motor in motors[:6]:
            motor.send_pos_vel.assert_called_once()
            assert motor.send_pos_vel.call_args.args[1] == pytest.approx(velocity)
        assert motors[6].send_force_pos.call_args.args[1] == pytest.approx(velocity)

    def test_send_action_pos_vel_uses_velocity_caps(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(
            mock_motorbridge,
            control_mode="pos_vel",
            gripper_control_mode="force_pos",
        )
        robot.connect()
        controller = mock_motorbridge.Controller.from_dm_serial.return_value
        motors = list(controller.mock_motors)

        for motor in motors:
            motor.get_state.return_value = None
        robot.send_action(np.zeros(7, dtype=np.float32))

        for motor, velocity in zip(motors[:6], REBOT_B601_DM_POS_VEL_DEG_S[:6], strict=True):
            assert motor.send_pos_vel.call_args.args[1] == pytest.approx(math.radians(velocity))
        assert motors[6].send_force_pos.call_args.args[1] == pytest.approx(
            math.radians(REBOT_B601_DM_POS_VEL_DEG_S[6]),
        )

    def test_send_action_mit_custom_gains(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, control_mode="mit", gripper_control_mode="mit", mit_kp=5.0, mit_kd=1.0)
        robot.connect()
        controller = mock_motorbridge.Controller.from_dm_serial.return_value
        motors = list(controller.mock_motors)

        robot.send_action(np.zeros(7, dtype=np.float32))

        for motor in motors:
            motor.send_mit.assert_called_once_with(0.0, 0.0, 5.0, 1.0, 0.0)

    def test_send_action_max_relative_target_clamps(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, control_mode="mit", max_relative_target=5.0)
        robot.connect()
        controller = mock_motorbridge.Controller.from_dm_serial.return_value
        motors = list(controller.mock_motors)

        for motor in motors:
            motor.get_state.return_value = _MotorState(pos=math.radians(10.0))
        action = np.array([200.0, 170.0, -200.0, 45.0, 90.0, 90.0, 0.0], dtype=np.float32)
        robot.send_action(action)

        # shoulder_pan present=10, mapped target=-145, limited to a 5-degree step -> 5
        motors[0].send_mit.assert_called_once_with(
            math.radians(5.0),
            0.0,
            REBOT_B601_DM_MIT_KP["shoulder_pan"],
            REBOT_B601_DM_MIT_KD["shoulder_pan"],
            0.0,
        )

    @pytest.mark.parametrize("control_mode", ["bad", "MIT", "position"])
    def test_invalid_control_mode_raises(self, mock_motorbridge: MagicMock, control_mode: str) -> None:
        with pytest.raises(ValueError, match="Invalid control_mode"):
            _create_robot(mock_motorbridge, control_mode=control_mode)

    @pytest.mark.parametrize("gripper_control_mode", ["bad", "FORCE", "mitm"])
    def test_invalid_gripper_control_mode_raises(
        self,
        mock_motorbridge: MagicMock,
        gripper_control_mode: str,
    ) -> None:
        with pytest.raises(ValueError, match="Invalid gripper_control_mode"):
            _create_robot(mock_motorbridge, gripper_control_mode=gripper_control_mode)

    def test_invalid_mit_kp_dict_raises(self, mock_motorbridge: MagicMock) -> None:
        with pytest.raises(ValueError, match="mit_kp"):
            _create_robot(mock_motorbridge, mit_kp={"shoulder_pan": 1.0})

    @pytest.mark.parametrize("gain", [-1.0, math.inf, math.nan])
    @pytest.mark.parametrize("gain_name", ["mit_kp", "mit_kd"])
    def test_invalid_mit_scalar_gain_raises(
        self,
        mock_motorbridge: MagicMock,
        gain_name: str,
        gain: float,
    ) -> None:
        with pytest.raises(ValueError, match=gain_name):
            _create_robot(mock_motorbridge, **{gain_name: gain})

    @pytest.mark.parametrize("max_relative_target", [0.0, -1.0, math.inf, math.nan])
    def test_invalid_max_relative_target_raises(
        self,
        mock_motorbridge: MagicMock,
        max_relative_target: float,
    ) -> None:
        with pytest.raises(ValueError, match="max_relative_target must be a finite positive value"):
            _create_robot(mock_motorbridge, max_relative_target=max_relative_target)

    @pytest.mark.parametrize("goal_time", [0.0, -0.1, math.inf, math.nan])
    def test_send_action_rejects_non_positive_goal_time(
        self,
        mock_motorbridge: MagicMock,
        goal_time: float,
    ) -> None:
        robot = _create_robot(mock_motorbridge)
        robot.connect()

        with pytest.raises(ValueError, match="goal_time must be a finite positive value"):
            robot.send_action(np.zeros(7, dtype=np.float32), goal_time=goal_time)

    def test_send_action_wrong_shape_raises(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)
        robot.connect()

        with pytest.raises(ValueError, match="Expected action shape"):
            robot.send_action(np.zeros(6, dtype=np.float32))

    def test_send_action_disconnected_raises(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)

        with pytest.raises(ConnectionError, match="not connected"):
            robot.send_action(np.zeros(7, dtype=np.float32))

    def test_disable_enable_torque(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        controller = mock_motorbridge.Controller.from_dm_serial.return_value

        robot.disable_torque()
        controller.disable_all.assert_called()

        robot.enable_torque()
        controller.enable_all.assert_called()
