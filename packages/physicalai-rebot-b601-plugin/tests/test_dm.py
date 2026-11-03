from __future__ import annotations

import itertools
import math
import sys
from dataclasses import dataclass
from importlib import import_module
from importlib.machinery import ModuleSpec
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, call

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
def mock_motorbridge(monkeypatch: pytest.MonkeyPatch) -> Generator[MagicMock]:
    module = _make_mock_motorbridge()
    sys.modules.pop("physicalai_rebot_b601_plugin.dm", None)
    sys.modules.pop("physicalai_rebot_b601_plugin", None)
    pkg = sys.modules.get("physicalai_rebot_b601_plugin")
    if pkg is not None and hasattr(pkg, "dm"):
        del cast("_RebotPackageModule", pkg).dm
    # setitem restores only this key; patch.dict(sys.modules) would also drop
    # every module imported during the test (e.g. loguru, multiprocessing).
    monkeypatch.setitem(sys.modules, "motorbridge", module)
    import_module("physicalai_rebot_b601_plugin.dm")
    yield module
    sys.modules.pop("physicalai_rebot_b601_plugin.dm", None)
    sys.modules.pop("physicalai_rebot_b601_plugin", None)


# Turns off the feedforward and motion shaping so tests can check the raw command mapping.
_RAW_CONTROL: dict[str, Any] = {
    "gravity_compensation": False,
    "velocity_feedforward": False,
    "approach_speed": None,
    "park_on_disconnect": False,
}


def _create_robot(mock_motorbridge: MagicMock, *, raw: bool = True, hold: bool = False, **kwargs: Any) -> Any:
    from physicalai_rebot_b601_plugin import ReBotB601DM

    _ = mock_motorbridge
    robot = ReBotB601DM(**({**_RAW_CONTROL, **kwargs} if raw else kwargs))
    if not hold:
        # Skip the hold command sent on enable, so tests see only the commands they send.
        robot._hold_after_enable = robot._reset_control_state
    return robot


class TestReBotB601DMConstruction:
    def test_defaults(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, raw=False)

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
        robot = _create_robot(mock_motorbridge, raw=False, port="/dev/ttyACM1", can_adapter="socketcan")

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
    def test_echoing_observation_holds_current_pose(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, control_mode="mit", gripper_control_mode="mit")
        robot.connect()
        motors = list(mock_motorbridge.Controller.from_dm_serial.return_value.mock_motors)
        motor_degrees = [-20.0, -30.0, -40.0, 10.0, -20.0, 15.0, -60.0]
        for motor, degrees in zip(motors, motor_degrees, strict=True):
            motor.get_state.return_value = _MotorState(pos=math.radians(degrees))

        robot.send_action(robot.get_observation().joint_positions)

        for motor, degrees in zip(motors, motor_degrees, strict=True):
            assert motor.send_mit.call_args.args[0] == pytest.approx(math.radians(degrees), abs=1e-5)

    def test_send_action_mit_maps_clips_and_sends(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, control_mode="mit", gripper_control_mode="mit")
        robot.connect()
        controller = mock_motorbridge.Controller.from_dm_serial.return_value
        motors = list(controller.mock_motors)

        action = np.array([200.0, 170.0, -500.0, -45.0, -100.0, 100.0, 100.0], dtype=np.float32)
        robot.send_action(action)

        expected_targets = (150.0, 1.0, -200.0, -45.0, -90.0, 90.0, 0.0)
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
        action = np.full(7, -10.0, dtype=np.float32)
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

        # shoulder_pan present=10, target=150, limited to a 5-degree public-frame step -> 15
        motors[0].send_mit.assert_called_once_with(
            math.radians(15.0),
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

    @pytest.mark.parametrize("max_relative_target", [0.0, -2.0, math.inf, math.nan])
    def test_invalid_max_relative_target_raises(
        self,
        mock_motorbridge: MagicMock,
        max_relative_target: float,
    ) -> None:
        with pytest.raises(ValueError, match="max_relative_target must be -1, None, or a finite positive value"):
            _create_robot(mock_motorbridge, max_relative_target=max_relative_target)

    def test_minus_one_disables_max_relative_target(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, max_relative_target=-1.0)

        assert robot.max_relative_target is None

    def test_set_zero_position_disables_torque_and_zeroes_every_motor(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        controller = mock_motorbridge.Controller.from_dm_serial.return_value
        controller.reset_mock()

        robot.set_zero_position()

        controller.disable_all.assert_called_once()
        for motor in controller.mock_motors:
            motor.set_zero_position.assert_called_once()

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


class _FakeClock:
    """Stands in for the driver module's ``time``: ``sleep`` only advances ``monotonic``."""

    def __init__(self) -> None:
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(seconds, 0.0)


@pytest.fixture
def clock(mock_motorbridge: MagicMock, monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    fake = _FakeClock()
    monkeypatch.setattr(sys.modules["physicalai_rebot_b601_plugin.dm"], "time", fake)
    return fake


def _follow_commands(motors: list[MagicMock]) -> None:
    """Make every mock motor report the last position it was commanded, like an ideal servo."""
    for motor in motors:
        state = {"pos": motor.get_state.return_value.pos}

        def remember(pos: float, *_args: float, _state: dict[str, float] = state) -> None:
            _state["pos"] = pos

        motor.send_mit.side_effect = remember
        motor.send_pos_vel.side_effect = remember
        motor.send_force_pos.side_effect = remember
        motor.get_state.side_effect = lambda _state=state: _MotorState(pos=_state["pos"])


def _dm_motors(mock_motorbridge: MagicMock) -> list[MagicMock]:
    return list(mock_motorbridge.Controller.from_dm_serial.return_value.mock_motors)


class TestReBotB601DMControl:
    @pytest.mark.parametrize(("gain_name", "gain"), [("mit_kd", 10.0), ("mit_kp", 600.0)])
    def test_gain_above_encodable_range_raises(self, mock_motorbridge: MagicMock, gain_name: str, gain: float) -> None:
        with pytest.raises(ValueError, match="exceeds the 4340P MIT limit"):
            _create_robot(mock_motorbridge, **{gain_name: gain})

    def test_default_mit_gains_are_encodable(self) -> None:
        from physicalai_rebot_b601_plugin.constants import REBOT_B601_DM_MOTOR_MODELS, REBOT_B601_MIT_GAIN_LIMITS

        for joint, model in REBOT_B601_DM_MOTOR_MODELS.items():
            kp_limit, kd_limit = REBOT_B601_MIT_GAIN_LIMITS[model]
            assert REBOT_B601_DM_MIT_KP[joint] <= kp_limit
            assert REBOT_B601_DM_MIT_KD[joint] <= kd_limit

    def test_mit_sends_gravity_torque_of_the_commanded_pose(self, mock_motorbridge: MagicMock) -> None:
        from physicalai_rebot_b601_plugin import get_urdf_path
        from physicalai_rebot_b601_plugin.constants import REBOT_B601_DM_GRAVITY_URDF, REBOT_B601_URDF_ARM_JOINTS
        from physicalai_rebot_b601_plugin.gravity import GravityModel

        robot = _create_robot(
            mock_motorbridge,
            control_mode="mit",
            gripper_control_mode="mit",
            gravity_compensation=True,
        )
        robot.connect()
        motors = _dm_motors(mock_motorbridge)
        pose = np.array([10.0, -60.0, -90.0, 20.0, 5.0, -30.0, -100.0], dtype=np.float32)

        robot.send_action(pose)

        expected = GravityModel(get_urdf_path() / REBOT_B601_DM_GRAVITY_URDF, REBOT_B601_URDF_ARM_JOINTS).torques(
            np.radians(pose[:6]),
        )
        assert np.abs(expected[1:4]).max() > 1.0  # the pose loads the shoulder, elbow and wrist
        for motor, torque in zip(motors[:6], expected, strict=True):
            assert motor.send_mit.call_args.args[4] == pytest.approx(torque, abs=1e-6)
        assert motors[6].send_mit.call_args.args[4] == 0.0

    def test_velocity_feedforward_follows_the_commanded_targets(
        self,
        mock_motorbridge: MagicMock,
        clock: _FakeClock,
    ) -> None:
        from physicalai_rebot_b601_plugin.dm import _VELOCITY_FILTER_ALPHA

        robot = _create_robot(mock_motorbridge, control_mode="mit", velocity_feedforward=True)
        robot.connect()
        motors = _dm_motors(mock_motorbridge)

        robot.send_action(np.zeros(7, dtype=np.float32))
        assert motors[0].send_mit.call_args.args[1] == 0.0
        clock.sleep(0.02)
        robot.send_action(np.array([0.2, 0, 0, 0, 0, 0, 0], dtype=np.float32))  # 10 deg/s, within the limits

        assert motors[0].send_mit.call_args.args[1] == pytest.approx(_VELOCITY_FILTER_ALPHA * math.radians(10.0))
        assert motors[1].send_mit.call_args.args[1] == 0.0

    def test_first_command_eases_in_from_the_measured_pose(
        self,
        mock_motorbridge: MagicMock,
        clock: _FakeClock,
    ) -> None:
        robot = _create_robot(mock_motorbridge, control_mode="mit", approach_speed=30.0)
        robot.connect()
        motors = _dm_motors(mock_motorbridge)
        for motor in motors:
            motor.get_state.return_value = _MotorState(pos=0.0)
        _follow_commands(motors)
        target = np.array([60.0, -30.0, 0.0, 0.0, 0.0, 0.0, -200.0], dtype=np.float32)

        pan = []
        for _ in range(200):
            robot.send_action(target)
            pan.append(math.degrees(motors[0].send_mit.call_args.args[0]))
            clock.sleep(1 / 30)

        steps = np.diff([0.0, *pan])
        assert 0.0 < pan[0] < 1.0
        assert steps.min() >= 0.0
        assert steps.max() <= 30.0 / 30 + 1e-6
        assert pan[-1] == pytest.approx(60.0)
        lift = math.degrees(motors[1].send_mit.call_args.args[0])
        assert lift == pytest.approx(-30.0)
        # The gripper is not slowed down: it gets its target on the first command.
        assert motors[6].send_force_pos.call_args_list[0].args[0] == pytest.approx(math.radians(-200.0))

    def test_jump_after_tracking_eases_in_again(self, mock_motorbridge: MagicMock, clock: _FakeClock) -> None:
        robot = _create_robot(mock_motorbridge, control_mode="mit", approach_speed=30.0)
        robot.connect()
        motors = _dm_motors(mock_motorbridge)
        for motor in motors:
            motor.get_state.return_value = _MotorState(pos=0.0)
        _follow_commands(motors)
        for _ in range(3):
            robot.send_action(np.zeros(7, dtype=np.float32))
            clock.sleep(1 / 30)
        # A 5-degree step is a fast but real motion: tracked within the acceleration limit (much faster
        # than the 30 deg/s approach).
        for _ in range(8):
            robot.send_action(np.array([5.0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
            clock.sleep(1 / 30)
        assert math.degrees(motors[0].send_mit.call_args.args[0]) == pytest.approx(5.0)

        robot.send_action(np.array([45.0, 0, 0, 0, 0, 0, 0], dtype=np.float32))

        assert 5.0 < math.degrees(motors[0].send_mit.call_args.args[0]) < 6.0

    def _commanded_deg(self, motors: list[MagicMock]) -> np.ndarray:
        return np.array([math.degrees(m.send_mit.call_args.args[0]) for m in motors[:6]])

    def test_fast_targets_are_held_to_the_speed_limit(self, mock_motorbridge: MagicMock, clock: _FakeClock) -> None:
        from physicalai_rebot_b601_plugin.constants import REBOT_B601_MAX_SPEED_DEG_S

        robot = _create_robot(mock_motorbridge, control_mode="mit")
        robot.connect()
        motors = _dm_motors(mock_motorbridge)
        _follow_commands(motors)
        dt = 1 / 60
        commands = []
        signs = np.array([1, -1, -1, 1, 1, 1])  # toward each joint's open range
        for k in range(30):  # every joint target moving at 400 deg/s for half a second
            robot.send_action(np.append(signs * 400.0 * dt * k, 0.0).astype(np.float32))
            commands.append(self._commanded_deg(motors))
            clock.sleep(dt)
        speed = np.abs(np.diff(np.array(commands), axis=0)).max(axis=0) / dt
        limits = np.array([REBOT_B601_MAX_SPEED_DEG_S[m] for m in ("4340P",) * 3 + ("4310",) * 3])
        assert np.all(speed <= limits + 1e-6)
        assert speed[0] == pytest.approx(limits[0])

    def test_motion_limits_can_be_turned_off(self, mock_motorbridge: MagicMock, clock: _FakeClock) -> None:
        robot = _create_robot(mock_motorbridge, control_mode="mit", motion_limits=False)
        robot.connect()
        motors = _dm_motors(mock_motorbridge)
        _follow_commands(motors)
        robot.send_action(np.zeros(7, dtype=np.float32))
        clock.sleep(1 / 60)
        robot.send_action(np.array([8.0, 0, 0, 0, 0, 0, 0], dtype=np.float32))  # 480 deg/s in one tick

        assert self._commanded_deg(motors)[0] == pytest.approx(8.0)

    def test_acceleration_is_limited_from_rest(self, mock_motorbridge: MagicMock, clock: _FakeClock) -> None:
        from physicalai_rebot_b601_plugin.constants import REBOT_B601_MAX_ACCEL_DEG_S2

        robot = _create_robot(mock_motorbridge, control_mode="mit")
        robot.connect()
        motors = _dm_motors(mock_motorbridge)
        _follow_commands(motors)
        dt = 1 / 60
        robot.send_action(np.zeros(7, dtype=np.float32))
        clock.sleep(dt)
        robot.send_action(np.array([8.0, 0, 0, 0, 0, 0, 0], dtype=np.float32))  # 480 deg/s in one tick

        first_step = self._commanded_deg(motors)[0]
        assert first_step == pytest.approx(REBOT_B601_MAX_ACCEL_DEG_S2["4340P"] * dt * dt)

    def test_slow_targets_are_tracked_without_lag(self, mock_motorbridge: MagicMock, clock: _FakeClock) -> None:
        robot = _create_robot(mock_motorbridge, control_mode="mit")
        robot.connect()
        motors = _dm_motors(mock_motorbridge)
        _follow_commands(motors)
        dt = 1 / 60
        for k in range(30):  # 60 deg/s, smoothly started
            target = 60.0 * dt * k * min(1.0, k / 10)
            robot.send_action(np.array([target, -target, 0, 0, 0, 0, 0], dtype=np.float32))
            clock.sleep(dt)

        assert self._commanded_deg(motors)[:2] == pytest.approx([target, -target])

    def test_catching_up_does_not_overshoot_a_still_target(self, mock_motorbridge: MagicMock, clock: _FakeClock) -> None:
        robot = _create_robot(mock_motorbridge, control_mode="mit")
        robot.connect()
        motors = _dm_motors(mock_motorbridge)
        _follow_commands(motors)
        dt = 1 / 60
        robot.send_action(np.zeros(7, dtype=np.float32))
        clock.sleep(dt)
        trace = []
        for _ in range(30):
            robot.send_action(np.array([9.0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
            trace.append(self._commanded_deg(motors)[0])
            clock.sleep(dt)

        assert max(trace) <= 9.0 + 1e-9
        assert trace[-1] == pytest.approx(9.0)

    def test_limp_motors_raise_instead_of_being_commanded_on(self, mock_motorbridge: MagicMock, clock: _FakeClock) -> None:
        robot = _create_robot(mock_motorbridge, control_mode="mit")
        robot.connect()
        motors = _dm_motors(mock_motorbridge)
        _follow_commands(motors)
        robot.send_action(np.array([0, -30.0, -30.0, 0, 0, 0, 0], dtype=np.float32))
        robot.get_observation()
        for motor in motors:  # power dropout: positions fall back to rest, no torque
            motor.get_state.side_effect = None
            motor.get_state.return_value = _MotorState(pos=0.0, torq=0.0)

        with pytest.raises(ConnectionError, match="stopped driving"):
            for _ in range(20):
                robot.get_observation()

    def test_loaded_motors_far_from_the_command_do_not_raise(self, mock_motorbridge: MagicMock, clock: _FakeClock) -> None:
        robot = _create_robot(mock_motorbridge, control_mode="mit")
        robot.connect()
        motors = _dm_motors(mock_motorbridge)
        robot.send_action(np.array([0, -30.0, -30.0, 0, 0, 0, 0], dtype=np.float32))
        for motor in motors:  # pushing hard against an obstacle: far from the command, but driving
            motor.get_state.return_value = _MotorState(pos=0.0, torq=8.0)

        for _ in range(20):
            robot.get_observation()

    def test_large_step_after_a_pause_eases_in(self, mock_motorbridge: MagicMock, clock: _FakeClock) -> None:
        robot = _create_robot(mock_motorbridge, control_mode="mit", approach_speed=30.0)
        robot.connect()
        motors = _dm_motors(mock_motorbridge)
        for motor in motors:
            motor.get_state.return_value = _MotorState(pos=0.0)
        _follow_commands(motors)
        for _ in range(3):
            robot.send_action(np.zeros(7, dtype=np.float32))
            clock.sleep(1 / 30)
        clock.sleep(2.0)  # the command stream stalls

        robot.send_action(np.array([0, -30.0, 0, 0, 0, 0, 0], dtype=np.float32))

        # Accelerating from rest over the (capped) 0.1 s interval moves the joint under 2 degrees.
        assert -2.0 < math.degrees(motors[1].send_mit.call_args.args[0]) < 0.0

    def test_enable_torque_eases_in_again(self, mock_motorbridge: MagicMock, clock: _FakeClock) -> None:
        robot = _create_robot(mock_motorbridge, control_mode="mit", approach_speed=30.0)
        robot.connect()
        motors = _dm_motors(mock_motorbridge)
        for motor in motors:
            motor.get_state.return_value = _MotorState(pos=0.0)
        _follow_commands(motors)
        for _ in range(60):
            robot.send_action(np.array([20.0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
            clock.sleep(1 / 30)
        assert math.degrees(motors[0].send_mit.call_args.args[0]) == pytest.approx(20.0)

        robot.disable_torque()
        robot.enable_torque()
        robot.send_action(np.array([-20.0, 0, 0, 0, 0, 0, 0], dtype=np.float32))

        assert 19.0 < math.degrees(motors[0].send_mit.call_args.args[0]) < 20.0

    @pytest.mark.parametrize("control_mode", ["mit", "pos_vel"])
    def test_enabling_torque_holds_the_measured_pose(self, mock_motorbridge: MagicMock, control_mode: str) -> None:
        robot = _create_robot(mock_motorbridge, hold=True, control_mode=control_mode, gravity_compensation=True)
        motors = _dm_motors(mock_motorbridge)

        robot.connect()

        send = "send_mit" if control_mode == "mit" else "send_pos_vel"
        for index, motor in enumerate(motors[:6], 1):
            # The mock motors report index * 10 degrees; without this hold the arm would hang limp.
            assert getattr(motor, send).call_args.args[0] == pytest.approx(math.radians(index * 10.0))
        if control_mode == "mit":
            assert motors[2].send_mit.call_args.args[1] == 0.0  # no velocity feedforward
            assert motors[2].send_mit.call_args.args[4] != 0.0  # gravity feedforward

        robot.disable_torque()
        for motor in motors:
            motor.reset_mock()
        robot.enable_torque()
        assert getattr(motors[0], send).call_args.args[0] == pytest.approx(math.radians(10.0))

    def test_first_command_after_the_hold_eases_in(self, mock_motorbridge: MagicMock, clock: _FakeClock) -> None:
        robot = _create_robot(mock_motorbridge, hold=True, control_mode="mit", approach_speed=30.0)
        motors = _dm_motors(mock_motorbridge)
        for motor in motors:
            motor.get_state.return_value = _MotorState(pos=0.0)
        _follow_commands(motors)
        robot.connect()

        robot.send_action(np.array([60.0, 0, 0, 0, 0, 0, 0], dtype=np.float32))

        assert 0.0 < math.degrees(motors[0].send_mit.call_args.args[0]) < 1.0

    def test_enable_without_feedback_does_not_fail(self, mock_motorbridge: MagicMock, clock: _FakeClock) -> None:
        robot = _create_robot(mock_motorbridge, hold=True, control_mode="mit")
        motors = _dm_motors(mock_motorbridge)
        motors[3].get_state.return_value = None

        robot.connect()

        assert robot.is_connected()
        motors[0].send_mit.assert_not_called()

    @pytest.mark.parametrize("control_mode", ["mit", "pos_vel"])
    def test_disconnect_parks_in_two_stages(
        self,
        mock_motorbridge: MagicMock,
        clock: _FakeClock,
        control_mode: str,
    ) -> None:
        robot = _create_robot(mock_motorbridge, control_mode=control_mode, park_on_disconnect=True)
        robot.connect()
        motors = _dm_motors(mock_motorbridge)
        start = [30.0, -60.0, -90.0, 20.0, 10.0, -40.0, -100.0]
        for motor, degrees in zip(motors, start, strict=True):
            motor.get_state.return_value = _MotorState(pos=math.radians(degrees))
        _follow_commands(motors)
        send = "send_mit" if control_mode == "mit" else "send_pos_vel"
        t0 = clock.now

        robot.disconnect()

        calls = [getattr(motor, send).call_args_list for motor in motors[:6]]
        trace = np.degrees([[joint[k].args[0] for joint in calls] for k in range(len(calls[0]))])
        stage_one = trace[:, [0, 3, 4, 5]]
        stage_two = trace[:, [1, 2]]
        first_stage_end = int(np.argmax(np.all(np.abs(stage_one) < 1e-6, axis=1)))
        assert np.allclose(stage_two[: first_stage_end + 1], [-60.0, -90.0])  # shoulder and elbow wait
        assert np.allclose(trace[-1], 0.0, atol=1e-6)
        assert np.abs(np.diff(trace, axis=0)).max() <= 30.0 / 50 + 1e-6  # peaks at 30 deg/s
        assert clock.now - t0 >= 2 * 1.0  # each stage takes at least a second
        mock_motorbridge.Controller.from_dm_serial.return_value.disable_all.assert_called()

    def test_park_ends_with_a_damped_soft_release(self, mock_motorbridge: MagicMock, clock: _FakeClock) -> None:
        robot = _create_robot(mock_motorbridge, control_mode="mit", gravity_compensation=True, park_on_disconnect=True)
        robot.connect()
        motors = _dm_motors(mock_motorbridge)
        _follow_commands(motors)

        robot.disconnect()

        elbow = [c.args for c in motors[2].send_mit.call_args_list]
        kps = [args[2] for args in elbow]
        assert kps[-1] == 0.0  # stiffness faded out
        assert elbow[-1][4] == 0.0  # gravity feedforward faded out
        assert elbow[-1][3] == REBOT_B601_DM_MIT_KD["elbow_flex"]  # damping kept until torque is released
        fade = kps[kps.index(REBOT_B601_DM_MIT_KP["elbow_flex"], len(kps) - 60) :]
        assert all(a >= b for a, b in itertools.pairwise(fade))

    def test_soft_release_waits_for_the_arm_to_settle(self, mock_motorbridge: MagicMock, clock: _FakeClock) -> None:
        from physicalai_rebot_b601_plugin import dm

        robot = _create_robot(mock_motorbridge, control_mode="mit")
        robot.connect()
        motors = _dm_motors(mock_motorbridge)
        for motor in motors:
            motor.get_state.return_value = _MotorState(pos=0.0)
        robot._last_command_deg = np.zeros(7)
        motors[3].get_state.return_value = _MotorState(pos=0.0, vel=0.5)  # the wrist keeps sinking
        start = clock.now

        robot._release_softly()

        assert clock.now - start == pytest.approx(dm._RELEASE_FADE_S + dm._RELEASE_SETTLE_MAX_S, abs=0.05)
        motors[3].get_state.return_value = _MotorState(pos=0.0)
        start = clock.now
        robot._release_softly()
        assert clock.now - start < dm._RELEASE_FADE_S + 2 * dm._RELEASE_STILL_S

    def test_disconnect_skips_park_while_torque_is_off(self, mock_motorbridge: MagicMock, clock: _FakeClock) -> None:
        robot = _create_robot(mock_motorbridge, control_mode="mit", park_on_disconnect=True)
        robot.connect()
        motors = _dm_motors(mock_motorbridge)
        robot.disable_torque()

        robot.disconnect()

        for motor in motors:
            motor.send_mit.assert_not_called()

    def test_park_stops_when_a_joint_is_blocked(self, mock_motorbridge: MagicMock, clock: _FakeClock) -> None:
        robot = _create_robot(mock_motorbridge, control_mode="mit", park_on_disconnect=True)
        robot.connect()
        motors = _dm_motors(mock_motorbridge)
        for motor in motors:
            motor.get_state.return_value = _MotorState(pos=math.radians(-90.0))  # never moves

        robot.disconnect()

        last = [math.degrees(m.send_mit.call_args.args[0]) for m in motors[:6]]
        assert last[0] > -90.0 + 1.0
        assert last[0] < 0.0  # stopped before reaching the rest pose
        mock_motorbridge.Controller.from_dm_serial.return_value.disable_all.assert_called()

    @pytest.mark.parametrize("speed", [0.0, -5.0, math.inf, math.nan])
    def test_invalid_speeds_raise(self, mock_motorbridge: MagicMock, speed: float) -> None:
        with pytest.raises(ValueError, match="approach_speed"):
            _create_robot(mock_motorbridge, approach_speed=speed)
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        with pytest.raises(ValueError, match="speed must be a finite positive value"):
            robot.park(speed=speed)
