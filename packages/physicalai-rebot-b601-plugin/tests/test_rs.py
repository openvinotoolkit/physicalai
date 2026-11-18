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
    REBOT_B601_RS_GRIPPER_MAX_TORQUE_NM,
    REBOT_B601_RS_JOINT_DIRECTIONS,
    REBOT_B601_RS_JOINT_ORDER,
)

if TYPE_CHECKING:
    from collections.abc import Generator

_mock_motorbridge = MagicMock()
_mock_motorbridge.__spec__ = ModuleSpec("motorbridge", None)
sys.modules.setdefault("motorbridge", _mock_motorbridge)


class _RebotPackageModule:
    dm: object
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
def mock_motorbridge(monkeypatch: pytest.MonkeyPatch) -> Generator[MagicMock]:
    module = _make_mock_motorbridge()
    pkg = sys.modules.get("physicalai_rebot_b601_plugin")
    sys.modules.pop("physicalai_rebot_b601_plugin.dm", None)
    sys.modules.pop("physicalai_rebot_b601_plugin.rs", None)
    sys.modules.pop("physicalai_rebot_b601_plugin", None)
    if pkg is not None and hasattr(pkg, "dm"):
        del cast("_RebotPackageModule", pkg).dm
    if pkg is not None and hasattr(pkg, "rs"):
        del cast("_RebotPackageModule", pkg).rs
    # setitem restores only this key; patch.dict(sys.modules) would also drop
    # every module imported during the test (e.g. loguru, multiprocessing).
    monkeypatch.setitem(sys.modules, "motorbridge", module)
    import_module("physicalai_rebot_b601_plugin.rs")
    yield module
    sys.modules.pop("physicalai_rebot_b601_plugin.dm", None)
    sys.modules.pop("physicalai_rebot_b601_plugin.rs", None)
    sys.modules.pop("physicalai_rebot_b601_plugin", None)


# Turns off the feedforward and motion shaping so tests can check the raw command mapping.
_RAW_CONTROL: dict[str, Any] = {
    "gravity_compensation": False,
    "velocity_feedforward": False,
    "approach_speed": None,
    "park_on_disconnect": False,
}


def _fake_gripper_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Patch the RS module's clock; returns the mutable current time in seconds."""
    from physicalai_rebot_b601_plugin import rs

    now = [10.0]
    fake_time = MagicMock()
    fake_time.monotonic.side_effect = lambda: now[0]
    monkeypatch.setattr(rs, "time", fake_time)
    return now


def _create_robot(mock_motorbridge: MagicMock, *, raw: bool = True, hold: bool = False, **kwargs: Any) -> Any:
    from physicalai_rebot_b601_plugin import ReBotB601RS

    _ = mock_motorbridge
    robot = ReBotB601RS(**({**_RAW_CONTROL, **kwargs} if raw else kwargs))
    if not hold:
        # Skip the hold command sent on enable, so tests see only the commands they send.
        robot._hold_after_enable = robot._reset_control_state
    return robot


class TestReBotB601RSConstruction:
    def test_defaults(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, raw=False)


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

    def test_rs_inherits_dm_implementation(self, mock_motorbridge: MagicMock) -> None:
        from physicalai.robot.interface import Robot
        from physicalai_rebot_b601_plugin import ReBotB601DM, ReBotB601RS

        robot = ReBotB601RS()

        assert issubclass(ReBotB601RS, ReBotB601DM)
        assert isinstance(robot, Robot)

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

    @pytest.mark.parametrize("name", ["gripper_mit_torque_limit", "gripper_mit_hold_torque_limit"])
    def test_gripper_torque_limit_above_motor_peak_raises(self, mock_motorbridge: MagicMock, name: str) -> None:
        with pytest.raises(ValueError, match="must not exceed"):
            _create_robot(mock_motorbridge, **{name: REBOT_B601_RS_GRIPPER_MAX_TORQUE_NM + 0.1})

    def test_exports_recipe_and_device_identity(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, raw=False, port="can1", mit_kp={"shoulder_lift": 40.0})

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
        robot = _create_robot(mock_motorbridge, gripper_max_speed=-1)
        robot.connect()
        controller = mock_motorbridge.Controller.return_value
        motors = list(controller.mock_motors)
        motors[6].get_state.return_value = _MotorState(pos=0.0, vel=0.0)

        action = np.array([200.0, 200.0, -500.0, -45.0, -100.0, 100.0, -270.0], dtype=np.float32)
        robot.send_action(action)

        motors[0].send_mit.assert_called_once_with(math.radians(-145.0), 0.0, 50.0, 3.0, 0.0)
        motors[1].send_mit.assert_called_once_with(0.0, 0.0, 150.0, 10.0, 0.0)
        motors[2].send_mit.assert_called_once_with(math.radians(200.0), 0.0, 150.0, 10.0, 0.0)
        motors[3].send_mit.assert_called_once_with(math.radians(45.0), 0.0, 50.0, 5.0, 0.0)
        motors[4].send_mit.assert_called_once_with(math.radians(90.0), 0.0, 50.0, 4.0, 0.0)
        motors[5].send_mit.assert_called_once_with(math.radians(-90.0), 0.0, 50.0, 4.0, 0.0)
        # The gripper servos on the motor itself; starting from rest it gets the moving limit.
        # First command after connecting: the cap is unknown, so the gripper is let go before it is written.
        assert motors[6].send_mit.call_args_list == [
            call(0.0, 0.0, 0.0, 0.8, 0.0),
            call(math.radians(270.0), 0.0, 30.0, 0.8, 0.0),
        ]
        motors[6].robstride_write_param_f32.assert_called_once_with(0x700B, 1.5)

    def test_gripper_holds_with_the_hold_limit_once_stalled_for_a_while(
        self, mock_motorbridge: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = _fake_gripper_clock(monkeypatch)
        robot = _create_robot(mock_motorbridge, gripper_max_speed=-1)
        robot.connect()
        gripper = mock_motorbridge.Controller.return_value.mock_motors[6]
        close = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        gripper.get_state.return_value = _MotorState(pos=math.radians(90.0))  # on an object

        limits = []
        for _ in range(12):  # 100 Hz
            robot.send_action(close)
            limits.append(gripper.robstride_write_param_f32.call_args.args[1])
            now[0] += 0.01

        # The moving limit until it has not moved for 0.1 s, then the hold limit.
        assert limits[:10] == [1.5] * 10
        assert limits[-1] == 0.5
        gripper.send_mit.assert_called_with(0.0, 0.0, 30.0, 0.8, 0.0)

    def test_gripper_target_moves_no_faster_than_the_max_speed(
        self, mock_motorbridge: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from physicalai_rebot_b601_plugin import rs

        now = [10.0]
        fake_time = MagicMock()
        fake_time.monotonic.side_effect = lambda: now[0]
        monkeypatch.setattr(rs, "time", fake_time)
        robot = _create_robot(mock_motorbridge, gripper_max_speed=200.0)
        robot.connect()
        gripper = mock_motorbridge.Controller.return_value.mock_motors[6]
        gripper.get_state.return_value = _MotorState(pos=math.radians(5.0))
        close = np.array([0, 0, 0, 0, 0, 0, -90.0], dtype=np.float32)  # public -90 is motor +90

        targets = []
        for _ in range(3):
            robot.send_action(close)
            targets.append(math.degrees(gripper.send_mit.call_args.args[0]))
            now[0] += 0.05

        # From the measured 5 deg: 4 deg in the nominal first 20 ms, then 10 deg per 50 ms.
        assert targets == pytest.approx([9.0, 19.0, 29.0])

    @pytest.mark.parametrize("speed", [0.0, -2.0, math.nan])
    def test_invalid_gripper_max_speed_raises(self, mock_motorbridge: MagicMock, speed: float) -> None:
        with pytest.raises(ValueError, match="gripper_max_speed"):
            _create_robot(mock_motorbridge, gripper_max_speed=speed)

    def test_gripper_torque_limit_is_written_only_when_it_changes_or_is_due(
        self, mock_motorbridge: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = _fake_gripper_clock(monkeypatch)
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        gripper = mock_motorbridge.Controller.return_value.mock_motors[6]
        gripper.get_state.return_value = _MotorState(pos=0.0)
        gripper.robstride_write_param_f32.reset_mock()

        for _ in range(10):  # held still for 0.2 s: moving limit, then the hold limit once
            now[0] += 0.02
            robot.send_action(np.zeros(7, dtype=np.float32))
        assert [c.args[1] for c in gripper.robstride_write_param_f32.call_args_list] == [1.5, 0.5]
        now[0] += 0.5
        robot.send_action(np.zeros(7, dtype=np.float32))

        assert gripper.robstride_write_param_f32.call_count == 3  # refreshed

    def test_disconnect_lets_the_gripper_go_before_restoring_its_full_torque_range(
        self, mock_motorbridge: MagicMock
    ) -> None:
        robot = _create_robot(mock_motorbridge, park_on_disconnect=False)
        robot.connect()
        controller = mock_motorbridge.Controller.return_value
        gripper = controller.mock_motors[6]
        manager = MagicMock()
        manager.attach_mock(gripper.send_mit, "send_mit")
        manager.attach_mock(gripper.robstride_write_param_f32, "write")
        manager.attach_mock(controller.disable_all, "disable_all")

        robot.disconnect()

        assert manager.mock_calls[:3] == [
            call.send_mit(0.0, 0.0, 0.0, 0.8, 0.0),
            call.write(0x700B, REBOT_B601_RS_GRIPPER_MAX_TORQUE_NM),
            call.disable_all(),
        ]

    def test_gripper_open_stall_cuts_torque_after_half_a_second(
        self, mock_motorbridge: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = _fake_gripper_clock(monkeypatch)
        robot = _create_robot(mock_motorbridge, gripper_max_speed=-1)
        robot.connect()
        gripper = mock_motorbridge.Controller.return_value.mock_motors[6]
        gripper.get_state.return_value = _MotorState(pos=0.0)  # told to open, never moves
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -270.0], dtype=np.float32)

        sent = []
        for _ in range(60):  # 100 Hz
            robot.send_action(action)
            sent.append(gripper.send_mit.call_args.args)
            now[0] += 0.01

        assert sent[:50] == [(math.radians(270.0), 0.0, 30.0, 0.8, 0.0)] * 50
        assert sent[-1] == (0.0, 0.0, 0.0, 0.8, 0.0)  # let go, damping only

    def test_gripper_open_stall_warns_once_while_torque_stays_cut(
        self, mock_motorbridge: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = _fake_gripper_clock(monkeypatch)
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        gripper = mock_motorbridge.Controller.return_value.mock_motors[6]
        gripper.get_state.return_value = _MotorState(pos=0.0)
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -270.0], dtype=np.float32)

        with patch("physicalai_rebot_b601_plugin.rs.logger") as logger:
            for _ in range(200):
                robot.send_action(action)
                now[0] += 0.01

        assert gripper.send_mit.call_args.args[2] == 0.0
        logger.warning.assert_called_once()

    def test_gripper_starting_from_rest_at_100_hz_is_not_an_open_stall(
        self, mock_motorbridge: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Measured on the arm: the speed passes the stall threshold 50-60 ms after a start."""
        now = _fake_gripper_clock(monkeypatch)
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        gripper = mock_motorbridge.Controller.return_value.mock_motors[6]
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -90.0], dtype=np.float32)  # open to 90 deg

        position = 0.0
        with patch("physicalai_rebot_b601_plugin.rs.logger") as logger:
            for tick in range(40):
                if tick >= 6:  # starts moving after 60 ms, then follows at 400 deg/s
                    position = min(position + math.radians(4.0), math.radians(90.0))
                gripper.get_state.return_value = _MotorState(pos=position)
                robot.send_action(action)
                now[0] += 0.01

        logger.warning.assert_not_called()
        assert gripper.send_mit.call_args.args[2] == 30.0

    def test_gripper_opening_after_resting_closed_is_not_an_open_stall(
        self, mock_motorbridge: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Seen on the arm: closed for a while (starting an episode), then told to open, it was cut at once."""
        now = _fake_gripper_clock(monkeypatch)
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        gripper = mock_motorbridge.Controller.return_value.mock_motors[6]
        closed = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        gripper.get_state.return_value = _MotorState(pos=0.0)
        for _ in range(100):  # one second at rest, closed
            robot.send_action(closed)
            now[0] += 0.01

        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -90.0], dtype=np.float32)
        position = 0.0
        limits = []
        with patch("physicalai_rebot_b601_plugin.rs.logger") as logger:
            for tick in range(40):
                if tick >= 6:  # starts moving after 60 ms
                    position = min(position + math.radians(4.0), math.radians(90.0))
                gripper.get_state.return_value = _MotorState(pos=position)
                robot.send_action(action)
                limits.append(gripper.robstride_write_param_f32.call_args.args[1])
                now[0] += 0.01

        logger.warning.assert_not_called()
        assert gripper.send_mit.call_args.args[2] == 30.0
        assert limits[0] == 1.5  # opening gets the moving limit at once, not the hold limit

    def test_gripper_open_stall_after_resting_still_waits_half_a_second(
        self, mock_motorbridge: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = _fake_gripper_clock(monkeypatch)
        robot = _create_robot(mock_motorbridge, gripper_max_speed=-1)
        robot.connect()
        gripper = mock_motorbridge.Controller.return_value.mock_motors[6]
        gripper.get_state.return_value = _MotorState(pos=0.0)
        closed = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(100):
            robot.send_action(closed)
            now[0] += 0.01

        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -270.0], dtype=np.float32)
        sent = []
        for _ in range(60):  # told to open, never moves (at its open stop)
            robot.send_action(action)
            sent.append(gripper.send_mit.call_args.args)
            now[0] += 0.01

        assert sent[:50] == [(math.radians(270.0), 0.0, 30.0, 0.8, 0.0)] * 50
        assert sent[-1] == (0.0, 0.0, 0.0, 0.8, 0.0)

    def test_gripper_gets_torque_back_once_told_to_close(
        self, mock_motorbridge: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = _fake_gripper_clock(monkeypatch)
        robot = _create_robot(mock_motorbridge, gripper_max_speed=-1)
        robot.connect()
        gripper = mock_motorbridge.Controller.return_value.mock_motors[6]
        gripper.get_state.return_value = _MotorState(pos=math.radians(260.0))  # stuck short of fully open
        for _ in range(60):
            robot.send_action(np.array([0.0] * 6 + [-270.0], dtype=np.float32))
            now[0] += 0.01
        assert gripper.send_mit.call_args.args[2] == 0.0

        robot.send_action(np.array([0.0] * 6 + [-100.0], dtype=np.float32))
        assert gripper.send_mit.call_args.args[2] == 30.0

    def test_gripper_moving_restarts_the_open_stall_clock(
        self, mock_motorbridge: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = _fake_gripper_clock(monkeypatch)
        robot = _create_robot(mock_motorbridge, gripper_max_speed=-1)
        robot.connect()
        gripper = mock_motorbridge.Controller.return_value.mock_motors[6]
        action = np.array([0.0] * 6 + [-270.0], dtype=np.float32)

        position = 0.0
        sent = []
        for tick in range(130):  # still for 60 ms, opens for 0.6 s, then stuck short of fully open
            if 6 <= tick < 66:
                position += math.radians(2.0)
            gripper.get_state.return_value = _MotorState(pos=position)
            robot.send_action(action)
            sent.append(gripper.send_mit.call_args.args[2])
            now[0] += 0.01

        assert sent[:116] == [30.0] * 116  # moving for longer than 0.5 s, then a fresh 0.5 s once stuck
        assert sent[-1] == 0.0

    def test_gripper_open_stall_cuts_though_the_opening_error_jitters(
        self, mock_motorbridge: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = _fake_gripper_clock(monkeypatch)
        robot = _create_robot(mock_motorbridge, gripper_max_speed=-1)
        robot.connect()
        gripper = mock_motorbridge.Controller.return_value.mock_motors[6]
        stuck = math.radians(200.0)
        gripper.get_state.return_value = _MotorState(pos=stuck)

        for tick in range(60):  # at its open stop, the leader just past and just under the threshold
            error = 0.049 if tick % 20 == 19 else 0.051
            robot.send_action(np.array([0.0] * 6 + [-math.degrees(stuck + error)], dtype=np.float32))
            now[0] += 0.01

        assert gripper.send_mit.call_args.args[2] == 0.0

    def test_gripper_gets_a_fresh_open_allowance_after_feedback_returns(
        self, mock_motorbridge: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = _fake_gripper_clock(monkeypatch)
        robot = _create_robot(mock_motorbridge, gripper_max_speed=-1)
        robot.connect()
        gripper = mock_motorbridge.Controller.return_value.mock_motors[6]
        action = np.array([0.0] * 6 + [-270.0], dtype=np.float32)
        gripper.get_state.return_value = _MotorState(pos=0.0)
        robot.send_action(action)  # the open-stall clock starts

        gripper.get_state.return_value = None
        for _ in range(60):  # no feedback for 0.6 s: damping only
            now[0] += 0.01
            robot.send_action(action)
        gripper.get_state.return_value = _MotorState(pos=0.0)
        now[0] += 0.01
        robot.send_action(action)

        assert gripper.send_mit.call_args.args[2] == 30.0

    def test_gripper_on_an_object_told_to_open_gets_the_new_reference_before_the_higher_limit(
        self, mock_motorbridge: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = _fake_gripper_clock(monkeypatch)
        robot = _create_robot(mock_motorbridge, gripper_max_speed=-1)
        robot.connect()
        gripper = mock_motorbridge.Controller.return_value.mock_motors[6]
        gripper.get_state.return_value = _MotorState(pos=math.radians(90.0))  # on an object
        for _ in range(20):
            robot.send_action(np.zeros(7, dtype=np.float32))
            now[0] += 0.01
        assert gripper.robstride_write_param_f32.call_args == call(0x700B, 0.5)

        gripper.reset_mock()
        robot.send_action(np.array([0.0] * 6 + [-180.0], dtype=np.float32))

        assert [c[0] for c in gripper.method_calls if c[0] in {"send_mit", "robstride_write_param_f32"}] == [
            "send_mit",
            "robstride_write_param_f32",
        ]
        assert gripper.send_mit.call_args.args[0] == math.radians(180.0)
        assert gripper.robstride_write_param_f32.call_args == call(0x700B, 1.5)

    def test_gripper_with_an_unknown_torque_limit_gets_the_limit_before_the_reference(
        self, mock_motorbridge: MagicMock
    ) -> None:
        robot = _create_robot(mock_motorbridge, gripper_max_speed=-1)
        robot.connect()
        gripper = mock_motorbridge.Controller.return_value.mock_motors[6]
        robot._reset_control_state()  # as after a reconnect: the firmware limit is unknown
        gripper.get_state.return_value = _MotorState(pos=math.radians(90.0))
        gripper.reset_mock()

        robot.send_action(np.zeros(7, dtype=np.float32))  # told to close

        sent = [c for c in gripper.method_calls if c[0] in {"send_mit", "robstride_write_param_f32"}]
        # An old closing reference may still run: let go first, then the cap, then the new reference.
        assert [c[0] for c in sent] == ["send_mit", "robstride_write_param_f32", "send_mit"]
        assert sent[0].args == (math.radians(90.0), 0.0, 0.0, 0.8, 0.0)

    def test_gripper_holding_an_object_keeps_the_hold_limit_across_a_feedback_gap(
        self, mock_motorbridge: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = _fake_gripper_clock(monkeypatch)
        robot = _create_robot(mock_motorbridge)  # default slewing
        robot.connect()
        gripper = mock_motorbridge.Controller.return_value.mock_motors[6]
        on_object = _MotorState(pos=math.radians(90.0))
        for _ in range(60):  # closed onto an object, the slewed target reaches closed
            gripper.get_state.return_value = on_object
            robot.send_action(np.zeros(7, dtype=np.float32))
            now[0] += 0.01
        assert gripper.robstride_write_param_f32.call_args == call(0x700B, 0.5)

        limits = []
        for _ in range(5):  # feedback drops out now and then while it is told to open
            gripper.get_state.return_value = None
            robot.send_action(np.array([0.0] * 6 + [-180.0], dtype=np.float32))
            now[0] += 0.01
            gripper.get_state.return_value = on_object
            robot.send_action(np.array([0.0] * 6 + [-180.0], dtype=np.float32))
            if gripper.send_mit.call_args.args[0] < on_object.pos:  # the slewed target still closes
                limits.append(gripper.robstride_write_param_f32.call_args.args[1])
            now[0] += 0.01

        assert limits
        assert set(limits) == {0.5}

    def test_gripper_lowers_a_higher_hold_limit_before_the_new_reference(
        self, mock_motorbridge: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = _fake_gripper_clock(monkeypatch)
        robot = _create_robot(
            mock_motorbridge, gripper_max_speed=-1, gripper_mit_torque_limit=0.5, gripper_mit_hold_torque_limit=1.5
        )
        robot.connect()
        gripper = mock_motorbridge.Controller.return_value.mock_motors[6]
        gripper.get_state.return_value = _MotorState(pos=1.0)
        for _ in range(20):  # still at its target: the (higher) hold limit
            robot.send_action(np.array([0.0] * 6 + [-math.degrees(1.0)], dtype=np.float32))
            now[0] += 0.01
        assert gripper.robstride_write_param_f32.call_args == call(0x700B, 1.5)

        gripper.reset_mock()
        gripper.get_state.return_value = _MotorState(pos=1.005)  # moves, and is told to close
        robot.send_action(np.zeros(7, dtype=np.float32))

        assert [c[0] for c in gripper.method_calls if c[0] in {"send_mit", "robstride_write_param_f32"}] == [
            "robstride_write_param_f32",
            "send_mit",
        ]
        assert gripper.robstride_write_param_f32.call_args == call(0x700B, 0.5)

    def test_send_action_max_relative_target_clamps(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, max_relative_target=5.0)
        robot.connect()
        motors = list(mock_motorbridge.Controller.return_value.mock_motors)
        for motor in motors:
            motor.get_state.return_value = _MotorState(pos=math.radians(10.0))

        robot.send_action(np.array([-145.0, 170.0, 0.0, -12.0, 0.0, 0.0, 0.0], dtype=np.float32))

        # Raw motor position 10 is public -10; limiting happens before the -1 motor conversion.
        motors[0].send_mit.assert_called_once_with(math.radians(15.0), 0.0, 50.0, 3.0, 0.0)
        motors[1].send_mit.assert_called_once_with(math.radians(5.0), 0.0, 150.0, 10.0, 0.0)
        motors[3].send_mit.assert_called_once_with(math.radians(12.0), 0.0, 50.0, 5.0, 0.0)

    def test_send_action_max_relative_target_scales_gripper_step(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, max_relative_target=10.0, gripper_max_speed=-1)
        robot.connect()
        motors = list(mock_motorbridge.Controller.return_value.mock_motors)
        for motor in motors:
            motor.get_state.return_value = _MotorState(pos=0.0)

        robot.send_action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -30.0], dtype=np.float32))

        # A 10-degree public-frame step maps to 10 motor degrees; there is no magnitude scaling.
        assert motors[6].send_mit.call_args.args[0] == pytest.approx(math.radians(10.0))

    @pytest.mark.parametrize("max_relative_target", [0.0, -2.0, math.inf, math.nan])
    def test_invalid_max_relative_target_raises(self, mock_motorbridge: MagicMock, max_relative_target: float) -> None:
        with pytest.raises(ValueError, match="max_relative_target must be -1, None, or a finite positive value"):
            _create_robot(mock_motorbridge, max_relative_target=max_relative_target)

    def test_minus_one_disables_max_relative_target(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, max_relative_target=-1.0)

        assert robot.max_relative_target is None

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

        motors[0].send_mit.assert_called_once_with(math.radians(30.0), 0.0, 50.0, 3.0, 0.0)

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
                "wrist_roll": 0.5,
            },
        )
        robot.connect()
        controller = mock_motorbridge.Controller.return_value
        motors = list(controller.mock_motors)

        robot.send_action(np.zeros(7, dtype=np.float32))

        for motor, expected_kd in zip(motors[:6], (1.0, 2.0, 3.0, 4.0, 5.0, 0.5), strict=True):
            _target, _vel, kp, kd, _tau = motor.send_mit.call_args.args
            assert (kp, kd) == (40.0, expected_kd)

    def test_partial_mit_gain_dict_uses_defaults_for_other_joints(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, mit_kp={"shoulder_lift": 40.0, "elbow_flex": 45.0})
        robot.connect()
        controller = mock_motorbridge.Controller.return_value
        motors = list(controller.mock_motors)

        robot.send_action(np.zeros(7, dtype=np.float32))

        kps = [motor.send_mit.call_args.args[2] for motor in motors[:6]]
        assert kps == [50.0, 40.0, 45.0, 50.0, 50.0, 50.0]


class TestReBotB601RSControl:
    def test_gain_limits_follow_the_motor_model(self, mock_motorbridge: MagicMock) -> None:
        # RS-06 joints encode damping up to 100, RS-00 joints only up to 5.
        _create_robot(mock_motorbridge, mit_kd={"shoulder_lift": 20.0})
        with pytest.raises(ValueError, match="exceeds the rs-00 MIT limit"):
            _create_robot(mock_motorbridge, mit_kd={"wrist_flex": 6.0})
        with pytest.raises(ValueError, match="exceeds the rs-00 MIT limit"):
            _create_robot(mock_motorbridge, mit_kp=600.0)

    def test_gravity_torque_is_sent_in_the_motor_frame(self, mock_motorbridge: MagicMock) -> None:
        from physicalai_rebot_b601_plugin import get_urdf_path
        from physicalai_rebot_b601_plugin.constants import REBOT_B601_RS_GRAVITY_URDF, REBOT_B601_URDF_ARM_JOINTS
        from physicalai_rebot_b601_plugin.gravity import GravityModel

        robot = _create_robot(mock_motorbridge, gravity_compensation=True)
        robot.connect()
        motors = list(mock_motorbridge.Controller.return_value.mock_motors)
        pose = np.array([0.0, -40.0, -80.0, 30.0, 0.0, 0.0, 0.0], dtype=np.float32)

        robot.send_action(pose)

        public = GravityModel(get_urdf_path() / REBOT_B601_RS_GRAVITY_URDF, REBOT_B601_URDF_ARM_JOINTS).torques(
            np.radians(pose[:6]),
        )
        for name, motor, torque in zip(REBOT_B601_RS_JOINT_ORDER, motors[:6], public, strict=False):
            # Motor angles are -1 x public angles, so motor torques are -1 x public torques too.
            assert motor.send_mit.call_args.args[4] == pytest.approx(torque / REBOT_B601_RS_JOINT_DIRECTIONS[name])
        assert abs(public[2]) > 1.0

    def test_observation_torques_are_in_the_public_frame(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        motors = list(mock_motorbridge.Controller.return_value.mock_motors)
        motors[1].get_state.return_value = _MotorState(pos=0.0, torq=2.5)

        obs = robot.get_observation()

        assert obs.sensor_data is not None
        assert obs.sensor_data["torques"][1] == pytest.approx(-2.5)

    def test_velocity_feedforward_uses_the_motor_frame(self, mock_motorbridge: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        from physicalai_rebot_b601_plugin import dm

        now = [10.0]
        fake_time = MagicMock()
        fake_time.monotonic.side_effect = lambda: now[0]
        monkeypatch.setattr(dm, "time", fake_time)
        robot = _create_robot(mock_motorbridge, velocity_feedforward=True)
        robot.connect()
        motors = list(mock_motorbridge.Controller.return_value.mock_motors)

        robot.send_action(np.zeros(7, dtype=np.float32))
        now[0] += 0.02
        robot.send_action(np.array([0.0, -0.2, 0, 0, 0, 0, 0], dtype=np.float32))

        # Public -0.2 degree in 20 ms is -10 deg/s; the motor turns the other way. Filtered by half.
        assert motors[1].send_mit.call_args.args[1] == pytest.approx(0.5 * math.radians(10.0))

    def test_gripper_stall_uses_the_measured_command_interval(
        self,
        mock_motorbridge: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        now = _fake_gripper_clock(monkeypatch)
        robot = _create_robot(mock_motorbridge)
        robot.connect()
        gripper = mock_motorbridge.Controller.return_value.mock_motors[6]
        gripper.get_state.return_value = _MotorState(pos=0.0, vel=0.0)
        robot.send_action(np.zeros(7, dtype=np.float32))

        # 0.02 rad per 100 ms is 0.2 rad/s: stalled (below 0.25), though not at a nominal 20 ms.
        for position in (0.02, 0.04):
            now[0] += 0.1
            gripper.get_state.return_value = _MotorState(pos=position, vel=0.0)
            robot.send_action(np.zeros(7, dtype=np.float32))

        assert gripper.robstride_write_param_f32.call_args == call(0x700B, 0.5)


class TestReBotB601RSLiveControl:
    def test_defaults_are_full_gravity_and_partial_velocity_feedforward(self, mock_motorbridge: MagicMock) -> None:
        settings = _create_robot(mock_motorbridge, raw=False).set_control()

        assert settings["gravity_compensation"] == 1.0
        assert settings["velocity_feedforward"] == 0.3
        assert settings["motion_limits"] is True

    @pytest.mark.parametrize("kwargs", [{"gravity_compensation": 1.2}, {"velocity_feedforward": float("nan")}])
    def test_constructor_rejects_feedforward_outside_0_to_1(self, mock_motorbridge: MagicMock, kwargs: dict) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            _create_robot(mock_motorbridge, **kwargs)

    def test_velocity_scale_scales_the_reference_velocity(
        self,
        mock_motorbridge: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from physicalai_rebot_b601_plugin import dm

        now = [10.0]
        fake_time = MagicMock()
        fake_time.monotonic.side_effect = lambda: now[0]
        monkeypatch.setattr(dm, "time", fake_time)
        robot = _create_robot(mock_motorbridge, velocity_feedforward=0.3)
        robot.connect()
        motors = list(mock_motorbridge.Controller.return_value.mock_motors)

        robot.send_action(np.zeros(7, dtype=np.float32))
        now[0] += 0.02
        robot.send_action(np.array([0.0, -0.2, 0, 0, 0, 0, 0], dtype=np.float32))

        # -10 deg/s public, filtered by half, scaled by 0.3; the motor turns the other way.
        assert motors[1].send_mit.call_args.args[1] == pytest.approx(0.3 * 0.5 * math.radians(10.0))

    def test_set_control_changes_feedforward_limits_and_gains(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge)
        robot.set_control(
            gravity_compensation=0.5,
            velocity_feedforward=0.0,
            motion_limits=False,
            mit_kd={"shoulder_lift": 16.0},
        )

        settings = robot.set_control()
        assert settings["gravity_compensation"] == 0.5
        assert settings["velocity_feedforward"] == 0.0
        assert settings["motion_limits"] is False
        assert settings["mit_kd"]["shoulder_lift"] == 16.0
        assert settings["mit_kd"]["elbow_flex"] == 10.0  # unchanged joints keep their gains
        assert "gripper" not in settings["mit_kd"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"gravity_compensation": 1.5},
            {"velocity_feedforward": -0.1},
            {"mit_kd": {"wrist_flex": 6.0}},  # RS-00 damping max is 5
            {"mit_kp": {"gripper": 10.0}},
            {"mit_kp": {"shoulder_lift": float("nan")}},
        ],
    )
    def test_set_control_rejects_invalid_settings(self, mock_motorbridge: MagicMock, kwargs: dict) -> None:
        robot = _create_robot(mock_motorbridge)
        before = robot.set_control()

        with pytest.raises(ValueError):
            robot.set_control(**kwargs)
        assert robot.set_control() == before

    def test_gravity_scale_scales_the_feedforward(self, mock_motorbridge: MagicMock) -> None:
        robot = _create_robot(mock_motorbridge, gravity_compensation=True)
        command = np.array([0.0, -60.0, -70.0, -20.0, 0.0, 0.0, 0.0])
        full = robot._gravity_torques(command)
        robot.set_control(gravity_compensation=0.25)

        assert robot._gravity_torques(command) == pytest.approx(0.25 * full)

