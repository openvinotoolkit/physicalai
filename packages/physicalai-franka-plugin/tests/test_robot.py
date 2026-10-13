from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar
from unittest.mock import patch

import numpy as np
import pytest
from physicalai.config import to_config

from physicalai_franka_plugin.constants import FR3_JOINT_LIMITS_RAD
from physicalai_franka_plugin.robot import FrankaFR3

if TYPE_CHECKING:
    from collections.abc import Generator


class _FakeDuration:
    def __init__(self, seconds: float) -> None:
        self._seconds = seconds

    def to_sec(self) -> float:
        return self._seconds


class _FakeErrors:
    def __init__(self, *, has_errors: bool = False) -> None:
        self._has_errors = has_errors

    def __bool__(self) -> bool:
        return self._has_errors


def _make_state(q: np.ndarray, *, has_errors: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        q=np.asarray(q, dtype=np.float64),
        dq=np.zeros(7, dtype=np.float64),
        tau_J=np.zeros(7, dtype=np.float64),
        tau_ext_hat_filtered=np.zeros(7, dtype=np.float64),
        current_errors=_FakeErrors(has_errors=has_errors),
        time=_FakeDuration(0.001),
    )


class _FakeJointPositions:
    def __init__(self, q: object) -> None:
        self.q = np.asarray(q, dtype=np.float64)


class _FakeControl:
    def __init__(self) -> None:
        self.writes: list[np.ndarray] = []
        self.reads = 0

    def readOnce(self) -> tuple[SimpleNamespace, _FakeDuration]:  # noqa: N802 - mirrors the pylibfranka API
        self.reads += 1
        time.sleep(0.001)
        return _make_state(np.zeros(7)), _FakeDuration(0.001)

    def writeOnce(self, command: _FakeJointPositions) -> None:  # noqa: N802 - mirrors the pylibfranka API
        self.writes.append(command.q.copy())


class _FakeGripper:
    def __init__(self, host: str) -> None:
        self.host = host
        self.width = 0.04
        self.moved: list[tuple[float, float]] = []

    def read_once(self) -> SimpleNamespace:
        return SimpleNamespace(width=self.width, temperature=25.0)

    def move(self, width: float, speed: float) -> None:
        self.width = width
        self.moved.append((width, speed))

    def stop(self) -> None:
        pass


class _FakeRobot:
    def __init__(self, host: str, realtime_config: str, *, has_errors: bool = False) -> None:
        self.host = host
        self.realtime_config = realtime_config
        self.control: _FakeControl | None = None
        self.recovered = False
        self.stopped = False
        self._has_errors = has_errors

    def read_once(self) -> SimpleNamespace:
        return _make_state(np.zeros(7), has_errors=self._has_errors)

    def automatic_error_recovery(self) -> None:
        self.recovered = True
        self._has_errors = False

    def start_joint_position_control(self, mode: str) -> _FakeControl:
        self.control = _FakeControl()
        return self.control

    def stop(self) -> None:
        self.stopped = True


class _RealtimeConfig:
    kEnforce = "kEnforce"  # noqa: N815 - mirrors the pylibfranka enum member
    kIgnore = "kIgnore"  # noqa: N815 - mirrors the pylibfranka enum member


class _ControllerMode:
    JointImpedance = "JointImpedance"
    CartesianImpedance = "CartesianImpedance"


@dataclass
class _FakePylibfranka:
    robots: list[_FakeRobot] = None  # type: ignore[assignment]
    grippers: list[_FakeGripper] = None  # type: ignore[assignment]
    robot_has_errors: bool = False

    def __post_init__(self) -> None:
        self.robots = []
        self.grippers = []

    def Robot(self, host: str, realtime_config: str) -> _FakeRobot:  # noqa: N802 - mirrors the pylibfranka API
        robot = _FakeRobot(host, realtime_config, has_errors=self.robot_has_errors)
        self.robots.append(robot)
        return robot

    def Gripper(self, host: str) -> _FakeGripper:  # noqa: N802 - mirrors the pylibfranka API
        gripper = _FakeGripper(host)
        self.grippers.append(gripper)
        return gripper

    JointPositions: ClassVar[type[_FakeJointPositions]] = _FakeJointPositions
    RealtimeConfig: ClassVar[type[_RealtimeConfig]] = _RealtimeConfig
    ControllerMode: ClassVar[type[_ControllerMode]] = _ControllerMode


@pytest.fixture
def fake_pylibfranka() -> Generator[_FakePylibfranka]:
    fake = _FakePylibfranka()
    with patch.dict(sys.modules, {"pylibfranka": fake}):
        yield fake


class TestFrankaFR3Construction:
    def test_defaults(self) -> None:
        robot = FrankaFR3("172.16.0.2")

        assert robot.host == "172.16.0.2"
        assert robot.gripper_enabled is True
        assert robot.realtime_config == "kEnforce"
        assert robot.goal_time == pytest.approx(0.1)
        assert robot.joint_names == [*FrankaFR3.JOINT_ORDER, "gripper"]
        assert robot.device_ids == ()

    def test_arm_only_joint_layout(self) -> None:
        robot = FrankaFR3("172.16.0.2", gripper_enabled=False)

        assert robot.joint_names == FrankaFR3.JOINT_ORDER

    def test_invalid_host_raises(self) -> None:
        with pytest.raises(ValueError, match="host must be a non-empty"):
            FrankaFR3("")

    def test_invalid_realtime_config_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid realtime_config"):
            FrankaFR3("172.16.0.2", realtime_config="kSometimes")

    def test_invalid_goal_time_raises(self) -> None:
        with pytest.raises(ValueError, match="goal_time must be positive"):
            FrankaFR3("172.16.0.2", goal_time=0.0)

    def test_exports_recipe(self) -> None:
        robot = FrankaFR3("172.16.0.2", gripper_enabled=False)

        assert to_config(robot)["init_args"] == {
            "host": "172.16.0.2",
            "gripper_enabled": False,
        }

    def test_connect_requires_pylibfranka(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delitem(sys.modules, "pylibfranka", raising=False)
        robot = FrankaFR3("172.16.0.2")

        with pytest.raises(ImportError, match="pylibfranka"):
            robot.connect()


class TestFrankaFR3ConnectDisconnect:
    def test_connect_reads_state_and_starts_loop(self, fake_pylibfranka: _FakePylibfranka) -> None:
        robot = FrankaFR3("172.16.0.2")

        robot.connect()
        assert robot.is_connected()
        assert fake_pylibfranka.robots[0].host == "172.16.0.2"
        assert fake_pylibfranka.robots[0].realtime_config == "kEnforce"
        assert fake_pylibfranka.robots[0].control is not None

        robot.disconnect()
        assert not robot.is_connected()
        assert fake_pylibfranka.robots[0].stopped

    def test_connect_is_idempotent(self, fake_pylibfranka: _FakePylibfranka) -> None:
        robot = FrankaFR3("172.16.0.2")

        robot.connect()
        first_control = fake_pylibfranka.robots[0].control
        robot.connect()
        assert fake_pylibfranka.robots[0].control is first_control
        robot.disconnect()

    def test_auto_recovery(self, fake_pylibfranka: _FakePylibfranka) -> None:
        robot = FrankaFR3("172.16.0.2")

        robot.connect()
        assert not fake_pylibfranka.robots[0].recovered
        robot.disconnect()

    def test_auto_recovery_on_error(self, fake_pylibfranka: _FakePylibfranka) -> None:
        fake_pylibfranka.robot_has_errors = True
        robot = FrankaFR3("172.16.0.2")

        robot.connect()
        assert fake_pylibfranka.robots[0].recovered
        robot.disconnect()

    def test_disconnect_without_connect_is_noop(self) -> None:
        robot = FrankaFR3("172.16.0.2")

        robot.disconnect()


class TestFrankaFRObservation:
    def test_observation_shape_and_sensor_data(self, fake_pylibfranka: _FakePylibfranka) -> None:
        robot = FrankaFR3("172.16.0.2")
        robot.connect()
        try:
            observation = robot.get_observation()

            assert observation.joint_positions.shape == (8,)
            assert observation.sensor_data is not None
            assert observation.sensor_data["velocities"].shape == (7,)
            assert observation.sensor_data["torques"].shape == (7,)
            assert observation.sensor_data["external_torques"].shape == (7,)
            assert observation.state is observation.joint_positions
        finally:
            robot.disconnect()

    def test_arm_only_observation_shape(self, fake_pylibfranka: _FakePylibfranka) -> None:
        robot = FrankaFR3("172.16.0.2", gripper_enabled=False)
        robot.connect()
        try:
            observation = robot.get_observation()

            assert observation.joint_positions.shape == (7,)
        finally:
            robot.disconnect()

    def test_get_observation_before_connect_raises(self) -> None:
        robot = FrankaFR3("172.16.0.2")

        with pytest.raises(ConnectionError, match="not connected"):
            robot.get_observation()


class TestFrankaFR3SendAction:
    def test_send_action_before_connect_raises(self) -> None:
        robot = FrankaFR3("172.16.0.2")

        with pytest.raises(ConnectionError, match="not connected"):
            robot.send_action(np.zeros(8))

    def test_send_action_validates_shape(self, fake_pylibfranka: _FakePylibfranka) -> None:
        robot = FrankaFR3("172.16.0.2")
        robot.connect()
        try:
            with pytest.raises(ValueError, match="Expected action shape"):
                robot.send_action(np.zeros(7))
        finally:
            robot.disconnect()

    def test_send_action_converges_to_clipped_target(self, fake_pylibfranka: _FakePylibfranka) -> None:
        robot = FrankaFR3("172.16.0.2", goal_time=0.001)
        robot.connect()
        try:
            robot.send_action(np.array([200.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04]), goal_time=0.001)
            time.sleep(0.05)

            control = fake_pylibfranka.robots[0].control
            assert control is not None
            expected = np.array(
                [
                    FR3_JOINT_LIMITS_RAD["fr3_joint1"][1],
                    np.radians(10.0),
                    0.0,
                    FR3_JOINT_LIMITS_RAD["fr3_joint4"][1],
                    0.0,
                    FR3_JOINT_LIMITS_RAD["fr3_joint6"][0],
                    0.0,
                ],
            )
            np.testing.assert_allclose(control.writes[-1], expected, atol=1e-6)
        finally:
            robot.disconnect()

    def test_gripper_move_only_on_change(self, fake_pylibfranka: _FakePylibfranka) -> None:
        robot = FrankaFR3("172.16.0.2", goal_time=0.001)
        robot.connect()
        try:
            action = np.zeros(8)
            action[-1] = 0.03
            robot.send_action(action)
            time.sleep(0.02)
            robot.send_action(action)
            time.sleep(0.02)

            assert fake_pylibfranka.grippers[0].moved == [(0.03, pytest.approx(0.05))]
        finally:
            robot.disconnect()
