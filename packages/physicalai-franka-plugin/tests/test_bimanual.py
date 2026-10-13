from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from physicalai.robot import Robot
from physicalai_franka_plugin.bimanual import BimanualFrankaFR3


class _FakeArm:
    def __init__(self, host: str, *, gripper_enabled: bool = False) -> None:
        self.host = host
        self.gripper_enabled = gripper_enabled
        self.connected = False
        self.actions: list[tuple[np.ndarray, float]] = []

    @property
    def joint_names(self) -> list[str]:
        names = [f"fr3_joint{i}" for i in range(1, 8)]
        return [*names, "gripper"] if self.gripper_enabled else names

    @property
    def device_ids(self) -> tuple[str, ...]:
        return ()

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def get_observation(self) -> SimpleNamespace:
        return SimpleNamespace(
            joint_positions=np.arange(len(self.joint_names), dtype=np.float32),
            timestamp=1.0,
            sensor_data={"velocities": np.zeros(len(self.joint_names), dtype=np.float32)},
            images=None,
        )

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        self.actions.append((np.asarray(action), goal_time))


def test_bimanual_protocol_and_joint_order() -> None:
    robot = BimanualFrankaFR3(_FakeArm("left"), _FakeArm("right"))  # type: ignore[arg-type]

    assert isinstance(robot, Robot)
    assert robot.joint_names == [
        *(f"left_fr3_joint{i}" for i in range(1, 8)),
        *(f"right_fr3_joint{i}" for i in range(1, 8)),
    ]
    assert robot.device_ids == ()


def test_bimanual_connect_observe_and_send_action() -> None:
    left = _FakeArm("left")
    right = _FakeArm("right")
    robot = BimanualFrankaFR3(left, right)  # type: ignore[arg-type]

    robot.connect()
    assert robot.is_connected()
    observation = robot.get_observation()
    assert observation.joint_positions.shape == (14,)
    assert observation.sensor_data is not None
    assert observation.sensor_data["velocities"].shape == (14,)

    action = np.arange(14, dtype=np.float32)
    robot.send_action(action, goal_time=0.25)
    np.testing.assert_array_equal(left.actions[0][0], action[:7])
    np.testing.assert_array_equal(right.actions[0][0], action[7:])
    assert left.actions[0][1] == pytest.approx(0.25)
    robot.disconnect()
    assert not robot.is_connected()


def test_bimanual_rejects_mismatched_layout_or_hosts() -> None:
    with pytest.raises(ValueError, match="same gripper_enabled"):
        BimanualFrankaFR3(_FakeArm("left"), _FakeArm("right", gripper_enabled=True))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="different FCI hosts"):
        BimanualFrankaFR3(_FakeArm("same"), _FakeArm("same"))  # type: ignore[arg-type]
