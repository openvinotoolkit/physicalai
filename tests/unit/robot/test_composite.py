# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ruff: file-ignore[assert, no-self-use, undocumented-public-class, undocumented-public-method]

"""Tests for composite robot behavior without hardware dependencies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from physicalai.config import Config
from physicalai.robot import BimanualRobot, CompositeRobot, Robot
from tests.unit.robot.transport.fake import FakeRobot


@dataclass
class _Observation:
    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict | None = None

    @property
    def state(self) -> np.ndarray:
        return self.joint_positions


class _Robot:
    def __init__(
        self,
        name: str,
        positions: list[float],
        *,
        device_ids: tuple[str, ...] = (),
        sensor_data: dict[str, np.ndarray] | None = None,
        fail_connect: bool = False,
        fail_disconnect: bool = False,
    ) -> None:
        self.name = name
        self._positions = np.array(positions, dtype=np.float32)
        self._device_ids = device_ids
        self._sensor_data = sensor_data
        self._fail_connect = fail_connect
        self._fail_disconnect = fail_disconnect
        self._connected = False
        self.actions: list[np.ndarray] = []
        self.disconnect_calls = 0

    @property
    def device_ids(self) -> tuple[str, ...]:
        return self._device_ids

    @property
    def joint_names(self) -> list[str]:
        return [f"{self.name}_{index}" for index in range(len(self._positions))]

    def connect(self) -> None:
        if self._fail_connect:
            msg = f"could not connect {self.name}"
            raise ConnectionError(msg)
        self._connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._connected = False
        if self._fail_disconnect:
            msg = f"could not disconnect {self.name}"
            raise RuntimeError(msg)

    def is_connected(self) -> bool:
        return self._connected

    def get_observation(self) -> _Observation:
        return _Observation(self._positions, timestamp=1.0, sensor_data=self._sensor_data)

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        _ = goal_time
        self.actions.append(action.copy())


class TestCompositeRobot:
    def test_protocol_joint_names_devices_and_actions(self) -> None:
        base = _Robot("joint", [1.0], device_ids=("serial:base", "serial:shared"))
        arm = _Robot("joint", [2.0, 3.0], device_ids=("serial:arm", "serial:shared"))
        robot = CompositeRobot({"base": base, "arm": arm})

        assert isinstance(robot, Robot)
        assert robot.joint_names == ["base_joint_0", "arm_joint_0", "arm_joint_1"]
        assert robot.device_ids == ("serial:arm", "serial:base", "serial:shared")

        robot.send_action(np.array([10.0, 20.0, 30.0], dtype=np.float32), goal_time=0.2)
        np.testing.assert_array_equal(base.actions[0], [10.0])
        np.testing.assert_array_equal(arm.actions[0], [20.0, 30.0])

    @pytest.mark.parametrize("action", [np.zeros(2), np.zeros((3, 1))])
    def test_action_requires_exact_vector_shape(self, action: np.ndarray) -> None:
        robot = CompositeRobot({"base": _Robot("a", [1.0]), "arm": _Robot("b", [2.0, 3.0])})

        with pytest.raises(ValueError, match="Expected action shape"):
            robot.send_action(action)

    def test_observation_intersects_sensor_keys_across_all_components(self) -> None:
        robot = CompositeRobot(
            {
                "one": _Robot("one", [1.0], sensor_data={"shared": np.array([1]), "one": np.array([2])}),
                "two": _Robot("two", [2.0], sensor_data={"shared": np.array([3]), "two": np.array([4])}),
                "three": _Robot("three", [3.0], sensor_data={"shared": np.array([5])}),
            },
        )

        observation = robot.get_observation()
        np.testing.assert_array_equal(observation.joint_positions, [1.0, 2.0, 3.0])
        assert observation.sensor_data is not None
        assert set(observation.sensor_data) == {"shared"}
        np.testing.assert_array_equal(observation.sensor_data["shared"], [1, 3, 5])

    def test_connect_rolls_back_only_components_connected_by_the_call(self) -> None:
        already_connected = _Robot("first", [1.0])
        already_connected.connect()
        failing = _Robot("second", [2.0], fail_connect=True)
        robot = CompositeRobot({"first": already_connected, "second": failing})

        with pytest.raises(ConnectionError, match="second"):
            robot.connect()

        assert already_connected.is_connected()
        assert already_connected.disconnect_calls == 0

    def test_disconnect_attempts_every_component(self) -> None:
        failing = _Robot("first", [1.0], fail_disconnect=True)
        other = _Robot("second", [2.0])
        robot = CompositeRobot({"first": failing, "second": other})

        with pytest.raises(RuntimeError, match="first"):
            robot.disconnect()

        assert failing.disconnect_calls == 1
        assert other.disconnect_calls == 1

    def test_rejects_empty_components_and_invalid_names(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            CompositeRobot({})
        with pytest.raises(ValueError, match="Component names"):
            CompositeRobot({" arm ": _Robot("arm", [1.0])})

    def test_config_round_trip_preserves_named_components(self) -> None:
        robot = CompositeRobot({"base": FakeRobot("base"), "arm": FakeRobot("arm")})

        config = Config.from_instance(robot)
        restored = config.instantiate()

        assert config.class_path == "physicalai.robot.CompositeRobot"
        assert isinstance(restored, CompositeRobot)
        assert restored.joint_names == [
            "base_joint_0",
            "base_joint_1",
            "base_joint_2",
            "base_joint_3",
            "base_joint_4",
            "base_joint_5",
            "arm_joint_0",
            "arm_joint_1",
            "arm_joint_2",
            "arm_joint_3",
            "arm_joint_4",
            "arm_joint_5",
        ]


class TestBimanualRobot:
    def test_uses_left_right_component_names(self) -> None:
        left = _Robot("joint", [1.0])
        right = _Robot("joint", [2.0])
        robot = BimanualRobot(left=left, right=right)

        assert robot.left is left
        assert robot.right is right
        assert robot.joint_names == ["left_joint_0", "right_joint_0"]
