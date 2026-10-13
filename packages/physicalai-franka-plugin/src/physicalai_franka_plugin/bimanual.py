# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Bimanual Franka Research 3 follower driver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from physicalai.config import export_config

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import RobotObservation
    from physicalai_franka_plugin.robot import FrankaFR3


@dataclass
class BimanualFrankaFR3Observation:
    """Merged left-then-right observation from two Franka FR3 arms."""

    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None

    @property
    def state(self) -> np.ndarray:
        """Joint-position state vector."""
        return self.joint_positions


@export_config(class_path="physicalai_franka_plugin.BimanualFrankaFR3")
class BimanualFrankaFR3:
    """Compose two compatible :class:`FrankaFR3` follower arms.

    Joint positions and actions are ordered left arm first, followed by the
    right arm. Both children must either include a Franka Hand gripper or both
    be arm-only, keeping the action layout deterministic.
    """

    def __init__(self, left: FrankaFR3, right: FrankaFR3) -> None:
        """Initialize the bimanual driver.

        Raises:
            ValueError: If the arms do not use the same gripper layout or host.
        """
        if left.gripper_enabled != right.gripper_enabled:
            msg = "Both FR3 arms must use the same gripper_enabled layout."
            raise ValueError(msg)
        if left.host == right.host:
            msg = "left and right FR3 arms must use different FCI hosts."
            raise ValueError(msg)
        self._left = left
        self._right = right

    @property
    def left(self) -> FrankaFR3:
        """Left FR3 child driver."""
        return self._left

    @property
    def right(self) -> FrankaFR3:
        """Right FR3 child driver."""
        return self._right

    @property
    def device_ids(self) -> tuple[str, ...]:
        """FCI arms do not exclusively own local physical devices."""
        return ()

    @property
    def joint_names(self) -> list[str]:
        """Left-prefixed names followed by right-prefixed names."""
        left_names = [f"left_{name}" for name in self._left.joint_names]
        right_names = [f"right_{name}" for name in self._right.joint_names]
        return left_names + right_names

    def connect(self) -> None:
        """Connect both arms, rolling the left connection back on failure."""
        if self.is_connected():
            return
        self._left.connect()
        try:
            self._right.connect()
        except Exception:
            self._left.disconnect()
            raise

    def disconnect(self) -> None:
        """Stop and disconnect both arms, attempting every cleanup action."""
        first_error: Exception | None = None
        for arm in (self._left, self._right):
            try:
                arm.disconnect()
            except Exception as exc:  # ruff: ignore[blind-except] - never leave the other arm running
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def is_connected(self) -> bool:
        """Return whether both constituent arms are connected."""
        return self._left.is_connected() and self._right.is_connected()

    def get_observation(self) -> RobotObservation:
        """Return a merged observation from both arms."""
        left = self._left.get_observation()
        right = self._right.get_observation()
        sensor_data: dict[str, np.ndarray] | None = None
        if left.sensor_data is not None and right.sensor_data is not None:
            sensor_data = {
                name: np.concatenate([left.sensor_data[name], right.sensor_data[name]])
                for name in left.sensor_data.keys() & right.sensor_data.keys()
            }
        return BimanualFrankaFR3Observation(
            joint_positions=np.concatenate([left.joint_positions, right.joint_positions]),
            timestamp=min(left.timestamp, right.timestamp),
            sensor_data=sensor_data,
        )

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Split and send a left-then-right joint-position action.

        Raises:
            ValueError: If the action layout does not match both arms.
        """
        action = np.asarray(action)
        expected_shape = (len(self.joint_names),)
        if action.shape != expected_shape:
            msg = f"Expected action shape {expected_shape}, got {action.shape}"
            raise ValueError(msg)
        split = len(self._left.joint_names)
        self._left.send_action(action[:split], goal_time=goal_time)
        self._right.send_action(action[split:], goal_time=goal_time)
