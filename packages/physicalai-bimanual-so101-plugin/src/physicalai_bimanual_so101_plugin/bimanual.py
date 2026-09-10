"""Bimanual SO-101 robot arm driver.

Wraps two :class:`physicalai.robot.so101.SO101` instances (left + right) behind the
:class:`~physicalai.robot.Robot` protocol, prefixing all joint names with
``left_`` / ``right_``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from physicalai.config import export_config
from physicalai.robot import Robot

from physicalai_bimanual_so101_plugin.constants import (
    BIMANUAL_SO101_JOINT_ORDER,
    NUM_BIMANUAL_JOINTS,
    NUM_SINGLE_ARM_JOINTS,
)

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import RobotObservation
    from physicalai.robot.so101 import SO101


@dataclass
class BimanualSO101Observation:
    """Merged observation from a bimanual SO-101 robot.

    Attributes:
        joint_positions: Array of shape ``(12,)`` — left (6) then right (6)
            in the underlying arm units (for example, normalized or ticks).
        timestamp: ``time.monotonic()`` at capture (from left arm).
        sensor_data: Merged sensor data of shape ``(12,)`` or ``None``.
        images: Always ``None`` — no built-in camera support.
    """

    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None

    @property
    def state(self) -> np.ndarray:
        """Joint positions as the canonical state vector."""
        return self.joint_positions


@export_config(class_path="physicalai_bimanual_so101_plugin.BimanualSO101")
class BimanualSO101(Robot):
    """Two-arm SO-101 driver composing a left and right :class:`SO101`.

    Args:
        left: Left arm driver.
        right: Right arm driver.

    Raises:
        ValueError: If the two arms have different roles or driver types.
    """

    JOINT_ORDER: ClassVar[tuple[str, ...]] = BIMANUAL_SO101_JOINT_ORDER
    NUM_JOINTS: int = NUM_BIMANUAL_JOINTS

    def __init__(self, left: SO101, right: SO101) -> None:
        """Initialize the bimanual driver with left and right SO101 arms.

        Raises:
            ValueError: If arms have different roles or different driver types.
        """
        if left.role != right.role:
            msg = f"Both arms must have the same role; got left={left.role!r}, right={right.role!r}."
            raise ValueError(msg)
        if type(left) is not type(right):
            msg = (
                f"Both arms must be the same driver type; got left={type(left).__name__}, right={type(right).__name__}."
            )
            raise ValueError(msg)

        self._left = left
        self._right = right

    @property
    def role(self) -> str:
        """Shared role of both arms."""
        return self._left.role

    @property
    def device_ids(self) -> tuple[str, ...]:
        """Sorted, deduplicated device identifiers for both arms."""
        return tuple(sorted(set(self._left.device_ids) | set(self._right.device_ids)))

    @property
    def joint_names(self) -> list[str]:
        """Prefixed joint names ordered as left arm then right arm."""
        left_names = [f"left_{n}" for n in self._left.joint_names]
        right_names = [f"right_{n}" for n in self._right.joint_names]
        return left_names + right_names

    def connect(self) -> None:
        """Connect both arms, rolling back left if right connection fails."""
        self._left.connect()
        try:
            self._right.connect()
        except Exception:
            self._left.disconnect()
            raise

    def disconnect(self) -> None:
        """Disconnect both arms, attempting both even if one fails."""
        first_error: Exception | None = None

        try:
            self._left.disconnect()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive against hardware failures
            first_error = exc

        try:
            self._right.disconnect()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive against hardware failures
            if first_error is None:
                first_error = exc

        if first_error is not None:
            raise first_error

    def is_connected(self) -> bool:
        """Return ``True`` when both arms report connected."""
        return self._left.is_connected() and self._right.is_connected()

    def get_observation(self) -> RobotObservation:
        """Read and merge observations from left and right arms.

        Returns:
            Merged bimanual observation with concatenated joint positions.
        """
        left_obs = self._left.get_observation()
        right_obs = self._right.get_observation()

        positions = np.concatenate([left_obs.joint_positions, right_obs.joint_positions])

        sensor_data: dict[str, np.ndarray] | None = None
        if left_obs.sensor_data is not None and right_obs.sensor_data is not None:
            shared_keys = left_obs.sensor_data.keys() & right_obs.sensor_data.keys()
            sensor_data = {
                key: np.concatenate([left_value, right_obs.sensor_data[key]])
                for key, left_value in left_obs.sensor_data.items()
                if key in shared_keys
            }

        return BimanualSO101Observation(
            joint_positions=positions,
            timestamp=left_obs.timestamp,
            sensor_data=sensor_data,
        )

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Split and send a 12-dim action to each arm.

        Raises:
            RuntimeError: If called while the robot role is ``leader``.
            ValueError: If ``action`` does not have shape ``(12,)``.
        """
        if self.role == "leader":
            msg = "Cannot send actions to a leader robot."
            raise RuntimeError(msg)

        expected_shape = (NUM_BIMANUAL_JOINTS,)
        if action.shape != expected_shape:
            msg = f"Expected action shape {expected_shape}, got {action.shape}"
            raise ValueError(msg)

        self._left.send_action(action[:NUM_SINGLE_ARM_JOINTS], goal_time=goal_time)
        self._right.send_action(action[NUM_SINGLE_ARM_JOINTS:], goal_time=goal_time)
