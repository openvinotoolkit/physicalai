# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Composable robot implementations.

``CompositeRobot`` presents independently controlled robots as one robot by
concatenating their joint vectors. ``BimanualRobot`` provides the conventional
left/right specialization used by two-arm integrations.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar, TypeVarTuple

import numpy as np

from physicalai.config import export_config
from physicalai.robot.interface import Robot, RobotObservation

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from physicalai.capture.frame import Frame


RobotTs = TypeVarTuple("RobotTs")
LeftRobotT = TypeVar("LeftRobotT", bound=Robot)
RightRobotT = TypeVar("RightRobotT", bound=Robot)


@dataclass
class CompositeRobotObservation:
    """Merged observation from a :class:`CompositeRobot`.

    Attributes:
        joint_positions: Concatenated component joint positions in component order.
        timestamp: Timestamp from the first component observation.
        sensor_data: Sensor arrays shared by every component, concatenated in
            component order.
        images: Always ``None``. External cameras belong on ``RobotRuntime``;
            embedded-image composition needs an explicit naming policy.
    """

    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None

    @property
    def state(self) -> np.ndarray:
        """Joint positions as the canonical composite state vector."""
        return self.joint_positions


@export_config(class_path="physicalai.robot.CompositeRobot")
class CompositeRobot(Generic[*RobotTs]):
    """Combine named robot components behind the :class:`Robot` protocol.

    Args:
        robots: Non-empty mapping of stable component names to robots. Mapping
            order defines the joint, observation, and action-vector order.

    Component names prefix joint names, preventing collisions between robots
    that use the same underlying joint names.
    """

    def __init__(self, robots: Mapping[str, Robot]) -> None:
        """Initialize a composite from named component robots.

        Raises:
            TypeError: If a component does not satisfy the Robot protocol.
            ValueError: If components are empty or have invalid names.
        """
        if not robots:
            msg = "CompositeRobot requires at least one component robot."
            raise ValueError(msg)

        self._robots = dict(robots)
        for name, robot in self._robots.items():
            if not isinstance(name, str) or not name or name.strip() != name:
                msg = f"Component names must be non-empty strings without surrounding whitespace; got {name!r}."
                raise ValueError(msg)
            if not isinstance(robot, Robot):
                msg = f"Component {name!r} does not satisfy the Robot protocol."
                raise TypeError(msg)

    @property
    def robots(self) -> Mapping[str, Robot]:
        """Component robots in their declared order."""
        return self._robots

    @property
    def device_ids(self) -> tuple[str, ...]:
        """Sorted, deduplicated device ids across all components."""
        return tuple(sorted({device_id for robot in self._robots.values() for device_id in robot.device_ids}))

    @property
    def joint_names(self) -> list[str]:
        """Component-prefixed joint names in declared component order.

        Raises:
            ValueError: If two component/joint pairs produce the same prefixed name.
        """
        joint_names: list[str] = []
        seen: dict[str, tuple[str, str]] = {}
        for component_name, robot in self._robots.items():
            for joint_name in robot.joint_names:
                aggregate = f"{component_name}_{joint_name}"
                previous = seen.get(aggregate)
                if previous is not None:
                    previous_component, previous_joint = previous
                    msg = (
                        f"Joint name {aggregate!r} comes from both "
                        f"{previous_component!r} joint {previous_joint!r} and "
                        f"{component_name!r} joint {joint_name!r}. "
                        "Prefixed joint names must be unique."
                    )
                    raise ValueError(msg)
                seen[aggregate] = (component_name, joint_name)
                joint_names.append(aggregate)
        return joint_names

    def connect(self) -> None:
        """Connect every component, rolling back components connected by this call."""
        self._connect_with_rollback()

    def _connect_with_rollback(self) -> list[Callable[[], None]]:
        """Connect components and return callbacks for connections opened here.

        This private helper lets a containing composite roll back a nested
        composite without disconnecting any of its components that were already
        connected before this attempt.

        Returns:
            Disconnect callbacks for connections opened by this call.

        """
        rollbacks: list[Callable[[], None]] = []
        try:
            for robot in self._robots.values():
                rollbacks.extend(self._connect_component(robot))
        except Exception:
            for rollback in reversed(rollbacks):
                with suppress(Exception):
                    rollback()
            raise
        return rollbacks

    @staticmethod
    def _connect_component(robot: Robot) -> list[Callable[[], None]]:
        """Connect one component and return callbacks for newly opened connections.

        Returns:
            Disconnect callbacks for connections opened by this call.
        """
        if isinstance(robot, CompositeRobot):
            return robot._connect_with_rollback()  # noqa: SLF001
        was_connected = robot.is_connected()
        robot.connect()
        return [] if was_connected else [robot.disconnect]

    def disconnect(self) -> None:
        """Disconnect every component and re-raise the first disconnect failure."""
        first_error: Exception | None = None
        for robot in self._robots.values():
            try:
                robot.disconnect()
            except Exception as exc:  # noqa: BLE001 # pragma: no cover - defensive against hardware failures
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def is_connected(self) -> bool:
        """Return whether every component is connected."""
        return all(robot.is_connected() for robot in self._robots.values())

    def get_observation(self) -> RobotObservation:
        """Read and merge component observations in declared component order.

        Returns:
            A merged composite observation.
        """
        observations = [robot.get_observation() for robot in self._robots.values()]
        positions = np.concatenate([observation.joint_positions for observation in observations])

        sensor_data: dict[str, np.ndarray] | None = None
        sensor_maps: list[dict[str, np.ndarray]] = []
        for observation in observations:
            if observation.sensor_data is None:
                break
            sensor_maps.append(observation.sensor_data)
        else:
            shared_keys = set(sensor_maps[0])
            for sensor_map in sensor_maps[1:]:
                shared_keys &= sensor_map.keys()
            sensor_data = {
                key: np.concatenate([sensor_map[key] for sensor_map in sensor_maps])
                for key in sensor_maps[0]
                if key in shared_keys
            }

        return CompositeRobotObservation(
            joint_positions=positions,
            timestamp=observations[0].timestamp,
            sensor_data=sensor_data,
        )

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Split an exact-size action vector and send each slice to its component.

        Raises:
            ValueError: If the action does not exactly match the aggregate joint shape.
        """
        expected_shape = (len(self.joint_names),)
        if action.shape != expected_shape:
            msg = f"Expected action shape {expected_shape}, got {action.shape}"
            raise ValueError(msg)

        offset = 0
        for robot in self._robots.values():
            joint_count = len(robot.joint_names)
            robot.send_action(action[offset : offset + joint_count], goal_time=goal_time)
            offset += joint_count


@export_config(class_path="physicalai.robot.BimanualRobot")
class BimanualRobot(CompositeRobot[LeftRobotT, RightRobotT], Generic[LeftRobotT, RightRobotT]):
    """Two named robot components with conventional ``left`` and ``right`` prefixes."""

    def __init__(self, left: LeftRobotT, right: RightRobotT) -> None:
        """Initialize the composite left and right robots."""
        self._left = left
        self._right = right
        super().__init__({"left": left, "right": right})

    @property
    def left(self) -> LeftRobotT:
        """The left component robot."""
        return self._left

    @property
    def right(self) -> RightRobotT:
        """The right component robot."""
        return self._right
