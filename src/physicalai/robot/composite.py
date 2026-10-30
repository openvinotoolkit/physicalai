# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Composable robot implementations.

``CompositeRobot`` presents independently controlled robots as one robot by
concatenating their joint vectors. ``BimanualRobot`` provides the conventional
left/right specialization used by two-arm integrations.
"""

from __future__ import annotations

import json
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar, TypeVarTuple, cast

import numpy as np

from physicalai.config import export_config
from physicalai.robot.interface import Robot, RobotObservation

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from physicalai.capture.frame import Frame
    from physicalai.robot.handover import HandoverCapabilities


RobotTs = TypeVarTuple("RobotTs")
LeftRobotT = TypeVar("LeftRobotT", bound=Robot)
RightRobotT = TypeVar("RightRobotT", bound=Robot)


@dataclass
class CompositeRobotObservation:
    """Merged observation from a :class:`CompositeRobot`.

    Attributes:
        joint_positions: Concatenated component joint positions in component order.
        timestamp: First component's capture time when every component last advanced.
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
        self._last_complete_observation: tuple[tuple[float, ...], CompositeRobotObservation] | None = None
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

    @property
    def handover_capabilities(self) -> HandoverCapabilities | None:
        """Compose declared coordinate conventions and per-joint limits in order."""
        from physicalai.robot.handover import HandoverCapabilities, validate_handover_capabilities  # noqa: PLC0415

        capabilities = []
        for name, robot in self._robots.items():
            capability = validate_handover_capabilities(
                getattr(robot, "handover_capabilities", None), num_joints=len(robot.joint_names)
            )
            if capability is None:
                return None
            capabilities.append((name, capability))
        return HandoverCapabilities(
            position_space=json.dumps([(name, item.position_space) for name, item in capabilities]),
            position_tolerance=tuple(value for _, item in capabilities for value in item.position_tolerance),
            max_position_step=tuple(value for _, item in capabilities for value in item.max_position_step),
            supports_alignment=(
                all(item.supports_alignment for _, item in capabilities)
                and all(callable(getattr(robot, "set_torque", None)) for robot in self._robots.values())
            ),
        )

    def set_torque(self, *, enabled: bool) -> None:
        """Set every component's torque, rolling back a partial enable.

        Component write failures propagate.

        Args:
            enabled: Whether to enable torque.

        Raises:
            TypeError: If enabled is not boolean.
            NotImplementedError: If a component lacks torque control.
            RuntimeError: If torque enable and its rollback both fail.
        """
        if not isinstance(enabled, bool):
            msg = "enabled must be a bool"
            raise TypeError(msg)
        methods = [getattr(robot, "set_torque", None) for robot in self._robots.values()]
        if enabled and not all(callable(method) for method in methods):
            msg = "Every component must support torque control before enabling torque"
            raise NotImplementedError(msg)
        errors: list[Exception] = []
        for method in methods:
            if not callable(method):
                errors.append(NotImplementedError("A component does not support torque control"))
                continue
            try:
                method(enabled=enabled)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
                if enabled:
                    break
        if errors:
            if enabled:
                try:
                    self.set_torque(enabled=False)
                except Exception as exc:  # noqa: BLE001
                    msg = f"Torque enable failed: {errors[0]}; rollback also failed: {exc}"
                    raise RuntimeError(msg) from errors[0]
            raise errors[0]

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
        if rollbacks:
            self._last_complete_observation = None
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
        self._last_complete_observation = None
        if first_error is not None:
            raise first_error

    def is_connected(self) -> bool:
        """Return whether every component is connected."""
        return all(robot.is_connected() for robot in self._robots.values())

    def get_observation(self) -> RobotObservation:
        """Read components, retaining the last complete snapshot while any is stale.

        Returns:
            A merged composite observation.
        """
        return self._get_observation(fresh=False)

    def get_fresh_observation(self) -> RobotObservation:
        """Capture each component after this call, including nested shared owners.

        Returns:
            A merged observation from fresh component reads. Capture failures propagate.
        """
        return self._get_observation(fresh=True)

    def _get_observation(self, *, fresh: bool) -> RobotObservation:
        """Read and merge components in order.

        Returns:
            A merged composite observation.

        Raises:
            ConnectionError: If a requested fresh component capture did not advance.
        """
        observations: list[RobotObservation] = []
        for robot in self._robots.values():
            capture = getattr(robot, "get_fresh_observation", None) if fresh else None
            observations.append(cast("RobotObservation", capture() if callable(capture) else robot.get_observation()))
        timestamps = tuple(float(observation.timestamp) for observation in observations)
        if self._last_complete_observation is not None:
            previous_timestamps, previous_observation = self._last_complete_observation
            if not all(current > previous for current, previous in zip(timestamps, previous_timestamps, strict=True)):
                if fresh:
                    msg = "A component's fresh capture did not advance"
                    raise ConnectionError(msg)
                return deepcopy(previous_observation)
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

        observation = CompositeRobotObservation(
            joint_positions=positions,
            timestamp=timestamps[0],
            sensor_data=sensor_data,
        )
        self._last_complete_observation = timestamps, deepcopy(observation)
        return observation

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
