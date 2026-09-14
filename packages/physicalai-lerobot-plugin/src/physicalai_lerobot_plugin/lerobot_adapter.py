# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Adapter that wraps a lerobot.robots.robot.Robot into PhysicalAI's Robot protocol.

Joint order, observation keys, and action keys are auto-detected from the
lerobot robot's observation dict on the first ``get_observation()`` call
(typically made during ``connect()``).
"""

from __future__ import annotations

import dataclasses
import importlib
import pkgutil
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import numpy as np
from loguru import logger

from physicalai.capture.frame import Frame
from physicalai.config import export_config
from physicalai.robot.device_ids import device_id_from_serial_port
from physicalai_lerobot_plugin.constants import VALID_ROLES, trust_unverified_plugins

if TYPE_CHECKING:
    from physicalai.robot.interface import RobotObservation


class _LeRobotLike(Protocol):
    observation_features: dict[str, Any]
    is_connected: bool

    def connect(self, calibrate: bool = True) -> None: ...  # noqa: FBT001, FBT002
    def disconnect(self) -> None: ...
    def get_observation(self) -> dict[str, Any]: ...
    def send_action(self, action: dict[str, Any]) -> None: ...


class _TeleoperatorLike(Protocol):
    action_features: dict[str, Any]
    is_connected: bool

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_action(self) -> dict[str, Any]: ...
    def send_feedback(self, action: dict[str, Any]) -> None: ...


@dataclass
class LeRobotAdapterObservation:
    """Observation from a LeRobot-wrapped robot.

    Attributes:
        joint_positions: Array of shape ``(N,)`` matching ``joint_names`` order.
        timestamp: ``time.monotonic()`` at capture.
        sensor_data: Optional auxiliary sensor readings.
        images: Optional built-in camera frames.
    """

    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None

    @property
    def state(self) -> np.ndarray:
        """Joint positions as the primary state vector."""
        return self.joint_positions


_DIM_THRESHOLD_IMAGE: int = 2

_POSITION_KEY_SUFFIX: str = ".pos"
_POSITION_KEY_LONG_SUFFIX: str = "_position"
_POSITION_KEY_FALLBACK_SUFFIX: str = "pos"
_ALLOWED_CONFIG_PACKAGE_ROOTS: frozenset[str] = frozenset({
    "lerobot.robots",
    "lerobot.teleoperators",
})


def _strip_position_suffix(key: str) -> str:
    """Strip a trailing position marker to derive the joint name.

    Prefers the canonical ``.pos`` suffix, then falls back to ``_position`` or
    ``pos`` so auto-detection stays correct for varied observation keys.

    Returns:
        The joint name with the trailing position marker removed.
    """
    if key.endswith(_POSITION_KEY_SUFFIX):
        return key[: -len(_POSITION_KEY_SUFFIX)]
    if key.endswith(_POSITION_KEY_LONG_SUFFIX):
        return key[: -len(_POSITION_KEY_LONG_SUFFIX)]
    if key.endswith(_POSITION_KEY_FALLBACK_SUFFIX):
        return key[: -len(_POSITION_KEY_FALLBACK_SUFFIX)]
    return key


def _collect_device_ports(value: object) -> list[str]:
    """Recursively collect ``port`` string values from nested config data.

    Args:
        value: Config kwargs or a nested container/dataclass within them.

    Returns:
        All non-empty ``port`` string values found at any nesting depth.
    """
    ports: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "port" and isinstance(item, str) and item:
                ports.append(item)
            ports.extend(_collect_device_ports(item))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            ports.extend(_collect_device_ports(item))
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            ports.extend(_collect_device_ports(getattr(value, field.name)))
    return ports


def _is_allowed_dynamic_import(module_name: str) -> bool:
    """Return whether a module name is trusted for dynamic importing."""
    return module_name.startswith("lerobot.") or (
        trust_unverified_plugins() and module_name.startswith(("lerobot_robot_", "lerobot_teleoperator_"))
    )


def _device_ids(config_kwargs: dict[str, Any]) -> tuple[str, ...]:
    """Return stable serial-device identities without touching hardware."""
    ports = sorted(set(_collect_device_ports(config_kwargs)))
    return tuple(device_id_from_serial_port(port) for port in ports)


def _import_config_modules(package_name: str) -> None:
    """Import LeRobot config modules so their registered types are available after spawn.

    Raises:
        ValueError: If a non-whitelisted package root is requested.
    """
    if package_name not in _ALLOWED_CONFIG_PACKAGE_ROOTS:
        msg = f"Unsupported package root for config imports: {package_name!r}"
        raise ValueError(msg)

    if package_name == "lerobot.robots":
        package = importlib.import_module("lerobot.robots")
    else:
        package = importlib.import_module("lerobot.teleoperators")

    for _importer, module_name, is_package in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}."):
        if "config" in module_name and not is_package and _is_allowed_dynamic_import(module_name):
            # The exact package root and discovered module prefix are allowlisted.
            # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
            importlib.import_module(module_name)


def _register_third_party_plugins() -> None:
    if not trust_unverified_plugins():
        return
    from lerobot.utils.import_utils import register_third_party_plugins  # noqa: PLC0415

    register_third_party_plugins()


def _robot_config_class(type_name: str) -> type:
    _import_config_modules("lerobot.robots")
    _register_third_party_plugins()
    from lerobot.robots.config import RobotConfig  # noqa: PLC0415

    choices = RobotConfig.get_known_choices()
    logger.info("Resolving LeRobot robot config type={!r}; available types={}", type_name, sorted(choices))
    return choices[type_name]


def _teleoperator_config_class(type_name: str) -> type:
    _import_config_modules("lerobot.teleoperators")
    _register_third_party_plugins()
    from lerobot.teleoperators.config import TeleoperatorConfig  # noqa: PLC0415

    choices = TeleoperatorConfig.get_known_choices()
    logger.info("Resolving LeRobot teleoperator config type={!r}; available types={}", type_name, sorted(choices))
    return choices[type_name]


@export_config(
    class_path="physicalai_lerobot_plugin.lerobot_adapter.LeRobotAdapter",
)
class LeRobotAdapter:
    """Wraps a lerobot Robot into PhysicalAI's Robot protocol.

    The lerobot ``Robot`` instance is NOT created at construction time — only
    the config class and its kwargs are stored.  This makes the adapter safe
    for multiprocessing (``dynamixel-sdk`` ``PortHandler`` objects cannot be
    pickled).  The actual lerobot robot is created lazily inside
    ``connect()``.

    If a pre-built robot is available (e.g. when the builder runs in the main
    process) it can be passed as the private ``_robot`` parameter, which also
    triggers eager joint-order discovery from ``observation_features``.
    """

    def __init__(
        self,
        config_type: str,
        config_kwargs: dict[str, Any],
        *,
        role: Literal["leader", "follower"] = "follower",
        _robot: object | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            config_type: Registered LeRobot ``RobotConfig`` type name.
            config_kwargs: Resolved keyword-args for the config dataclass.
            role: ``"follower"`` (full control) or ``"leader"`` (read-only).

        Raises:
            ValueError: If role is invalid.
        """
        if role not in VALID_ROLES:
            msg = f"Invalid role {role!r}. Must be one of {sorted(VALID_ROLES)}."
            raise ValueError(msg)

        self._config_type = config_type
        self._config_kwargs = config_kwargs
        self._role = role
        self._robot: _LeRobotLike | None = cast("_LeRobotLike | None", _robot)
        self._joint_order: list[str] | None = None
        self._obs_position_keys: list[str] | None = None
        self._act_position_keys: list[str] | None = None
        self._num_joints: int | None = None
        self._image_sequences: dict[str, int] = {}

        if _robot is not None:
            features: Any = cast("_LeRobotLike", _robot).observation_features
            if isinstance(features, dict):
                self._ensure_joint_order(features)

    def __getstate__(self) -> dict[str, object]:
        """Return pickle-safe state without the live lerobot instance."""
        return {
            "_config_type": self._config_type,
            "_config_kwargs": self._config_kwargs,
            "_role": self._role,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore state and reset runtime-only members after unpickling.

        Raises:
            TypeError: If serialized state values have unexpected types.
        """
        config_type = state.get("_config_type")
        config_kwargs = state.get("_config_kwargs")
        role = state.get("_role")
        if not isinstance(config_type, str):
            msg = "Invalid state: _config_type must be a string"
            raise TypeError(msg)
        if not isinstance(config_kwargs, dict):
            msg = "Invalid state: _config_kwargs must be a mapping"
            raise TypeError(msg)
        if role not in VALID_ROLES:
            msg = "Invalid state: _role must be 'leader' or 'follower'"
            raise TypeError(msg)
        self._config_type = config_type
        self._config_kwargs = cast("dict[str, Any]", config_kwargs)
        self._role = cast("Literal['leader', 'follower']", role)
        self._robot = None
        self._joint_order = None
        self._obs_position_keys = None
        self._act_position_keys = None
        self._num_joints = None
        self._image_sequences = {}

    @staticmethod
    def _physicalai_normalize_captured_init_args(init_args: dict[str, object]) -> None:
        """Exclude the live device injected by the catalog builder from config exports."""
        init_args.pop("_robot", None)

    def _ensure_robot(self) -> None:
        if self._robot is not None:
            logger.info("Using pre-built LeRobot robot for config type={!r}: {!r}", self._config_type, self._robot)
            return
        from lerobot.robots import make_robot_from_config  # noqa: PLC0415

        logger.info("Building LeRobot robot config type={!r} with kwargs={!r}", self._config_type, self._config_kwargs)
        lerobot_config = _robot_config_class(self._config_type)(**self._config_kwargs)
        logger.info("Built LeRobot robot config: {!r}", lerobot_config)
        self._robot = cast("_LeRobotLike", make_robot_from_config(lerobot_config))
        logger.info("Built LeRobot robot: {!r}", self._robot)

    def _ensure_joint_order(self, obs: dict[str, Any]) -> None:
        """Discover joint order from a lerobot observation dict.

        Keys ending with ``.pos`` are treated as joint position keys, sorted
        alphabetically, and stripped of the suffix to produce the joint names.

        Args:
            obs: The lerobot observation dict.
        """
        if self._joint_order is not None:
            return
        pos_keys = sorted(k for k in obs if k.endswith(_POSITION_KEY_SUFFIX))
        if not pos_keys:
            pos_keys = sorted(k for k in obs if "position" in k.lower() or k.endswith("pos"))
        self._joint_order = [_strip_position_suffix(k) for k in pos_keys]
        self._num_joints = len(self._joint_order)
        self._obs_position_keys = list(pos_keys)
        self._act_position_keys = list(pos_keys)

    def _require_joint_order(self) -> None:
        """Raise if joint order has not been discovered yet.

        Raises:
            RuntimeError: If ``connect()`` has not been called.
        """
        if self._joint_order is None:
            msg = "Joint order not yet discovered. Call connect() first or ensure the robot has been observed."
            raise RuntimeError(msg)

    def _require_position_keys(self) -> tuple[list[str], list[str]]:
        self._require_joint_order()
        obs_keys = self._obs_position_keys
        act_keys = self._act_position_keys
        if obs_keys is None or act_keys is None:
            msg = "Joint position keys not yet discovered. Call connect() first."
            raise RuntimeError(msg)
        return obs_keys, act_keys

    @property
    def NUM_JOINTS(self) -> int:  # noqa: N802
        """Number of joints (available after first observation)."""
        self._require_joint_order()
        return self._num_joints  # type: ignore[return-value]

    @property
    def joint_names(self) -> list[str]:
        """Ordered joint names matching the position/action array."""
        self._require_joint_order()
        return self._joint_order  # type: ignore[return-value]

    @property
    def robot(self) -> _LeRobotLike | None:
        """The wrapped lerobot Robot instance (may be None before connect)."""
        return self._robot

    @property
    def device_ids(self) -> tuple[str, ...]:
        """Stable identities of serial devices exclusively owned by this robot."""
        return _device_ids(self._config_kwargs)

    def connect(self) -> None:
        """Open the connection and discover joint order.

        Idempotent — no-op if already connected.
        Creates the lerobot robot from stored config on first call.

        Raises:
            ConnectionError: If the connection or initial observation fails.
        """
        if self.is_connected():
            return
        try:
            self._connect_robot()
        except Exception as e:
            self.disconnect()
            msg = f"Failed to connect LeRobot {self._robot}: {e}"
            raise ConnectionError(msg) from e

    def _connect_robot(self) -> None:
        """Build (if needed), connect, and observe the LeRobot robot.

        Raises:
            RuntimeError: If robot construction did not initialize the robot instance.
        """
        self._ensure_robot()
        logger.info("Connecting LeRobot robot config type={!r}", self._config_type)
        robot = self._robot
        if robot is None:
            msg = "Robot is not initialized"
            raise RuntimeError(msg)
        robot.connect(calibrate=True)
        logger.info("Connected LeRobot robot config type={!r}; reading initial observation", self._config_type)
        obs = robot.get_observation()
        self._ensure_joint_order(obs)
        logger.info("LeRobot robot config type={!r} ready with joints={}", self._config_type, self._joint_order)

    def disconnect(self) -> None:
        """Close the connection. Safe to call multiple times."""
        if not self.is_connected():
            return
        try:
            robot = self._robot
            if robot is None:
                return
            robot.disconnect()
        except Exception:  # noqa: BLE001
            logger.exception("Error during LeRobot disconnect")

    def is_connected(self) -> bool:
        """Check if the robot is currently connected.

        Returns:
            True if connected.
        """
        if self._robot is None:
            return False
        robot = self._robot
        if robot is None:
            return False
        return robot.is_connected

    def get_observation(self) -> RobotObservation:
        """Read current joint positions and return an observation.

        Joint order is auto-detected on the first call.

        Returns:
            LeRobotAdapterObservation with joint positions and sensor data.

        Raises:
            ConnectionError: If no robot instance is connected.
        """
        robot = self._robot
        if robot is None:
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)
        lerobot_obs = robot.get_observation()
        self._ensure_joint_order(lerobot_obs)
        self._require_joint_order()

        obs_keys, act_keys = self._require_position_keys()

        positions = np.array([lerobot_obs[key] for key in obs_keys], dtype=np.float32)

        sensor_data: dict[str, np.ndarray] = {}
        images: dict[str, Frame] = {}
        obs_key_set = set(obs_keys)
        act_key_set = set(act_keys)
        for key, value in lerobot_obs.items():
            if key in obs_key_set or key in act_key_set:
                continue
            if isinstance(value, np.ndarray) and value.ndim >= _DIM_THRESHOLD_IMAGE:
                sequence = self._image_sequences.get(key, 0)
                self._image_sequences[key] = sequence + 1
                images[key] = Frame(data=cast("Any", value), timestamp=time.monotonic(), sequence=sequence)
            elif isinstance(value, np.ndarray):
                sensor_data[key] = value
            elif isinstance(value, (int, float)):
                sensor_data[key] = np.array([value], dtype=np.float32)

        return LeRobotAdapterObservation(
            joint_positions=positions,
            timestamp=time.monotonic(),
            sensor_data=sensor_data or None,
            images=images or None,
        )

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Send a joint command to the robot.

        Args:
            action: Array of shape ``(N,)`` matching ``joint_names`` order.
            goal_time: Time to reach the goal in seconds (ignored).

        Raises:
            RuntimeError: If called in leader role or before ``connect()``.
            ValueError: If action has incorrect shape.
            ConnectionError: If no robot instance is connected.
        """
        _ = goal_time
        if self._role == "leader":
            msg = "Cannot send actions to a leader arm."
            raise RuntimeError(msg)

        self._require_joint_order()
        if action.shape != (self._num_joints,):
            msg = f"Expected action shape ({self._num_joints},), got {action.shape}"
            raise ValueError(msg)

        action_dict: dict[str, Any] = {}
        _, act_keys = self._require_position_keys()
        for i, key in enumerate(act_keys):
            action_dict[key] = float(action[i])

        robot = self._robot
        if robot is None or not robot.is_connected:
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)
        robot.send_action(action_dict)


@export_config(
    class_path="physicalai_lerobot_plugin.lerobot_adapter.LeRobotTeleoperatorAdapter",
)
class LeRobotTeleoperatorAdapter:
    """Wraps a lerobot ``Teleoperator`` into PhysicalAI's ``Robot`` protocol.

    Teleoperators produce actions (e.g. from a human guiding a leader arm) and
    can receive feedback.  This adapter maps:
    * ``teleoperator.get_action()`` → ``get_observation()``
    * ``teleoperator.send_feedback()`` → ``send_action()`` (follower role only)

    In leader role the adapter exposes the teleoperator's state as a read-only
    observation source — ``send_action()`` raises ``RuntimeError``.

    The actual ``Teleoperator`` instance is NOT created at construction time —
    only the config class and kwargs are stored, making the adapter safe for
    multiprocessing.
    """

    def __init__(
        self,
        config_type: str,
        config_kwargs: dict[str, Any],
        *,
        role: Literal["leader", "follower"] = "leader",
        _teleoperator: object | None = None,
    ) -> None:
        """Initialize the teleoperator adapter.

        Args:
            config_type: Registered LeRobot ``TeleoperatorConfig`` type name.
            config_kwargs: Resolved keyword-args for the config dataclass.
            role: ``"leader"`` (read-only, default) or ``"follower"`` (full).

        Raises:
            ValueError: If role is invalid.
        """
        if role not in VALID_ROLES:
            msg = f"Invalid role {role!r}. Must be one of {sorted(VALID_ROLES)}."
            raise ValueError(msg)

        self._config_type = config_type
        self._config_kwargs = config_kwargs
        self._role = role
        self._teleoperator: _TeleoperatorLike | None = cast("_TeleoperatorLike | None", _teleoperator)
        self._joint_order: list[str] | None = None
        self._action_position_keys: list[str] | None = None
        self._num_joints: int | None = None
        self._image_sequences: dict[str, int] = {}

        if _teleoperator is not None:
            features: Any = cast("_TeleoperatorLike", _teleoperator).action_features
            if isinstance(features, dict):
                self._ensure_joint_order(features)

    def __getstate__(self) -> dict[str, object]:
        """Return pickle-safe state without the live teleoperator instance."""
        return {
            "_config_type": self._config_type,
            "_config_kwargs": self._config_kwargs,
            "_role": self._role,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore state and reset runtime-only members after unpickling.

        Raises:
            TypeError: If serialized state values have unexpected types.
        """
        config_type = state.get("_config_type")
        config_kwargs = state.get("_config_kwargs")
        role = state.get("_role")
        if not isinstance(config_type, str):
            msg = "Invalid state: _config_type must be a string"
            raise TypeError(msg)
        if not isinstance(config_kwargs, dict):
            msg = "Invalid state: _config_kwargs must be a mapping"
            raise TypeError(msg)
        if role not in VALID_ROLES:
            msg = "Invalid state: _role must be 'leader' or 'follower'"
            raise TypeError(msg)
        self._config_type = config_type
        self._config_kwargs = cast("dict[str, Any]", config_kwargs)
        self._role = cast("Literal['leader', 'follower']", role)
        self._teleoperator = None
        self._joint_order = None
        self._action_position_keys = None
        self._num_joints = None
        self._image_sequences = {}

    @staticmethod
    def _physicalai_normalize_captured_init_args(init_args: dict[str, object]) -> None:
        """Exclude the live device injected by the catalog builder from config exports."""
        init_args.pop("_teleoperator", None)

    def _ensure_teleoperator(self) -> None:
        if self._teleoperator is not None:
            logger.info(
                "Using pre-built LeRobot teleoperator for config type={!r}: {!r}",
                self._config_type,
                self._teleoperator,
            )
            return
        from lerobot.teleoperators import make_teleoperator_from_config  # noqa: PLC0415

        logger.info(
            "Building LeRobot teleoperator config type={!r} with kwargs={!r}",
            self._config_type,
            self._config_kwargs,
        )
        teleop_config = _teleoperator_config_class(self._config_type)(**self._config_kwargs)
        logger.info("Built LeRobot teleoperator config: {!r}", teleop_config)
        self._teleoperator = cast("_TeleoperatorLike", make_teleoperator_from_config(teleop_config))
        logger.info("Built LeRobot teleoperator: {!r}", self._teleoperator)

    def _ensure_joint_order(self, action: dict[str, Any]) -> None:
        if self._joint_order is not None:
            return
        pos_keys = sorted(k for k in action if k.endswith(_POSITION_KEY_SUFFIX))
        action_keys = pos_keys or sorted(action)
        self._joint_order = [_strip_position_suffix(k) for k in action_keys]
        self._num_joints = len(self._joint_order)
        self._action_position_keys = action_keys

    def _require_joint_order(self) -> None:
        if self._joint_order is None:
            msg = "Joint order not yet discovered. Call connect() first."
            raise RuntimeError(msg)

    def _require_action_keys(self) -> list[str]:
        self._require_joint_order()
        action_keys = self._action_position_keys
        if action_keys is None:
            msg = "Joint action keys not yet discovered. Call connect() first."
            raise RuntimeError(msg)
        return action_keys

    @property
    def NUM_JOINTS(self) -> int:  # noqa: N802
        """Number of joints (available after first observation)."""
        self._require_joint_order()
        return self._num_joints  # type: ignore[return-value]

    @property
    def joint_names(self) -> list[str]:
        """Ordered joint names matching the position/action array."""
        self._require_joint_order()
        return self._joint_order  # type: ignore[return-value]

    @property
    def robot(self) -> _TeleoperatorLike | None:
        """The wrapped lerobot Teleoperator instance (may be None before connect)."""
        return self._teleoperator

    @property
    def device_ids(self) -> tuple[str, ...]:
        """Stable identities of serial devices exclusively owned by this teleoperator."""
        return _device_ids(self._config_kwargs)

    def connect(self) -> None:
        """Open the connection and discover joint order.

        Idempotent — no-op if already connected.
        Creates the teleoperator from stored config on first call.

        Raises:
            ConnectionError: If the connection or initial observation fails.
        """
        if self.is_connected():
            return
        try:
            self._connect_teleoperator()
        except Exception as e:
            self.disconnect()
            msg = f"Failed to connect LeRobot teleoperator {self._teleoperator}: {e}"
            raise ConnectionError(msg) from e

    def _connect_teleoperator(self) -> None:
        """Build (if needed), connect, and read the initial teleoperator action.

        Raises:
            RuntimeError: If teleoperator construction did not initialize the device instance.
        """
        self._ensure_teleoperator()
        logger.info("Connecting LeRobot teleoperator config type={!r}", self._config_type)
        teleop = self._teleoperator
        if teleop is None:
            msg = "Teleoperator is not initialized"
            raise RuntimeError(msg)
        teleop.connect()
        logger.info(
            "Connected LeRobot teleoperator config type={!r}; reading initial action",
            self._config_type,
        )
        action = teleop.get_action()
        self._ensure_joint_order(action)
        logger.info(
            "LeRobot teleoperator config type={!r} ready with joints={}",
            self._config_type,
            self._joint_order,
        )

    def disconnect(self) -> None:
        """Close the connection. Safe to call multiple times."""
        if not self.is_connected():
            return
        try:
            teleop = self._teleoperator
            if teleop is None:
                return
            teleop.disconnect()
        except Exception:  # noqa: BLE001
            logger.exception("Error during LeRobot teleoperator disconnect")

    def is_connected(self) -> bool:
        """Check if the teleoperator is currently connected.

        Returns:
            True if connected.
        """
        if self._teleoperator is None:
            return False
        teleop = self._teleoperator
        if teleop is None:
            return False
        return teleop.is_connected

    def get_observation(self) -> RobotObservation:
        """Read current teleoperator action and return as observation.

        Joint order is auto-detected on the first call.

        Returns:
            LeRobotAdapterObservation with joint positions and sensor data.

        Raises:
            ConnectionError: If no teleoperator instance is connected.
        """
        teleop = self._teleoperator
        if teleop is None:
            msg = "Teleoperator is not connected. Call connect() first."
            raise ConnectionError(msg)
        action = teleop.get_action()
        self._ensure_joint_order(action)
        self._require_joint_order()

        action_keys = self._require_action_keys()
        positions = np.array([action[key] for key in action_keys], dtype=np.float32)

        sensor_data: dict[str, np.ndarray] = {}
        images: dict[str, Frame] = {}
        key_set = set(action_keys)
        for key, value in action.items():
            if key in key_set:
                continue
            if isinstance(value, np.ndarray) and value.ndim >= _DIM_THRESHOLD_IMAGE:
                sequence = self._image_sequences.get(key, 0)
                self._image_sequences[key] = sequence + 1
                images[key] = Frame(data=cast("Any", value), timestamp=time.monotonic(), sequence=sequence)
            elif isinstance(value, np.ndarray):
                sensor_data[key] = value
            elif isinstance(value, (int, float)):
                sensor_data[key] = np.array([value], dtype=np.float32)

        return LeRobotAdapterObservation(
            joint_positions=positions,
            timestamp=time.monotonic(),
            sensor_data=sensor_data or None,
            images=images or None,
        )

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Send a feedback command to the teleoperator.

        In leader role this raises ``RuntimeError`` (read-only).
        In follower role the action is forwarded as feedback to the teleoperator.

        Args:
            action: Array of shape ``(N,)`` matching ``joint_names`` order.
            goal_time: Time to reach the goal in seconds (ignored).

        Raises:
            RuntimeError: If called in leader role or before ``connect()``.
            ValueError: If action has incorrect shape.
            ConnectionError: If no teleoperator instance is connected.
        """
        _ = goal_time
        if self._role == "leader":
            msg = "Cannot send actions to a leader arm."
            raise RuntimeError(msg)

        self._require_joint_order()
        if action.shape != (self._num_joints,):
            msg = f"Expected action shape ({self._num_joints},), got {action.shape}"
            raise ValueError(msg)

        feedback_dict: dict[str, Any] = {}
        action_keys = self._require_action_keys()
        for i, key in enumerate(action_keys):
            feedback_dict[key] = float(action[i])

        teleop = self._teleoperator
        if teleop is None or not teleop.is_connected:
            msg = "Teleoperator is not connected. Call connect() first."
            raise ConnectionError(msg)
        teleop.send_feedback(feedback_dict)
