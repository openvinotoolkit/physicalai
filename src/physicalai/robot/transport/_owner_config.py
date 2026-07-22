# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Serializable owner construction config for shared robots.

Named ``RobotOwnerConfig`` (not ``RobotSpec``) to avoid colliding with the
unrelated ``physicalai.inference.manifest.RobotSpec`` (a manifest-schema
pydantic model). Used by the foreground serve path and by the owner
subprocess stdin handshake.

The owner must construct the robot driver itself: a live serial/socket
handle cannot cross a process boundary (D15). Only an importable
``robot_class`` plus JSON-serializable ``robot_kwargs`` survive that
boundary — arbitrary robot types, including third-party plugins, work
without any registry lookup here.

Security: *robot_class* is trusted local application/config input, exactly
like a jsonargparse ``class_path`` (``docs/development/security.md`` rules
4, 9, 11). It must never originate from network-received data (e.g. a
``/metadata`` payload) — that would let an untrusted peer choose an
arbitrary module to import.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ._importing import import_dotted_path

if TYPE_CHECKING:
    from physicalai.robot.interface import Robot

DEFAULT_RATE_HZ = 100.0
"""Owner loop rate used when no per-instance override is given.

Bounds action pickup to ~10 ms when the driver keeps pace; a slower
blocking driver naturally runs below this without accumulating scheduling
debt. Override per instance with ``rate_hz`` when hardware measurements
justify a different value for a specific robot class.
"""


def normalize_robot_class(robot_class: type | str) -> str:
    """Normalize a robot class reference to an importable dotted path.

    An explicit string path is trusted and returned unchanged. A class
    object is converted via ``cls.__module__ + "." + cls.__qualname__``,
    because Python does not retain the re-export path through which a
    class happened to be imported by the caller.

    Args:
        robot_class: A class object, or its dotted import path as a string.

    Returns:
        The normalized dotted path.

    Raises:
        TypeError: If *robot_class* is neither a string nor a class.
        ValueError: If a class object is a local class (defined inside a
            function) — those have no stable import path and cannot be
            reconstructed in the owner subprocess.
    """
    if isinstance(robot_class, str):
        return robot_class
    if not isinstance(robot_class, type):
        msg = f"robot_class must be a class or a dotted path string, got {type(robot_class).__name__}"
        raise TypeError(msg)

    path = f"{robot_class.__module__}.{robot_class.__qualname__}"
    if "<locals>" in robot_class.__qualname__:
        msg = f"robot_class {path!r} is a local class and cannot be auto-spawned; give it a module-level definition"
        raise ValueError(msg)
    return path


@dataclass(frozen=True)
class RobotOwnerConfig:
    """Everything the owner needs to construct and run a robot.

    Warning:
        ``allow_remote=True`` exposes an unauthenticated physical ``/action``
        endpoint beyond localhost. Any peer that can reach the owner's Zenoh
        session can move the robot. Use only on an isolated robot-cell
        network (VLAN/firewall) or with Zenoh ACL/TLS.

    Attributes:
        name: The robot's logical name (keys the Zenoh topics).
        robot_class: Normalized importable dotted path to the driver class.
        robot_kwargs: JSON-serializable keyword arguments forwarded to the
            driver constructor (e.g. ``calibration`` as a file path).
        allow_remote: Whether the owner's Zenoh session is reachable beyond
            localhost. Fixed for the owner's lifetime once spawned.
            ``True`` exposes an unauthenticated physical ``/action`` endpoint
            — use only on an isolated robot-cell network or with Zenoh
            ACL/TLS. Default ``False`` keeps the owner unreachable off-host.
        rate_hz: Owner loop rate.
        idle_timeout: Seconds with zero subscribers before self-exit.
    """

    name: str
    robot_class: str
    robot_kwargs: dict[str, Any] = field(default_factory=dict)
    allow_remote: bool = False
    rate_hz: float = DEFAULT_RATE_HZ
    idle_timeout: float | None = 10.0

    def __post_init__(self) -> None:
        """Validate ``rate_hz`` and that ``robot_kwargs`` is JSON-serializable.

        Raises:
            ValueError: If ``rate_hz`` is not finite and positive, or
                ``robot_kwargs`` contains non-JSON-serializable values.
        """
        if (
            isinstance(self.rate_hz, bool)
            or not isinstance(self.rate_hz, (int, float))
            or not math.isfinite(self.rate_hz)
            or self.rate_hz <= 0
        ):
            msg = f"rate_hz must be finite and greater than zero, got {self.rate_hz!r}"
            raise ValueError(msg)
        if self.idle_timeout is not None and (
            isinstance(self.idle_timeout, bool)
            or not isinstance(self.idle_timeout, (int, float))
            or not math.isfinite(self.idle_timeout)
            or self.idle_timeout <= 0
        ):
            msg = f"idle_timeout must be finite and greater than zero, got {self.idle_timeout!r}"
            raise ValueError(msg)
        from ._ids import validate_name  # noqa: PLC0415

        validate_name(self.name)
        if not isinstance(self.robot_class, str) or not self.robot_class.strip() or "." not in self.robot_class:
            msg = f"robot_class must be a nonempty dotted path, got {self.robot_class!r}"
            raise ValueError(msg)
        try:
            json.dumps(self.robot_kwargs)
        except TypeError as exc:
            msg = f"robot_kwargs must be JSON-serializable (e.g. paths as str, not objects): {exc}"
            raise ValueError(msg) from exc

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary for the stdin handshake.

        Returns:
            Dictionary with every dataclass field.
        """
        return {
            "name": self.name,
            "robot_class": self.robot_class,
            "robot_kwargs": self.robot_kwargs,
            "allow_remote": self.allow_remote,
            "rate_hz": self.rate_hz,
            "idle_timeout": self.idle_timeout,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> RobotOwnerConfig:
        """Deserialize from a JSON dictionary.

        Args:
            data: Dictionary produced by :meth:`to_json_dict`.

        Returns:
            A new :class:`RobotOwnerConfig` instance.
        """
        return cls(
            name=data["name"],
            robot_class=data["robot_class"],
            robot_kwargs=data.get("robot_kwargs", {}),
            allow_remote=data.get("allow_remote", False),
            rate_hz=data.get("rate_hz", DEFAULT_RATE_HZ),
            idle_timeout=data.get("idle_timeout", 10.0),
        )

    def build(self) -> Robot:
        """Instantiate the robot driver described by this config.

        Returns:
            A new, not-yet-connected driver instance.

        Raises:
            TypeError: If ``robot_class`` does not resolve to a class.
        """
        cls = import_dotted_path(self.robot_class)
        if not isinstance(cls, type):
            msg = f"robot_class {self.robot_class!r} does not resolve to a class (got {type(cls).__name__})"
            raise TypeError(msg)
        return cls(**self.robot_kwargs)
