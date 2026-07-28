# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Serializable owner construction config for shared robots.

Named ``RobotOwnerConfig`` (not ``RobotSpec``) to avoid colliding with the
unrelated ``physicalai.inference.manifest.RobotSpec`` (a manifest-schema
pydantic model). Used by the foreground serve path and by the owner
subprocess stdin handshake.

The owner must construct the robot driver itself: a live serial/socket
handle cannot cross a process boundary (D15). Only a local
:class:`~physicalai.config.ComponentConfig` (``class_path`` + ``init_args``)
survives that boundary — arbitrary robot types, including third-party
plugins, work without any registry lookup here.

Private stdin is ``robot: ComponentConfig`` only. The owner envelope is
validated schema-positively: required ``robot``, known transport keys, and
rejection of unknown keys before import or hardware access. Public
``SharedRobot`` and ``physicalai robot serve`` use the same
``robot`` / ``--robot`` shape.

Security: ``class_path`` is local application/config input, exactly like a
jsonargparse ``class_path`` (``docs/development/security.md`` rules 4, 9,
11). It must never originate from network-received data (e.g. a
``/metadata`` payload) — that would let a peer choose an arbitrary module to
import.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from physicalai.config import (
    ComponentConfig,
    instantiate,
    normalize_component_config,
    validate_envelope,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from physicalai.robot.interface import Robot

DEFAULT_RATE_HZ = 100.0
"""Owner loop rate used when no per-instance override is given.

Bounds action pickup to ~10 ms when the driver keeps pace; a slower
blocking driver naturally runs below this without accumulating scheduling
debt. Override per instance with ``rate_hz`` when hardware measurements
justify a different value for a specific robot class.
"""

# Allowed keys on owner stdin handshake payloads. Everything else is an
# unknown-key schema error.
# Keep this the single allowlist — :meth:`RobotOwnerConfig.from_json_dict`
# and any future reconfigure path should share :func:`validate_owner_config`.
_OWNER_ENVELOPE_KEYS = frozenset({
    "name",
    "robot",
    "allow_remote",
    "rate_hz",
    "idle_timeout",
})


def validate_owner_config(data: Mapping[str, Any]) -> Mapping[str, object]:
    """Validate an owner stdin payload schema-positively.

    Returns:
        The validated ``robot`` ComponentConfig mapping (see
        :func:`normalize_robot_config` for the JSON-serializability check).
    """
    return validate_envelope(
        data,
        component_key="robot",
        allowed_keys=_OWNER_ENVELOPE_KEYS,
        envelope_name="owner",
    )


def normalize_robot_config(robot: Mapping[str, object]) -> ComponentConfig:
    """Validate a robot ComponentConfig without importing its ``class_path``.

    Returns:
        A validated config whose ``class_path`` is a dotted import path.
    """
    return normalize_component_config(
        robot,
        component_key="robot",
        class_label="robot_class",
        json_hint=" (e.g. paths as str, not objects)",
    )


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
        robot: Trusted construction config (``class_path`` + ``init_args``).
        allow_remote: Whether the owner's Zenoh session is reachable beyond
            localhost. Fixed for the owner's lifetime once spawned.
            ``True`` exposes an unauthenticated physical ``/action`` endpoint
            — use only on an isolated robot-cell network or with Zenoh
            ACL/TLS. Default ``False`` keeps the owner unreachable off-host.
        rate_hz: Owner loop rate.
        idle_timeout: Seconds with zero subscribers before self-exit.
    """

    name: str
    robot: Mapping[str, object]
    allow_remote: bool = False
    rate_hz: float = DEFAULT_RATE_HZ
    idle_timeout: float | None = 10.0

    def __post_init__(self) -> None:
        """Validate transport fields and normalize ``robot`` to a public ComponentConfig.

        Raises:
            ValueError: If ``rate_hz`` / ``idle_timeout`` are invalid, or
                ``robot`` is not a JSON-serializable ComponentConfig.
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
        object.__setattr__(self, "robot", normalize_robot_config(self.robot))

    @property
    def robot_class(self) -> str:
        """Public ``class_path`` advertised on network metadata as ``robot_class``."""
        return str(self.robot["class_path"])

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary for the stdin handshake.

        Returns:
            Dictionary with every dataclass field (new ``robot:`` shape only).

        Raises:
            TypeError: If ``robot.init_args`` is not a mapping.
        """
        init_args = self.robot["init_args"]
        if not isinstance(init_args, dict):
            msg = f"robot.init_args must be a mapping, got {type(init_args).__name__}"
            raise TypeError(msg)
        return {
            "name": self.name,
            "robot": {
                "class_path": self.robot["class_path"],
                "init_args": dict(init_args),
            },
            "allow_remote": self.allow_remote,
            "rate_hz": self.rate_hz,
            "idle_timeout": self.idle_timeout,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> RobotOwnerConfig:
        """Deserialize from a JSON dictionary.

        Uses :func:`validate_owner_config` so the owner envelope is validated
        schema-positively (required ``robot``, known transport keys, unknown
        keys rejected) before any import or hardware access.

        Args:
            data: Dictionary produced by :meth:`to_json_dict`.

        Returns:
            A new :class:`RobotOwnerConfig` instance.

        Raises:
            TypeError: If ``data`` is not a mapping.
            ValueError: If required ``name`` is missing or not a string.
        """
        if not isinstance(data, dict):
            msg = f"owner config must be a mapping, got {type(data).__name__}"
            raise TypeError(msg)
        if "name" not in data:
            msg = "owner config missing required 'name'"
            raise ValueError(msg)
        if not isinstance(data["name"], str):
            msg = f"owner config 'name' must be a string, got {type(data['name']).__name__}"
            raise TypeError(msg)

        robot = validate_owner_config(data)
        return cls(
            name=data["name"],
            robot=robot,
            allow_remote=data.get("allow_remote", False),
            rate_hz=data.get("rate_hz", DEFAULT_RATE_HZ),
            idle_timeout=data.get("idle_timeout", 10.0),
        )

    def build(self) -> Robot:
        """Instantiate the robot driver described by this config.

        Owner stdin is a parent→child local handshake only — never pass
        network metadata to this path. Uses :func:`physicalai.config.instantiate`
        on the ``robot`` ComponentConfig, then verifies the
        :class:`~physicalai.robot.Robot` protocol.

        Returns:
            A new, not-yet-connected driver instance.

        Raises:
            TypeError: If the instantiated object does not satisfy ``Robot``.
        """
        from physicalai.robot.interface import Robot  # noqa: PLC0415

        driver = instantiate(self.robot)  # type: ignore[arg-type]
        if not isinstance(driver, Robot):
            msg = f"{self.robot_class!r} does not satisfy the Robot protocol (got {type(driver).__name__})"
            raise TypeError(msg)
        return driver
