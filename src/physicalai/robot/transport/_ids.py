# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Robot id derivation and Zenoh key builders.

The ``robot_id`` keys the Zenoh topics. It must be deterministic from the
connection parameters so a second same-machine instance re-derives the same
key and attaches to the existing owner instead of spawning a competing one.
"""

from __future__ import annotations

import hashlib
import socket
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

KEY_PREFIX = "physicalai/robot"

META_WILDCARD = f"{KEY_PREFIX}/**/meta"
"""Selector enumerating the ``/meta`` queryable of every reachable robot."""


def derive_device_id(robot_kwargs: Mapping[str, object]) -> str:
    """Derive a device id from robot construction kwargs.

    Serial robots (SO-101) use the symlink-resolved basename of ``port`` so
    that ``/dev/ttyUSB0`` and ``/dev/serial/by-id/...`` map to the same id.
    Network robots (Trossen) use ``ip``. ``role`` is intentionally excluded —
    the id keys on the physical connection, not leader/follower.

    Args:
        robot_kwargs: Constructor kwargs containing ``port`` or ``ip``.

    Returns:
        A device id string suitable for key and lock-file naming.

    Raises:
        ValueError: If neither ``port`` nor ``ip`` is present.
    """
    port = robot_kwargs.get("port")
    if isinstance(port, str) and port:
        if port.startswith("/dev/"):
            return Path(port).resolve().name
        return Path(port).name

    ip = robot_kwargs.get("ip")
    if isinstance(ip, str) and ip:
        return ip

    msg = "cannot derive a device id: robot kwargs contain neither 'port' nor 'ip'"
    raise ValueError(msg)


def derive_robot_id(
    robot_type: str,
    robot_kwargs: Mapping[str, object],
    *,
    robot_id: str | None = None,
    host: str | None = None,
) -> str:
    """Derive the Zenoh robot id: ``physicalai/robot/{type}/{host}/{device_id}``.

    Args:
        robot_type: Logical robot type (e.g. ``"so101"``).
        robot_kwargs: Constructor kwargs used for device-id derivation.
        robot_id: Explicit override; returned as-is (prefixed) when given.
            Useful for network-only, manually-launched-owner setups.
        host: Host component override. Defaults to the local hostname.

    Returns:
        The full robot id used as the Zenoh key prefix for this robot.
    """
    if robot_id is not None:
        if robot_id.startswith(f"{KEY_PREFIX}/"):
            return robot_id
        return f"{KEY_PREFIX}/{robot_id}"
    host = host or socket.gethostname()
    device_id = derive_device_id(robot_kwargs)
    return f"{KEY_PREFIX}/{robot_type}/{host}/{device_id}"


_PORT_BASE = 17000
_PORT_RANGE = 1000


def derive_endpoint_port(robot_id: str) -> int:
    """Deterministic TCP port for the owner's Zenoh listen endpoint.

    Multicast scouting is not available on every host (e.g. macOS local
    network privacy, locked-down LANs). A port derived from the robot id
    lets same-host subscribers connect deterministically without any
    discovery mechanism; Zenoh retries the endpoint in the background, so
    a subscriber session opened before the owner exists attaches as soon
    as the owner starts listening.

    Args:
        robot_id: The full robot id.

    Returns:
        A port in ``[17000, 17999]``.
    """
    digest = hashlib.sha256(robot_id.encode()).digest()
    return _PORT_BASE + int.from_bytes(digest[:4], "big") % _PORT_RANGE


def state_key(robot_id: str) -> str:
    """Key for the owner-to-subscribers state stream.

    Returns:
        The ``{robot_id}/state`` key expression.
    """
    return f"{robot_id}/state"


def action_key(robot_id: str) -> str:
    """Key for the subscribers-to-owner action stream.

    Returns:
        The ``{robot_id}/action`` key expression.
    """
    return f"{robot_id}/action"


def meta_key(robot_id: str) -> str:
    """Key for the owner-answered discovery/validation queryable.

    Returns:
        The ``{robot_id}/meta`` key expression.
    """
    return f"{robot_id}/meta"
