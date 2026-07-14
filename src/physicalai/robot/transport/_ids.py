# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Robot naming and Zenoh key builders.

``name`` is a required, caller-chosen logical identifier — it keys the
Zenoh topics directly and never needs a live driver instance to resolve
(unlike the superseded connection-derived ``robot_id``). Physical device
identity (:attr:`~physicalai.robot.interface.Robot.device_ids`) is a
separate concern, used only for host-local exclusivity locking by the
owner — see ``_lock.py``.
"""

from __future__ import annotations

import hashlib
import re
import socket

KEY_PREFIX = "physicalai/robot"

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
"""One safe Zenoh key segment: no ``/``, no wildcards (``*``/``**``/``$*``)."""

METADATA_WILDCARD = f"{KEY_PREFIX}/*/metadata"
"""Selector enumerating the ``/metadata`` queryable of every reachable robot."""

_PORT_BASE = 20000
_PORT_RANGE = 40000
"""Unprivileged range 20000-59999 for the deterministic local rendezvous port."""


def validate_name(name: str) -> str:
    """Validate a robot's logical name as one safe Zenoh key segment.

    Args:
        name: Caller-chosen logical identifier, e.g. ``"left-arm"``.

    Returns:
        *name*, unchanged, once validated.

    Raises:
        ValueError: If *name* is empty or contains anything other than
            ASCII letters, digits, ``_``, or ``-``.
    """
    if not name or not _NAME_RE.match(name):
        msg = f"invalid robot name {name!r}: must be a non-empty string of letters, digits, '_', or '-'"
        raise ValueError(msg)
    return name


def robot_prefix(name: str) -> str:
    """Return the Zenoh key prefix for a robot's topics.

    Args:
        name: The robot's logical name.

    Returns:
        ``physicalai/robot/{name}``.
    """
    return f"{KEY_PREFIX}/{validate_name(name)}"


def state_key(name: str) -> str:
    """Key for the owner-to-subscribers state stream.

    Returns:
        The ``{prefix}/state`` key expression.
    """
    return f"{robot_prefix(name)}/state"


def action_key(name: str) -> str:
    """Key for the subscribers-to-owner action stream.

    Returns:
        The ``{prefix}/action`` key expression.
    """
    return f"{robot_prefix(name)}/action"


def metadata_key(name: str) -> str:
    """Key for the owner-answered discovery/validation queryable.

    Returns:
        The ``{prefix}/metadata`` key expression.
    """
    return f"{robot_prefix(name)}/metadata"


def derive_endpoint_port(name: str) -> int:
    """Deterministic loopback TCP port for the owner's Zenoh listen endpoint.

    Local rendezvous cannot rely on multicast scouting (unavailable on some
    hosts, and disabled entirely in local-only/``allow_remote=False`` mode),
    so the owner and its subscribers instead derive the same port from the
    robot's full key prefix. A collision on this port is treated as a
    startup failure rather than silently retried on another port, because
    an independent subscriber has no way to discover a shifted port.

    Args:
        name: The robot's logical name.

    Returns:
        A port in ``[20000, 59999]``.
    """
    digest = hashlib.sha256(robot_prefix(name).encode()).digest()
    return _PORT_BASE + int.from_bytes(digest[:4], "big") % _PORT_RANGE


def default_host() -> str:
    """Return the local hostname for informational ``/metadata`` reporting.

    Returns:
        The local hostname.
    """
    return socket.gethostname()
