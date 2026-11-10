# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared Zenoh configuration and deterministic endpoint helpers."""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import zenoh

_PORT_BASE = 20000
_PORT_RANGE = 40000
_MAX_TCP_PORT = 65535
_KEY_PREFIX_PARTS = 3
_MODEL_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z", re.ASCII)


def model_key_prefix(model_name: str) -> str:
    """Return the validated inference namespace for a model name.

    Raises:
        ValueError: If the model name is not a valid key segment.
    """
    if not _MODEL_NAME_RE.fullmatch(model_name) or model_name in {".", ".."}:
        msg = "model_name must be a non-empty key segment containing only letters, digits, '.', '_' or '-'"
        raise ValueError(msg)
    return f"physicalai/inference/{model_name}"


def derive_port(namespace: str, name: str) -> int:
    """Derive a stable unprivileged TCP port for one named namespace.

    Returns:
        The deterministic TCP port for this namespace and name.
    """
    key_prefix = f"physicalai/{namespace}/{name}"
    digest = hashlib.sha256(key_prefix.encode("utf-8")).digest()
    return _PORT_BASE + int.from_bytes(digest[:4], "big") % _PORT_RANGE


def derive_endpoint_port(key_prefix: str) -> int:
    """Derive a port from a full ``physicalai`` key prefix.

    Returns:
        The deterministic TCP port for the key prefix.
    """
    parts = key_prefix.split("/", maxsplit=2)
    if len(parts) == _KEY_PREFIX_PARTS and parts[0] == "physicalai":
        return derive_port(parts[1], parts[2])
    digest = hashlib.sha256(key_prefix.encode("utf-8")).digest()
    return _PORT_BASE + int.from_bytes(digest[:4], "big") % _PORT_RANGE


def endpoint_for_key(key_prefix: str, host: str, port: int | None = None) -> str:
    """Build a TCP endpoint, deriving its port from the key when omitted.

    Returns:
        A Zenoh TCP endpoint with an IPv6-aware host representation.

    Raises:
        ValueError: If the host or port is invalid.
    """
    if not host or "/" in host or any(char.isspace() for char in host):
        msg = "host must be a non-empty hostname or IP address"
        raise ValueError(msg)
    resolved_port = derive_endpoint_port(key_prefix) if port is None else port
    if not 1 <= resolved_port <= _MAX_TCP_PORT:
        msg = "port must be between 1 and 65535"
        raise ValueError(msg)
    formatted_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"tcp/{formatted_host}:{resolved_port}"


def make_zenoh_config(
    mode: str,
    *,
    connect_endpoints: Sequence[str] | None = None,
    listen_endpoints: Sequence[str] | None = None,
    multicast_enabled: bool | None = False,
    gossip_enabled: bool | None = False,
    config: zenoh.Config | None = None,
) -> zenoh.Config:
    """Build Zenoh config with explicit role, endpoints, and discovery policy.

    Returns:
        The supplied or newly created configuration with the requested settings.
    """
    import zenoh  # noqa: PLC0415

    config = config or zenoh.Config()
    config.insert_json5("mode", json.dumps(mode))
    if connect_endpoints is not None:
        config.insert_json5("connect/endpoints", json.dumps(list(connect_endpoints)))
    if listen_endpoints is not None:
        config.insert_json5("listen/endpoints", json.dumps(list(listen_endpoints)))
    if multicast_enabled is not None:
        config.insert_json5("scouting/multicast/enabled", json.dumps(multicast_enabled))
    if gossip_enabled is not None:
        config.insert_json5("scouting/gossip/enabled", json.dumps(gossip_enabled))
    return config


def open_zenoh_session(
    mode: str,
    *,
    connect_endpoints: Sequence[str] | None = None,
    listen_endpoints: Sequence[str] | None = None,
    multicast_enabled: bool | None = False,
    gossip_enabled: bool | None = False,
    config: zenoh.Config | None = None,
) -> zenoh.Session:
    """Open a Zenoh session using the shared role and discovery defaults.

    Returns:
        The configured open session.
    """
    import zenoh  # noqa: PLC0415

    return zenoh.open(
        make_zenoh_config(
            mode,
            connect_endpoints=connect_endpoints,
            listen_endpoints=listen_endpoints,
            multicast_enabled=multicast_enabled,
            gossip_enabled=gossip_enabled,
            config=config,
        )
    )
