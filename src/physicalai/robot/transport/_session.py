# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Zenoh session helper shared by owner and subscriber endpoints.

Secure by default: unless the caller explicitly opts into ``allow_remote``,
multicast/gossip scouting is disabled and the owner's listen endpoint is
bound to loopback only, so the transport is unreachable off-host — not
merely undiscoverable. The caller that spawns an owner fixes this scope
for the owner's entire lifetime; a later attacher's ``allow_remote`` value
only configures its own session, never the running owner's.
"""

from __future__ import annotations

from typing import Any

from physicalai.transport._zenoh import endpoint_for_key, open_zenoh_session

from ._ids import robot_prefix


def open_session(name: str | None = None, *, listen: bool = False, allow_remote: bool = False) -> Any:  # noqa: ANN401
    """Open a Zenoh session pinned to peer mode.

    Peer mode (no router hop) roughly halves latency for the two-endpoint
    owner-subscriber topology. A router is only needed for cross-subnet
    discovery, which stays a deferred option.

    A deterministic TCP endpoint derived from *name* is always used for
    same-host rendezvous, because multicast scouting is not available on
    every host (e.g. macOS local-network privacy, locked-down LANs) and is
    disabled outright when ``allow_remote`` is False: the owner listens on
    the derived port, subscribers connect to it on localhost (Zenoh
    retries in the background until the owner is up).

    Args:
        name: Robot name for endpoint derivation; ``None`` opens a
            scouting-only session with no fixed endpoint (used by
            :func:`discover_robots`).
        listen: True for the owner (listen on the derived port), False for
            subscribers (connect to localhost).
        allow_remote: When False (default), disables multicast/gossip
            scouting and binds the owner's listen endpoint to ``127.0.0.1``
            only — the session is unreachable off-host. When True, enables
            scouting and binds the owner to ``0.0.0.0`` for cross-host
            reachability; the deployer is responsible for network
            isolation (see ``docs/development/security.md`` rule 12).

    Returns:
        The open Zenoh session.
    """
    connect_endpoints: list[str] | None = None
    listen_endpoints: list[str] | None = None
    if name is not None:
        if listen:
            bind_host = "0.0.0.0" if allow_remote else "127.0.0.1"  # noqa: S104  # nosec B104: explicit remote opt-in
            listen_endpoints = [endpoint_for_key(robot_prefix(name), bind_host)]
            connect_endpoints = []
        else:
            connect_endpoints = [endpoint_for_key(robot_prefix(name), "127.0.0.1")]
            listen_endpoints = []
    return open_zenoh_session(
        mode="peer",
        connect_endpoints=connect_endpoints,
        listen_endpoints=listen_endpoints,
        multicast_enabled=allow_remote,
        gossip_enabled=allow_remote,
    )
