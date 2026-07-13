# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Zenoh session helper shared by owner and subscriber endpoints."""

from __future__ import annotations

from typing import Any

from ._ids import derive_endpoint_port


def open_session(robot_id: str | None = None, *, listen: bool = False) -> Any:  # noqa: ANN401
    """Open a Zenoh session pinned to peer mode.

    Peer mode (no router hop) roughly halves latency for the two-endpoint
    owner-subscriber topology. A router is only needed for cross-subnet
    discovery, which stays a deferred option.

    In addition to default multicast scouting, a deterministic TCP endpoint
    derived from *robot_id* is used so same-host spawn-or-attach works on
    hosts without multicast: the owner listens on it, subscribers connect
    to it on localhost (Zenoh retries in the background until the owner is
    up).

    Args:
        robot_id: Robot id for endpoint derivation; ``None`` opens a
            default (scouting-only) session.
        listen: True for the owner (listen on the derived port on all
            interfaces), False for subscribers (connect to localhost).

    Returns:
        The open Zenoh session.
    """
    import zenoh  # noqa: PLC0415

    config = zenoh.Config()
    config.insert_json5("mode", '"peer"')
    if robot_id is not None:
        port = derive_endpoint_port(robot_id)
        if listen:
            config.insert_json5("listen/endpoints", f'["tcp/0.0.0.0:{port}"]')
        else:
            config.insert_json5("connect/endpoints", f'["tcp/127.0.0.1:{port}"]')
    return zenoh.open(config)
