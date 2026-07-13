# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Zenoh transport for shared robots.

One owner process holds the exclusive hardware connection; any number of
:class:`SharedRobot` subscribers read state (pull) and send actions
(fire-and-forget) over Zenoh. Mirrors the structure of the capture
transport (``SharedCamera``) with a network-capable transport, because
robot state/action payloads are tiny and may cross hosts.

Security note: this transport is designed for a **trusted robot-cell
network**. It provides no authentication on ``/action`` — any peer that can
reach the owner's Zenoh session can move the physical robot. Isolating the
network (VLAN/firewall, or Zenoh ACL/TLS) is the deployer's responsibility.

Requires the ``transport`` extra::

    pip install physicalai[transport]
"""

from __future__ import annotations

from ._shared_robot import SharedRobot, discover_robots

__all__ = ["SharedRobot", "discover_robots"]
