# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""A virtual leader arm that reads its joint positions from the simulation.

The simulation publishes a leader pose on ``GET /leader`` every control tick:
the autopilot's targets when it runs in ``leader`` mode, otherwise the arm's
current joints (so teleoperation holds still). Physical AI Studio builds this
class for the *MuJoCo SO-101 Virtual Leader* catalog entry and records what it
reads as the follower's actions.

This module must not import MuJoCo: Studio's environment may not carry the
simulator's version, and a leader needs only HTTP.
"""

from __future__ import annotations

import http.client
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from physicalai.config import export_config

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame
from physicalai_mujoco_so101_plugin.constants import SO101_JOINT_ORDER

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
DEFAULT_HTTP_PORT = 8080


@dataclass
class VirtualLeaderObservation:
    """Leader pose read from the simulation."""

    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None

    @property
    def state(self) -> np.ndarray:
        """Joint positions represented as the robot state."""
        return self.joint_positions


@export_config(class_path="physicalai_mujoco_so101_plugin.virtual_leader.MuJoCoVirtualLeader")
class MuJoCoVirtualLeader:
    """Read-only leader arm backed by the simulation's ``GET /leader`` endpoint."""

    def __init__(
        self,
        http_port: int = DEFAULT_HTTP_PORT,
        host: str = "127.0.0.1",
        timeout_s: float = 1.0,
    ) -> None:
        """Point at a simulation's HTTP server on this machine.

        Raises:
            ValueError: If `host` is not a loopback address or `http_port` is out of range.
        """
        if host not in LOOPBACK_HOSTS:
            msg = f"The virtual leader only reads from this machine; got host {host!r}"
            raise ValueError(msg)
        if not 1 <= int(http_port) <= 65535:  # noqa: PLR2004
            msg = f"http_port must be 1..65535, got {http_port}"
            raise ValueError(msg)
        self.http_port = int(http_port)
        self.host = host
        self.timeout_s = float(timeout_s)
        self.joint_names = list(SO101_JOINT_ORDER)
        self._conn: http.client.HTTPConnection | None = None
        self._last_seq: int | None = None
        self._last: VirtualLeaderObservation | None = None

    @property
    def device_ids(self) -> tuple[str, ...]:
        """The simulation, not this reader, owns the arm."""
        return ()

    def connect(self) -> None:
        """Open the connection and check the simulation publishes a single-arm leader pose.

        Raises:
            ConnectionError: If the simulation is unreachable or publishes another joint layout.
        """
        self._conn = http.client.HTTPConnection(self.host, self.http_port, timeout=self.timeout_s)
        try:
            payload = self._fetch()
        except ConnectionError:
            self.disconnect()
            raise
        names = payload.get("joint_names")
        if not isinstance(names, list) or names != self.joint_names:
            self.disconnect()
            msg = f"The simulation publishes leader joints {names}, expected {self.joint_names}"
            raise ConnectionError(msg)

    def disconnect(self) -> None:
        """Close the connection."""
        if self._conn is not None:
            self._conn.close()
        self._conn = None
        self._last_seq = None
        self._last = None

    def is_connected(self) -> bool:
        """Report whether `connect` succeeded and `disconnect` was not called since.

        Returns:
            ``True`` while connected.
        """
        return self._conn is not None

    def get_observation(self) -> VirtualLeaderObservation:
        """Return the latest leader pose.

        The timestamp only advances when the simulation has ticked since the
        last read, so a frozen simulation reads as a stalled leader.

        Returns:
            The pose, in the follower's joint units.

        Raises:
            ConnectionError: If not connected, or the simulation stopped answering.
        """
        if self._conn is None:
            msg = "Virtual leader is not connected. Call connect() first."
            raise ConnectionError(msg)
        payload = self._fetch()
        seq = payload.get("seq")
        if not isinstance(seq, int):
            msg = "Leader pose has no sequence number; is the simulation up to date?"
            raise ConnectionError(msg)
        if self._last is not None and seq == self._last_seq:
            return self._last
        try:
            positions = np.asarray(payload.get("joint_positions"), dtype=np.float32)
        except (TypeError, ValueError) as exc:
            msg = "Leader pose has no numeric joint positions"
            raise ConnectionError(msg) from exc
        if positions.shape != (len(self.joint_names),):
            msg = f"Leader pose has shape {positions.shape}, expected ({len(self.joint_names)},)"
            raise ConnectionError(msg)
        self._last_seq = seq
        self._last = VirtualLeaderObservation(
            joint_positions=positions,
            timestamp=time.monotonic(),
            sensor_data={"autopilot": np.asarray([payload.get("mode") == "leader"], dtype=np.float32)},
        )
        return self._last

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:  # noqa: PLR6301
        """Ignore actions: a leader arm is only read."""
        _ = action, goal_time

    def _fetch(self) -> dict[str, object]:
        conn = self._conn
        if conn is None:
            msg = "Virtual leader is not connected. Call connect() first."
            raise ConnectionError(msg)
        try:
            conn.request("GET", "/leader")
            response = conn.getresponse()
            body = response.read()
        except (OSError, http.client.HTTPException) as exc:
            conn.close()  # reconnects on the next request
            msg = f"Simulation at {self.host}:{self.http_port} is not answering: {exc}"
            raise ConnectionError(msg) from exc
        if response.status != 200:  # noqa: PLR2004
            msg = f"GET /leader returned HTTP {response.status}"
            raise ConnectionError(msg)
        try:
            payload = json.loads(body)
        except ValueError as exc:
            msg = "GET /leader returned invalid JSON"
            raise ConnectionError(msg) from exc
        if not isinstance(payload, dict):
            msg = "GET /leader did not return a JSON object"
            raise ConnectionError(msg)
        return payload
