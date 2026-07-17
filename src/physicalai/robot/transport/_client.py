# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Long-lived remote client for shared robot discovery and attachment."""

from __future__ import annotations

import time
from typing import Any, Self

from loguru import logger

from ._session import open_session
from ._shared_robot import SharedRobot, discover_robots

_PROBE_TIMEOUT = 1.0
_RETRY_INTERVAL = 0.2


class SharedRobotClient:
    """Manage one reusable remote Zenoh session for shared robot clients.

    The client is attach-only: it never constructs a hardware driver or
    starts an owner process. Its session remains open across discovery and
    attached robots, avoiding repeat scouting after the first remote call.

    Args:
        allow_remote: Must be ``True`` to acknowledge the trusted-network
            transport boundary.
    """

    def __init__(self, *, allow_remote: bool = False) -> None:
        """Create a remote shared-robot client without opening a session yet.

        Raises:
            ValueError: If remote transport was not explicitly enabled.
        """
        if not allow_remote:
            msg = "SharedRobotClient requires allow_remote=True"
            raise ValueError(msg)
        self._session: Any = None
        self._robots: list[SharedRobot] = []
        self._closed = False

    def __enter__(self) -> Self:
        """Enter the client context.

        Returns:
            This open client.
        """
        self._require_open()
        return self

    def __exit__(self, *_args: object) -> None:
        """Disconnect attached robots and close the shared session."""
        self.close()

    def discover(self, timeout: float = 2.0) -> list[dict[str, Any]]:
        """Enumerate reachable remote shared robots.

        Args:
            timeout: Total discovery budget, including any initial Zenoh
                scouting needed to establish remote routes.

        Returns:
            One metadata record per answering owner.
        """
        if timeout <= 0:
            return []

        session = self._get_session()
        deadline = time.monotonic() + timeout
        robots: dict[object, dict[str, Any]] = {}
        while (remaining := deadline - time.monotonic()) > 0:
            for metadata in discover_robots(session=session, timeout=min(_PROBE_TIMEOUT, remaining)):
                robots[metadata.get("name")] = metadata
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(_RETRY_INTERVAL, remaining))
        return list(robots.values())

    def attach(self, name: str, *, connect_timeout: float = 10.0) -> SharedRobot:
        """Create an attach-only robot that reuses this client's session.

        Args:
            name: The remote owner's logical name.
            connect_timeout: Overall budget for the robot's :meth:`connect`.

        Returns:
            An attach-only shared robot using this client's session.
        """
        robot = SharedRobot.attach(
            name,
            allow_remote=True,
            connect_timeout=connect_timeout,
            _session=self._get_session(),
        )
        self._robots.append(robot)
        return robot

    def close(self) -> None:
        """Disconnect attached robots and close the shared Zenoh session."""
        if self._closed:
            return
        self._closed = True
        for robot in [r for r in self._robots if r.is_connected()]:
            logger.info(f"Disconnecting SharedRobot {robot.name!r} as SharedRobotClient closes")
            robot.disconnect()
        self._robots.clear()
        if self._session is not None:
            self._session.close()
            self._session = None

    def _get_session(self) -> Any:  # noqa: ANN401
        self._require_open()
        if self._session is None:
            self._session = open_session(allow_remote=True)
        return self._session

    def _require_open(self) -> None:
        if self._closed:
            msg = "SharedRobotClient is closed"
            raise RuntimeError(msg)
