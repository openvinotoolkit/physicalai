# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Long-lived client for shared robot discovery and attachment."""

from __future__ import annotations

import time
from typing import Any, Self

from loguru import logger

from ._session import open_session
from ._shared_robot import SharedRobot, discover_robots

_COLD_DISCOVERY_TIMEOUT = 1.0
_WARM_DISCOVERY_TIMEOUT = 0.1
_RETRY_INTERVAL = 0.2


class SharedRobotClient:
    """Manage one reusable Zenoh session for shared robot clients.

    The client is attach-only: it never constructs a hardware driver or
    starts an owner process. Its session remains open across discovery and
    attached robots, avoiding repeat scouting after the first remote call.

    Args:
        allow_remote: Whether the shared session can discover and attach to
            owners beyond localhost. Defaults to ``False``.
    """

    def __init__(self, *, allow_remote: bool = False) -> None:
        """Create a shared-robot client without opening a session yet."""
        self._allow_remote = allow_remote
        self._session: Any = None
        self._robots: list[SharedRobot] = []
        self._closed = False
        self._has_discovered = False

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

    def discover(self, timeout: float | None = None) -> list[dict[str, Any]]:
        """Enumerate reachable remote shared robots.

        Args:
            timeout: Total discovery budget. When omitted, the first call
                uses one second for Zenoh scouting; later calls use 0.1
                seconds with the warmed session.

        Returns:
            One metadata record per answering owner.
        """
        budget = timeout
        if budget is None:
            budget = _WARM_DISCOVERY_TIMEOUT if self._has_discovered else _COLD_DISCOVERY_TIMEOUT
        if budget <= 0:
            return []

        session = self._get_session()
        self._has_discovered = True
        deadline = time.monotonic() + budget
        robots: dict[object, dict[str, Any]] = {}
        while (remaining := deadline - time.monotonic()) > 0:
            for metadata in discover_robots(session=session, timeout=min(_COLD_DISCOVERY_TIMEOUT, remaining)):
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
            allow_remote=self._allow_remote,
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
            self._session = open_session(allow_remote=self._allow_remote)
        return self._session

    def _require_open(self) -> None:
        if self._closed:
            msg = "SharedRobotClient is closed"
            raise RuntimeError(msg)
