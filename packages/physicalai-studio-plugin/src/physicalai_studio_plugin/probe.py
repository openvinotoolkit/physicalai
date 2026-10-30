# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Probe and scanner protocols used by plugin robot integrations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from .schemas import SerialPortInfo

_PayloadT_contra = TypeVar("_PayloadT_contra", bound=BaseModel, contravariant=True)


class PortScanner(Protocol):
    """Duck-type for serial port scanners (e.g. RobotConnectionManager).

    Used by RobotProbe to accept scan results without coupling to a specific
    manager implementation.
    """

    async def find_robots(self) -> None:
        """Refresh discovered robot connections."""

    @property
    def robots(self) -> list[SerialPortInfo]:
        """Currently discovered robot connections."""


@runtime_checkable
class RobotProbe(Protocol[_PayloadT_contra]):
    """Hardware interaction interface for a robot type.

    Each built-in or plugin-provided robot type implements this protocol
    to encapsulate device discovery, visual identification, and online
    status checking. The type parameter ``_PayloadT`` is the robot's
    payload model (a pydantic ``BaseModel`` subclass).
    """

    async def discover(self, manager: PortScanner, *, passive: bool = False) -> list[SerialPortInfo]:
        """Discover robot candidates and return current connection metadata.

        With ``passive=True``, only enumerate OS/cached evidence: do not open
        serial connections, ping network endpoints, connect drivers, or move
        hardware. Return candidates, not an assertion of their robot models.
        The default preserves existing plugin behavior.

        Returns:
            Available connection candidates.

        Raises:
            NotImplementedError: If passive discovery cannot be provided safely.
        """

    async def identify(
        self,
        payload: _PayloadT_contra,
        manager: PortScanner | None,
        joint: str | None = None,
    ) -> None:
        """Trigger physical identification behavior for a robot."""

    async def is_online(
        self,
        payload: _PayloadT_contra,
        manager: PortScanner | None = None,
        *,
        passive: bool = False,
    ) -> bool:
        """Return whether the robot appears online using the requested evidence.

        With ``passive=True``, use only OS/cached evidence and never initiate
        hardware or network I/O. If such evidence cannot establish status,
        raise ``NotImplementedError`` instead of returning a misleading false.

        Returns:
            Whether the requested connection appears online.

        Raises:
            NotImplementedError: If passive status cannot be established.
        """
