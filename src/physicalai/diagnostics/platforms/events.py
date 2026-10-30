# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Optional native change notification with a portable polling fallback."""

from __future__ import annotations

import select
import socket
import sys
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator

_NETLINK_KOBJECT_UEVENT = 15
_RELEVANT = (b"SUBSYSTEM=usb\x00", b"SUBSYSTEM=tty\x00", b"SUBSYSTEM=video4linux\x00", b"SUBSYSTEM=hidraw\x00")


class ChangeSource(Protocol):
    """A wakeup hint, not an authoritative hardware inventory."""

    detail: str

    def wait(self, timeout: float) -> None:
        """Wait for a relevant change or a periodic reconciliation deadline."""


class PollChanges:
    """Portable fallback; polling cannot see a complete replug between samples."""

    detail = "Polling; unplug/replug completed between samples can be missed."

    @staticmethod
    def wait(timeout: float) -> None:
        """Sleep until the next complete reconciliation."""
        time.sleep(timeout)


class LinuxChanges:
    """Kernel USB/tty/video/HID notifications, with periodic full reconciliation."""

    detail = "Linux kernel device notifications plus periodic polling reconciliation."

    def __init__(self, connection: socket.socket) -> None:
        """Use a socket opened before collecting the baseline inventory."""
        self.connection = connection
        self.failed = False

    def wait(self, timeout: float) -> None:
        """Wake early on relevant events; retain periodic refreshes for lost events."""
        if self.failed:
            time.sleep(timeout)
            return
        try:
            self._wait(timeout)
        except (OSError, ValueError) as exc:
            self.failed = True
            self.detail = f"Native device notifications unavailable ({exc}); using polling reconciliation."
            time.sleep(timeout)

    def _wait(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            readable, _, _ = select.select([self.connection], [], [], max(0, deadline - time.monotonic()))
            if not readable:
                return
            data = self.connection.recv(65536)
            if any(part in data for part in _RELEVANT):
                # Coalesce composite-device bursts before reconciling sysfs/udev.
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
                for _ in range(64):
                    if not select.select([self.connection], [], [], 0)[0]:
                        break
                    self.connection.recv(65536)
                return


@contextmanager
def change_source() -> Iterator[ChangeSource]:
    """Subscribe before baseline collection; fall back safely without elevation.

    Yields:
        Native Linux notifications when accessible, otherwise polling.
    """
    connection: socket.socket | None = None
    if sys.platform == "linux":
        try:
            connection = socket.socket(socket.AF_NETLINK, socket.SOCK_DGRAM, _NETLINK_KOBJECT_UEVENT)
            connection.bind((0, 1))
        except OSError:
            if connection is not None:
                connection.close()
            connection = None
    try:
        yield LinuxChanges(connection) if connection is not None else PollChanges()
    finally:
        if connection is not None:
            connection.close()
