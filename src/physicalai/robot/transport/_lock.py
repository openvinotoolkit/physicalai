# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Single-owner arbiter: a self-managed, user-scoped lock file.

The Zenoh ``/meta`` probe has a check-then-act gap: two cold-starting
processes can both see "no owner" and both try to grab the same hardware.
The lock file is the transport-agnostic arbiter, working identically for
serial and IP backends. The user-scoped cache directory (not a
world-writable ``/tmp`` path) avoids the CWE-377 predictable-path race.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
from pathlib import Path
from typing import Self


def _lock_dir() -> Path:
    """Return the user-scoped lock directory, creating it if needed.

    Returns:
        The ``~/.cache/physicalai/robot-locks`` directory path.
    """
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home) if cache_home else Path.home() / ".cache"
    lock_dir = base / "physicalai" / "robot-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir


def lock_path(device_id: str) -> Path:
    """Deterministic lock-file path for a device id.

    Args:
        device_id: The same device id used in the robot id derivation.

    Returns:
        Path to ``{lock_dir}/{device_id}.lock`` (path separators sanitized).
    """
    safe = device_id.replace(os.sep, "_").replace("/", "_")
    return _lock_dir() / f"{safe}.lock"


class RobotLock:
    """Exclusive, non-blocking advisory lock on a device id.

    Held for the owner's lifetime; released on process exit even after a
    crash (``flock`` locks die with the file descriptor).

    Args:
        device_id: Device id derived from the robot connection params.
    """

    def __init__(self, device_id: str) -> None:
        self._path = lock_path(device_id)
        self._fd: int | None = None

    @property
    def path(self) -> Path:
        """The lock-file path."""
        return self._path

    def acquire(self) -> bool:
        """Try to acquire the lock without blocking.

        Returns:
            True if this process now holds the lock, False if another
            process already holds it.
        """
        if self._fd is not None:
            return True
        fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        self._fd = fd
        return True

    def release(self) -> None:
        """Release the lock. No-op when not held."""
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def __enter__(self) -> Self:
        """Acquire on entry.

        Returns:
            This lock instance.

        Raises:
            RuntimeError: If the lock is already held by another process.
        """
        if not self.acquire():
            msg = f"robot lock already held: {self._path}"
            raise RuntimeError(msg)
        return self

    def __exit__(self, *args: object) -> None:
        """Release on exit."""
        self.release()
