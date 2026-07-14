# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Host-local exclusivity locks: one name lock, N device locks.

Two distinct namespaces share this primitive:

- ``name`` locks serialize concurrent same-name owner creation (the Zenoh
  ``/metadata`` probe-then-declare sequence is not atomic — see
  :mod:`_owner_worker`).
- ``device`` locks are the actual single-owner arbiter over physical
  hardware, keyed by :attr:`~physicalai.robot.interface.Robot.device_ids`.

Both use a self-managed, user-scoped ``flock`` file (not a world-writable
``/tmp`` path, avoiding the CWE-377 predictable-path race). The lock
belongs to an open file descriptor, so the OS releases it on process exit
— including a crash or ``SIGKILL`` — without any cleanup code running.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from collections.abc import Sequence

NAME_KIND = "name"
DEVICE_KIND = "device"


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


def lock_path(kind: str, identity: str) -> Path:
    """Deterministic lock-file path for a namespaced identity.

    Hashing (rather than sanitizing the raw identity into a filename) keeps
    the ``name`` and ``device`` namespaces from ever colliding on disk even
    when a name and a device id happen to be equal strings, and avoids any
    filesystem-unsafe-character handling for identities like network
    addresses or scheme-qualified device ids (``tcp:192.168.1.2``).

    Args:
        kind: Lock namespace — ``"name"`` or ``"device"``.
        identity: The name or device id being locked.

    Returns:
        Path to ``{lock_dir}/{sha256(kind:identity)}.lock``.
    """
    digest = hashlib.sha256(f"{kind}:{identity}".encode()).hexdigest()
    return _lock_dir() / f"{digest}.lock"


class NamedLock:
    """Exclusive, non-blocking advisory lock on one namespaced identity.

    Held for the owner's lifetime; released on process exit even after a
    crash (``flock`` locks die with the file descriptor). Lock files are
    never deleted on release — the kernel lock, not file existence,
    determines ownership.

    Args:
        kind: Lock namespace — ``"name"`` or ``"device"``.
        identity: The name or device id being locked.
        owner_name: The robot name claiming this lock, recorded in the
            lock file's diagnostic contents (not used for arbitration).
    """

    def __init__(self, kind: str, identity: str, *, owner_name: str | None = None) -> None:
        self._kind = kind
        self._identity = identity
        self._owner_name = owner_name
        self._path = lock_path(kind, identity)
        self._fd: int | None = None

    @property
    def path(self) -> Path:
        """The lock-file path."""
        return self._path

    @property
    def kind(self) -> str:
        """The lock namespace (``"name"`` or ``"device"``)."""
        return self._kind

    @property
    def identity(self) -> str:
        """The name or device id this lock guards."""
        return self._identity

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
        diagnostics = {
            "kind": self._kind,
            "identity": self._identity,
            "owner_name": self._owner_name,
            "pid": os.getpid(),
        }
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps(diagnostics).encode())
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
            msg = f"{self._kind} lock already held: {self._identity}"
            raise RuntimeError(msg)
        return self

    def __exit__(self, *args: object) -> None:
        """Release on exit."""
        self.release()


class LockContention(Exception):  # noqa: N818 — internal signal, not a user-facing error condition
    """Internal signal naming which lock (kind + identity) blocked acquisition.

    Raised only by :func:`acquire_locks`; callers translate it into the
    appropriate structured worker error (``name_lock_contention`` or
    ``device_lock_contention``) rather than letting it escape as-is.
    """

    def __init__(self, kind: str, identity: str) -> None:
        """Store which lock could not be acquired."""
        super().__init__(f"{kind} lock already held: {identity}")
        self.kind = kind
        self.identity = identity


@dataclass
class OwnedLocks:
    """The full set of locks a single owner is holding."""

    name_lock: NamedLock
    device_locks: list[NamedLock] = field(default_factory=list)

    def release_all(self) -> None:
        """Release device locks, then the name lock, in reverse acquisition order."""
        for lock in reversed(self.device_locks):
            lock.release()
        self.name_lock.release()


def acquire_locks(name: str, device_ids: Sequence[str]) -> OwnedLocks:
    """Acquire the name lock, then every device lock, in sorted order.

    Lock ordering is always name-lock-first, then sorted+deduplicated
    device locks — consistent ordering across concurrent processes
    prevents deadlock when two owners' device sets overlap. Any partial
    acquisition is rolled back before raising, so a failed call never
    leaves stray locks held.

    Args:
        name: The robot's logical name.
        device_ids: Physical device ids to lock exclusively (empty for a
            virtual robot with no exclusively-owned hardware).

    Returns:
        The acquired :class:`OwnedLocks`.

    Raises:
        LockContention: Naming the first lock (name, or a specific device)
            that another process already holds.
    """
    name_lock = NamedLock(NAME_KIND, name, owner_name=name)
    if not name_lock.acquire():
        raise LockContention(NAME_KIND, name)

    device_locks: list[NamedLock] = []
    try:
        for device_id in sorted(set(device_ids)):
            device_lock = NamedLock(DEVICE_KIND, device_id, owner_name=name)
            if not device_lock.acquire():
                raise LockContention(DEVICE_KIND, device_id)  # noqa: TRY301
            device_locks.append(device_lock)
    except LockContention:
        for lock in reversed(device_locks):
            lock.release()
        name_lock.release()
        raise

    return OwnedLocks(name_lock=name_lock, device_locks=device_locks)
