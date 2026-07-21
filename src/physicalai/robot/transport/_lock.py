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
import stat
import tempfile
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
        A private directory below ``XDG_RUNTIME_DIR`` when available, or a
        per-user directory below the platform temporary directory otherwise.

    Raises:
        RuntimeError: If the resulting directory is not private or is not
            owned by the current user.
    """
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime_dir) if runtime_dir else Path(tempfile.gettempdir()) / f"physicalai-{os.getuid()}"

    lock_dir = base / "physicalai" / "robot-locks" if runtime_dir else base / "robot-locks"
    lock_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    lock_dir_stat = lock_dir.stat()
    if not stat.S_ISDIR(lock_dir_stat.st_mode) or lock_dir_stat.st_uid != os.getuid() or lock_dir_stat.st_mode & 0o077:
        msg = f"robot lock directory must be private and owned by this user: {lock_dir}"
        raise RuntimeError(msg)
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


def _read_live_name_diagnostics(path: Path) -> dict[str, object] | None:
    """Parse and liveness-check one lock file's diagnostics as a name lock.

    Shared by :func:`_active_owner_name` and :func:`active_owner_device_ids`
    -- both need the same "well-formed ``name`` lock, owner still alive"
    gate before trusting any field in the diagnostics.

    Returns:
        The diagnostics dict if *path* is a well-formed, live ``name``-kind
        lock file, else ``None``.
    """
    try:
        diagnostics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(diagnostics, dict) or diagnostics.get("kind") != NAME_KIND:
        return None

    pid = diagnostics.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        # Signal 0 sends nothing; the kernel only validates that pid exists
        # and is signalable by us, giving a liveness check with no side effect.
        os.kill(pid, 0)
    except OSError:
        return None

    # A PID that is alive but has already released the lock (e.g. after a clean
    # shutdown or after a caller used acquire_locks/release_all for verification)
    # is not an active owner.  Confirm by attempting a non-blocking exclusive
    # trylock: success means nobody holds it; EWOULDBLOCK means someone does.
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return None
    held = False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Acquired the lock — no process is currently holding it.
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        # Could not acquire — some process holds the lock; this is a live owner.
        held = True
    finally:
        os.close(fd)
    return diagnostics if held else None


def _active_owner_name(path: Path) -> str | None:
    """Return the live owner name recorded in one lock file, if any."""
    diagnostics = _read_live_name_diagnostics(path)
    if diagnostics is None:
        return None
    name = diagnostics.get("identity")
    return name if isinstance(name, str) else None


def registered_owner_names() -> list[str]:
    """Return names recorded by owner processes that are still alive.

    Name-lock files double as a host-local discovery registry. Files are
    intentionally retained after release, so the recorded PID is checked
    before returning an entry. The result is only a set of candidates:
    callers still confirm liveness through the owner's metadata queryable.

    Returns:
        Sorted, deduplicated robot names whose recorded owner PID exists.
    """
    names: set[str] = set()
    for path in _lock_dir().glob("*.lock"):
        if name := _active_owner_name(path):
            names.add(name)
    return sorted(names)


def active_owner_device_ids(name: str) -> tuple[str, ...] | None:
    """Return a live owner's device identities from its private name lock.

    Returns:
        Sorted device identities for a live owner, an empty tuple for a
        virtual owner, or ``None`` when no live, compatible diagnostic is
        available.
    """
    diagnostics = _read_live_name_diagnostics(lock_path(NAME_KIND, name))
    if diagnostics is None or diagnostics.get("identity") != name:
        return None

    device_ids = diagnostics.get("device_ids")
    if not isinstance(device_ids, list) or not all(isinstance(device_id, str) for device_id in device_ids):
        return None
    return tuple(sorted(set(device_ids)))


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
        device_ids: Owner device identities recorded only in a name-lock
            diagnostic for same-host race recovery.
    """

    def __init__(
        self,
        kind: str,
        identity: str,
        *,
        owner_name: str | None = None,
        device_ids: Sequence[str] | None = None,
    ) -> None:
        self._kind = kind
        self._identity = identity
        self._owner_name = owner_name
        self._device_ids = tuple(sorted(set(device_ids or ())))
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
        diagnostics: dict[str, object] = {
            "kind": self._kind,
            "identity": self._identity,
            "owner_name": self._owner_name,
            "pid": os.getpid(),
        }
        if self._kind == NAME_KIND:
            diagnostics["device_ids"] = list(self._device_ids)
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
    normalized_device_ids = tuple(sorted(set(device_ids)))
    name_lock = NamedLock(NAME_KIND, name, owner_name=name, device_ids=normalized_device_ids)
    if not name_lock.acquire():
        raise LockContention(NAME_KIND, name)

    device_locks: list[NamedLock] = []
    try:
        for device_id in normalized_device_ids:
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
