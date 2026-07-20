# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from physicalai.robot.transport._lock import (
    LockContention,
    NamedLock,
    active_owner_device_ids,
    acquire_locks,
    lock_path,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def lock_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    return runtime_dir / "physicalai" / "robot-locks"


class TestLockPath:
    def test_user_scoped_hashed_path(self, lock_dir: Path) -> None:
        path = lock_path("device", "ttyUSB0")
        assert path.parent == lock_dir
        assert path.suffix == ".lock"
        assert path.stem != "ttyUSB0"  # hashed, not the raw identity

    def test_falls_back_to_user_scoped_temporary_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        monkeypatch.setattr(
            "physicalai.robot.transport._lock.tempfile.gettempdir", lambda: str(tmp_path)
        )

        path = lock_path("device", "ttyUSB0")

        assert path.parent == tmp_path / f"physicalai-{os.getuid()}" / "robot-locks"

    def test_deterministic(self, lock_dir: Path) -> None:
        assert lock_path("device", "ttyUSB0") == lock_path("device", "ttyUSB0")

    def test_namespaces_never_collide(self, lock_dir: Path) -> None:
        """Equal raw strings in different namespaces must not share a lock file."""
        assert lock_path("name", "x") != lock_path("device", "x")


class TestNamedLock:
    def test_acquire_release(self, lock_dir: Path) -> None:
        lock = NamedLock("device", "dev0")
        assert lock.acquire()
        lock.release()

    def test_acquire_idempotent(self, lock_dir: Path) -> None:
        lock = NamedLock("device", "dev1")
        assert lock.acquire()
        assert lock.acquire()
        lock.release()

    def test_reacquire_after_release(self, lock_dir: Path) -> None:
        lock = NamedLock("device", "dev2")
        assert lock.acquire()
        lock.release()
        assert lock.acquire()
        lock.release()

    def test_context_manager(self, lock_dir: Path) -> None:
        with NamedLock("device", "dev3") as lock:
            assert lock.path.exists()

    def test_diagnostic_contents(self, lock_dir: Path) -> None:
        lock = NamedLock("device", "dev4", owner_name="left-arm")
        lock.acquire()
        try:
            diagnostics = json.loads(lock.path.read_text())
            assert diagnostics["kind"] == "device"
            assert diagnostics["identity"] == "dev4"
            assert diagnostics["owner_name"] == "left-arm"
            assert isinstance(diagnostics["pid"], int)
        finally:
            lock.release()

    def test_second_process_blocked(self, lock_dir: Path) -> None:
        """flock is per-process; a second process must fail to acquire."""
        device_id = f"race-{uuid4().hex[:8]}"
        lock = NamedLock("device", device_id)
        assert lock.acquire()

        code = textwrap.dedent(f"""
            from physicalai.robot.transport._lock import NamedLock
            raise SystemExit(0 if not NamedLock("device", {device_id!r}).acquire() else 1)
        """)
        result = subprocess.run(
            [sys.executable, "-c", code],
            env={"XDG_RUNTIME_DIR": str(lock_dir.parent.parent), "PYTHONPATH": "src"},
            check=False,
            capture_output=True,
        )
        lock.release()
        assert result.returncode == 0, result.stderr.decode()

    def test_second_process_wins_after_release(self, lock_dir: Path) -> None:
        device_id = f"free-{uuid4().hex[:8]}"
        lock = NamedLock("device", device_id)
        assert lock.acquire()
        lock.release()

        code = textwrap.dedent(f"""
            from physicalai.robot.transport._lock import NamedLock
            raise SystemExit(0 if NamedLock("device", {device_id!r}).acquire() else 1)
        """)
        result = subprocess.run(
            [sys.executable, "-c", code],
            env={"XDG_RUNTIME_DIR": str(lock_dir.parent.parent), "PYTHONPATH": "src"},
            check=False,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr.decode()


class TestAcquireLocks:
    def test_acquires_name_and_devices(self, lock_dir: Path) -> None:
        owned = acquire_locks("left-arm", ["serial:ttyUSB0"])
        try:
            assert owned.name_lock.acquire()  # idempotent re-check
            assert len(owned.device_locks) == 1
        finally:
            owned.release_all()

    def test_empty_device_ids_valid_for_virtual_robot(self, lock_dir: Path) -> None:
        owned = acquire_locks("virtual-bot", [])
        try:
            assert owned.device_locks == []
            assert active_owner_device_ids("virtual-bot") == ()
        finally:
            owned.release_all()

    def test_name_lock_diagnostic_records_device_ids(self, lock_dir: Path) -> None:
        owned = acquire_locks("left-arm", ["serial:ttyUSB1", "serial:ttyUSB0", "serial:ttyUSB0"])
        try:
            diagnostics = json.loads(owned.name_lock.path.read_text())
            assert diagnostics["device_ids"] == ["serial:ttyUSB0", "serial:ttyUSB1"]
            assert active_owner_device_ids("left-arm") == ("serial:ttyUSB0", "serial:ttyUSB1")
        finally:
            owned.release_all()

    def test_legacy_or_malformed_name_lock_has_no_device_ids(self, lock_dir: Path) -> None:
        path = lock_path("name", "left-arm")
        path.write_text(json.dumps({"kind": "name", "identity": "left-arm", "pid": os.getpid()}))

        assert active_owner_device_ids("left-arm") is None

    def test_sorted_and_deduplicated(self, lock_dir: Path) -> None:
        owned = acquire_locks("left-arm", ["tcp:2", "tcp:1", "tcp:1"])
        try:
            identities = [lock.identity for lock in owned.device_locks]
            assert identities == ["tcp:1", "tcp:2"]
        finally:
            owned.release_all()

    def test_name_contention_raises_and_holds_nothing(self, lock_dir: Path) -> None:
        first = acquire_locks("left-arm", ["serial:ttyUSB0"])
        try:
            with pytest.raises(LockContention) as exc_info:
                acquire_locks("left-arm", ["serial:ttyUSB1"])
            assert exc_info.value.kind == "name"
            # The failed attempt's device lock must not remain held.
            assert NamedLock("device", "serial:ttyUSB1").acquire()
        finally:
            first.release_all()

    def test_device_contention_rolls_back_name_lock(self, lock_dir: Path) -> None:
        first = acquire_locks("left-arm", ["serial:ttyUSB0"])
        try:
            with pytest.raises(LockContention) as exc_info:
                acquire_locks("right-arm", ["serial:ttyUSB0"])
            assert exc_info.value.kind == "device"
            # The failed attempt's name lock must not remain held.
            assert NamedLock("name", "right-arm").acquire()
        finally:
            first.release_all()

    def test_owner_crash_releases_via_process_exit(self, lock_dir: Path) -> None:
        device_id = f"crash-{uuid4().hex[:8]}"
        code = textwrap.dedent(f"""
            from physicalai.robot.transport._lock import acquire_locks
            acquire_locks({device_id!r}, [{device_id!r}])
            import os
            os._exit(0)
        """)
        result = subprocess.run(
            [sys.executable, "-c", code],
            env={"XDG_RUNTIME_DIR": str(lock_dir.parent.parent), "PYTHONPATH": "src"},
            check=False,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr.decode()
        # Process exit (even os._exit, skipping cleanup) releases flock.
        owned = acquire_locks(device_id, [device_id])
        owned.release_all()
