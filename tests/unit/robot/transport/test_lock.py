# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from physicalai.robot.transport._lock import RobotLock, lock_path

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def lock_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    return tmp_path


class TestLockPath:
    def test_user_scoped_path(self, lock_dir: Path) -> None:
        path = lock_path("ttyUSB0")
        assert path == lock_dir / "physicalai" / "robot-locks" / "ttyUSB0.lock"
        assert path.parent.is_dir()

    def test_separators_sanitized(self, lock_dir: Path) -> None:
        path = lock_path("a/b/c")
        assert path.name == "a_b_c.lock"


class TestRobotLock:
    def test_acquire_release(self, lock_dir: Path) -> None:
        lock = RobotLock("dev0")
        assert lock.acquire()
        lock.release()

    def test_acquire_idempotent(self, lock_dir: Path) -> None:
        lock = RobotLock("dev1")
        assert lock.acquire()
        assert lock.acquire()
        lock.release()

    def test_reacquire_after_release(self, lock_dir: Path) -> None:
        lock = RobotLock("dev2")
        assert lock.acquire()
        lock.release()
        assert lock.acquire()
        lock.release()

    def test_context_manager(self, lock_dir: Path) -> None:
        with RobotLock("dev3") as lock:
            assert lock.path.exists()

    def test_second_process_blocked(self, lock_dir: Path) -> None:
        """flock is per-process; a second process must fail to acquire."""
        device_id = f"race-{uuid4().hex[:8]}"
        lock = RobotLock(device_id)
        assert lock.acquire()

        code = textwrap.dedent(f"""
            from physicalai.robot.transport._lock import RobotLock
            raise SystemExit(0 if not RobotLock({device_id!r}).acquire() else 1)
        """)
        result = subprocess.run(
            [sys.executable, "-c", code],
            env={"XDG_CACHE_HOME": str(lock_dir), "PYTHONPATH": "src"},
            check=False,
            capture_output=True,
        )
        lock.release()
        assert result.returncode == 0, result.stderr.decode()

    def test_second_process_wins_after_release(self, lock_dir: Path) -> None:
        device_id = f"free-{uuid4().hex[:8]}"
        lock = RobotLock(device_id)
        assert lock.acquire()
        lock.release()

        code = textwrap.dedent(f"""
            from physicalai.robot.transport._lock import RobotLock
            raise SystemExit(0 if RobotLock({device_id!r}).acquire() else 1)
        """)
        result = subprocess.run(
            [sys.executable, "-c", code],
            env={"XDG_CACHE_HOME": str(lock_dir), "PYTHONPATH": "src"},
            check=False,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr.decode()
