# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for robot transport tests."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

if TYPE_CHECKING:
    from pathlib import Path

HAS_ZENOH = importlib.util.find_spec("zenoh") is not None

requires_zenoh = pytest.mark.skipif(not HAS_ZENOH, reason="eclipse-zenoh not installed")

FAKE_ROBOT_CLASS = "tests.unit.robot.transport.fake.FakeRobot"


@pytest.fixture(autouse=True)
def isolated_lock_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep lock files out of the user cache; inherited by owner subprocesses."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def unique_id() -> str:
    """A unique robot id suffix per test to avoid cross-test interference."""
    return f"test/{uuid4().hex[:8]}"
