"""Test configuration for the MuJoCo SO-101 plugin."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
SLOW_TESTS_ENV = "PHYSICALAI_MUJOCO_SLOW_TESTS"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip this plugin's slow simulation tests when CI found no plugin changes.

    CI sets ``PHYSICALAI_MUJOCO_SLOW_TESTS=false`` for pull requests that leave
    the plugin alone. Locally, on pushes to main and on pull requests that touch
    the plugin, every test runs. Only tests in this directory are affected, so
    other packages keep their own ``slow`` tests.
    """
    _ = config
    if os.environ.get(SLOW_TESTS_ENV, "true").strip().lower() != "false":
        return
    skip = pytest.mark.skip(reason="MuJoCo plugin unchanged; its slow simulation tests run when it changes")
    for item in items:
        if "slow" in item.keywords and _TESTS_DIR in Path(item.path).resolve().parents:
            item.add_marker(skip)
