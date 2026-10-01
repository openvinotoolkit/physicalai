# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Deterministic replay checks for checked-in fuzz regression seeds."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path

import pytest

_SEEDS_ROOT = Path(__file__).parent / "seeds"
_HARNESS_SEEDS = {
    "fuzz_manifest": _SEEDS_ROOT / "fuzz_manifest",
}


def _seed_cases() -> list[pytest.ParameterSet]:
    cases: list[pytest.ParameterSet] = []
    for harness_name, seed_dir in _HARNESS_SEEDS.items():
        if not seed_dir.is_dir():
            pytest.fail(f"configured seed directory does not exist: {seed_dir}")

        seed_paths = sorted(seed_dir.iterdir())
        if not seed_paths:
            pytest.fail(f"configured seed directory is empty: {seed_dir}")

        for seed_path in seed_paths:
            if seed_path.is_symlink():
                pytest.fail(f"checked-in fuzz seed must not be a symlink: {seed_path}")
            if not seed_path.is_file():
                pytest.fail(f"checked-in fuzz seed must be a regular file: {seed_path}")
            cases.append(pytest.param(harness_name, seed_path, id=f"{harness_name}/{seed_path.name}"))
    return cases


@pytest.mark.parametrize(("harness_name", "seed_path"), _seed_cases())
def test_checked_in_seed_replays(harness_name: str, seed_path: Path) -> None:
    """Replay one checked-in seed through its allowlisted harness."""
    harness = importlib.import_module(harness_name)
    test_one_input = getattr(harness, "test_one_input", None)
    assert isinstance(test_one_input, Callable), f"{harness_name} does not expose test_one_input()"

    try:
        test_one_input(seed_path.read_bytes())
    except Exception as exc:
        pytest.fail(f"{harness_name} failed to replay {seed_path}: {exc!r}")
