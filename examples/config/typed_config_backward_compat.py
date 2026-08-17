#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Demonstrate typed-config backward compatibility after the jsonargparse migration.

Run from the repository root::

    uv run python examples/config/typed_config_backward_compat.py

The script writes temporary YAML files in the shape produced on ``main`` (enum
*values* in ``init_args``, ``class_path``/``init_args`` envelope) and shows that
the current branch loads them, saves in the jsonargparse-native shape, and
round-trips ``@export_config`` recipes.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from physicalai.config import Config, export_config


class Mode(Enum):
    FAST = "fast"


@dataclass
class Nested:
    value: int


@dataclass
class AppConfig(Config):
    nested: Nested
    mode: Mode
    shape: tuple[int, int]


@export_config
class Worker:
    def __init__(self, config: AppConfig) -> None:
        self.config = config


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    expected = AppConfig(Nested(7), Mode.FAST, (2, 4))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # Legacy envelope YAML (main): enum stored as value "fast", not name "FAST".
        legacy_path = root / "legacy_main_style.yaml"
        legacy_path.write_text(
            "class_path: __main__.AppConfig\n"
            "init_args:\n"
            "  nested:\n"
            "    value: 7\n"
            "  mode: fast\n"
            "  shape: [2, 4]\n",
            encoding="utf-8",
        )
        _section("Load legacy YAML (enum value + envelope)")
        loaded = AppConfig.load(legacy_path)
        print("legacy file:\n", legacy_path.read_text(), sep="")
        print("loaded:", loaded)
        assert loaded == expected

        # Save on this branch → jsonargparse-native names in init_args.
        current_path = root / "current_branch_style.yaml"
        loaded.save(current_path)
        _section("Save after load (current branch wire shape)")
        print(current_path.read_text())
        roundtrip = AppConfig.load(current_path)
        assert roundtrip == expected

        # Mapping load with enum value (no file).
        _section("Load mapping with legacy enum value")
        from_mapping = AppConfig.load(
            {"nested": {"value": 7}, "mode": "fast", "shape": [2, 4]},
        )
        assert from_mapping == expected

        # @export_config recipe round-trip (separate path from typed load).
        _section("Export_config recipe instantiate")
        worker = Worker(expected)
        recipe = Config.from_instance(worker)
        restored_worker = recipe.instantiate()
        assert isinstance(restored_worker, Worker)
        assert restored_worker.config == expected

    print("\nAll backward-compatibility checks passed.")


if __name__ == "__main__":
    main()
