# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff:file-ignore[undocumented-public-module, undocumented-public-class, undocumented-public-function, undocumented-public-init, magic-value-comparison, assert]

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytest
from physicalai.config import Config, ConfigError, export_config


class Mode(Enum):
    FAST = "fast"


@dataclass
class Nested:
    value: int


@dataclass
class TypedConfig(Config):
    nested: Nested
    mode: Mode
    shape: tuple[int, int]


class Target:
    def __init__(self, value: int) -> None:
        self.value = value


@export_config
class ConfigTarget:
    def __init__(self, config: TypedConfig) -> None:
        self.config = config


def test_direct_recipe_round_trip(tmp_path: Path) -> None:
    config = Config(f"{__name__}.Target", {"value": 7})
    path = tmp_path / "recipe.yaml"

    config.save(path)
    restored = Config.load(path)

    assert restored.to_dict() == config.to_dict()
    assert isinstance(restored.instantiate(), Target)


def test_typed_dataclass_semantics(tmp_path: Path) -> None:
    config = TypedConfig(Nested(3), Mode.FAST, (2, 4))
    path = tmp_path / "typed.yaml"
    config.save(path)

    restored = TypedConfig.load(path)
    saved_text = path.read_text()

    assert restored == config
    assert config.to_dict() == {"nested": {"value": 3}, "mode": "FAST", "shape": [2, 4]}
    assert config.to_jsonargparse()["init_args"] == config.to_dict()
    assert "class_path:" in saved_text
    assert "init_args:" in saved_text
    assert "class_path: tests.unit.config.test_config.Nested" not in saved_text


def test_typed_config_load_legacy_envelope_file(tmp_path: Path) -> None:
    path = tmp_path / "legacy.yaml"
    path.write_text(
        "class_path: tests.unit.config.test_config.TypedConfig\n"
        "init_args:\n"
        "  nested:\n"
        "    value: 3\n"
        "  mode: FAST\n"
        "  shape: [2, 4]\n",
        encoding="utf-8",
    )

    restored = TypedConfig.load(path)

    assert restored == TypedConfig(Nested(3), Mode.FAST, (2, 4))


def test_typed_config_load_bare_fields_file(tmp_path: Path) -> None:
    path = tmp_path / "bare.yaml"
    path.write_text(
        "nested:\n  value: 3\nmode: FAST\nshape: [2, 4]\n",
        encoding="utf-8",
    )

    restored = TypedConfig.load(path)

    assert restored == TypedConfig(Nested(3), Mode.FAST, (2, 4))


def test_typed_config_load_legacy_enum_value_in_envelope(tmp_path: Path) -> None:
    path = tmp_path / "legacy-enum-value.yaml"
    path.write_text(
        "class_path: tests.unit.config.test_config.TypedConfig\n"
        "init_args:\n"
        "  nested:\n"
        "    value: 3\n"
        "  mode: fast\n"
        "  shape: [2, 4]\n",
        encoding="utf-8",
    )

    restored = TypedConfig.load(path)

    assert restored == TypedConfig(Nested(3), Mode.FAST, (2, 4))


def test_typed_config_load_accepts_mapping() -> None:
    config = TypedConfig.load({"nested": {"value": 3}, "mode": "FAST", "shape": [2, 4]})

    assert config == TypedConfig(Nested(3), Mode.FAST, (2, 4))


def test_typed_config_load_accepts_legacy_enum_values_in_mapping() -> None:
    config = TypedConfig.load({"nested": {"value": 3}, "mode": "fast", "shape": [2, 4]})

    assert config == TypedConfig(Nested(3), Mode.FAST, (2, 4))


def test_typed_dataclass_dynamic_instantiation() -> None:
    config = TypedConfig(Nested(3), Mode.FAST, (2, 4))

    restored = Config.from_dict(config.to_jsonargparse()).instantiate()

    assert restored == config


def test_typed_dataclass_nested_in_exported_instance() -> None:
    instance = ConfigTarget(TypedConfig(Nested(3), Mode.FAST, (2, 4)))

    restored = Config.from_instance(instance).instantiate()

    assert isinstance(restored, ConfigTarget)
    assert restored.config == instance.config


def test_general_instantiation_delegates_recipe_to_strict_config() -> None:
    target = Config(f"{__name__}.Target", {"value": 9}).instantiate()
    assert isinstance(target, Target)
    assert target.value == 9


@dataclass
class PathConfig(Config):
    root: Path


def test_typed_dataclass_strict_rejects_extra_keys() -> None:
    with pytest.raises(TypeError, match="Unexpected keys"):
        TypedConfig.from_dict(
            {"nested": {"value": 1}, "mode": "FAST", "shape": [1, 1], "epochs": 1},
            strict=True,
        )


def test_typed_dataclass_non_strict_drops_unknown_top_level_keys() -> None:
    """strict=False should silently drop unknown top-level keys instead of raising.

    Regression test: passing strict=False previously still forwarded the
    full mapping to the underlying parser, which rejected unknown keys
    regardless of strict.
    """
    restored = TypedConfig.from_dict(
        {"nested": {"value": 1}, "mode": "FAST", "shape": [1, 1], "epochs": 1, "extra_field": "ignored"},
        strict=False,
    )
    assert restored.nested.value == 1
    assert restored.mode is Mode.FAST
    assert restored.shape == (1, 1)


def test_typed_dataclass_non_strict_still_rejects_nested_unknown_keys() -> None:
    """strict=False only filters unknown keys at the top level, not nested ones.

    This matches the pre-existing (pre-regression) behavior: nested unknown
    keys were never tolerated by strict=False, only top-level ones. The
    underlying parser raises its own error type (not necessarily TypeError)
    for nested rejections, so we only assert that *some* exception is raised.
    """
    with pytest.raises(Exception, match="extra_nested_field"):
        TypedConfig.from_dict(
            {"nested": {"value": 1, "extra_nested_field": "rejected"}, "mode": "FAST", "shape": [1, 1]},
            strict=False,
        )


def test_typed_dataclass_path_round_trip() -> None:
    cfg = PathConfig(Path("/tmp/data"))
    restored = PathConfig.from_dict(cfg.to_dict())
    assert restored.root == Path("/tmp/data")
