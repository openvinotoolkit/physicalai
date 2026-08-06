# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff:file-ignore[undocumented-public-module, undocumented-public-class, undocumented-public-function, undocumented-public-init, magic-value-comparison, assert]

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytest
from physicalai.config import Config, ConfigError, FromConfig, export_config, instantiate_obj


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


class Target(FromConfig):
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

    assert restored == config
    assert config.to_dict() == {"nested": {"value": 3}, "mode": "fast", "shape": [2, 4]}
    assert config.to_jsonargparse()["init_args"] == config.to_dict()


def test_typed_dataclass_dynamic_instantiation() -> None:
    config = TypedConfig(Nested(3), Mode.FAST, (2, 4))

    restored = instantiate_obj(config.to_jsonargparse())

    assert restored == config


def test_typed_dataclass_nested_in_exported_instance() -> None:
    instance = ConfigTarget(TypedConfig(Nested(3), Mode.FAST, (2, 4)))

    restored = Config.from_instance(instance).instantiate()

    assert isinstance(restored, ConfigTarget)
    assert restored.config == instance.config


def test_general_instantiation_delegates_recipe_to_strict_config() -> None:
    target = instantiate_obj({"class_path": f"{__name__}.Target", "init_args": {"value": 9}})
    assert isinstance(target, Target)
    assert target.value == 9


def test_instantiate_config_class_path_builds_nested_recipe() -> None:
    from physicalai.config import instantiate

    cfg = instantiate(
        {
            "class_path": "physicalai.config.Config",
            "init_args": {
                "class_path": "builtins.dict",
                "init_args": {"k": {"class_path": "builtins.int", "init_args": {}}},
            },
        },
    )

    assert isinstance(cfg, Config)
    assert cfg.class_path == "builtins.dict"
    nested = cfg.init_args["k"]
    assert isinstance(nested, dict)
    assert nested["class_path"] == "builtins.int"


def test_instantiate_config_class_path_requires_inner_recipe() -> None:
    import pytest

    from physicalai.config import ConfigError, instantiate

    with pytest.raises(ConfigError, match="class_path"):
        instantiate({"class_path": "physicalai.config.Config", "init_args": {}})


@dataclass
class PathConfig(Config):
    root: Path


def test_typed_dataclass_strict_rejects_extra_keys() -> None:
    with pytest.raises(TypeError, match="Unexpected keys"):
        TypedConfig.from_dict(
            {"nested": {"value": 1}, "mode": "fast", "shape": [1, 1], "epochs": 1},
            strict=True,
        )


def test_typed_dataclass_path_round_trip() -> None:
    cfg = PathConfig(Path("/tmp/data"))
    restored = PathConfig.from_dict(cfg.to_dict())
    assert restored.root == Path("/tmp/data")


def test_instantiate_recursive_depth_limit() -> None:
    nested: dict[str, object] = {"value": 1}
    current: dict[str, object] = nested
    for _ in range(12):
        child: dict[str, object] = {"value": 1}
        current["child"] = child
        current = child

    class Leaf:
        def __init__(self, **kwargs: object) -> None:
            pass

    with pytest.raises(ConfigError, match="nesting depth exceeds"):
        instantiate_obj({"nested": nested}, target_cls=Leaf)
