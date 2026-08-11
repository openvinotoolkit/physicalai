# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff:file-ignore[missing-type-function-argument, missing-return-type-undocumented-public-function, class-as-data-structure, undocumented-public-init, magic-value-comparison, no-self-use, float-equality-comparison, assert]

"""Tests for FromConfig, instantiate_obj, and typed dataclass Config."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

import numpy as np
import pytest
from pydantic import BaseModel

from physicalai.config import Config, from_config
from physicalai.config.loading import import_class, instantiate_obj
from physicalai.config.mixin import FromConfig


class SampleModel(FromConfig):
    def __init__(self, hidden_size: int, num_layers: int = 3, **kwargs: object) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.kwargs = kwargs


class SampleModelConfig(BaseModel):
    hidden_size: int = 128
    num_layers: int = 3


@dataclass
class SampleModelDataclassConfig(Config):
    hidden_size: int = 128
    num_layers: int = 3


class NestedComponent:
    def __init__(self, value: int) -> None:
        self.value = value


class ParentModel(FromConfig):
    def __init__(self, component: NestedComponent, components: list[NestedComponent] | None = None) -> None:
        self.component = component
        self.components = components or []


@from_config
class DecoratedModel:
    def __init__(self, hidden_size: int, num_layers: int = 3) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers


@dataclasses.dataclass
class SampleModelDataclass:
    hidden_size: int = 128
    num_layers: int = 3


class ActivationType(StrEnum):
    RELU = "relu"
    GELU = "gelu"


@dataclass
class SimpleConfig(Config):
    hidden_size: int = 128
    num_layers: int = 3


@dataclass
class NestedConfig(Config):
    model: SimpleConfig = field(default_factory=SimpleConfig)
    learning_rate: float = 0.001


@dataclass
class ComplexConfig(Config):
    activation: ActivationType = ActivationType.RELU
    layers: tuple = (64, 128)
    weights: np.ndarray = field(default_factory=lambda: np.array([1.0, 2.0]))


class TestInstantiateObj:
    def test_from_dict(self) -> None:
        result = instantiate_obj({"class_path": "builtins.dict", "init_args": {"key": "value"}})
        assert result == {"key": "value"}

    def test_from_dict_with_key(self) -> None:
        config = {"model": {"class_path": "builtins.dict", "init_args": {"size": 128}}}
        assert instantiate_obj(config, key="model") == {"size": 128}

    def test_nested_instantiation(self) -> None:
        config = {
            "class_path": "builtins.dict",
            "init_args": {"nested": {"class_path": "builtins.dict", "init_args": {"k": "v"}}},
        }
        result = instantiate_obj(config)
        assert isinstance(result, dict)
        assert result["nested"] == {"k": "v"}

    def test_from_file(self, tmp_path) -> None:
        (tmp_path / "config.yaml").write_text("class_path: builtins.dict\ninit_args:\n  key: value")
        assert instantiate_obj(tmp_path / "config.yaml") == {"key": "value"}

    def test_missing_class_path_raises(self) -> None:
        with pytest.raises(ValueError, match="class_path"):
            instantiate_obj({"init_args": {}})

    def test_invalid_import_raises(self) -> None:
        with pytest.raises(ImportError):
            import_class("nonexistent.module.Class")

    def test_import_class_imports_symbol(self) -> None:
        assert import_class("builtins.dict") is dict


class TestFromConfigMixin:
    def test_from_dict(self) -> None:
        model = SampleModel.from_dict({"hidden_size": 256, "num_layers": 4})
        assert model.hidden_size == 256
        assert model.num_layers == 4

    def test_from_dict_with_key(self) -> None:
        model = SampleModel.from_dict({"model": {"hidden_size": 512, "num_layers": 6}}, key="model")
        assert model.hidden_size == 512

    def test_from_pydantic(self) -> None:
        model = SampleModel.from_pydantic(SampleModelConfig(hidden_size=256))
        assert model.hidden_size == 256

    def test_from_dataclass(self) -> None:
        model = SampleModel.from_dataclass(SampleModelDataclass(hidden_size=512))
        assert model.hidden_size == 512

    def test_from_yaml(self, tmp_path) -> None:
        (tmp_path / "config.yaml").write_text("hidden_size: 1024\nnum_layers: 8")
        model = SampleModel.from_yaml(tmp_path / "config.yaml")
        assert model.hidden_size == 1024

    def test_from_config_unified(self) -> None:
        assert SampleModel.from_config({"hidden_size": 128, "num_layers": 3}).hidden_size == 128
        assert SampleModel.from_config(SampleModelConfig()).hidden_size == 128
        assert SampleModel.from_config(SampleModelDataclass()).hidden_size == 128

    def test_concrete_class_accepts_jsonargparse_config(self) -> None:
        config = {
            "class_path": f"{SampleModel.__module__}.SampleModel",
            "init_args": {"hidden_size": 256, "num_layers": 4},
        }
        model = SampleModel.from_config(config)
        assert model.hidden_size == 256
        assert model.num_layers == 4

    def test_concrete_class_accepts_config_dataclass(self) -> None:
        model = SampleModel.from_config(SampleModelDataclassConfig(hidden_size=384, num_layers=5))
        assert model.hidden_size == 384
        assert model.num_layers == 5

    def test_nested_class_path_values_in_direct_args(self) -> None:
        model = ParentModel.from_config(
            {
                "component": {
                    "class_path": f"{NestedComponent.__module__}.NestedComponent",
                    "init_args": {"value": 10},
                },
                "components": [
                    {
                        "class_path": f"{NestedComponent.__module__}.NestedComponent",
                        "init_args": {"value": 20},
                    },
                ],
            },
        )
        assert isinstance(model.component, NestedComponent)
        assert model.component.value == 10
        assert isinstance(model.components[0], NestedComponent)
        assert model.components[0].value == 20

    def test_from_config_decorator(self) -> None:
        decorated_model_cls = cast("Any", DecoratedModel)
        model = decorated_model_cls.from_config({"hidden_size": 512, "num_layers": 6})
        assert model.hidden_size == 512
        assert model.num_layers == 6

    def test_from_config_decorator_with_yaml(self, tmp_path) -> None:
        decorated_model_cls = cast("Any", DecoratedModel)
        path = tmp_path / "decorated.yaml"
        path.write_text("hidden_size: 640\nnum_layers: 7")
        model = decorated_model_cls.from_config(path)
        assert model.hidden_size == 640
        assert model.num_layers == 7

    def test_recursive_parameter(self) -> None:
        @dataclass
        class Nested:
            size: int = 64

        @dataclass
        class Parent:
            hidden_size: int = 128
            nested: Nested = field(default_factory=Nested)

        class Model(FromConfig):
            def __init__(self, hidden_size: int, nested: Nested | None = None) -> None:
                self.hidden_size = hidden_size
                self.nested = nested

        parent = Parent()
        assert isinstance(Model.from_dataclass(parent, recursive=False).nested, Nested)
        assert isinstance(Model.from_dataclass(parent, recursive=True).nested, dict)


class TestConfigSerialization:
    def test_to_jsonargparse(self) -> None:
        result = SimpleConfig(hidden_size=256).to_jsonargparse()
        assert "class_path" in result
        assert result["init_args"]["hidden_size"] == 256

    def test_to_dict(self) -> None:
        result = SimpleConfig(hidden_size=256).to_dict()
        assert "class_path" not in result
        assert result["hidden_size"] == 256

    def test_from_dict(self) -> None:
        config = SimpleConfig.from_dict({"hidden_size": 512, "num_layers": 8})
        assert config.hidden_size == 512

    def test_from_dict_nested(self) -> None:
        config = NestedConfig.from_dict({"model": {"hidden_size": 256, "num_layers": 4}, "learning_rate": 0.01})
        assert isinstance(config.model, SimpleConfig)
        assert config.model.hidden_size == 256

    def test_round_trip(self) -> None:
        original = NestedConfig(model=SimpleConfig(hidden_size=512), learning_rate=0.005)
        restored = NestedConfig.from_dict(original.to_dict())
        assert restored.model.hidden_size == 512
        assert restored.learning_rate == 0.005

    def test_type_conversions(self) -> None:
        result = ComplexConfig(
            activation=ActivationType.GELU,
            layers=(32, 64),
            weights=np.array([[1.0, 2.0]]),
        ).to_jsonargparse()
        assert result["init_args"]["activation"] == "GELU"
        assert result["init_args"]["layers"] == [32, 64]
        assert result["init_args"]["weights"] == [[1.0, 2.0]]


class TestConfigSaveLoad:
    def test_save_load_jsonargparse(self, tmp_path) -> None:
        path = tmp_path / "config.yaml"
        SimpleConfig(hidden_size=256).save(path)
        assert SimpleConfig.load(path).hidden_size == 256

    def test_save_load_dict_format(self, tmp_path) -> None:
        path = tmp_path / "config.yaml"
        SimpleConfig(hidden_size=512).save(path, format="dict")
        assert SimpleConfig.load(path).hidden_size == 512

    def test_invalid_extension_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="Unsupported file extension"):
            SimpleConfig().save(tmp_path / "config.json")

    def test_not_dataclass_raises(self) -> None:
        class NotDataclass(Config):
            pass

        with pytest.raises(TypeError, match="Config subclasses must be dataclasses"):
            NotDataclass("builtins.dict")
