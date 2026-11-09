# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: file-ignore[undocumented-public-module, undocumented-public-class, undocumented-public-method, undocumented-public-init, magic-value-comparison, no-self-use, float-equality-comparison, assert]

"""Tests for the typed dataclass Config API."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
import pytest

from physicalai.config import Config


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
    layers: tuple[int, ...] = (64, 128)
    weights: np.ndarray = field(default_factory=lambda: np.array([1.0, 2.0]))


def test_to_jsonargparse_preserves_class_envelope() -> None:
    result = SimpleConfig(hidden_size=256).to_jsonargparse()

    assert result["class_path"] == f"{SimpleConfig.__module__}.SimpleConfig"
    assert result["init_args"]["hidden_size"] == 256


def test_to_dict_contains_plain_dataclass_fields() -> None:
    result = SimpleConfig(hidden_size=256).to_dict()

    assert "class_path" not in result
    assert result == {"hidden_size": 256, "num_layers": 3}


def test_from_dict_reconstructs_typed_config() -> None:
    config = SimpleConfig.from_dict({"hidden_size": 512, "num_layers": 8})

    assert config == SimpleConfig(hidden_size=512, num_layers=8)


def test_from_dict_reconstructs_nested_typed_config() -> None:
    config = NestedConfig.from_dict({"model": {"hidden_size": 256, "num_layers": 4}, "learning_rate": 0.01})

    assert isinstance(config.model, SimpleConfig)
    assert config.model == SimpleConfig(hidden_size=256, num_layers=4)
    assert config.learning_rate == 0.01


def test_nested_typed_config_round_trip() -> None:
    original = NestedConfig(model=SimpleConfig(hidden_size=512), learning_rate=0.005)

    restored = NestedConfig.from_dict(original.to_dict())

    assert restored == original


def test_to_jsonargparse_serializes_enums_tuples_and_arrays() -> None:
    result = ComplexConfig(
        activation=ActivationType.GELU,
        layers=(32, 64),
        weights=np.array([[1.0, 2.0]]),
    ).to_jsonargparse()

    assert result["init_args"]["activation"] == "GELU"
    assert result["init_args"]["layers"] == [32, 64]
    assert result["init_args"]["weights"] == [[1.0, 2.0]]


def test_save_load_jsonargparse(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    config = SimpleConfig(hidden_size=256)

    config.save(path)

    assert SimpleConfig.load(path) == config


def test_save_load_dict_format(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    config = SimpleConfig(hidden_size=512)

    config.save(path, format="dict")

    assert SimpleConfig.load(path) == config


def test_save_rejects_invalid_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported file extension"):
        SimpleConfig().save(tmp_path / "config.json")


def test_config_subclasses_must_be_dataclasses() -> None:
    class NotDataclass(Config):
        pass

    with pytest.raises(TypeError, match="Config subclasses must be dataclasses"):
        NotDataclass("builtins.dict")
