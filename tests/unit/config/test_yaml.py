# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff:file-ignore[undocumented-public-module, undocumented-public-class, undocumented-public-method, undocumented-public-init, undocumented-magic-method, magic-value-comparison, no-self-use, assert]

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from physicalai.config import (
    ComponentConfigError,
    export_config,
    instantiate,
    load_yaml,
    save_yaml,
    to_config,
    to_yaml,
)

if TYPE_CHECKING:
    from pathlib import Path


@export_config
class Gadget:
    def __init__(self, size: int, label: str = "gadget") -> None:
        self.size = size
        self.label = label


@export_config
class Holder:
    def __init__(self, gadget: Gadget, note: str | None = None) -> None:
        self.gadget = gadget
        self.note = note


class TestToYaml:
    def test_live_component_round_trips_through_yaml(self) -> None:
        holder = Holder(Gadget(3, label="inner"), note="hi")

        text = to_yaml(holder)
        rebuilt = instantiate(yaml.safe_load(text))

        assert isinstance(rebuilt, Holder)
        assert rebuilt.note == "hi"
        assert rebuilt.gadget.size == 3
        assert rebuilt.gadget.label == "inner"

    def test_accepts_existing_component_config_mapping(self) -> None:
        config = to_config(Gadget(7))

        text = to_yaml(config)
        loaded = yaml.safe_load(text)

        assert loaded == {"class_path": f"{__name__}.Gadget", "init_args": {"size": 7}}

    def test_rejects_malformed_mapping(self) -> None:
        with pytest.raises(ComponentConfigError, match="class_path"):
            to_yaml({"init_args": {"size": 1}})

    def test_rejects_non_exportable_object(self) -> None:
        with pytest.raises(ComponentConfigError):
            to_yaml(object())


class TestSaveLoadYaml:
    def test_save_then_load_then_instantiate(self, tmp_path: Path) -> None:
        target = tmp_path / "gadget.yaml"
        save_yaml(Gadget(5), target)

        loaded = load_yaml(target)
        rebuilt = instantiate(loaded)

        assert isinstance(rebuilt, Gadget)
        assert rebuilt.size == 5
        assert rebuilt.label == "gadget"

    def test_load_rejects_non_mapping_document(self, tmp_path: Path) -> None:
        target = tmp_path / "list.yaml"
        target.write_text("- 1\n- 2\n", encoding="utf-8")

        with pytest.raises(ComponentConfigError, match="must be a mapping"):
            load_yaml(target)
