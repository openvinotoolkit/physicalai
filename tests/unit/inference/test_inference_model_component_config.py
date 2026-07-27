# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff:file-ignore[undocumented-public-module, undocumented-public-class, undocumented-public-method, undocumented-public-init, assert, magic-value-comparison]

"""Construction round-trips for path-rooted ``InferenceModel`` ``@export_config``."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from physicalai.config import ComponentConfigError, instantiate, is_config_exportable, to_config


def _make_export_dir(tmp_path: Path, *, backend: str = "openvino") -> Path:
    export_dir = tmp_path / "exports"
    export_dir.mkdir(exist_ok=True)
    artifact = "act.xml" if backend == "openvino" else f"act.{backend}"
    manifest = {
        "format": "policy_package",
        "version": "1.0",
        "policy": {
            "name": "act",
            "source": {"class_path": "physicalai.policies.act.ACT"},
        },
        "model": {
            "artifacts": {backend: artifact},
            "runner": {"class_path": "physicalai.inference.runners.SinglePass", "init_args": {}},
        },
    }
    with (export_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f)
    (export_dir / artifact).touch()
    if artifact.endswith(".xml"):
        (export_dir / artifact.replace(".xml", ".bin")).touch()
    return export_dir


def _assert_construction_round_trip(model: object) -> dict[str, Any]:
    assert is_config_exportable(model)
    config = to_config(model)
    wire: dict[str, Any] = json.loads(json.dumps(config))
    restored = instantiate(wire)
    assert type(restored) is type(model)
    assert to_config(restored) == wire
    return wire


@pytest.fixture
def mock_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.input_names = []
    adapter.output_names = []
    adapter.default_device.return_value = "cpu"
    return adapter


@pytest.fixture
def _patch_adapter(mock_adapter: MagicMock) -> Generator[MagicMock, None, None]:
    with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
        yield mock_adapter


class TestInferenceModelComponentConfig:
    def test_path_rooted_round_trip(self, tmp_path: Path, _patch_adapter: MagicMock) -> None:
        from physicalai.inference import InferenceModel

        export_dir = _make_export_dir(tmp_path)
        model = InferenceModel(
            export_dir=export_dir,
            policy_name="act",
            backend="openvino",
            device="cpu",
        )
        wire = _assert_construction_round_trip(model)
        assert wire["class_path"] == "physicalai.inference.InferenceModel"
        assert wire["init_args"] == {
            "export_dir": str(export_dir),
            "policy_name": "act",
            "backend": "openvino",
            "device": "cpu",
        }

    def test_relative_export_dir_as_given(
        self, tmp_path: Path, _patch_adapter: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from physicalai.inference import InferenceModel

        monkeypatch.chdir(tmp_path)
        _make_export_dir(tmp_path)
        rel = Path("exports")

        model = InferenceModel(export_dir=rel, backend="openvino", device="cpu")
        wire = _assert_construction_round_trip(model)
        assert wire["init_args"]["export_dir"] == "exports"

    def test_scalar_adapter_kwargs_round_trip(self, tmp_path: Path, _patch_adapter: MagicMock) -> None:
        from physicalai.inference import InferenceModel

        model = InferenceModel(
            export_dir=_make_export_dir(tmp_path),
            backend="openvino",
            device="cpu",
            num_threads=4,
        )
        wire = _assert_construction_round_trip(model)
        assert wire["init_args"]["num_threads"] == 4

    def test_omitted_overrides_stay_omitted(self, tmp_path: Path, _patch_adapter: MagicMock) -> None:
        from physicalai.inference import InferenceModel

        model = InferenceModel(export_dir=_make_export_dir(tmp_path), backend="openvino", device="cpu")
        wire = to_config(model)
        for key in ("runner", "preprocessors", "postprocessors", "callbacks"):
            assert key not in wire["init_args"]

    def test_live_runner_fails(self, tmp_path: Path, _patch_adapter: MagicMock) -> None:
        from physicalai.inference import InferenceModel
        from physicalai.inference.runners import SinglePass

        model = InferenceModel(
            export_dir=_make_export_dir(tmp_path),
            backend="openvino",
            device="cpu",
            runner=SinglePass(),
        )
        with pytest.raises(ComponentConfigError, match=r"init_args\.runner"):
            to_config(model)

    def test_live_preprocessors_fail(self, tmp_path: Path, _patch_adapter: MagicMock) -> None:
        from physicalai.inference import InferenceModel

        model = InferenceModel(
            export_dir=_make_export_dir(tmp_path),
            backend="openvino",
            device="cpu",
            preprocessors=[MagicMock()],
        )
        with pytest.raises(ComponentConfigError, match=r"init_args\.preprocessors"):
            to_config(model)

    def test_live_postprocessors_fail(self, tmp_path: Path, _patch_adapter: MagicMock) -> None:
        from physicalai.inference import InferenceModel

        model = InferenceModel(
            export_dir=_make_export_dir(tmp_path),
            backend="openvino",
            device="cpu",
            postprocessors=[MagicMock()],
        )
        with pytest.raises(ComponentConfigError, match=r"init_args\.postprocessors"):
            to_config(model)

    def test_live_callbacks_fail(self, tmp_path: Path, _patch_adapter: MagicMock) -> None:
        from physicalai.inference import InferenceModel

        model = InferenceModel(
            export_dir=_make_export_dir(tmp_path),
            backend="openvino",
            device="cpu",
            callbacks=[MagicMock()],
        )
        with pytest.raises(ComponentConfigError, match=r"init_args\.callbacks"):
            to_config(model)

    def test_non_scalar_dict_adapter_kwarg_fails(self, tmp_path: Path, _patch_adapter: MagicMock) -> None:
        from physicalai.inference import InferenceModel

        model = InferenceModel(
            export_dir=_make_export_dir(tmp_path),
            backend="openvino",
            device="cpu",
            config_blob={"a": 1},
        )
        with pytest.raises(ComponentConfigError, match=r"init_args\.config_blob"):
            to_config(model)

    def test_non_scalar_list_adapter_kwarg_fails(self, tmp_path: Path, _patch_adapter: MagicMock) -> None:
        from physicalai.inference import InferenceModel

        model = InferenceModel(
            export_dir=_make_export_dir(tmp_path),
            backend="openvino",
            device="cpu",
            tags=["x", "y"],
        )
        with pytest.raises(ComponentConfigError, match=r"init_args\.tags"):
            to_config(model)
