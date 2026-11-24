# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from physicalai.inference.component_factory import component_registry
from physicalai.inference.constants import IMAGES, STATE, TASK
from physicalai.inference.preprocessors import VLAJEPAPreprocessor
from physicalai.inference.preprocessors.vla_jepa import _area_weights, _bicubic_weights, _resize

_EMBODIED = "<|embodied_action|>"
_PAD = "<|image_pad|>"


@pytest.fixture()
def preprocessor(tmp_path: Path) -> VLAJEPAPreprocessor:
    words = ["[UNK]", _PAD, _EMBODIED, "pick", "the", "cup", "do"]
    tokenizer = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    # Two 64x64 views -> 4x4 patches, 2x2 merge -> 4 pads per image.
    return VLAJEPAPreprocessor(
        artifact=str(tmp_path / "tokenizer.json"),
        prompt=f"{' '.join([_PAD] * 4)} {' '.join([_PAD] * 4)} do {{instruction}} {_EMBODIED} {_EMBODIED}",
        image_keys=["images.wrist", "images.front"],
        embodied_token_id=2,
        num_embodied_tokens=2,
        noise_shape=[3, 2],
        image_resolution=(48, 48),
        vision_resolution=(64, 64),
    )


def _inputs(**overrides: object) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    inputs: dict[str, Any] = {
        "images.wrist": rng.integers(0, 256, (1, 30, 40, 3), dtype=np.uint8),
        "images.front": rng.integers(0, 256, (1, 30, 40, 3), dtype=np.uint8),
        STATE: np.ones((1, 6), dtype=np.float32),
        TASK: ["pick the cup"],
    }
    return {**inputs, **overrides}


def test_registered_type() -> None:
    assert component_registry.resolve("vla_jepa") == "physicalai.inference.preprocessors.VLAJEPAPreprocessor"


def test_graph_inputs(preprocessor: VLAJEPAPreprocessor) -> None:
    result = preprocessor(_inputs())
    np.testing.assert_array_equal(result["input_ids"], [[1] * 8 + [6, 3, 4, 5, 2, 2]])
    np.testing.assert_array_equal(result["attention_mask"], np.ones((1, 14)))
    np.testing.assert_array_equal(result["embodied_positions"], [[12, 13]])
    # 2 images x 16 patches, 3 channels x 2 frames x 16 x 16 values each.
    assert result["pixel_values"].shape == (32, 1536)
    assert result["pixel_values"].dtype == np.float32
    assert result["noise"].shape == (1, 3, 2)
    np.testing.assert_array_equal(result[STATE], np.ones((1, 6)))


def test_camera_layouts_match(preprocessor: VLAJEPAPreprocessor) -> None:
    """Runtime frames (nested, channels-last uint8) match flat channels-first float inputs."""
    inputs = _inputs()
    images = {name: inputs.pop(f"images.{name}") for name in ("wrist", "front")}
    nested = {**inputs, IMAGES: images}
    channels_first = {
        **_inputs(),
        **{key: value.transpose(0, 3, 1, 2) / 255 for key, value in _inputs().items() if key.startswith(IMAGES)},
    }
    np.testing.assert_allclose(
        preprocessor(nested)["pixel_values"],
        preprocessor(channels_first)["pixel_values"],
        atol=1e-6,
    )


def test_resize_weights() -> None:
    """Area pooling averages whole bins and both resizes preserve constant images."""
    image = np.arange(16, dtype=np.float64).reshape(1, 4, 4)
    np.testing.assert_allclose(_resize(image, (_area_weights(4, 2), _area_weights(4, 2)))[0], [[2.5, 4.5], [10.5, 12.5]])
    constant = np.full((3, 7, 9), 0.25)
    for weights in (_area_weights, _bicubic_weights):
        np.testing.assert_allclose(_resize(constant, (weights(7, 5), weights(9, 12))), 0.25)
    np.testing.assert_allclose(_bicubic_weights(5, 5), np.eye(5), atol=1e-12)


def test_rejects_invalid_inputs(preprocessor: VLAJEPAPreprocessor) -> None:
    with pytest.raises(KeyError, match="images.front"):
        preprocessor({key: value for key, value in _inputs().items() if key != "images.front"})
    with pytest.raises(ValueError, match="batch size one"):
        preprocessor(_inputs(**{"images.wrist": np.zeros((2, 30, 40, 3), dtype=np.uint8)}))
    with pytest.raises(ValueError, match="one task"):
        preprocessor(_inputs(**{TASK: ["pick", "cup"]}))
    with pytest.raises(ValueError, match="embodied-action"):
        preprocessor(_inputs(**{TASK: [_EMBODIED]}))
    with pytest.raises(ValueError, match="Flow noise"):
        preprocessor(_inputs(noise=np.full((1, 3, 2), np.nan)))
    with pytest.raises(ValueError, match="placeholder"):
        VLAJEPAPreprocessor(artifact="", prompt="", image_keys=[], embodied_token_id=0, num_embodied_tokens=0, noise_shape=[])
