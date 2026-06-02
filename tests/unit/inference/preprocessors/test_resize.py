# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from physicalai.inference.constants import IMAGES
from physicalai.inference.preprocessors import Preprocessor, ResizeMode, ResizePreprocessor


class TestResizePreprocessor:
    def test_is_preprocessor(self) -> None:
        prep = ResizePreprocessor(image_resolution=(64, 64))
        assert isinstance(prep, Preprocessor)

    def test_stretch_no_aspect_no_padding(self) -> None:
        prep = ResizePreprocessor(image_resolution=(64, 64), mode=ResizeMode.STRETCH)
        img = np.random.rand(1, 3, 32, 16).astype(np.float32)
        result = prep({IMAGES: img})
        assert result[IMAGES].shape == (1, 3, 64, 64)

    def test_letterbox_preserves_aspect_ratio_with_padding(self) -> None:
        prep = ResizePreprocessor(image_resolution=(64, 64), mode=ResizeMode.LETTERBOX)
        img = np.random.rand(1, 3, 32, 16).astype(np.float32)
        result = prep({IMAGES: img})
        # Padded back to the exact target resolution.
        assert result[IMAGES].shape == (1, 3, 64, 64)

    def test_padding_uses_configured_pad_value(self) -> None:
        prep = ResizePreprocessor(
            image_resolution=(64, 64),
            mode=ResizeMode.LETTERBOX,
            pad_value=7,
        )
        img = np.ones((1, 1, 32, 16), dtype=np.float32)

        result = prep({IMAGES: img})
        out = result[IMAGES]

        assert out.shape == (1, 1, 64, 64)
        # 32x16 scales to 64x32, so left and right pads are 16 pixels each.
        assert np.all(out[:, :, :, :16] == 7)
        assert np.all(out[:, :, :, 48:] == 7)
        assert np.allclose(out[:, :, :, 16:48], 1.0)

    def test_letterbox_skips_padding_when_aspect_already_matches(self) -> None:
        prep = ResizePreprocessor(image_resolution=(64, 64), mode=ResizeMode.LETTERBOX)
        img = np.random.rand(1, 3, 32, 32).astype(np.float32)
        result = prep({IMAGES: img})
        assert result[IMAGES].shape == (1, 3, 64, 64)

    def test_letterbox_clamps_to_minimum_size_one(self) -> None:
        prep = ResizePreprocessor(image_resolution=(512, 512), mode=ResizeMode.LETTERBOX, pad_value=7)
        img = np.ones((1, 1, 1, 640), dtype=np.float32)

        result = prep({IMAGES: img})
        out = result[IMAGES]

        # 1x640 downscaled to 512-wide would produce height=0 without clamping.
        assert out.shape == (1, 1, 512, 512)
        assert np.all(out[:, :, :255, :] == 7)
        assert np.allclose(out[:, :, 255:256, :], 1.0)
        assert np.all(out[:, :, 256:, :] == 7)

    def test_nested_image_dict(self) -> None:
        prep = ResizePreprocessor(image_resolution=(64, 64), mode=ResizeMode.STRETCH)
        images = {"cam0": np.random.rand(1, 3, 32, 32).astype(np.float32)}
        result = prep({IMAGES: images})
        assert result[IMAGES]["cam0"].shape == (1, 3, 64, 64)

    def test_flat_image_keys(self) -> None:
        prep = ResizePreprocessor(image_resolution=(64, 64), mode=ResizeMode.STRETCH)
        inputs = {
            "images.cam0": np.random.rand(1, 3, 32, 32).astype(np.float32),
            "images.cam0.is_pad": np.zeros((1,), dtype=bool),
        }
        result = prep(inputs)
        assert result["images.cam0"].shape == (1, 3, 64, 64)
        # is_pad keys are left untouched.
        assert result["images.cam0.is_pad"].shape == (1,)

    def test_invalid_ndim_raises(self) -> None:
        prep = ResizePreprocessor(image_resolution=(64, 64))
        img = np.random.rand(3, 32, 32).astype(np.float32)
        with pytest.raises(ValueError, match="expected"):
            prep({IMAGES: img})
