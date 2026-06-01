# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from physicalai.inference.constants import IMAGES
from physicalai.inference.preprocessors import Preprocessor, ResizePreprocessor


class TestResizePreprocessor:
    def test_is_preprocessor(self) -> None:
        prep = ResizePreprocessor(image_resolution=(64, 64))
        assert isinstance(prep, Preprocessor)

    def test_stretch_no_aspect_no_padding(self) -> None:
        prep = ResizePreprocessor(image_resolution=(64, 64), padding=False, keep_aspect_ratio=False)
        img = np.random.rand(1, 3, 32, 16).astype(np.float32)
        result = prep({IMAGES: img})
        assert result[IMAGES].shape == (1, 3, 64, 64)

    def test_keep_aspect_ratio_with_padding(self) -> None:
        prep = ResizePreprocessor(image_resolution=(64, 64), padding=True, keep_aspect_ratio=True)
        img = np.random.rand(1, 3, 32, 16).astype(np.float32)
        result = prep({IMAGES: img})
        # Padded back to the exact target resolution.
        assert result[IMAGES].shape == (1, 3, 64, 64)

    def test_keep_aspect_ratio_without_padding(self) -> None:
        prep = ResizePreprocessor(image_resolution=(64, 64), padding=False, keep_aspect_ratio=True)
        img = np.random.rand(1, 3, 32, 16).astype(np.float32)
        result = prep({IMAGES: img})
        # Aspect ratio preserved (32:16 -> 64:32), no padding applied.
        assert result[IMAGES].shape == (1, 3, 64, 32)

    def test_nested_image_dict(self) -> None:
        prep = ResizePreprocessor(image_resolution=(64, 64), keep_aspect_ratio=False)
        images = {"cam0": np.random.rand(1, 3, 32, 32).astype(np.float32)}
        result = prep({IMAGES: images})
        assert result[IMAGES]["cam0"].shape == (1, 3, 64, 64)

    def test_flat_image_keys(self) -> None:
        prep = ResizePreprocessor(image_resolution=(64, 64), keep_aspect_ratio=False)
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
