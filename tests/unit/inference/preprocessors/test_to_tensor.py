# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from physicalai.inference.constants import IMAGES
from physicalai.inference.preprocessors import Preprocessor, ToFloatTensorPreprocessor


class TestToFloatTensorPreprocessor:
    def test_is_preprocessor(self) -> None:
        prep = ToFloatTensorPreprocessor()
        assert isinstance(prep, Preprocessor)

    def test_single_array_is_normalized(self) -> None:
        prep = ToFloatTensorPreprocessor()
        img = np.full((1, 3, 8, 8), 255, dtype=np.uint8)
        result = prep({IMAGES: img})
        out = result[IMAGES]
        assert out.dtype == np.float32
        assert np.allclose(out, 1.0)

    def test_zero_maps_to_zero(self) -> None:
        prep = ToFloatTensorPreprocessor()
        img = np.zeros((1, 3, 8, 8), dtype=np.uint8)
        result = prep({IMAGES: img})
        assert np.allclose(result[IMAGES], 0.0)

    def test_midpoint_value(self) -> None:
        prep = ToFloatTensorPreprocessor()
        img = np.full((1, 3, 4, 4), 128, dtype=np.uint8)
        result = prep({IMAGES: img})
        assert np.allclose(result[IMAGES], 128 / 255.0)

    def test_float_input_is_unchanged(self) -> None:
        prep = ToFloatTensorPreprocessor()
        img = np.random.rand(1, 3, 8, 8).astype(np.float32)
        result = prep({IMAGES: img})
        assert result[IMAGES].dtype == np.float32
        assert np.array_equal(result[IMAGES], img)

    def test_nested_image_dict(self) -> None:
        prep = ToFloatTensorPreprocessor()
        images = {"cam0": np.full((1, 3, 8, 8), 255, dtype=np.uint8)}
        result = prep({IMAGES: images})
        out = result[IMAGES]["cam0"]
        assert out.dtype == np.float32
        assert np.allclose(out, 1.0)

    def test_flat_image_keys(self) -> None:
        prep = ToFloatTensorPreprocessor()
        inputs = {
            "images.cam0": np.full((1, 3, 8, 8), 255, dtype=np.uint8),
            "images.cam0.is_pad": np.zeros((1,), dtype=bool),
        }
        result = prep(inputs)
        assert result["images.cam0"].dtype == np.float32
        assert np.allclose(result["images.cam0"], 1.0)
        # is_pad keys are left untouched.
        assert result["images.cam0.is_pad"].dtype == bool

    def test_channels_last_is_transposed_to_channels_first(self) -> None:
        prep = ToFloatTensorPreprocessor()
        chw = np.random.randint(0, 256, size=(1, 3, 8, 6), dtype=np.uint8)
        hwc = np.transpose(chw, (0, 2, 3, 1))
        result = prep({IMAGES: hwc})
        out = result[IMAGES]
        assert out.dtype == np.float32
        assert out.shape == (1, 3, 8, 6)
        np.testing.assert_allclose(out, chw.astype(np.float32) / 255.0)

    def test_channels_first_layout_is_preserved(self) -> None:
        prep = ToFloatTensorPreprocessor()
        img = np.random.randint(0, 256, size=(1, 3, 8, 6), dtype=np.uint8)
        result = prep({IMAGES: img})
        assert result[IMAGES].shape == (1, 3, 8, 6)

    def test_channels_last_float_input_is_transposed(self) -> None:
        prep = ToFloatTensorPreprocessor()
        img = np.random.rand(1, 8, 6, 3).astype(np.float32)
        result = prep({IMAGES: img})
        assert result[IMAGES].dtype == np.float32
        assert result[IMAGES].shape == (1, 3, 8, 6)
        np.testing.assert_array_equal(result[IMAGES], np.transpose(img, (0, 3, 1, 2)))

    def test_non_4d_array_layout_is_unchanged(self) -> None:
        prep = ToFloatTensorPreprocessor()
        img = np.full((8, 8, 3), 255, dtype=np.uint8)
        result = prep({IMAGES: img})
        assert result[IMAGES].shape == (8, 8, 3)
        assert result[IMAGES].dtype == np.float32
