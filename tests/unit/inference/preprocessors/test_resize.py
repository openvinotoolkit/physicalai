# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from physicalai.inference.constants import IMAGES
from physicalai.inference.preprocessors import ImageLayout, Preprocessor, ResizeMode, ResizePreprocessor


class TestResizePreprocessor:
    def test_is_preprocessor(self) -> None:
        prep = ResizePreprocessor(image_layout="BCHW", image_resolution=(64, 64))
        assert isinstance(prep, Preprocessor)

    def test_stretch_no_aspect_no_padding(self) -> None:
        prep = ResizePreprocessor(image_layout="BCHW", image_resolution=(64, 64), mode=ResizeMode.STRETCH)
        img = np.random.rand(1, 3, 32, 16).astype(np.float32)
        result = prep({IMAGES: img})
        assert result[IMAGES].shape == (1, 3, 64, 64)

    def test_letterbox_preserves_aspect_ratio_with_padding(self) -> None:
        prep = ResizePreprocessor(image_layout="BCHW", image_resolution=(64, 64), mode=ResizeMode.LETTERBOX)
        img = np.random.rand(1, 3, 32, 16).astype(np.float32)
        result = prep({IMAGES: img})
        # Padded back to the exact target resolution.
        assert result[IMAGES].shape == (1, 3, 64, 64)

    def test_padding_uses_configured_pad_value(self) -> None:
        prep = ResizePreprocessor(
            image_layout="BCHW",
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
        prep = ResizePreprocessor(image_layout="BCHW", image_resolution=(64, 64), mode=ResizeMode.LETTERBOX)
        img = np.random.rand(1, 3, 32, 32).astype(np.float32)
        result = prep({IMAGES: img})
        assert result[IMAGES].shape == (1, 3, 64, 64)

    def test_letterbox_clamps_to_minimum_size_one(self) -> None:
        prep = ResizePreprocessor(image_layout="BCHW", image_resolution=(512, 512), mode=ResizeMode.LETTERBOX, pad_value=7)
        img = np.ones((1, 1, 1, 640), dtype=np.float32)

        result = prep({IMAGES: img})
        out = result[IMAGES]

        # 1x640 downscaled to 512-wide would produce height=0 without clamping.
        assert out.shape == (1, 1, 512, 512)
        assert np.all(out[:, :, :255, :] == 7)
        assert np.allclose(out[:, :, 255:256, :], 1.0)
        assert np.all(out[:, :, 256:, :] == 7)

    def test_nested_image_dict(self) -> None:
        prep = ResizePreprocessor(image_layout="BCHW", image_resolution=(64, 64), mode=ResizeMode.STRETCH)
        images = {"cam0": np.random.rand(1, 3, 32, 32).astype(np.float32)}
        result = prep({IMAGES: images})
        assert result[IMAGES]["cam0"].shape == (1, 3, 64, 64)

    def test_flat_image_keys(self) -> None:
        prep = ResizePreprocessor(image_layout="BCHW", image_resolution=(64, 64), mode=ResizeMode.STRETCH)
        inputs = {
            "images.cam0": np.random.rand(1, 3, 32, 32).astype(np.float32),
            "images.cam0.is_pad": np.zeros((1,), dtype=bool),
        }
        result = prep(inputs)
        assert result["images.cam0"].shape == (1, 3, 64, 64)
        # is_pad keys are left untouched.
        assert result["images.cam0.is_pad"].shape == (1,)

    def test_invalid_ndim_raises(self) -> None:
        prep = ResizePreprocessor(image_layout="BCHW", image_resolution=(64, 64))
        img = np.random.rand(3, 32, 32).astype(np.float32)
        with pytest.raises(ValueError, match="expected"):
            prep({IMAGES: img})

    def test_unsupported_dtype_raises(self) -> None:
        prep = ResizePreprocessor(image_layout="BCHW", image_resolution=(64, 64))
        img = np.zeros((1, 3, 32, 32), dtype=np.int32)
        with pytest.raises(ValueError, match="Unsupported image dtype"):
            prep({IMAGES: img})

    def test_uint8_is_normalized_to_unit_range(self) -> None:
        prep = ResizePreprocessor(image_layout="BCHW", image_resolution=(32, 32), mode=ResizeMode.STRETCH)
        img = np.full((1, 3, 32, 32), 255, dtype=np.uint8)
        result = prep({IMAGES: img})
        out = result[IMAGES]
        assert out.dtype == np.float32
        assert np.allclose(out, 1.0)

    def test_float_output_is_float32(self) -> None:
        prep = ResizePreprocessor(image_layout="BCHW", image_resolution=(32, 32), mode=ResizeMode.STRETCH)
        img = np.random.rand(1, 3, 16, 16).astype(np.float64)
        result = prep({IMAGES: img})
        assert result[IMAGES].dtype == np.float32

    def test_channels_last_input_returns_channels_first(self) -> None:
        prep = ResizePreprocessor(image_layout="BHWC", image_resolution=(64, 64), mode=ResizeMode.STRETCH)
        img = np.random.rand(1, 32, 16, 3).astype(np.float32)
        result = prep({IMAGES: img})
        # Channels-last input is converted to channels-first output.
        assert result[IMAGES].shape == (1, 3, 64, 64)

    def test_channels_last_grayscale_returns_channels_first(self) -> None:
        # Regression: C=1 in BHWC was misidentified as channels-first, producing
        # garbage output with H in the channel position.
        prep = ResizePreprocessor(image_layout="BHWC", image_resolution=(64, 64), mode=ResizeMode.STRETCH)
        img = np.random.rand(1, 47, 32, 1).astype(np.float32)  # (B, H, W, C=1)
        result = prep({IMAGES: img})
        assert result[IMAGES].shape == (1, 1, 64, 64)

    def test_channels_last_two_channel_returns_channels_first(self) -> None:
        # Regression: C=2 in BHWC was misidentified as channels-first.
        prep = ResizePreprocessor(image_layout="BHWC", image_resolution=(64, 64), mode=ResizeMode.STRETCH)
        img = np.random.rand(1, 22, 32, 2).astype(np.float32)  # (B, H, W, C=2)
        result = prep({IMAGES: img})
        assert result[IMAGES].shape == (1, 2, 64, 64)

    def test_channels_last_rgba_returns_channels_first(self) -> None:
        prep = ResizePreprocessor(image_layout="BHWC", image_resolution=(32, 32), mode=ResizeMode.STRETCH)
        img = np.random.rand(1, 16, 16, 4).astype(np.float32)  # (B, H, W, C=4)
        result = prep({IMAGES: img})
        assert result[IMAGES].shape == (1, 4, 32, 32)

    @pytest.mark.parametrize("image_layout", [ImageLayout.BCHW, ImageLayout.BHWC, "BCHW", "BHWC"])
    def test_explicit_layout_preserves_pixels_for_ambiguous_shape(self, image_layout: ImageLayout | str) -> None:
        prep = ResizePreprocessor(image_resolution=(3, 3), image_layout=image_layout)
        img = np.arange(27, dtype=np.float32).reshape(1, 3, 3, 3) / 26
        expected = img if image_layout == "BCHW" else img.transpose(0, 3, 1, 2)
        np.testing.assert_array_equal(prep({IMAGES: img})[IMAGES], expected)

    @pytest.mark.parametrize("image_layout", ["HWC", "CHW", "AUTO", "", "BHWW"])
    def test_invalid_layout_raises(self, image_layout: str) -> None:
        with pytest.raises(ValueError):
            ResizePreprocessor(image_resolution=(64, 64), image_layout=image_layout)

    def test_layout_is_required(self) -> None:
        with pytest.raises(TypeError, match="image_layout"):
            ResizePreprocessor(image_resolution=(64, 64))  # pyrefly: ignore [missing-argument]

    @pytest.mark.parametrize("mode", [ResizeMode.STRETCH, ResizeMode.LETTERBOX])
    @pytest.mark.parametrize("presentation", ["single", "nested", "flat"])
    def test_layouts_produce_identical_resized_pixels(self, mode: ResizeMode, presentation: str) -> None:
        chw = np.arange(2 * 3 * 7 * 4, dtype=np.float32).reshape(2, 3, 7, 4) / 167
        outputs = []
        for image_layout, img in [(ImageLayout.BCHW, chw), (ImageLayout.BHWC, chw.transpose(0, 2, 3, 1))]:
            prep = ResizePreprocessor(image_resolution=(8, 12), image_layout=image_layout, mode=mode)
            if presentation == "nested":
                result = prep({IMAGES: {"top": img, "wrist": img.copy()}})[IMAGES]
                np.testing.assert_array_equal(result["top"], result["wrist"])
                outputs.append(result["top"])
            elif presentation == "flat":
                result = prep({"images.top": img, "images.wrist": img.copy()})
                np.testing.assert_array_equal(result["images.top"], result["images.wrist"])
                out = result["images.top"]
                assert isinstance(out, np.ndarray)
                outputs.append(out)
            else:
                out = prep({IMAGES: img})[IMAGES]
                assert isinstance(out, np.ndarray)
                outputs.append(out)
        assert outputs[0].shape == (2, 3, 8, 12)
        np.testing.assert_array_equal(outputs[0], outputs[1])

    def test_channels_last_uint8_is_normalized(self) -> None:
        prep = ResizePreprocessor(image_layout="BHWC", image_resolution=(32, 32), mode=ResizeMode.STRETCH)
        img = np.full((1, 16, 16, 3), 255, dtype=np.uint8)
        result = prep({IMAGES: img})
        out = result[IMAGES]
        assert out.shape == (1, 3, 32, 32)
        assert out.dtype == np.float32
        assert np.allclose(out, 1.0)

    def test_uint8_resize_matches_float_within_quantization_error(self) -> None:
        prep = ResizePreprocessor(image_layout="BCHW", image_resolution=(96, 96), mode=ResizeMode.LETTERBOX)
        rng = np.random.default_rng(0)

        # Random fp32 image in [0, 1] and its uint8 quantized counterpart.
        img_float = rng.random((1, 3, 48, 64), dtype=np.float32)
        img_uint8 = (img_float * 255.0).round().astype(np.uint8)

        out_float = prep({IMAGES: img_float})[IMAGES]
        out_uint8 = prep({IMAGES: img_uint8})[IMAGES]

        assert out_float.shape == out_uint8.shape
        # Average per-pixel difference should be on the order of the uint8
        # quantization step (1/255 ~= 0.0039).
        mean_abs_diff = np.mean(np.abs(out_float - out_uint8))
        assert mean_abs_diff < 1.0 / 255.0

    def test_zero_height_raises(self) -> None:
        # Regression for fuzzer crash: previously raised ZeroDivisionError.
        prep = ResizePreprocessor(image_layout="BCHW", image_resolution=(64, 64))
        img = np.zeros((1, 3, 0, 64), dtype=np.float32)
        with pytest.raises(ValueError, match="zero spatial dimension"):
            prep({IMAGES: img})

    def test_zero_width_raises(self) -> None:
        prep = ResizePreprocessor(image_layout="BCHW", image_resolution=(64, 64))
        img = np.zeros((1, 3, 64, 0), dtype=np.float32)
        with pytest.raises(ValueError, match="zero spatial dimension"):
            prep({IMAGES: img})
