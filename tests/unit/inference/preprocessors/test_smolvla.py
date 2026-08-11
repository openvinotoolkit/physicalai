# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from physicalai.inference.constants import IMAGE_MASKS, IMAGES
from physicalai.inference.preprocessors import Preprocessor, ResizeSmolVLA


class TestResizeSmolVLAInit:
    def test_is_preprocessor(self) -> None:
        prep = ResizeSmolVLA()
        assert isinstance(prep, Preprocessor)

    def test_default_resolution(self) -> None:
        prep = ResizeSmolVLA()
        assert prep.image_resolution == (512, 512)

    def test_custom_resolution(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(256, 256))
        assert prep.image_resolution == (256, 256)


class TestResizeSmolVLACall:
    def test_output_keys(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        img = np.random.rand(1, 3, 64, 64).astype(np.float32)
        result = prep({IMAGES: img})
        assert IMAGES in result
        assert IMAGE_MASKS in result

    def test_output_shape(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        img = np.random.rand(1, 3, 32, 32).astype(np.float32)
        result = prep({IMAGES: img})
        # 1 image → stacked with extra dim: (1, batch, channels, H, W)
        assert result[IMAGES].shape[2] == 3
        assert result[IMAGES].shape[3] == 64
        assert result[IMAGES].shape[4] == 64

    def test_pixel_range_normalised(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        img = np.ones((1, 3, 64, 64), dtype=np.float32)
        result = prep({IMAGES: img})
        # input 1.0 → 1.0 * 2 - 1 = 1.0
        np.testing.assert_allclose(result[IMAGES].max(), 1.0, atol=1e-5)

    def test_pixel_range_zeros(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        img = np.zeros((1, 3, 64, 64), dtype=np.float32)
        result = prep({IMAGES: img})
        # input 0.0 → 0.0 * 2 - 1 = -1.0
        np.testing.assert_allclose(result[IMAGES].min(), -1.0, atol=1e-5)

    def test_masks_are_boolean_ones(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        img = np.random.rand(2, 3, 64, 64).astype(np.float32)
        result = prep({IMAGES: img})
        assert result[IMAGE_MASKS].dtype == np.bool_
        assert result[IMAGE_MASKS].all()

    def test_preserves_other_keys(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        img = np.random.rand(1, 3, 64, 64).astype(np.float32)
        result = prep({IMAGES: img, "task": "pick up"})
        assert result["task"] == "pick up"

    def test_multiple_image_keys(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        inputs = {
            f"{IMAGES}.0": np.random.rand(1, 3, 64, 64).astype(np.float32),
            f"{IMAGES}.1": np.random.rand(1, 3, 64, 64).astype(np.float32),
        }
        result = prep(inputs)
        # Two images stacked
        assert result[IMAGES].shape[0] == 2

    def test_non_square_image_padded(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        # Wide image: height < width
        img = np.random.rand(1, 3, 32, 64).astype(np.float32)
        result = prep({IMAGES: img})
        assert result[IMAGES].shape[3] == 64
        assert result[IMAGES].shape[4] == 64

    def test_dict_images_stacked(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        inputs = {
            IMAGES: {
                "top": np.random.rand(1, 3, 48, 48).astype(np.float32),
                "wrist": np.random.rand(1, 3, 32, 64).astype(np.float32),
            },
        }
        result = prep(inputs)
        assert result[IMAGES].shape == (2, 1, 3, 64, 64)
        assert result[IMAGE_MASKS].shape == (2, 1)
        assert result[IMAGE_MASKS].all()


class TestResizeSmolVLADtypeAndLayout:
    def test_uint8_input_normalised(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        # uint8 255 → 1.0 → 1.0 * 2 - 1 = 1.0
        img = np.full((1, 3, 64, 64), 255, dtype=np.uint8)
        result = prep({IMAGES: img})
        assert result[IMAGES].dtype == np.float32
        assert result[IMAGES].shape[1:] == img.shape
        np.testing.assert_allclose(result[IMAGES].max(), 1.0, atol=1e-5)

    def test_uint8_zeros_normalised(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        # uint8 0 → 0.0 → 0.0 * 2 - 1 = -1.0
        img = np.zeros((1, 3, 64, 64), dtype=np.uint8)
        result = prep({IMAGES: img})
        assert result[IMAGES].dtype == np.float32
        assert result[IMAGES].shape[1:] == img.shape
        np.testing.assert_allclose(result[IMAGES].min(), -1.0, atol=1e-5)

    def test_non_float32_dtype_converted(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        img = np.ones((1, 3, 64, 64), dtype=np.float64)
        result = prep({IMAGES: img})
        assert result[IMAGES].shape[1:] == img.shape
        assert result[IMAGES].dtype == np.float32

    def test_unsupported_dtype_raises(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        img = np.ones((1, 3, 64, 64), dtype=np.int32)
        with pytest.raises(ValueError, match="Unsupported image dtype"):
            prep({IMAGES: img})

    def test_out_of_range_float_clamped_to_pixel_bounds(self) -> None:
        # Regression: arbitrary float values previously caused output far outside [-1, 1].
        prep = ResizeSmolVLA(image_resolution=(32, 32))
        img = np.full((1, 3, 32, 32), -1e30, dtype=np.float32)
        result = prep({IMAGES: img})
        out = result[IMAGES]
        assert float(out.min()) >= -1.0 - 1e-5
        assert float(out.max()) <= 1.0 + 1e-5

    def test_nan_float_input_clamped_to_pixel_bounds(self) -> None:
        # Regression: NaN propagates through np.clip unchanged; nan_to_num handles it.
        prep = ResizeSmolVLA(image_resolution=(32, 32))
        img = np.full((1, 3, 32, 32), float("nan"), dtype=np.float32)
        result = prep({IMAGES: img})
        out = result[IMAGES]
        assert not np.any(np.isnan(out))
        assert float(out.min()) >= -1.0 - 1e-5
        assert float(out.max()) <= 1.0 + 1e-5

    def test_channels_last_transposed(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        # (B, H, W, C) input should be transposed to (B, C, H, W) internally
        img = np.random.rand(1, 32, 32, 3).astype(np.float32)
        result = prep({IMAGES: img})
        assert result[IMAGES].shape[2] == 3
        assert result[IMAGES].shape[3] == 64
        assert result[IMAGES].shape[4] == 64

    def test_channels_last_uint8_matches_channels_first(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        chw = np.random.randint(0, 256, size=(1, 3, 48, 48), dtype=np.uint8)
        hwc = np.transpose(chw, (0, 2, 3, 1))
        result_chw = prep({IMAGES: chw})
        result_hwc = prep({IMAGES: hwc})
        np.testing.assert_array_equal(result_chw[IMAGES], result_hwc[IMAGES])

    def test_ambiguous_layout_raises(self) -> None:
        # Both dim-1 and dim-4 are in {1,2,3,4} — layout is indeterminate.
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        img = np.random.rand(1, 3, 3, 3).astype(np.float32)
        with pytest.raises(ValueError, match="ambiguous layout"):
            prep({IMAGES: img})

    def test_extreme_aspect_ratio_does_not_crash(self) -> None:
        # Regression: int(cur_height / ratio) could be 0 for extreme ratios, crashing cv2.
        prep = ResizeSmolVLA(image_resolution=(512, 512))
        img = np.zeros((1, 3, 1, 256), dtype=np.float32)  # very thin image
        result = prep({IMAGES: img})
        assert result[IMAGES].shape[2] >= 1
        assert result[IMAGES].shape[3] >= 1


class TestResizeSmolVLAResizeWithPad:
    def test_invalid_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="expected"):
            ResizeSmolVLA._resize_with_pad(np.zeros((3, 64, 64)), 64, 64)

    def test_no_pad_when_matching(self) -> None:
        img = np.random.rand(1, 3, 64, 64).astype(np.float32)
        result = ResizeSmolVLA._resize_with_pad(img, 64, 64)
        assert result.shape == (1, 3, 64, 64)

    def test_pads_to_target_size(self) -> None:
        img = np.random.rand(1, 3, 32, 64).astype(np.float32)
        result = ResizeSmolVLA._resize_with_pad(img, 64, 64)
        assert result.shape[2] == 64
        assert result.shape[3] == 64

    def test_zero_height_raises(self) -> None:
        # Regression for fuzzer crash: previously raised ZeroDivisionError.
        prep = ResizeSmolVLA(image_resolution=(512, 512))
        img = np.zeros((1, 3, 0, 64), dtype=np.float32)
        with pytest.raises(ValueError, match="zero spatial dimension"):
            prep({IMAGES: img})

    def test_zero_width_raises(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(512, 512))
        img = np.zeros((1, 3, 64, 0), dtype=np.float32)
        with pytest.raises(ValueError, match="zero spatial dimension"):
            prep({IMAGES: img})


class TestResizeSmolVLAImageKeyReorderMapInit:
    """Tests for image_key_reorder_map and num_cameras params added for physical-ai-studio compat."""

    def test_accepts_image_key_reorder_map(self) -> None:
        prep = ResizeSmolVLA(image_key_reorder_map={"left": 0, "right": 1})
        assert prep._image_key_reorder_map == {"left": 0, "right": 1}

    def test_prefixed_keys_normalised_to_bare(self) -> None:
        # Studio may export "images.wrist" or bare "wrist"; both must normalise to bare.
        prep = ResizeSmolVLA(image_key_reorder_map={"images.wrist": 0, "images.overhead": 1})
        assert prep._image_key_reorder_map == {"wrist": 0, "overhead": 1}

    def test_mixed_bare_and_prefixed_keys_normalised(self) -> None:
        prep = ResizeSmolVLA(image_key_reorder_map={"images.wrist": 0, "overhead": 1})
        assert prep._image_key_reorder_map == {"wrist": 0, "overhead": 1}

    def test_accepts_num_cameras(self) -> None:
        prep = ResizeSmolVLA(num_cameras=3)
        assert prep._num_cameras == 3

    def test_none_image_key_reorder_map_stays_none(self) -> None:
        prep = ResizeSmolVLA(image_key_reorder_map=None)
        assert prep._image_key_reorder_map is None

    def test_default_image_key_reorder_map_is_none(self) -> None:
        assert ResizeSmolVLA()._image_key_reorder_map is None

    def test_default_num_cameras_is_zero(self) -> None:
        assert ResizeSmolVLA()._num_cameras == 0


class TestResizeSmolVLAResolveImageOrder:
    def test_natural_order_when_no_map(self) -> None:
        prep = ResizeSmolVLA()
        assert prep._resolve_image_order(["b", "a", "c"]) == ["b", "a", "c"]

    def test_reorder_by_map(self) -> None:
        prep = ResizeSmolVLA(image_key_reorder_map={"right": 0, "left": 1})
        assert prep._resolve_image_order(["left", "right"]) == ["right", "left"]

    def test_reorder_by_map_with_prefixed_img_keys(self) -> None:
        # Studio exports bare keys; PolicySource sends images.-prefixed flat keys.
        prep = ResizeSmolVLA(image_key_reorder_map={"right": 0, "left": 1}, num_cameras=2)
        assert prep._resolve_image_order(["images.left", "images.right"]) == ["images.right", "images.left"]

    def test_num_cameras_fills_none_slots(self) -> None:
        prep = ResizeSmolVLA(image_key_reorder_map={"cam": 1}, num_cameras=3)
        layout = prep._resolve_image_order(["cam"])
        assert len(layout) == 3
        assert layout[1] == "cam"
        assert layout[0] is None
        assert layout[2] is None

    def test_num_cameras_missing_key_in_map_fills_none_slot(self) -> None:
        # Map describes all possible cameras; only a subset is present in inputs.
        prep = ResizeSmolVLA(image_key_reorder_map={"cam0": 0, "cam1": 1}, num_cameras=2)
        assert prep._resolve_image_order(["cam1"]) == [None, "cam1"]

    def test_exact_camera_count_no_none_slots(self) -> None:
        prep = ResizeSmolVLA(image_key_reorder_map={"a": 0, "b": 1}, num_cameras=2)
        assert prep._resolve_image_order(["a", "b"]) == ["a", "b"]

    def test_input_key_missing_from_map_raises(self) -> None:
        prep = ResizeSmolVLA(image_key_reorder_map={"known": 0}, num_cameras=2)
        with pytest.raises(ValueError, match="Missing slot mapping for image keys"):
            prep._resolve_image_order(["known", "unknown"])

    def test_num_cameras_smaller_than_input_count_raises(self) -> None:
        prep = ResizeSmolVLA(image_key_reorder_map={"a": 0, "b": 1, "c": 2}, num_cameras=2)
        with pytest.raises(ValueError, match="num_cameras"):
            prep._resolve_image_order(["a", "b", "c"])

    def test_slot_index_out_of_range_raises(self) -> None:
        prep = ResizeSmolVLA(image_key_reorder_map={"cam": 5}, num_cameras=2)
        with pytest.raises(ValueError, match="out of range"):
            prep._resolve_image_order(["cam"])

    def test_duplicate_slot_index_raises(self) -> None:
        prep = ResizeSmolVLA(image_key_reorder_map={"a": 0, "b": 0}, num_cameras=2)
        with pytest.raises(ValueError, match="Duplicate camera slot"):
            prep._resolve_image_order(["a", "b"])


class TestResizeSmolVLANoneCameraSlot:
    def test_single_camera_ndarray_bypasses_reorder_map(self) -> None:
        # {IMAGES: ndarray} takes the isinstance(np.ndarray) branch; map is never consulted.
        prep = ResizeSmolVLA(image_key_reorder_map={"cam": 0}, num_cameras=1, image_resolution=(8, 8))
        img = np.ones((1, 3, 8, 8), dtype=np.float32)
        result = prep({IMAGES: img})
        assert result[IMAGES].shape[0] == 1

    def test_single_flat_key_bypasses_reorder_map(self) -> None:
        # {"images.cam": ndarray} hits the len(img_keys)==1 branch; map is never consulted.
        prep = ResizeSmolVLA(image_key_reorder_map={"other": 0}, num_cameras=1, image_resolution=(8, 8))
        img = np.ones((1, 3, 8, 8), dtype=np.float32)
        result = prep({f"{IMAGES}.cam": img})
        assert result[IMAGES].shape[0] == 1

    def test_none_slot_produces_black_image(self) -> None:
        # num_cameras=2, only slot 1 supplied → slot 0 is None → black image
        prep = ResizeSmolVLA(image_key_reorder_map={"cam": 1}, num_cameras=2, image_resolution=(8, 8))
        img = np.ones((1, 3, 8, 8), dtype=np.float32)
        result = prep({IMAGES: {"cam": img}})
        assert result[IMAGES].shape[0] == 2
        np.testing.assert_allclose(result[IMAGES][0], -1.0, atol=1e-5)

    def test_none_slot_mask_is_false(self) -> None:
        prep = ResizeSmolVLA(image_key_reorder_map={"cam": 1}, num_cameras=2, image_resolution=(8, 8))
        img = np.ones((1, 3, 8, 8), dtype=np.float32)
        result = prep({IMAGES: {"cam": img}})
        assert not result[IMAGE_MASKS][0].any()

    def test_real_slot_mask_is_true(self) -> None:
        prep = ResizeSmolVLA(image_key_reorder_map={"cam": 1}, num_cameras=2, image_resolution=(8, 8))
        img = np.ones((1, 3, 8, 8), dtype=np.float32)
        result = prep({IMAGES: {"cam": img}})
        assert result[IMAGE_MASKS][1].all()


class TestResizeSmolVLAComponentSpecCompat:
    """Regression: physical-ai-studio exports image_key_reorder_map and num_cameras
    into ComponentSpec; instantiate_component must forward them to ResizeSmolVLA.__init__."""

    def test_instantiate_via_component_spec(self) -> None:
        from physicalai.inference.component_factory import instantiate_component
        from physicalai.inference.manifest import ComponentSpec

        spec = ComponentSpec(
            type="smolvla_resize",
            image_resolution=[64, 64],
            image_key_reorder_map={"left": 0, "right": 1},
            num_cameras=2,
        )
        prep = instantiate_component(spec)
        assert isinstance(prep, ResizeSmolVLA)
        assert prep._image_key_reorder_map == {"left": 0, "right": 1}
        assert prep._num_cameras == 2

    def test_instantiate_via_component_spec_empty_map(self) -> None:
        from physicalai.inference.component_factory import instantiate_component
        from physicalai.inference.manifest import ComponentSpec

        # Mirrors the default export when no reordering is needed; {} normalises to None.
        spec = ComponentSpec(type="smolvla_resize", image_resolution=[512, 512], image_key_reorder_map={}, num_cameras=0)
        prep = instantiate_component(spec)
        assert isinstance(prep, ResizeSmolVLA)
        assert not prep._image_key_reorder_map
        assert prep._num_cameras == 0
