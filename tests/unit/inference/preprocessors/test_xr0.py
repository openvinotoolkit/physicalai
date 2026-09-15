# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from physicalai.inference.constants import IMAGES, STATE, TASK
from physicalai.inference.preprocessors import Preprocessor, XR0Preprocessor
from physicalai.inference.preprocessors.xr0 import _build_pixel_grid, _render_chat_prompt


@pytest.fixture()
def preprocessor():
    return XR0Preprocessor(
        max_state_dim=32,
        image_factor=32,
        image_max_pixels=90000,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        rescale_factor=1.0 / 255.0,
        patch_size=16,
        merge_size=2,
    )


def _make_inputs(h: int = 256, w: int = 256, state_dim: int = 8) -> dict[str, object]:
    """Build a minimal XR0 observation dict (raw uint8 images, 1-D state)."""
    return {
        IMAGES: {
            "base": np.zeros((h, w, 3), dtype=np.uint8),
            "wrist_left": np.zeros((h, w, 3), dtype=np.uint8),
        },
        STATE: np.zeros((state_dim,), dtype=np.float32),
        TASK: "pick up the cup",
    }


class TestXR0PreprocessorInit:
    def test_is_preprocessor(self, preprocessor) -> None:
        assert isinstance(preprocessor, Preprocessor)

    def test_nonpositive_patch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            XR0Preprocessor(patch_size=0)


class TestXR0PreprocessorOutput:
    def test_output_keys(self, preprocessor) -> None:
        result = preprocessor(_make_inputs())
        assert set(result) == {"pixel_values", "state", TASK}

    def test_pixel_grid_shape_and_dtype(self, preprocessor) -> None:
        result = preprocessor(_make_inputs())
        # Flat patchified pixel_values: 2 views at 256x256 -> grid_h=grid_w=16,
        # 256 tokens/image, 2 images -> 512 rows; feature = C * tp * patch**2.
        assert result["pixel_values"].shape == (512, 3 * 2 * 16 * 16)
        assert result["pixel_values"].dtype == np.float32

    def test_pixel_grid_normalized(self, preprocessor) -> None:
        # Zero uint8 pixels -> (0 * 1/255 - 0.5) / 0.5 == -1.0 for every channel.
        result = preprocessor(_make_inputs())
        np.testing.assert_allclose(result["pixel_values"], -1.0, atol=1e-5)

    def test_state_padded_shape(self, preprocessor) -> None:
        result = preprocessor(_make_inputs(state_dim=8))
        assert result["state"].shape == (1, 1, 32)
        assert result["state"].dtype == np.float32

    def test_state_zero_padding(self, preprocessor) -> None:
        inputs = _make_inputs(state_dim=8)
        inputs[STATE] = np.arange(8, dtype=np.float32)
        result = preprocessor(inputs)
        np.testing.assert_allclose(result["state"][0, 0, :8], np.arange(8))
        np.testing.assert_allclose(result["state"][0, 0, 8:], 0.0)


class TestXR0PreprocessorPrompt:
    def test_task_is_single_element_list(self, preprocessor) -> None:
        result = preprocessor(_make_inputs())
        assert isinstance(result[TASK], list)
        assert len(result[TASK]) == 1

    def test_prompt_structure(self, preprocessor) -> None:
        prompt = preprocessor(_make_inputs())[TASK][0]
        assert prompt.startswith("<|im_start|>user\n")
        assert prompt.endswith("<|im_start|>assistant\n<cot></cot><|im_end|>\n")
        assert "pick up the cup" in prompt
        assert "# Base View" in prompt
        assert "# Left-Wrist View" in prompt

    def test_image_pad_token_count(self, preprocessor) -> None:
        # 256/16 = 16 patches per side; merged 2x2 -> (16*16)/(2*2) = 64 pads/image.
        prompt = preprocessor(_make_inputs())[TASK][0]
        assert prompt.count("<|image_pad|>") == 64 * 2

    def test_missing_task_defaults_empty(self, preprocessor) -> None:
        inputs = _make_inputs()
        inputs.pop(TASK)
        prompt = preprocessor(inputs)[TASK][0]
        assert "<|im_start|>user" in prompt

    def test_task_list_uses_first(self, preprocessor) -> None:
        inputs = _make_inputs()
        inputs[TASK] = ["open the drawer", "ignored"]
        prompt = preprocessor(inputs)[TASK][0]
        assert "open the drawer" in prompt
        assert "ignored" not in prompt


class TestXR0PreprocessorState:
    def test_3d_state_uses_last_timestep(self, preprocessor) -> None:
        inputs = _make_inputs(state_dim=8)
        state = np.zeros((1, 5, 8), dtype=np.float32)
        state[0, -1] = np.arange(8, dtype=np.float32)
        inputs[STATE] = state
        result = preprocessor(inputs)
        np.testing.assert_allclose(result["state"][0, 0, :8], np.arange(8))

    def test_normalized_state(self) -> None:
        prep = XR0Preprocessor(
            max_state_dim=32,
            patch_size=16,
            merge_size=2,
            normalize_state=True,
            state_mean=[1.0] * 8,
            state_std=[2.0] * 8,
        )
        inputs = _make_inputs(state_dim=8)
        inputs[STATE] = np.ones(8, dtype=np.float32) * 3.0
        result = prep(inputs)
        state = np.asarray(result["state"])
        # (3 - 1) / (2 + 1e-6) ~= 1.0 on the real dims; padded dims stay 0.
        np.testing.assert_allclose(state[0, 0, :8], 1.0, atol=1e-5)
        np.testing.assert_allclose(state[0, 0, 8:], 0.0, atol=1e-5)


class TestXR0PreprocessorExtractImages:
    def test_nested_dict_returns_array_per_view(self, preprocessor) -> None:
        views, images = preprocessor._extract_images(_make_inputs())
        assert views == ["base", "wrist_left"]
        assert len(images) == 2
        assert all(isinstance(image, np.ndarray) for image in images)
        assert all(image.dtype == np.uint8 and image.shape[-1] == 3 for image in images)

    def test_resized_to_patch_aligned(self, preprocessor) -> None:
        # 256 is a multiple of factor=32 and area 65536 < 90000 -> unchanged.
        _, images = preprocessor._extract_images(_make_inputs(h=256, w=256))
        assert all(image.shape == (256, 256, 3) for image in images)

    def test_odd_size_rounded_to_factor(self, preprocessor) -> None:
        # 250 rounds to nearest multiple of factor=32 -> 256.
        _, images = preprocessor._extract_images(_make_inputs(h=250, w=250))
        assert all(image.shape == (256, 256, 3) for image in images)

    def test_flattened_keys(self, preprocessor) -> None:
        inputs = {
            f"{IMAGES}.base": np.zeros((256, 256, 3), dtype=np.uint8),
            f"{IMAGES}.wrist_left": np.zeros((256, 256, 3), dtype=np.uint8),
        }
        views, images = preprocessor._extract_images(inputs)
        assert views == ["base", "wrist_left"]
        assert len(images) == 2

    def test_flattened_keys_skip_is_pad(self, preprocessor) -> None:
        inputs = {
            f"{IMAGES}.base": np.zeros((256, 256, 3), dtype=np.uint8),
            f"{IMAGES}.base.is_pad": np.zeros((1,), dtype=bool),
        }
        _, images = preprocessor._extract_images(inputs)
        assert len(images) == 1

    def test_all_views_used(self, preprocessor) -> None:
        # Every provided view is used, in the observation's insertion order.
        inputs = {
            IMAGES: {
                "base": np.zeros((256, 256, 3), dtype=np.uint8),
                "wrist_left": np.zeros((256, 256, 3), dtype=np.uint8),
                "wrist_right": np.zeros((256, 256, 3), dtype=np.uint8),
            },
        }
        views, images = preprocessor._extract_images(inputs)
        assert views == ["base", "wrist_left", "wrist_right"]
        assert len(images) == 3

    def test_selected_in_observation_order(self) -> None:
        # Views follow the observation key insertion order, so pixel_values stay
        # aligned with the prompt sections.
        prep = XR0Preprocessor(max_state_dim=32)
        base = np.zeros((256, 256, 3), dtype=np.uint8)
        wrist = np.full((256, 256, 3), 255, dtype=np.uint8)
        inputs: dict[str, object] = {IMAGES: {"wrist_left": wrist, "base": base}}
        views, images = prep._extract_images(inputs)
        # First image is wrist_left (all 255), second is base (all 0).
        assert views == ["wrist_left", "base"]
        assert images[0].max() == 255
        assert images[1].max() == 0

    def test_arbitrary_keys_used_in_order(self, preprocessor) -> None:
        # Non-reference view names are used as-is in insertion order.
        inputs = {
            f"{IMAGES}.camA": np.zeros((256, 256, 3), dtype=np.uint8),
            f"{IMAGES}.camB": np.zeros((256, 256, 3), dtype=np.uint8),
        }
        views, images = preprocessor._extract_images(inputs)
        assert views == ["camA", "camB"]
        assert len(images) == 2

    def test_no_images_raises(self, preprocessor) -> None:
        with pytest.raises(ValueError, match="at least one image"):
            preprocessor._extract_images({STATE: np.zeros((8,), dtype=np.float32)})


class TestBuildPixelGrid:
    def test_shape_and_dtype(self) -> None:
        images = [np.zeros((32, 48, 3), dtype=np.uint8) for _ in range(2)]
        grid = _build_pixel_grid(images, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 1.0 / 255.0)
        # (num_images, C, H, W) -- channels-first from (H, W, C) input.
        assert grid.shape == (2, 3, 32, 48)
        assert grid.dtype == np.float32

    def test_zero_pixels_normalized(self) -> None:
        images = [np.zeros((16, 16, 3), dtype=np.uint8)]
        grid = _build_pixel_grid(images, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 1.0 / 255.0)
        # (0 * 1/255 - 0.5) / 0.5 == -1.0 for every channel.
        np.testing.assert_allclose(grid, -1.0, atol=1e-5)

    def test_per_channel_stats(self) -> None:
        # Distinct per-channel value + stats verify the mean/std are applied per channel.
        arr = np.zeros((2, 2, 3), dtype=np.uint8)
        arr[..., 0], arr[..., 1], arr[..., 2] = 10, 20, 30
        grid = _build_pixel_grid([arr], (1.0, 2.0, 3.0), (2.0, 4.0, 6.0), 1.0)
        # (value * 1.0 - mean) / std per channel.
        np.testing.assert_allclose(grid[0, 0], (10 - 1.0) / 2.0, atol=1e-5)
        np.testing.assert_allclose(grid[0, 1], (20 - 2.0) / 4.0, atol=1e-5)
        np.testing.assert_allclose(grid[0, 2], (30 - 3.0) / 6.0, atol=1e-5)

    def test_hwc_to_chw_transpose(self) -> None:
        # Unique value per pixel/channel so a wrong transpose would be detected.
        arr = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)  # (H, W, C)
        grid = _build_pixel_grid([arr], (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1.0)
        np.testing.assert_allclose(grid[0], np.transpose(arr.astype(np.float32), (2, 0, 1)))


class TestRenderChatPrompt:
    def test_chat_envelope(self) -> None:
        prompt = _render_chat_prompt(["base"], [4], "pick up the cup")
        assert prompt.startswith("<|im_start|>user\n")
        assert prompt.endswith("<|im_start|>assistant\n<cot></cot><|im_end|>\n")

    def test_multi_view_header_and_titles(self) -> None:
        prompt = _render_chat_prompt(["base", "wrist_left"], [1, 1], "task")
        assert "The following observations are captured from multiple views.\n" in prompt
        assert "# Base View" in prompt
        assert "# Left-Wrist View" in prompt
        # View order preserved: base section precedes wrist-left section.
        assert prompt.index("# Base View") < prompt.index("# Left-Wrist View")

    def test_unknown_view_title_capitalized(self) -> None:
        prompt = _render_chat_prompt(["top_down"], [1], "task")
        assert "# Top Down View" in prompt

    def test_image_pad_expansion_per_view(self) -> None:
        prompt = _render_chat_prompt(["base", "wrist_left"], [3, 5], "task")
        assert prompt.count("<|image_pad|>") == 3 + 5
        # Each view wraps its pads in vision-start/end.
        assert prompt.count("<|vision_start|>") == 2
        assert prompt.count("<|vision_end|>") == 2

    def test_instruction_embedded(self) -> None:
        prompt = _render_chat_prompt(["base"], [1], "open the drawer")
        assert "Generate robot actions for the task:\nopen the drawer /no_cot" in prompt

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="must have the same length"):
            _render_chat_prompt(["base", "wrist_left"], [1], "task")


class TestPrepareState:
    @pytest.fixture()
    def small_preprocessor(self):
        # Small max_state_dim keeps the reference tensors tiny.
        return XR0Preprocessor(max_state_dim=4, patch_size=16, merge_size=2)

    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            # 1-D (D,) shorter than max_state_dim -> promoted to (1, D) then zero-padded.
            (np.array([1.0, 2.0], dtype=np.float32), np.array([[[1.0, 2.0, 0.0, 0.0]]], dtype=np.float32)),
            # 1-D exact length -> no padding.
            (
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32),
            ),
            # 1-D longer than max_state_dim -> truncated to first 4.
            (
                np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
                np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32),
            ),
            # 2-D (B, D) batch preserved, padded per row.
            (
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                np.array([[[1.0, 2.0, 0.0, 0.0]], [[3.0, 4.0, 0.0, 0.0]]], dtype=np.float32),
            ),
            # 3-D (B, T, D) -> last timestep only, then padded.
            (
                np.array([[[9.0, 9.0], [1.0, 2.0]]], dtype=np.float32),
                np.array([[[1.0, 2.0, 0.0, 0.0]]], dtype=np.float32),
            ),
            # 3-D multi-batch (B=2, T=2, D=2) -> per-sample last timestep, then padded.
            (
                np.array([[[9.0, 9.0], [1.0, 2.0]], [[8.0, 8.0], [3.0, 4.0]]], dtype=np.float32),
                np.array([[[1.0, 2.0, 0.0, 0.0]], [[3.0, 4.0, 0.0, 0.0]]], dtype=np.float32),
            ),
        ],
    )
    def test_prepare_state(self, small_preprocessor, state, expected) -> None:
        result = small_preprocessor._prepare_state({STATE: state})
        assert result.shape == expected.shape
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            # (value - mean) / (std + eps); real dims use given stats, padded dims stay 0.
            # mean=[1,1], std=[2,2]: (3-1)/2=1.0, (5-1)/2=2.0.
            (
                np.array([3.0, 5.0], dtype=np.float32),
                np.array([[[1.0, 2.0, 0.0, 0.0]]], dtype=np.float32),
            ),
            # 3-D (B, T, D) last timestep normalized the same way.
            (
                np.array([[[9.0, 9.0], [3.0, 5.0]]], dtype=np.float32),
                np.array([[[1.0, 2.0, 0.0, 0.0]]], dtype=np.float32),
            ),
        ],
    )
    def test_prepare_state_normalized(self, state, expected) -> None:
        prep = XR0Preprocessor(
            max_state_dim=4,
            patch_size=16,
            merge_size=2,
            normalize_state=True,
            state_mean=[1.0, 1.0],
            state_std=[2.0, 2.0],
        )
        result = prep._prepare_state({STATE: state})
        assert result.shape == expected.shape
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_missing_state_raises(self, small_preprocessor) -> None:
        with pytest.raises(ValueError, match="requires a 'state'"):
            small_preprocessor._prepare_state({})


class TestXR0PreprocessorCall:
    def test_output_structure_and_contiguity(self, preprocessor) -> None:
        result = preprocessor(_make_inputs())
        assert set(result) == {"pixel_values", "state", TASK}
        assert isinstance(result["pixel_values"], np.ndarray)
        assert result["pixel_values"].dtype == np.float32
        assert result["pixel_values"].flags["C_CONTIGUOUS"]
        assert isinstance(result["state"], np.ndarray)
        assert result["state"].dtype == np.float32
        assert result["state"].flags["C_CONTIGUOUS"]
        assert isinstance(result[TASK], list)
        assert len(result[TASK]) == 1
        assert isinstance(result[TASK][0], str)

    def test_pixel_grid_matches_view_count(self, preprocessor) -> None:
        # Flat patchified pixel_values: 2 views at 256x256 -> 256 tokens/image,
        # 2 images -> 512 rows; feature = C * temporal_patch_size * patch**2.
        result = preprocessor(_make_inputs())
        assert result["pixel_values"].shape == (512, 3 * 2 * 16 * 16)

    def test_pad_counts_derive_from_resized_dims(self, preprocessor) -> None:
        # Non-square 128x256 stays patch-aligned (multiples of factor=32).
        # patch_size=16 -> grid 8x16; merge 2x2 -> (8*16)/(2*2) = 32 pads/image.
        prompt = preprocessor(_make_inputs(h=128, w=256))[TASK][0]
        assert prompt.count("<|image_pad|>") == 32 * 2
        assert prompt.count("<|vision_start|>") == 2

    def test_fewer_images_than_views_truncates(self, preprocessor) -> None:
        # Only one flattened image -> single view drives pixel grid and prompt.
        inputs = {
            f"{IMAGES}.base": np.zeros((256, 256, 3), dtype=np.uint8),
            STATE: np.zeros((8,), dtype=np.float32),
            TASK: "pick up the cup",
        }
        result = preprocessor(inputs)
        # Single 256x256 view -> 256 patch rows; feature = C * tp * patch**2.
        assert result["pixel_values"].shape == (256, 3 * 2 * 16 * 16)
        prompt = result[TASK][0]
        assert "# Base View" in prompt
        assert "# Left-Wrist View" not in prompt
        assert prompt.count("<|vision_start|>") == 1
