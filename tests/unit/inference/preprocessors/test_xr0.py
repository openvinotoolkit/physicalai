# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from physicalai.inference.constants import IMAGES, STATE, TASK
from physicalai.inference.preprocessors import Preprocessor, XR0Preprocessor


@pytest.fixture()
def preprocessor():
    return XR0Preprocessor(
        camera_views=("base", "wrist_left"),
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

    def test_empty_camera_views_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one camera view"):
            XR0Preprocessor(camera_views=())

    def test_nonpositive_patch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            XR0Preprocessor(patch_size=0)


class TestXR0PreprocessorOutput:
    def test_output_keys(self, preprocessor) -> None:
        result = preprocessor(_make_inputs())
        assert set(result) == {"pixel_values", "state", TASK}

    def test_pixel_grid_shape_and_dtype(self, preprocessor) -> None:
        result = preprocessor(_make_inputs())
        # (num_images, C, H, W) for the two camera views at 256x256.
        assert result["pixel_values"].shape == (2, 3, 256, 256)
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
            camera_views=("base", "wrist_left"),
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
        # (3 - 1) / (2 + 1e-6) ~= 1.0 on the real dims; padded dims stay 0.
        np.testing.assert_allclose(result["state"][0, 0, :8], 1.0, atol=1e-5)
        np.testing.assert_allclose(result["state"][0, 0, 8:], 0.0, atol=1e-5)
