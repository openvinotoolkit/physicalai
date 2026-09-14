# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the runtime RLDX-1 frozen-geometry preprocessor."""

from __future__ import annotations

import numpy as np

from physicalai.inference.constants import TASK
from physicalai.inference.preprocessors.rldx1 import IMAGE_GRID_THW, PIXEL_VALUES, Rldx1Preprocessor


def test_single_sample_pixel_values_match_export_surface() -> None:
    """Single-sample inference emits the flattened patch matrix expected by export."""
    preprocessor = Rldx1Preprocessor(image_resolution=(32, 32), num_views=1, num_frames=1)
    image = np.zeros((1, 1, 32, 32, 3), dtype=np.uint8)

    outputs = preprocessor({"images": image, "task": "Pick up block"})

    assert outputs[PIXEL_VALUES].shape == (4, 1536)
    assert outputs[IMAGE_GRID_THW].shape == (1, 1, 3)


def test_batched_pixel_values_keep_batch_axis() -> None:
    """Batched calls preserve the leading batch axis for each sample's patch matrix."""
    preprocessor = Rldx1Preprocessor(image_resolution=(32, 32), num_views=1, num_frames=1)
    images = np.zeros((2, 1, 32, 32, 3), dtype=np.uint8)

    outputs = preprocessor({"images": images, "task": ["Pick", "Place"]})

    assert outputs[PIXEL_VALUES].shape == (2, 4, 1536)
    assert outputs[IMAGE_GRID_THW].shape == (2, 1, 3)


def test_state_is_zero_padded_to_max_state_dim() -> None:
    """Short state vectors fill the leading slots and zero-pad the tail."""
    preprocessor = Rldx1Preprocessor(image_resolution=(32, 32), num_views=1, num_frames=1, max_state_dim=64)
    image = np.zeros((1, 1, 32, 32, 3), dtype=np.uint8)
    state = np.arange(8, dtype=np.float32).reshape(1, 1, 8)

    outputs = preprocessor({"images": image, "task": "Pick up block", "state": state})

    assert outputs["state"].shape == (1, 1, 64)
    np.testing.assert_array_equal(outputs["state"][0, 0, :8], state[0, 0])
    np.testing.assert_array_equal(outputs["state"][0, 0, 8:], np.zeros(56, dtype=np.float32))


def test_state_without_time_axis_is_coerced_and_padded() -> None:
    """A ``(B, D)`` state input becomes ``(B, 1, max_state_dim)``."""
    preprocessor = Rldx1Preprocessor(image_resolution=(32, 32), num_views=1, num_frames=1, max_state_dim=64)
    image = np.zeros((1, 1, 32, 32, 3), dtype=np.uint8)
    state = np.arange(8, dtype=np.float32).reshape(1, 8)

    outputs = preprocessor({"images": image, "task": "Pick up block", "state": state})

    assert outputs["state"].shape == (1, 1, 64)
    np.testing.assert_array_equal(outputs["state"][0, 0, :8], state[0])


def test_task_only_mode_emits_normalized_text_without_markers() -> None:
    """Preprocessor emits normalized natural language for downstream tokenization."""
    preprocessor = Rldx1Preprocessor(image_resolution=(32, 32), num_views=1, num_frames=1)
    image = np.zeros((1, 1, 32, 32, 3), dtype=np.uint8)

    outputs = preprocessor({"images": image, "task": "Put the mug on the plate."})

    assert outputs[TASK] == ["put the mug on the plate"]