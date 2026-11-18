# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for exported Patch Policy image processing."""

from __future__ import annotations

import numpy as np
import pytest

from physicalai.inference.component_factory import instantiate_component
from physicalai.inference.manifest import ComponentSpec
from physicalai.inference.preprocessors import (
    PatchPolicyPreprocessor,
    Preprocessor,
    ResizePreprocessor,
    StatsNormalizer,
    ToFloatTensorPreprocessor,
)


def test_instantiates_from_registered_manifest_name() -> None:
    """The short component name resolves from exported manifests."""
    spec = ComponentSpec.model_validate({
        "type": "patch_policy",
        "camera_names": ["front"],
    })

    processor = instantiate_component(Preprocessor, spec)

    assert isinstance(processor, PatchPolicyPreprocessor)  # ruff: ignore[assert]


def test_orders_and_flattens_temporal_camera_views_without_changing_pixels() -> None:
    """Camera preparation preserves image values, dtype, and resolution."""
    processor = PatchPolicyPreprocessor(
        ["front", "wrist"],
        n_obs_steps=2,
    )
    front = np.full((1, 2, 3, 4, 5), 64, dtype=np.uint8)
    wrist = np.full((1, 2, 3, 4, 5), 128, dtype=np.uint8)

    result = processor({"images": {"wrist": wrist, "front": front}})

    np.testing.assert_equal(list(result), ["images.front", "images.wrist"])
    np.testing.assert_array_equal(result["images.front"], front)
    np.testing.assert_array_equal(result["images.wrist"], wrist)
    np.testing.assert_equal(result["images.front"].dtype, np.dtype(np.uint8))


def test_adds_batch_and_temporal_axes_for_single_images() -> None:
    """Unbatched images receive batch and temporal axes only."""
    processor = PatchPolicyPreprocessor(["front"])

    result = processor({"images.front": np.ones((3, 2, 2), dtype=np.float32)})

    np.testing.assert_equal(result["images.front"].shape, (1, 1, 3, 2, 2))


def test_infers_batched_channels_last_camera_layout() -> None:
    """Raw Runtime camera frames may arrive in BHWC layout."""
    processor = PatchPolicyPreprocessor(["front"])
    resizer = ResizePreprocessor((2, 2), mode="stretch")

    result = resizer(processor({"images": {"front": np.ones((1, 6, 5, 3), dtype=np.float32)}}))

    np.testing.assert_equal(result["images.front"].shape, (1, 1, 3, 2, 2))


def test_composes_with_stats_normalizer_using_camera_stats() -> None:
    """Per-camera statistics broadcast over batch, time, and pixels."""
    processor = PatchPolicyPreprocessor(["front", "wrist"])
    resizer = ResizePreprocessor((2, 2), image_layout="BCHW", mode="stretch")
    normalizer = StatsNormalizer(
        mode="mean_std",
        stats={
            "images.front": {
                "mean": np.full((3, 1, 1), 0.5, dtype=np.float32),
                "std": np.full((3, 1, 1), 0.5, dtype=np.float32),
            },
            "images.wrist": {
                "mean": np.full((3, 1, 1), 1.0, dtype=np.float32),
                "std": np.full((3, 1, 1), 2.0, dtype=np.float32),
            },
        },
    )
    inputs = {
        "images": {
            "front": np.ones((1, 3, 2, 2), dtype=np.float32),
            "wrist": np.full((1, 3, 2, 2), 3.0, dtype=np.float32),
        },
    }

    result = normalizer(resizer(ToFloatTensorPreprocessor()(processor(inputs))))

    np.testing.assert_allclose(result["images.front"], 1.0)
    np.testing.assert_allclose(result["images.wrist"], 1.0)


def test_shared_normalizer_and_float_transform_produce_model_dtype() -> None:
    """The shared pipeline converts normalized manifest-list values to float32."""
    normalizer = StatsNormalizer(stats={"images": {"mean": [1.0], "std": [2.0]}}, mode="mean_std")

    normalized = normalizer({"images": np.array([3.0], dtype=np.float32)})
    inputs: dict[str, np.ndarray | dict[str, np.ndarray]] = dict(normalized)
    result = ToFloatTensorPreprocessor()(inputs)
    image = result["images"]
    if not isinstance(image, np.ndarray):
        msg = "Expected the flat images value to remain a NumPy array."
        raise AssertionError(msg)

    np.testing.assert_equal(image.dtype, np.dtype(np.float32))
    np.testing.assert_allclose(image, [1.0])


def test_rejects_missing_or_misaligned_cameras() -> None:
    """Missing views and inconsistent leading dimensions fail clearly."""
    processor = PatchPolicyPreprocessor(["front", "wrist"])
    with pytest.raises(KeyError, match="wrist"):
        processor({"images": {"front": np.ones((1, 3, 2, 2), dtype=np.float32)}})

    with pytest.raises(ValueError, match="batch and temporal"):
        processor(
            {
                "images": {
                    "front": np.ones((1, 3, 2, 2), dtype=np.float32),
                    "wrist": np.ones((2, 3, 2, 2), dtype=np.float32),
                },
            },
        )


def test_rejects_unexpected_temporal_length() -> None:
    """Only singleton frames or full observation windows are accepted."""
    processor = PatchPolicyPreprocessor(["front"], n_obs_steps=2)

    with pytest.raises(ValueError, match="expected 1 or 2"):
        processor({"images": {"front": np.ones((1, 3, 3, 2, 2), dtype=np.float32)}})


def test_accepts_singleton_observation_without_repeating_history() -> None:
    """A current frame stays a singleton even when the policy uses a full window."""
    processor = PatchPolicyPreprocessor(["front"], n_obs_steps=3)
    result = processor({"images.front": np.ones((2, 1, 3, 2, 2), dtype=np.float32)})

    np.testing.assert_equal(result["images.front"].shape, (2, 1, 3, 2, 2))


def test_preserves_non_image_inputs_without_processing_actions() -> None:
    """Action values and padding metadata pass through without transformation."""
    processor = PatchPolicyPreprocessor(["front"])
    actions = np.array([[[100.0, -50.0]]], dtype=np.float64)
    padding = np.array([[False]])
    result = processor({
        "images.front": np.zeros((1, 3, 2, 2), dtype=np.uint8),
        "action": actions,
        "extra.action_is_pad": padding,
        "images.front.is_pad": padding,
    })

    assert result["action"] is actions  # ruff: ignore[assert]
    assert result["extra.action_is_pad"] is padding  # ruff: ignore[assert]
    assert result["images.front.is_pad"] is padding  # ruff: ignore[assert]
