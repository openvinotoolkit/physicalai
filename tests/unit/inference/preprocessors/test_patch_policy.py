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
    StatsNormalizer,
    ToFloatTensorPreprocessor,
)


def test_instantiates_from_registered_manifest_name() -> None:
    """The short component name resolves from exported manifests."""
    spec = ComponentSpec.model_validate({
        "type": "patch_policy",
        "camera_names": ["front"],
        "image_resolution": [2, 2],
        "image_layout": "BCHW",
    })

    processor = instantiate_component(Preprocessor, spec)

    assert isinstance(processor, PatchPolicyPreprocessor)  # ruff: ignore[assert]


def test_orders_resizes_and_stacks_temporal_camera_views() -> None:
    """Configured camera order controls the view axis for temporal inputs."""
    processor = PatchPolicyPreprocessor(
        ["front", "wrist"],
        (2, 3),
        image_layout="BCHW",
        n_obs_steps=2,
    )
    front = np.full((1, 2, 3, 4, 5), 64, dtype=np.uint8)
    wrist = np.full((1, 2, 3, 4, 5), 128, dtype=np.uint8)

    result = processor({"images": {"wrist": wrist, "front": front}})

    np.testing.assert_equal(result["images"].shape, (1, 2, 2, 3, 2, 3))
    np.testing.assert_equal(result["images"].dtype, np.dtype(np.float32))
    np.testing.assert_allclose(result["images"][:, :, 0], 64.0 / 255.0)
    np.testing.assert_allclose(result["images"][:, :, 1], 128.0 / 255.0)


def test_adds_batch_and_temporal_axes_for_single_images() -> None:
    """Unbatched images receive batch, temporal, and view axes."""
    processor = PatchPolicyPreprocessor(["front"], (2, 2), image_layout="BCHW")

    result = processor({"images.front": np.ones((3, 2, 2), dtype=np.float32)})

    np.testing.assert_equal(result["images"].shape, (1, 1, 1, 3, 2, 2))


def test_infers_batched_channels_last_camera_layout() -> None:
    """Raw Runtime camera frames may arrive in BHWC layout."""
    processor = PatchPolicyPreprocessor(["front"], (2, 2))

    result = processor({"images": {"front": np.ones((1, 6, 5, 3), dtype=np.float32)}})

    np.testing.assert_equal(result["images"].shape, (1, 1, 1, 3, 2, 2))


def test_composes_with_stats_normalizer_using_view_axis_stats() -> None:
    """Stacked camera statistics broadcast over batch, time, and pixels."""
    processor = PatchPolicyPreprocessor(["front", "wrist"], (2, 2), image_layout="BCHW")
    normalizer = StatsNormalizer(
        mode="mean_std",
        stats={
            "images": {
                "mean": np.array([0.5, 1.0], dtype=np.float32).reshape(2, 1, 1, 1),
                "std": np.array([0.5, 2.0], dtype=np.float32).reshape(2, 1, 1, 1),
            },
        },
    )
    inputs = {
        "images": {
            "front": np.ones((1, 3, 2, 2), dtype=np.float32),
            "wrist": np.full((1, 3, 2, 2), 3.0, dtype=np.float32),
        },
    }

    result = normalizer(processor(inputs))["images"]

    np.testing.assert_allclose(result[:, :, 0], 1.0)
    np.testing.assert_allclose(result[:, :, 1], 1.0)


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
    processor = PatchPolicyPreprocessor(["front", "wrist"], (2, 2), image_layout="BCHW")
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
    """The packed temporal axis must match the exported model contract."""
    processor = PatchPolicyPreprocessor(["front"], (2, 2), image_layout="BCHW", n_obs_steps=2)

    with pytest.raises(ValueError, match="expected 2"):
        processor({"images": {"front": np.ones((1, 3, 2, 2), dtype=np.float32)}})
