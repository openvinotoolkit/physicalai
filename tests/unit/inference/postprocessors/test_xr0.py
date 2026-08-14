# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from physicalai.inference.constants import ACTION
from physicalai.inference.postprocessors import Postprocessor, XR0Postprocessor


class TestXR0PostprocessorInit:
    def test_is_postprocessor(self) -> None:
        post = XR0Postprocessor(action_mean=[0.0] * 4, action_std=[1.0] * 4)
        assert isinstance(post, Postprocessor)

    def test_mismatched_stats_raise(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            XR0Postprocessor(action_mean=[0.0] * 4, action_std=[1.0] * 3)


class TestXR0PostprocessorCall:
    def test_denormalizes_action(self) -> None:
        post = XR0Postprocessor(action_mean=[1.0] * 4, action_std=[2.0] * 4)
        out = post({ACTION: np.zeros((1, 3, 4), dtype=np.float32)})
        # 0 * (2 + eps) + 1 == 1.0
        np.testing.assert_allclose(out[ACTION], 1.0, atol=1e-5)

    def test_slices_to_action_dim(self) -> None:
        post = XR0Postprocessor(action_mean=[0.0] * 8, action_std=[1.0] * 8, action_dim=6)
        out = post({ACTION: np.ones((1, 5, 8), dtype=np.float32)})
        assert out[ACTION].shape == (1, 5, 6)

    def test_none_action_dim_keeps_width(self) -> None:
        post = XR0Postprocessor(action_mean=[0.0] * 8, action_std=[1.0] * 8)
        out = post({ACTION: np.ones((1, 5, 8), dtype=np.float32)})
        assert out[ACTION].shape == (1, 5, 8)

    def test_missing_action_passthrough(self) -> None:
        post = XR0Postprocessor(action_mean=[0.0] * 4, action_std=[1.0] * 4)
        payload = {"other": np.zeros((2,), dtype=np.float32)}
        out = post(payload)
        assert "other" in out

    def test_roundtrip_with_preprocessor_convention(self) -> None:
        mean = np.array([0.5, -1.0, 2.0, 0.0], dtype=np.float32)
        std = np.array([1.5, 0.5, 3.0, 1.0], dtype=np.float32)
        post = XR0Postprocessor(action_mean=mean.tolist(), action_std=std.tolist())
        raw = np.random.rand(1, 4, 4).astype(np.float32)
        normalized = (raw - mean) / (std + 1e-6)
        out = post({ACTION: normalized})
        np.testing.assert_allclose(out[ACTION], raw, atol=1e-4)
