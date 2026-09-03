# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from physicalai.inference.constants import ACTION, STATE_PASSTHROUGH
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


class TestXR0PostprocessorDelta:
    """Delta mode re-adds the current-frame state to the denormalized delta."""

    def _delta_post(self, chunk: int = 5, dim: int = 8, action_dim: int = 6) -> XR0Postprocessor:
        # Identity delta stats (mean 0 / std 1) so the graph output is the raw delta.
        return XR0Postprocessor(
            action_mean=np.zeros((chunk, dim), dtype=np.float32).tolist(),
            action_std=np.ones((chunk, dim), dtype=np.float32).tolist(),
            action_dim=action_dim,
            action_mode="delta",
        )

    def test_adds_current_state(self) -> None:
        post = self._delta_post()
        delta = np.random.rand(1, 5, 8).astype(np.float32)
        state = np.random.rand(1, 1, 8).astype(np.float32)  # (B, T=1, D)
        out = post({ACTION: delta, STATE_PASSTHROUGH: state})
        # delta + state on the first action_dim=6 channels, sliced to 6.
        expected = delta[..., :6] + state[:, -1, :][:, None, :6]
        assert out[ACTION].shape == (1, 5, 6)
        np.testing.assert_allclose(out[ACTION], expected, atol=1e-5)

    def test_roundtrip_reconstructs_absolute(self) -> None:
        mean = np.random.rand(5, 8).astype(np.float32)
        std = (np.random.rand(5, 8) + 0.5).astype(np.float32)
        post = XR0Postprocessor(
            action_mean=mean.tolist(),
            action_std=std.tolist(),
            action_dim=6,
            action_mode="delta",
        )
        absolute = np.random.rand(1, 5, 6).astype(np.float32)
        state = np.random.rand(1, 1, 8).astype(np.float32)
        # Studio-side target: normalized delta = ((absolute - state) - mean) / (std + eps),
        # padded to width 8.
        delta = absolute - state[:, -1, :][:, None, :6]
        padded = np.zeros((1, 5, 8), dtype=np.float32)
        padded[..., :6] = delta
        normalized = (padded - mean) / (std + 1e-6)
        out = post({ACTION: normalized, STATE_PASSTHROUGH: state})
        np.testing.assert_allclose(out[ACTION], absolute, atol=1e-4)

    def test_delta_requires_state(self) -> None:
        post = self._delta_post()
        with pytest.raises(ValueError, match="state"):
            post({ACTION: np.zeros((1, 5, 8), dtype=np.float32)})

    def test_absolute_mode_ignores_state(self) -> None:
        post = XR0Postprocessor(action_mean=[0.0] * 8, action_std=[1.0] * 8, action_dim=6)
        action = np.ones((1, 5, 8), dtype=np.float32)
        with_state = post({ACTION: action, STATE_PASSTHROUGH: np.ones((1, 1, 8), dtype=np.float32)})[ACTION]
        without_state = post({ACTION: action})[ACTION]
        np.testing.assert_allclose(with_state, without_state, atol=1e-6)
