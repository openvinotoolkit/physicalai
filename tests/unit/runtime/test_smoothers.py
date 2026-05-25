from __future__ import annotations

import numpy as np
import pytest

from physicalai.runtime.smoothers import LerpSmoother, ReplaceSmoother


class TestReplaceSmoother:
    def test_merge_returns_incoming(self) -> None:
        smoother = ReplaceSmoother()
        remaining = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        incoming = np.array([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=np.float32)

        result = smoother.merge(remaining, incoming)

        np.testing.assert_array_equal(result, incoming)

    def test_empty_remaining_returns_incoming(self) -> None:
        smoother = ReplaceSmoother()
        remaining = np.empty((0, 2), dtype=np.float32)
        incoming = np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32)

        result = smoother.merge(remaining, incoming)

        np.testing.assert_array_equal(result, incoming)


class TestLerpSmoother:
    def test_lerp_weights_match_queue_mixer_formula(self) -> None:
        smoother = LerpSmoother(duration_frames=5)
        remaining = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]], dtype=np.float32)
        incoming = np.array(
            [[100.0, 100.0], [110.0, 110.0], [120.0, 120.0], [130.0, 130.0]],
            dtype=np.float32,
        )

        result = smoother.merge(remaining, incoming)

        expected = np.array(
            [[10.0, 10.0], [50.0, 50.0], [90.0, 90.0], [130.0, 130.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(result, expected)

    def test_incoming_shorter_than_remaining(self) -> None:
        smoother = LerpSmoother(duration_frames=99)
        remaining = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
        incoming = np.array([[6.0, 6.0], [7.0, 7.0]], dtype=np.float32)

        result = smoother.merge(remaining, incoming)

        # lerp_dur = min(n_remain=3, duration_frames=99) = 3
        # weights = [1.0, 2/3, 1/3]; n_blend = min(3,2) = 2
        # blended[0] = 1.0*1 + 0.0*6 = 1.0
        # blended[1] = (2/3)*2 + (1/3)*7 = 4/3 + 7/3 = 11/3
        expected = np.array([[1.0, 1.0], [11.0 / 3, 11.0 / 3]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_empty_remaining_returns_incoming(self) -> None:
        smoother = LerpSmoother(duration_frames=5)

        result = smoother.merge(
            np.empty((0, 2), dtype=np.float32),
            np.array([[1.0, 1.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(result, np.array([[1.0, 1.0]], dtype=np.float32))

    def test_single_remaining_single_incoming(self) -> None:
        smoother = LerpSmoother(duration_frames=5)

        result = smoother.merge(
            np.array([[1.0, 1.0]], dtype=np.float32),
            np.array([[2.0, 2.0]], dtype=np.float32),
        )
        # weight[0] = 1.0 -> blended = 1.0*1 + 0.0*2 = 1.0
        np.testing.assert_array_equal(result, np.array([[1.0, 1.0]], dtype=np.float32))

    def test_exact_numerical_blend_values(self) -> None:
        smoother = LerpSmoother(duration_frames=5)
        remaining = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        incoming = np.array(
            [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
            dtype=np.float32,
        )

        result = smoother.merge(remaining, incoming)

        # lerp_dur = min(3, 5) = 3; weights = [1.0, 2/3, 1/3]
        expected = np.array(
            [[1.0, 1.0], [4.0 / 3, 4.0 / 3], [5.0 / 3, 5.0 / 3]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_merge_is_stateless_for_same_arguments(self) -> None:
        smoother = LerpSmoother(duration_frames=5)
        remaining = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        incoming = np.array([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]], dtype=np.float32)

        first = smoother.merge(remaining, incoming)
        second = smoother.merge(remaining, incoming)

        np.testing.assert_array_equal(first, second)


def test_input_validation_mismatched_action_dim_raises_value_error() -> None:
    smoother = ReplaceSmoother()
    remaining = np.array([[1.0, 1.0]], dtype=np.float32)
    incoming = np.array([[2.0, 2.0, 2.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="action_dim"):
        smoother.merge(remaining, incoming)
