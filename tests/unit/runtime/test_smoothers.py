from __future__ import annotations

import numpy as np
import pytest

from physicalai.runtime.smoothers import LerpSmoother, ReplaceSmoother


class TestReplaceSmoother:
    def test_merge_drops_remaining_and_returns_incoming_offset(self) -> None:
        smoother = ReplaceSmoother()
        remaining = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        incoming = np.array([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=np.float32)

        result = smoother.merge(remaining, incoming, offset=1)

        np.testing.assert_array_equal(
            result,
            np.array([[3.0, 3.0], [4.0, 4.0]], dtype=np.float32),
        )

    def test_offset_zero_returns_all_incoming(self) -> None:
        smoother = ReplaceSmoother()
        remaining = np.array([[1.0, 1.0]], dtype=np.float32)
        incoming = np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32)

        result = smoother.merge(remaining, incoming, offset=0)

        np.testing.assert_array_equal(result, incoming)

    def test_offset_beyond_incoming_returns_empty_array(self) -> None:
        smoother = ReplaceSmoother()
        remaining = np.array([[1.0, 1.0]], dtype=np.float32)
        incoming = np.array([[2.0, 2.0]], dtype=np.float32)

        result = smoother.merge(remaining, incoming, offset=5)

        assert result.shape == (0, 2)
        assert result.dtype == incoming.dtype


class TestLerpSmoother:
    def test_lerp_weights_match_queue_mixer_formula(self) -> None:
        smoother = LerpSmoother(duration_frames=5)
        remaining = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]], dtype=np.float32)
        incoming = np.array(
            [[100.0, 100.0], [110.0, 110.0], [120.0, 120.0], [130.0, 130.0]],
            dtype=np.float32,
        )

        result = smoother.merge(remaining, incoming, offset=0)

        expected = np.array(
            [[10.0, 10.0], [50.0, 50.0], [90.0, 90.0], [130.0, 130.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(result, expected)

    def test_offset_aware_duration(self) -> None:
        smoother = LerpSmoother(duration_frames=99)
        remaining = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
        incoming = np.array([[4.0, 4.0], [5.0, 5.0], [6.0, 6.0], [7.0, 7.0]], dtype=np.float32)

        result = smoother.merge(remaining, incoming, offset=2)

        expected = np.array([[1.0, 1.0], [4.5, 4.5]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_edge_cases_empty_remaining_single_element_and_offset_beyond_chunk(self) -> None:
        smoother = LerpSmoother(duration_frames=5)

        empty_result = smoother.merge(
            np.empty((0, 2), dtype=np.float32),
            np.array([[1.0, 1.0]], dtype=np.float32),
            offset=0,
        )
        np.testing.assert_array_equal(empty_result, np.array([[1.0, 1.0]], dtype=np.float32))

        single_result = smoother.merge(
            np.array([[1.0, 1.0]], dtype=np.float32),
            np.array([[2.0, 2.0]], dtype=np.float32),
            offset=0,
        )
        np.testing.assert_array_equal(single_result, np.array([[1.0, 1.0]], dtype=np.float32))

        offset_beyond_result = smoother.merge(
            np.array([[1.0, 1.0]], dtype=np.float32),
            np.array([[2.0, 2.0]], dtype=np.float32),
            offset=5,
        )
        assert offset_beyond_result.shape == (0, 2)

    def test_exact_numerical_blend_values(self) -> None:
        smoother = LerpSmoother(duration_frames=5)
        remaining = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        incoming = np.array(
            [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
            dtype=np.float32,
        )

        result = smoother.merge(remaining, incoming, offset=1)

        expected = np.array(
            [[1.0, 1.0], [2.0, 2.0], [2.0, 2.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(result, expected)

    def test_merge_is_stateless_for_same_arguments(self) -> None:
        smoother = LerpSmoother(duration_frames=5)
        remaining = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        incoming = np.array([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]], dtype=np.float32)

        first = smoother.merge(remaining, incoming, offset=1)
        second = smoother.merge(remaining, incoming, offset=1)

        np.testing.assert_array_equal(first, second)


def test_input_validation_mismatched_action_dim_raises_value_error() -> None:
    smoother = ReplaceSmoother()
    remaining = np.array([[1.0, 1.0]], dtype=np.float32)
    incoming = np.array([[2.0, 2.0, 2.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="action_dim"):
        smoother.merge(remaining, incoming, offset=0)
