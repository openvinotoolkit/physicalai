# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from physicalai.inference.postprocessors import ActionChunkTrimmer, Postprocessor


class TestActionChunkTrimmer:
    def test_is_postprocessor(self) -> None:
        trimmer = ActionChunkTrimmer(n_action_steps=10)
        assert isinstance(trimmer, Postprocessor)

    def test_trims_temporal_axis_when_chunk_is_longer_than_limit(self) -> None:
        trimmer = ActionChunkTrimmer(n_action_steps=4)
        actions = np.arange(2 * 8 * 3).reshape(2, 8, 3)

        result = trimmer(actions)

        assert result.shape == (2, 4, 3)
        np.testing.assert_array_equal(result, actions[:, :4, :])

    def test_keeps_temporal_axis_when_chunk_matches_limit(self) -> None:
        trimmer = ActionChunkTrimmer(n_action_steps=8)
        actions = np.arange(2 * 8 * 3).reshape(2, 8, 3)

        result = trimmer(actions)

        assert result.shape == (2, 8, 3)
        np.testing.assert_array_equal(result, actions)

    def test_keeps_temporal_axis_when_chunk_is_shorter_than_limit(self) -> None:
        trimmer = ActionChunkTrimmer(n_action_steps=10)
        actions = np.arange(2 * 8 * 3).reshape(2, 8, 3)

        result = trimmer(actions)

        assert result.shape == (2, 8, 3)
        np.testing.assert_array_equal(result, actions)

    def test_non_temporal_array_is_passed_through(self) -> None:
        trimmer = ActionChunkTrimmer(n_action_steps=1)
        actions = np.arange(2 * 6).reshape(2, 6)

        result = trimmer(actions)

        assert result.shape == (2, 6)
        np.testing.assert_array_equal(result, actions)

    def test_repr(self) -> None:
        trimmer = ActionChunkTrimmer(n_action_steps=6)
        assert repr(trimmer) == "ActionChunkTrimmer(n_action_steps=6)"
