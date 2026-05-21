# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from physicalai.inference.constants import ACTION
from physicalai.inference.postprocessors import ActionChunkTrimmer, Postprocessor


class TestActionChunkTrimmer:
    def test_is_postprocessor(self) -> None:
        trimmer = ActionChunkTrimmer(n_action_steps=10)
        assert isinstance(trimmer, Postprocessor)

    def test_trims_temporal_axis_when_chunk_is_longer_than_limit(self) -> None:
        trimmer = ActionChunkTrimmer(n_action_steps=4)
        outputs = {
            ACTION: np.arange(2 * 8 * 3).reshape(2, 8, 3),
            "scores": np.array([0.9, 0.8]),
        }

        result = trimmer(outputs)

        assert result[ACTION].shape == (2, 4, 3)
        np.testing.assert_array_equal(result[ACTION], outputs[ACTION][:, :4, :])
        np.testing.assert_array_equal(result["scores"], np.array([0.9, 0.8]))

    def test_keeps_temporal_axis_when_chunk_matches_limit(self) -> None:
        trimmer = ActionChunkTrimmer(n_action_steps=8)
        outputs = {ACTION: np.arange(2 * 8 * 3).reshape(2, 8, 3)}

        result = trimmer(outputs)

        assert result[ACTION].shape == (2, 8, 3)
        np.testing.assert_array_equal(result[ACTION], outputs[ACTION])

    def test_keeps_temporal_axis_when_chunk_is_shorter_than_limit(self) -> None:
        trimmer = ActionChunkTrimmer(n_action_steps=10)
        outputs = {ACTION: np.arange(2 * 8 * 3).reshape(2, 8, 3)}

        result = trimmer(outputs)

        assert result[ACTION].shape == (2, 8, 3)
        np.testing.assert_array_equal(result[ACTION], outputs[ACTION])

    def test_non_temporal_array_is_passed_through(self) -> None:
        trimmer = ActionChunkTrimmer(n_action_steps=1)
        outputs = {ACTION: np.arange(2 * 6).reshape(2, 6)}

        result = trimmer(outputs)

        assert result[ACTION].shape == (2, 6)
        np.testing.assert_array_equal(result[ACTION], outputs[ACTION])

    def test_repr(self) -> None:
        trimmer = ActionChunkTrimmer(n_action_steps=6)
        assert repr(trimmer) == "ActionChunkTrimmer(n_action_steps=6)"
