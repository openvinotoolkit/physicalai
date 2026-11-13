# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor that draws the flow-matching start noise for exported policies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from physicalai.inference.constants import NOISE, STATE

from .base import Preprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence


class NoisePreprocessor(Preprocessor):
    """Add ``N(0, 1)`` start noise of shape ``(batch, *shape)`` under ``noise``, fresh on every call.

    Exports take the noise as a graph input because OpenVINO's GPU plugin returns the
    same values from in-graph random ops on every call. The batch size comes from
    ``state``. Noise already present in the inputs is kept, which makes a run
    reproducible.

    Args:
        shape: Per-sample noise shape, e.g. ``(chunk_size, max_action_dim)``.
        seed: Seed for the noise generator. ``None`` draws a fresh seed.
    """

    def __init__(self, shape: Sequence[int], seed: int | None = None) -> None:
        """Initialize the noise preprocessor."""
        super().__init__()
        self._shape = tuple(shape)
        self._rng = np.random.default_rng(seed)

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Add the start noise unless the inputs already carry it.

        Args:
            inputs: Observation dict with a ``state`` array of shape ``(batch, ...)``.

        Returns:
            The inputs, or a shallow copy of them with a ``(batch, *shape)`` float32 ``noise`` array.
        """
        if NOISE in inputs:
            return inputs
        # Copy so a reused observation dict doesn't keep this noise for the next call.
        noise = self._rng.standard_normal((inputs[STATE].shape[0], *self._shape), dtype=np.float32)
        return {**inputs, NOISE: noise}
