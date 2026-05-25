# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Chunk smoothers for runtime action queues."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import override

import numpy as np

_NDIM_2 = 2
_ERR_2D = "remaining and incoming must be 2D arrays"
_ERR_ACTION_DIM = "remaining and incoming must have the same action_dim"


class ChunkSmoother(ABC):
    """Merges a new action chunk into remaining actions from the previous chunk."""

    @abstractmethod
    def merge(self, remaining: np.ndarray, incoming: np.ndarray) -> np.ndarray:
        """Merge a previous remainder with a new incoming chunk."""
        raise NotImplementedError


class ReplaceSmoother(ChunkSmoother):
    """Replace remaining actions with the incoming chunk."""

    @override
    def merge(self, remaining: np.ndarray, incoming: np.ndarray) -> np.ndarray:
        """Return the incoming chunk (remaining is discarded)."""
        _validate_inputs(remaining, incoming)
        return incoming


class LerpSmoother(ChunkSmoother):
    """Blend overlapping actions and append the incoming tail."""

    def __init__(self, duration_frames: int = 5) -> None:
        """Create a smoother with a lerp window."""
        self.duration_frames = duration_frames

    @override
    def merge(self, remaining: np.ndarray, incoming: np.ndarray) -> np.ndarray:
        """Merge chunks using queue-mixer-style linear interpolation.

        Returns:
            The blended action chunk.
        """
        _validate_inputs(remaining, incoming)

        n_remain = len(remaining)
        lerp_dur = min(n_remain, self.duration_frames)

        weights = np.maximum(1.0 - np.arange(n_remain) / max(lerp_dur, 1), 0.0)
        weights = weights[:, np.newaxis]

        n_blend = min(n_remain, len(incoming))
        blended = weights[:n_blend] * remaining[:n_blend] + (1.0 - weights[:n_blend]) * incoming[:n_blend]

        return np.concatenate([blended, incoming[n_blend:]], axis=0).astype(np.float32)


def _validate_inputs(remaining: np.ndarray, incoming: np.ndarray) -> None:
    if remaining.ndim != _NDIM_2 or incoming.ndim != _NDIM_2:
        raise ValueError(_ERR_2D)
    if remaining.shape[1] != incoming.shape[1]:
        raise ValueError(_ERR_ACTION_DIM)
