# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections import deque

import numpy as np

from physicalai.runtime.smoothers import ChunkSmoother, ReplaceSmoother


class ActionQueue:
    """Thread-safe action queue with chunk smoothing."""

    def __init__(self, smoother: ChunkSmoother | None = None) -> None:
        self._smoother = smoother or ReplaceSmoother()
        self._deque: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._consecutive_holds = 0
        self._total_holds = 0
        self._total_pops = 0

    def push_chunk(self, chunk: np.ndarray, offset: int = 0) -> None:
        """Push an action chunk, blending with remaining actions via the smoother."""
        with self._lock:
            remaining = np.stack(list(self._deque)) if self._deque else np.empty((0, chunk.shape[1]), dtype=chunk.dtype)
            merged = self._smoother.merge(remaining, chunk, offset)
            self._deque.clear()
            self._deque.extend(merged)

    def pop(self) -> np.ndarray | None:
        """Pop the next action.

        Returns:
            Single action vector, or None if empty.
        """
        with self._lock:
            if not self._deque:
                self._consecutive_holds += 1
                self._total_holds += 1
                return None
            self._consecutive_holds = 0
            self._total_pops += 1
            return self._deque.popleft()

    @property
    def remaining(self) -> int:
        with self._lock:
            return len(self._deque)

    @property
    def consecutive_holds(self) -> int:
        return self._consecutive_holds

    @property
    def total_holds(self) -> int:
        return self._total_holds

    @property
    def total_pops(self) -> int:
        return self._total_pops

    def below_threshold(self, threshold: int) -> bool:
        with self._lock:
            return len(self._deque) < threshold

    def clear(self) -> None:
        with self._lock:
            self._deque.clear()
            self._consecutive_holds = 0
