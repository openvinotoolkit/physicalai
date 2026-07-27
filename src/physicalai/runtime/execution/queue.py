# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action queue protocol and the default chunk-smoothing implementation."""

from __future__ import annotations

import threading
from collections import deque
from typing import Protocol, runtime_checkable

import numpy as np

from physicalai.config import export_config
from physicalai.runtime.smoothers import ChunkSmoother, ReplaceSmoother


@runtime_checkable
class ActionQueue(Protocol):
    """Protocol for a thread-safe action queue."""

    def pop(self) -> np.ndarray | None:
        """Pop the next action.

        Returns:
            Single action vector, or None if empty.
        """
        ...

    @property
    def remaining(self) -> int:
        """Number of unconsumed actions in the queue."""
        ...

    @property
    def consecutive_holds(self) -> int:
        """Number of consecutive holds (resets on successful pop)."""
        ...

    @property
    def total_holds(self) -> int:
        """Total number of hold events (pop on empty queue)."""
        ...

    @property
    def total_pops(self) -> int:
        """Total number of actions popped."""
        ...

    def below_threshold(self, threshold: int) -> bool:
        """Check if remaining actions are below threshold."""
        ...

    def clear(self) -> None:
        """Clear all state from the queue."""
        ...

    def push_chunk(self, chunk: np.ndarray, offset: int = 0) -> None:
        """Push an action chunk into the queue."""
        ...

    def reset(self) -> None:
        """Clear queue and reset all counters for a fresh session."""
        ...


@export_config(class_path="physicalai.runtime.ChunkedActionQueue")
class ChunkedActionQueue:
    """Thread-safe action queue with chunk smoothing."""

    def __init__(self, smoother: ChunkSmoother | None = None) -> None:
        """Initialize the queue with an optional chunk smoother."""
        self._smoother = smoother or ReplaceSmoother()
        self._deque: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._consecutive_holds = 0
        self._total_holds = 0
        self._total_pops = 0

    def push_chunk(self, chunk: np.ndarray, offset: int = 0) -> None:
        """Push an action chunk, blending with remaining actions via the smoother."""
        with self._lock:
            incoming = chunk[offset:]
            remaining = np.stack(list(self._deque)) if self._deque else np.empty((0, chunk.shape[1]), dtype=chunk.dtype)
            merged = self._smoother.merge(remaining, incoming)
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

    def peek_remaining(self) -> np.ndarray | None:
        """Return copy of remaining actions without consuming them. Thread-safe."""
        with self._lock:
            if not self._deque:
                return None
            return np.stack(list(self._deque))

    @property
    def remaining(self) -> int:
        """Number of unconsumed actions in the queue."""
        with self._lock:
            return len(self._deque)

    @property
    def consecutive_holds(self) -> int:
        """Number of consecutive holds (resets on successful pop)."""
        return self._consecutive_holds

    @property
    def total_holds(self) -> int:
        """Total number of hold events (pop on empty queue)."""
        return self._total_holds

    @property
    def total_pops(self) -> int:
        """Total number of actions popped."""
        return self._total_pops

    def below_threshold(self, threshold: int) -> bool:
        """Check if remaining actions are below threshold.

        Returns:
            True if the number of remaining actions is below the threshold.
        """
        with self._lock:
            return len(self._deque) < threshold

    def clear(self) -> None:
        """Clear all state from the queue."""
        with self._lock:
            self._deque.clear()
            self._consecutive_holds = 0

    def reset(self) -> None:
        """Clear queue and reset all counters for a fresh session."""
        with self._lock:
            self._deque.clear()
            self._consecutive_holds = 0
            self._total_holds = 0
            self._total_pops = 0
