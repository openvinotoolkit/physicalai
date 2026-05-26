# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dual-track action queue for Real-Time Chunking (RTC).

Stores both raw (normalized) actions for ``prev_chunk_left_over``
feedback and postprocessed (denormalized) actions for robot execution.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class RTCActionQueue:
    """Thread-safe dual-track action queue for RTC inference.

    Maintains two parallel tracks:
    - **raw**: normalized model output, used as ``prev_chunk_left_over``
      for the next inference call.
    - **processed**: denormalized actions sent to the robot.

    A cursor tracks consumption. ``pop()`` returns one processed action.
    ``get_left_over()`` returns the unconsumed raw tail.
    ``merge()`` replaces both tracks, trimming stale prefix.

    All public methods are thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._raw: np.ndarray | None = None
        self._processed: np.ndarray | None = None
        self._cursor: int = 0
        self._total_pops: int = 0
        self._total_holds: int = 0
        self._consecutive_holds: int = 0

    @property
    def total_pops(self) -> int:
        """Total number of actions popped."""
        return self._total_pops

    @property
    def total_holds(self) -> int:
        """Total number of hold events (pop on empty queue)."""
        return self._total_holds

    @property
    def consecutive_holds(self) -> int:
        """Number of consecutive holds (resets on successful pop)."""
        return self._consecutive_holds

    @property
    def remaining(self) -> int:
        """Number of unconsumed actions in the queue."""
        with self._lock:
            if self._processed is None:
                return 0
            return max(0, len(self._processed) - self._cursor)

    def get_action_index(self) -> int:
        """Current consumption cursor (snapshot for delay cross-check).

        Returns:
            The current cursor position.
        """
        with self._lock:
            return self._cursor

    def pop(self) -> np.ndarray | None:
        """Pop one processed (denormalized) action.

        Returns:
            Action array of shape ``(action_dim,)``, or ``None`` if empty.
        """
        with self._lock:
            if self._processed is None or self._cursor >= len(self._processed):
                self._consecutive_holds += 1
                self._total_holds += 1
                return None
            action = self._processed[self._cursor].copy()
            self._cursor += 1
            self._consecutive_holds = 0
            self._total_pops += 1
            return action

    def get_left_over(self) -> np.ndarray | None:
        """Return unconsumed raw (normalized) actions.

        Called from the RTC background thread to build
        ``prev_chunk_left_over`` for the next inference call.

        Returns:
            Array of shape ``(remaining, action_dim)`` or ``None``.
        """
        with self._lock:
            if self._raw is None or self._cursor >= len(self._raw):
                return None
            return self._raw[self._cursor :].copy()

    def merge(
        self,
        raw: np.ndarray,
        processed: np.ndarray,
        action_index_before_inference: int | None = None,
    ) -> None:
        """Replace queue contents, trimming actions consumed during inference.

        Trim is derived from actual cursor movement (actions the robot
        consumed from the previous chunk while inference was running).
        For the first chunk, cursor hasn't moved so trim=0 and the
        full chunk is kept.

        Args:
            raw: Raw model output, shape ``(chunk_size, action_dim)``.
            processed: Postprocessed actions, shape ``(chunk_size, robot_dof)``.
            action_index_before_inference: Cursor snapshot taken before
                inference started. Used to compute how many actions were
                consumed during inference.
        """
        with self._lock:
            # Trim = actual actions consumed during inference
            if action_index_before_inference is not None:
                trim = max(0, self._cursor - action_index_before_inference)
            else:
                trim = 0

            # Clamp to array length
            trim = min(trim, len(raw), len(processed))

            self._raw = raw[trim:]
            self._processed = processed[trim:]
            self._cursor = 0

            logger.debug(
                "RTCActionQueue.merge: trim=%d, remaining=%d",
                trim,
                len(self._processed),
            )

    def below_threshold(self, threshold: int) -> bool:
        """Check if remaining actions are below threshold.

        Returns:
            True if remaining actions are fewer than threshold.
        """
        with self._lock:
            if self._processed is None:
                return True
            return (len(self._processed) - self._cursor) < threshold

    def clear(self) -> None:
        """Clear all state."""
        with self._lock:
            self._raw = None
            self._processed = None
            self._cursor = 0
            self._consecutive_holds = 0
