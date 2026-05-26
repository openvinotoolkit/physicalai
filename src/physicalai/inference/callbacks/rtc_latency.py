# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Latency tracker callback for RTC inference delay estimation.

Measures wall-clock inference latency over a sliding window and
exposes the worst-case value for computing ``inference_delay``.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, override

import numpy as np

from physicalai.inference.callbacks.base import Callback


class RTCLatencyTracker(Callback):
    """Track inference latency for RTC delay computation.

    Measures the duration of each ``model(inputs)`` call via
    ``on_predict_start`` / ``on_predict_end`` hooks. Exposes
    ``max_latency_s`` and ``compute_delay(fps)`` for the
    ``RTCExecution`` to read.

    Args:
        window_size: Number of recent measurements to retain.

    Examples:
        >>> tracker = RTCLatencyTracker()
        >>> model = InferenceModel.load("./exports/pi05_rtc", callbacks=[tracker])
        >>> # After some predictions:
        >>> delay = tracker.compute_delay(fps=30, chunk_size=50, execution_horizon=10)
    """

    def __init__(self, window_size: int = 100) -> None:  # noqa: D107
        self._window: deque[float] = deque(maxlen=window_size)
        self._start_time: float = 0.0

    @override
    def on_predict_start(self, inputs: dict[str, Any]) -> None:
        """Record prediction start time."""
        self._start_time = time.perf_counter()

    @override
    def on_predict_end(self, outputs: dict[str, Any]) -> None:
        """Record prediction duration."""
        elapsed = time.perf_counter() - self._start_time
        self._window.append(elapsed)

    @override
    def on_reset(self) -> None:
        """Clear recorded latencies."""
        self._window.clear()
        self._start_time = 0.0

    @property
    def max_latency_s(self) -> float:
        """Worst-case latency in seconds over the sliding window.

        Returns 0.0 if no measurements recorded yet.
        """
        return max(self._window) if self._window else 0.0

    @property
    def latest_latency_s(self) -> float:
        """Most recent inference latency in seconds."""
        return self._window[-1] if self._window else 0.0

    def percentile_s(self, q: float) -> float:
        """Compute a percentile of recorded latencies.

        Args:
            q: Percentile in [0, 100].

        Returns:
            The q-th percentile in seconds, or 0.0 if empty.
        """
        if not self._window:
            return 0.0
        return float(np.percentile(np.array(self._window), q))

    def compute_delay(self, fps: float) -> int:
        """Compute integer delay for the RTC model input.

        ``delay = ceil(max_latency * fps)``

        Args:
            fps: Robot control frequency in Hz.

        Returns:
            Integer delay (number of action steps consumed during inference).
        """
        if self.max_latency_s <= 0:
            return 0
        return math.ceil(self.max_latency_s * fps)

    @override
    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"RTCLatencyTracker("
            f"max={self.max_latency_s:.3f}s, "
            f"latest={self.latest_latency_s:.3f}s, "
            f"samples={len(self._window)})"
        )
