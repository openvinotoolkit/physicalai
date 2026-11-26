# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Latency tracker callback for RTC inference delay estimation.

Measures wall-clock inference latency over a sliding window and
exposes the worst-case value for computing ``inference_delay``.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import override

from physicalai.config import export_config
from physicalai.inference.callbacks.base import Callback

if TYPE_CHECKING:
    from collections.abc import Iterator

# Larger delays mean "longer than any chunk"; saturating avoids converting inf or 1e308 to an int.
_MAX_DELAY_TICKS = 1_000_000


@export_config(class_path="physicalai.inference.callbacks.RTCLatencyTracker")
class RTCLatencyTracker(Callback):
    """Track inference latency for RTC delay computation.

    ``RTCExecution`` times every inference it runs and records it here
    (:meth:`record`), so it does not need to be registered as a model
    callback. As a callback it measures each ``model(inputs)`` call via
    ``on_predict_start`` / ``on_predict_end``. Exposes ``max_latency_s``
    and ``compute_delay(fps)`` for the ``RTCExecution`` to read.

    Args:
        window_size: Number of recent measurements to retain.

    Examples:
        >>> tracker = RTCLatencyTracker()
        >>> execution = RTCExecution(fps=30, latency_tracker=tracker)
        >>> # After some inferences:
        >>> delay = tracker.compute_delay(fps=30)
    """

    def __init__(self, window_size: int = 100) -> None:  # noqa: D107
        self._window: deque[float] = deque(maxlen=window_size)
        self._start_time: float = 0.0
        # Estimate used while the window is empty (start of a run, after a reset).
        self._fallback_s: float = 0.0
        # Per thread: inside skip_callbacks() the model-callback hooks do nothing.
        self._skipping = threading.local()

    @contextmanager
    def skip_callbacks(self) -> Iterator[None]:
        """Ignore the model-callback hooks for calls made inside this block, on this thread.

        ``RTCExecution`` wraps its own model calls in this block and records
        them with :meth:`record`, so a tracker that is also registered with the
        model (the older wiring) gets one sample per inference. Other calls,
        before ``start()``, after ``stop()`` or from other threads, are still
        sampled through the hooks.
        """
        self._skipping.active = True
        try:
            yield
        finally:
            self._skipping.active = False

    def _skip(self) -> bool:
        return getattr(self._skipping, "active", False)

    def record(self, elapsed_s: float) -> None:
        """Add one measured latency, for callers that time the call themselves.

        ``RTCExecution`` times every inference it runs and records it here, so
        the tracker works without being registered as a model callback.
        """
        if elapsed_s > 0:
            self._window.append(float(elapsed_s))

    def seed(self, latency_s: float) -> None:
        """Drop the measurements and use ``latency_s`` until a new one arrives.

        Used for cold-start inferences: their duration (model compilation,
        kernel builds) must not stay in the worst-case window, but it is a
        better estimate of the delay than zero.
        """
        self._window.clear()
        self._fallback_s = max(float(latency_s), 0.0)

    @override
    def on_predict_start(self, inputs: dict[str, Any]) -> None:
        """Record prediction start time (not inside :meth:`skip_callbacks`)."""
        if self._skip():
            return
        self._start_time = time.perf_counter()

    @override
    def on_predict_end(self, outputs: dict[str, Any]) -> None:
        """Record prediction duration (not inside :meth:`skip_callbacks`)."""
        if self._skip():
            return
        elapsed = time.perf_counter() - self._start_time
        self._window.append(elapsed)

    @override
    def on_reset(self) -> None:
        """Clear recorded latencies, keeping the worst one as the estimate.

        A reset happens at every episode boundary. Forgetting the latency there
        would make the first re-plan of each episode assume zero delay.
        """
        if self._window:
            self._fallback_s = max(self._window)
        self._window.clear()
        self._start_time = 0.0

    @property
    def max_latency_s(self) -> float:
        """Worst-case latency in seconds over the sliding window.

        Returns the seeded estimate (0.0 unless :meth:`seed` or
        :meth:`on_reset` set one) if no measurement is recorded.
        """
        return max(self._window) if self._window else self._fallback_s

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
        ticks = self.max_latency_s * fps
        if not ticks > 0:
            return 0
        if not ticks < _MAX_DELAY_TICKS:
            # Saturate instead of overflowing; callers clamp it to the chunk.
            return _MAX_DELAY_TICKS
        # The epsilon keeps a latency of exactly n ticks from rounding up to n + 1.
        return math.ceil(ticks - 1e-6)

    @override
    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"RTCLatencyTracker("
            f"max={self.max_latency_s:.3f}s, "
            f"latest={self.latest_latency_s:.3f}s, "
            f"samples={len(self._window)})"
        )
