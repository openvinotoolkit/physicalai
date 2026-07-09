# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Periodic one-line console summary callback."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physicalai.runtime.events import LifecycleEvent, TickEvent


class ConsoleCallback:
    """Periodic one-line summary to stdout, throttled by step count.

    Prints every ``throttle_steps`` steps (default 30), i.e. roughly once
    per second at a 30 fps control loop; scale ``throttle_steps`` with the
    runtime's ``fps`` for a different cadence.
    """

    def __init__(self, throttle_steps: int = 30) -> None:  # noqa: D107
        self._throttle_steps = throttle_steps
        self._start_time: float | None = None
        self._last_report_time: float | None = None
        self._last_report_step: int = 0

    def on_tick(self, event: TickEvent) -> None:  # noqa: D102
        now = time.monotonic()
        if self._start_time is None:
            self._start_time = now
            self._last_report_time = now
            self._last_report_step = 0
        if event.step > 0 and event.step % self._throttle_steps != 0:
            return
        last_report_time = self._last_report_time or now
        dt = now - last_report_time
        elapsed = now - self._start_time
        actual_hz = (event.step - self._last_report_step) / dt if dt > 0 else 0.0
        self._last_report_time = now
        self._last_report_step = event.step
        print(  # noqa: T201
            f"[{elapsed:6.1f}s] step={event.step} "
            f"hz={actual_hz:.0f} "
            f"loop={event.loop_duration_s * 1000:.1f}ms"
            f"{' STALE' if event.stale_obs else ''}",
        )

    def on_lifecycle(self, event: LifecycleEvent) -> None:  # noqa: D102, PLR6301
        print(f"[lifecycle] {event.event}: {event.metadata}")  # noqa: T201
