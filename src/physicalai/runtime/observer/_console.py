# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from typing import Any


class ConsoleHandler:
    def __init__(self, target_fps: float = 30.0) -> None:
        self._target_fps = target_fps
        self._last_step = -1

    def __call__(self, session_id: str, topic: str, payload: dict[str, Any]) -> None:
        if topic == "tick":
            self._handle_tick(session_id, payload)
        elif topic == "inference":
            self._handle_inference(session_id, payload)
        elif topic == "lifecycle":
            self._handle_lifecycle(session_id, payload)

    def _handle_tick(self, session_id: str, payload: dict[str, Any]) -> None:
        step = payload.get("step", "?")
        loop_ms = payload.get("physicalai.runtime.loop_duration_s", 0) * 1000
        queue = payload.get("queue_remaining", "?")
        stale = " [STALE]" if payload.get("stale_obs") else ""
        actual_fps = 1000 / loop_ms if loop_ms > 0 else 0
        line = (
            f"\r[{session_id}] step={step}  "
            f"fps={actual_fps:.0f}/{self._target_fps:.0f}  "
            f"loop={loop_ms:.1f}ms  "
            f"queue={queue}{stale}    "
        )
        sys.stdout.write(line)
        sys.stdout.flush()

    @staticmethod
    def _handle_inference(session_id: str, payload: dict[str, Any]) -> None:
        latency_ms = payload.get("physicalai.runtime.inference_latency_s", 0) * 1000
        offset = payload.get("offset", "?")
        sys.stdout.write(f"\n[{session_id}] inference: latency={latency_ms:.0f}ms offset={offset}\n")
        sys.stdout.flush()

    @staticmethod
    def _handle_lifecycle(session_id: str, payload: dict[str, Any]) -> None:
        event = payload.get("event", "unknown")
        sys.stdout.write(f"\n[{session_id}] lifecycle: {event} {payload}\n")
        sys.stdout.flush()
