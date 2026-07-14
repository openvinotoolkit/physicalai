# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class TelemetryEmitter:
    def __init__(self, session_id: str | None = None) -> None:
        self._session_id = session_id or uuid.uuid4().hex[:8]
        self._session: Any = None
        self._msgpack: Any = None
        self._enabled = False
        try:
            import msgpack  # noqa: PLC0415
            import zenoh  # noqa: PLC0415

            self._msgpack = msgpack
            self._session = zenoh.open(zenoh.Config())
            self._enabled = True
        except ImportError:
            pass

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def session_id(self) -> str:
        return self._session_id

    @staticmethod
    def _pack(payload: dict[str, Any]) -> bytes:
        from ._codec import pack_payload  # noqa: PLC0415

        return pack_payload(payload)

    def emit_tick(
        self,
        *,
        step: int,
        timestamp: float,
        joint_positions: np.ndarray | None,
        action_sent: np.ndarray | None,
        loop_duration_s: float,
        sleep_time_s: float,
        stale_obs: bool = False,
    ) -> None:
        if not self._enabled:
            return
        payload = {
            "step": step,
            "timestamp": timestamp,
            "joint_positions": joint_positions,
            "action_sent": action_sent,
            "physicalai.runtime.loop_duration_s": loop_duration_s,
            "physicalai.runtime.sleep_time_s": sleep_time_s,
            "stale_obs": stale_obs,
        }
        self._session.put(f"physicalai/rt/{self._session_id}/tick", self._pack(payload))

    def emit_inference(
        self,
        *,
        latency_s: float,
        offset: int,
        chunk: np.ndarray,
    ) -> None:
        if not self._enabled:
            return
        payload = {
            "physicalai.runtime.inference_latency_s": latency_s,
            "offset": offset,
            "chunk": chunk,
        }
        self._session.put(f"physicalai/rt/{self._session_id}/inference", self._pack(payload))

    def emit_lifecycle(self, event: str, **metadata: Any) -> None:  # noqa: ANN401
        if not self._enabled:
            return
        payload = {
            "event": event,
            "timestamp": time.time(),
            **metadata,
        }
        self._session.put(f"physicalai/rt/{self._session_id}/lifecycle", self._pack(payload))

    def close(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                logger.debug("Error closing zenoh session", exc_info=True)
            self._session = None
            self._enabled = False
