# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from physicalai.runtime._telemetry import _decode_numpy  # noqa: PLC2701

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_MIN_TOPIC_PARTS = 4


def _unpack_numpy(obj: object) -> object:
    if isinstance(obj, dict) and obj.get("__np__"):
        return _decode_numpy(obj)
    if isinstance(obj, dict):
        return {k: _unpack_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unpack_numpy(v) for v in obj]
    return obj


class TelemetrySubscriber:
    def __init__(self, session_id: str | None = None) -> None:
        import msgpack  # noqa: PLC0415
        import zenoh  # noqa: PLC0415

        self._zenoh = zenoh
        self._msgpack = msgpack
        self._session = zenoh.open(zenoh.Config())
        self._handlers: list[Callable[[str, str, dict[str, Any]], None]] = []
        self._session_id = session_id
        self._sub: Any = None

    def add_handler(self, handler: Callable[[str, str, dict[str, Any]], None]) -> None:
        self._handlers.append(handler)

    def start(self) -> None:
        prefix = f"physicalai/rt/{self._session_id}/**" if self._session_id else "physicalai/rt/**"
        self._sub = self._session.declare_subscriber(prefix, self._on_event)

    def _on_event(self, sample: Any) -> None:  # noqa: ANN401
        try:
            key = str(sample.key_expr)
            parts = key.split("/")
            if len(parts) < _MIN_TOPIC_PARTS:
                return
            session_id = parts[2]
            topic = parts[3]
            payload = self._msgpack.unpackb(sample.payload.to_bytes(), raw=False)
            payload = _unpack_numpy(payload)
            if not isinstance(payload, dict):
                return
            for handler in self._handlers:
                try:
                    handler(session_id, topic, payload)
                except Exception:
                    logger.exception("Handler error")
        except Exception:
            logger.exception("Failed to decode telemetry event")

    def stop(self) -> None:
        if self._sub is not None:
            self._sub.undeclare()
            self._sub = None
        if self._session is not None:
            self._session.close()
            self._session = None
