# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ._codec import decode_payload

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_MIN_TOPIC_PARTS = 4


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

    def _decode_event(self, sample: Any) -> tuple[str, str, dict[str, Any]] | None:  # noqa: ANN401
        key = str(sample.key_expr)
        parts = key.split("/")
        if len(parts) < _MIN_TOPIC_PARTS:
            return None
        payload = self._msgpack.unpackb(sample.payload.to_bytes(), raw=False)
        payload = decode_payload(payload)
        if not isinstance(payload, dict):
            return None
        return parts[2], parts[3], payload

    def _on_event(self, sample: Any) -> None:  # noqa: ANN401
        try:
            event = self._decode_event(sample)
        except Exception:
            logger.exception("Failed to decode telemetry event")
            return
        if event is None:
            return
        session_id, topic, payload = event
        for handler in self._handlers:
            try:
                handler(session_id, topic, payload)
            except Exception:
                logger.exception("Handler error")

    def stop(self) -> None:
        if self._sub is not None:
            self._sub.undeclare()
            self._sub = None
        if self._session is not None:
            self._session.close()
            self._session = None
