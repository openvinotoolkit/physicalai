# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shipped callback implementations for the runtime callback bus."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from physicalai.runtime.events import InferenceEvent, LifecycleEvent, TickEvent

logger = logging.getLogger(__name__)


class ConsoleCallback:
    """Periodic one-line summary to stdout (~1 per second)."""

    def __init__(self, throttle_steps: int = 30) -> None:  # noqa: D107
        self._throttle_steps = throttle_steps
        self._start_time: float | None = None

    def on_tick(self, event: TickEvent) -> None:  # noqa: D102
        if self._start_time is None:
            self._start_time = time.monotonic()
        if event.step > 0 and event.step % self._throttle_steps != 0:
            return
        elapsed = time.monotonic() - self._start_time
        print(  # noqa: T201
            f"[{elapsed:6.1f}s] step={event.step} "
            f"queue={event.queue_remaining} "
            f"loop={event.loop_duration_s * 1000:.1f}ms"
            f"{' STALE' if event.stale_obs else ''}",
        )

    def on_lifecycle(self, event: LifecycleEvent) -> None:  # noqa: D102, PLR6301
        print(f"[lifecycle] {event.event}: {event.metadata}")  # noqa: T201


class JsonlCallback:
    """Append-only JSONL recording. Numpy arrays converted to lists."""

    def __init__(self, path: str | Path, *, record_chunks: bool = False) -> None:  # noqa: D107
        self._path = Path(path)
        self._file = self._path.open("a")
        self._record_chunks = record_chunks

    def on_tick(self, event: TickEvent) -> None:  # noqa: D102
        self._write(
            "tick",
            {
                "session_id": event.session_id,
                "step": event.step,
                "timestamp": event.timestamp,
                "joint_positions": _np_to_list(event.joint_positions),
                "action_sent": _np_to_list(event.action_sent),
                "queue_remaining": event.queue_remaining,
                "loop_duration_s": event.loop_duration_s,
                "sleep_time_s": event.sleep_time_s,
                "stale_obs": event.stale_obs,
            },
        )

    def on_inference(self, event: InferenceEvent) -> None:  # noqa: D102
        payload: dict[str, Any] = {
            "session_id": event.session_id,
            "timestamp": event.timestamp,
            "latency_s": event.latency_s,
            "offset": event.offset,
            "chunk_shape": list(event.chunk.shape),
        }
        if self._record_chunks:
            payload["chunk"] = event.chunk.tolist()
        self._write("inference", payload)

    def on_lifecycle(self, event: LifecycleEvent) -> None:  # noqa: D102
        self._write(
            "lifecycle",
            {
                "session_id": event.session_id,
                "timestamp": event.timestamp,
                "event": event.event,
                "metadata": event.metadata,
            },
        )

    def close(self) -> None:  # noqa: D102
        self._file.close()

    def _write(self, kind: str, payload: dict[str, Any]) -> None:
        record = {"type": kind, **payload}
        self._file.write(json.dumps(record, default=_json_default) + "\n")
        self._file.flush()


class ZenohCallback:
    """Publishes events over zenoh pub-sub using the existing TelemetryEmitter.

    Requires ``physicalai[telemetry]`` (zenoh + msgpack).  If not installed,
    construction raises ``ImportError``.

    Note: ``zenoh.open()`` runs synchronously on the control thread during the
    first ``on_lifecycle`` call.  This is typically <10ms but blocks the loop
    until the zenoh session is established.
    """

    def __init__(self) -> None:  # noqa: D107
        # Verify zenoh is importable at construction time (fail fast).
        from physicalai.runtime._telemetry import TelemetryEmitter  # noqa: PLC0415, PLC2701, F401

        self._emitter: Any = None
        self._initialized = False

    def on_lifecycle(self, event: LifecycleEvent) -> None:  # noqa: D102
        if not self._initialized:
            self._init_emitter(event.session_id)
        self._emitter.emit_lifecycle(event.event, **event.metadata)

    def on_tick(self, event: TickEvent) -> None:  # noqa: D102
        if not self._emitter or not self._emitter.enabled:
            return
        self._emitter.emit_tick(
            step=event.step,
            timestamp=event.timestamp,
            joint_positions=event.joint_positions,
            action_sent=event.action_sent,
            queue_remaining=event.queue_remaining,
            loop_duration_s=event.loop_duration_s,
            sleep_time_s=event.sleep_time_s,
            stale_obs=event.stale_obs,
        )

    def on_inference(self, event: InferenceEvent) -> None:  # noqa: D102
        if not self._emitter or not self._emitter.enabled:
            return
        self._emitter.emit_inference(
            latency_s=event.latency_s,
            offset=event.offset,
            chunk=event.chunk,
        )

    def close(self) -> None:  # noqa: D102
        if self._emitter:
            self._emitter.close()

    def _init_emitter(self, session_id: str) -> None:
        from physicalai.runtime._telemetry import TelemetryEmitter  # noqa: PLC0415, PLC2701

        self._emitter = TelemetryEmitter(session_id=session_id)
        self._initialized = True


class AsyncCallback:
    """Wraps a callback so all hooks run on a dedicated background thread.

    The control loop only pays deque.append per event. On overflow, oldest
    events are dropped.
    """

    def __init__(self, inner: Any, max_queue: int = 1024) -> None:  # noqa: D107, ANN401
        if hasattr(inner, "before_send_action"):
            msg = (
                f"{type(inner).__name__} defines before_send_action which requires "
                "synchronous request-response semantics incompatible with AsyncCallback"
            )
            raise TypeError(msg)
        self._inner = inner
        self._queue: deque[tuple[str, Any]] = deque(maxlen=max_queue)
        self._stop = threading.Event()
        self._has_work = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="AsyncCallbackWorker", daemon=True)
        self._thread.start()

    def on_tick(self, event: TickEvent) -> None:  # noqa: D102
        self._enqueue("on_tick", event)

    def on_inference(self, event: InferenceEvent) -> None:  # noqa: D102
        self._enqueue("on_inference", event)

    def on_lifecycle(self, event: LifecycleEvent) -> None:  # noqa: D102
        self._enqueue("on_lifecycle", event)

    def close(self) -> None:  # noqa: D102
        self._stop.set()
        self._has_work.set()
        self._thread.join(timeout=5.0)
        close_fn = getattr(self._inner, "close", None)
        if close_fn is not None:
            close_fn()

    def _enqueue(self, method: str, event: Any) -> None:  # noqa: ANN401
        self._queue.append((method, event))
        self._has_work.set()

    def _worker(self) -> None:
        while not self._stop.is_set():
            self._has_work.wait()
            self._has_work.clear()
            while self._queue:
                method, event = self._queue.popleft()
                fn = getattr(self._inner, method, None)
                if fn is not None:
                    try:
                        fn(event)
                    except Exception:
                        logger.exception("AsyncCallback inner %r.%s failed", self._inner, method)


def _np_to_list(arr: np.ndarray | None) -> list[float] | None:
    if arr is None:
        return None
    return arr.tolist()


def _json_default(obj: object) -> Any:  # noqa: ANN401
    import numpy as np  # noqa: PLC0415

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    msg = f"Object of type {type(obj)} is not JSON serializable"
    raise TypeError(msg)
