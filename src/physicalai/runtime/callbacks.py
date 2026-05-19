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
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np

    from physicalai.capture.camera import Camera
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

    _ACTION_HOOKS = ("before_send_action", "on_action_sent", "on_hold")

    def __init__(self, inner: Any, max_queue: int = 1024) -> None:  # noqa: D107, ANN401
        dropped = [h for h in self._ACTION_HOOKS if hasattr(inner, h)]
        if dropped:
            msg = (
                f"{type(inner).__name__} defines action hooks {dropped} which "
                "AsyncCallback does not forward (use synchronous attachment instead)"
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


class RerunCallback:
    """In-process Rerun logging for runtime visualization.

    Requires ``physicalai[observer-rerun]``.  Logs scalars and chunks every
    tick / inference event, and camera frames at ``image_decimation``-th tick.

    Do NOT wrap with :class:`AsyncCallback` — Rerun's SDK already batches I/O
    asynchronously.  The ``AsyncCallback`` guard (rejects inners with action
    hooks) is not triggered because this class defines none, but wrapping would
    double the buffering with no benefit.
    """

    def __init__(  # noqa: D107
        self,
        *,
        cameras: Mapping[str, Camera] | None = None,
        image_decimation: int = 3,
        mode: Literal["spawn", "save", "connect"] = "spawn",
        save_path: str | None = None,
        connect_addr: str = "127.0.0.1:9876",
        application_id: str = "physicalai-runtime",
    ) -> None:
        if mode == "save" and save_path is None:
            msg = "mode='save' requires save_path"
            raise ValueError(msg)
        # Fail fast if rerun-sdk is not installed.
        import rerun as rr  # noqa: PLC0415, F401

        self._cameras = cameras
        self._image_decimation = image_decimation
        self._mode = mode
        self._save_path = save_path
        self._connect_addr = connect_addr
        self._application_id = application_id

        self._last_step: int = 0
        self._fps: int = 30
        self._initialized = False
        self._camera_subscribers: dict[str, Any] = {}

    def on_lifecycle(self, event: LifecycleEvent) -> None:  # noqa: D102
        if event.event == "start" and not self._initialized:
            self._init_rerun(event.session_id, event.metadata)
        self._log_lifecycle_marker(event)

    def on_tick(self, event: TickEvent) -> None:  # noqa: D102
        import rerun as rr  # noqa: PLC0415

        self._last_step = event.step
        rr.set_time_sequence("step", event.step)
        rr.set_time_seconds("wall", event.timestamp)

        if event.joint_positions is not None:
            for i, val in enumerate(event.joint_positions):
                rr.log(f"robot/joint/{i}", rr.Scalar(float(val)))

        if event.action_sent is not None:
            for i, val in enumerate(event.action_sent):
                rr.log(f"robot/action/{i}", rr.Scalar(float(val)))

        rr.log("runtime/queue_remaining", rr.Scalar(float(event.queue_remaining)))
        rr.log("runtime/loop_duration_s", rr.Scalar(event.loop_duration_s))
        rr.log("runtime/sleep_time_s", rr.Scalar(event.sleep_time_s))
        rr.log("runtime/stale_obs", rr.Scalar(float(event.stale_obs)))

        if event.step % self._image_decimation == 0:
            self._log_camera_frames()

    def on_inference(self, event: InferenceEvent) -> None:  # noqa: D102
        import rerun as rr  # noqa: PLC0415

        horizon, dof = event.chunk.shape
        start_step = self._last_step + 1

        for k in range(horizon):
            rr.set_time_sequence("step", start_step + k)
            rr.set_time_seconds("wall", event.timestamp + k / self._fps)
            for i in range(dof):
                rr.log(f"robot/predicted/{i}", rr.Scalar(float(event.chunk[k, i])))

    def close(self) -> None:
        """Release independent camera subscribers."""
        for sub in self._camera_subscribers.values():
            try:
                sub.disconnect()
            except Exception:
                logger.exception("Error closing RerunCallback camera subscriber")
        self._camera_subscribers.clear()

    def _init_rerun(self, session_id: str, metadata: dict[str, Any]) -> None:
        import rerun as rr  # noqa: PLC0415

        rr.init(application_id=self._application_id, recording_id=session_id)
        if self._mode == "spawn":
            rr.spawn()
        elif self._mode == "save":
            rr.save(self._save_path)
        elif self._mode == "connect":
            rr.connect_tcp(self._connect_addr)

        self._fps = metadata.get("fps", 30)
        self._initialized = True

        self._open_camera_subscribers()

    def _open_camera_subscribers(self) -> None:
        from physicalai.capture.transport._shared_camera import SharedCamera  # noqa: PLC0415, PLC2701

        for name, cam in (self._cameras or {}).items():
            if isinstance(cam, SharedCamera):
                sub = SharedCamera(
                    camera_type=None,
                    service_name=cam.service_name,
                    validate_on_connect=False,
                )
                sub.connect()
                self._camera_subscribers[name] = sub
            else:
                logger.warning(
                    "RerunCallback: camera %r is not SharedCamera-backed; skipping image logging",
                    name,
                )

    def _log_camera_frames(self) -> None:
        import rerun as rr  # noqa: PLC0415

        for name, sub in self._camera_subscribers.items():
            try:
                frame = sub.read_latest()
                rr.log(f"camera/{name}", rr.Image(frame.data))
            except Exception:
                logger.debug("RerunCallback: failed to read camera %r", name, exc_info=True)

    def _log_lifecycle_marker(self, event: LifecycleEvent) -> None:
        import rerun as rr  # noqa: PLC0415

        rr.set_time_sequence("step", self._last_step)
        rr.set_time_seconds("wall", event.timestamp)
        rr.log(
            f"runtime/lifecycle/{event.event}",
            rr.TextLog(f"{event.event}: {event.metadata}"),
        )


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
