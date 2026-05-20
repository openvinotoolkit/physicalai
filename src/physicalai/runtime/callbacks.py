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
        log_images: bool = True,
        image_jpeg_quality: int | None = None,
        image_max_dim: int | None = None,
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
        self._log_images = log_images
        self._image_jpeg_quality = image_jpeg_quality
        self._image_max_dim = image_max_dim
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
        rr.set_time("step", sequence=event.step)
        rr.set_time("wall", timestamp=event.timestamp)

        if event.joint_positions is not None:
            # One plot with N overlaid series instead of N separate plots.
            rr.log("robot/joints", rr.Scalars([float(v) for v in event.joint_positions]))

        if event.action_sent is not None:
            rr.log("robot/actions", rr.Scalars([float(v) for v in event.action_sent]))

        rr.log("queue/remaining", rr.Scalars(float(event.queue_remaining)))
        rr.log("runtime/loop_duration_s", rr.Scalars(event.loop_duration_s))
        rr.log("runtime/sleep_time_s", rr.Scalars(event.sleep_time_s))
        rr.log("runtime/stale_obs", rr.Scalars(float(event.stale_obs)))

        if self._log_images and event.step % self._image_decimation == 0:
            self._log_camera_frames()

    def on_inference(self, event: InferenceEvent) -> None:  # noqa: D102
        import rerun as rr  # noqa: PLC0415

        horizon = event.chunk.shape[0]
        start_step = self._last_step + 1

        for k in range(horizon):
            rr.set_time("step", sequence=start_step + k)
            rr.set_time("wall", timestamp=event.timestamp + k / self._fps)
            rr.log("robot/predicted", rr.Scalars([float(v) for v in event.chunk[k]]))

        # Mark the inference event on the queue timeline (shows as a spike/refill).
        rr.set_time("step", sequence=self._last_step)
        rr.set_time("wall", timestamp=event.timestamp)
        rr.log("queue/inference", rr.Scalars(float(horizon)))

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
            # Rerun 0.22+ uses gRPC. Address like "127.0.0.1:9876" is wrapped
            # into the canonical rerun+http://host:port/proxy URL.
            addr = self._connect_addr
            url = addr if addr.startswith(("rerun+http://", "rerun+https://")) else f"rerun+http://{addr}/proxy"
            rr.connect_grpc(url=url)

        self._fps = metadata.get("fps", 30)
        self._initialized = True

        self._send_series_styles()
        self._open_camera_subscribers()
        self._send_default_blueprint()

    @staticmethod
    def _send_series_styles() -> None:
        """Set static visual style for series: solid lines for actions, dots for predicted."""
        import rerun as rr  # noqa: PLC0415

        # Actions: solid, 2px, blue
        rr.log("robot/actions", rr.SeriesLines(width=2.0, colors=[70, 130, 230, 255], names="actions"), static=True)
        # Predicted: large markers (circles), orange — clearly distinct from action lines
        rr.log(
            "robot/predicted",
            rr.SeriesPoints(marker_sizes=6.0, colors=[255, 140, 0, 230], names="predicted"),
            static=True,
        )
        # Joints: default styling (thin lines)
        rr.log("robot/joints", rr.SeriesLines(width=1.5), static=True)
        # Queue: green line; inference events: red spikes
        rr.log("queue/remaining", rr.SeriesLines(width=2.0, colors=[80, 200, 120, 255], names="queue"), static=True)
        rr.log(
            "queue/inference",
            rr.SeriesPoints(marker_sizes=8.0, colors=[220, 50, 50, 255], names="inference"),
            static=True,
        )

    def _send_default_blueprint(self) -> None:
        """Send a default blueprint: actions+predicted overlaid, queue, joints, cameras."""
        try:
            import rerun as rr  # noqa: PLC0415
            import rerun.blueprint as rrb  # noqa: PLC0415
        except ImportError:
            logger.debug("rerun.blueprint not available; skipping default blueprint")
            return

        camera_names = list((self._cameras or {}).keys()) if self._log_images else []

        views: list[Any] = [
            rrb.TimeSeriesView(
                origin="/robot",
                contents=["/robot/actions", "/robot/predicted"],
                name="Actions vs Predicted",
            ),
            rrb.TimeSeriesView(
                origin="/queue",
                name="Action Queue",
            ),
            rrb.TimeSeriesView(
                origin="/robot/joints",
                name="Joint State",
            ),
        ]
        if camera_names:
            views.append(
                rrb.Grid(
                    contents=[rrb.Spatial2DView(origin=f"/camera/{n}", name=n) for n in camera_names],
                    name="Cameras",
                )
            )

        blueprint = rrb.Blueprint(
            rrb.Vertical(*views),
            rrb.SelectionPanel(state="collapsed"),
            rrb.TimePanel(state="expanded"),
        )
        try:
            rr.send_blueprint(blueprint, make_active=True, make_default=True)
        except Exception:
            logger.debug("Failed to send Rerun blueprint", exc_info=True)

    def _open_camera_subscribers(self) -> None:
        if not self._log_images:
            return
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
                data = frame.data
                if self._image_max_dim is not None:
                    data = _downsample_to_max_dim(data, self._image_max_dim)
                img = rr.Image(data)
                if self._image_jpeg_quality is not None:
                    img = img.compress(jpeg_quality=self._image_jpeg_quality)
                rr.log(f"camera/{name}", img)
            except Exception:
                logger.debug("RerunCallback: failed to read camera %r", name, exc_info=True)

    def _log_lifecycle_marker(self, event: LifecycleEvent) -> None:
        import rerun as rr  # noqa: PLC0415

        rr.set_time("step", sequence=self._last_step)
        rr.set_time("wall", timestamp=event.timestamp)
        rr.log(
            f"runtime/lifecycle/{event.event}",
            rr.TextLog(f"{event.event}: {event.metadata}"),
        )


def _np_to_list(arr: np.ndarray | None) -> list[float] | None:
    if arr is None:
        return None
    return arr.tolist()


def _downsample_to_max_dim(data: np.ndarray, max_dim: int) -> np.ndarray:
    """Subsample image so the longer side is <= ``max_dim``. No-op if already smaller.

    Returns:
        Subsampled image. Does not modify input.
    """
    h, w = data.shape[:2]
    longer = max(h, w)
    if longer <= max_dim:
        return data
    stride = (longer + max_dim - 1) // max_dim  # ceil-divide
    return data[::stride, ::stride]


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
