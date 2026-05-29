# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shipped callback implementations for the runtime callback bus."""

from __future__ import annotations

import colorsys
import json
import logging
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from physicalai.capture.frame import Frame

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
                "joint_positions": _np_to_list(event.robot_observation.joint_positions),
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

    def on_tick(self, event: TickEvent) -> None:
        """Enqueue tick event, copying borrowed frame buffers to prevent dangling SHM refs.

        Zero-copy SharedCamera frames are views into iceoryx2 shared memory that become
        invalid on the next read_latest() call. Since the background worker may process
        this event after the next tick, we use dataclasses.replace to produce a new
        TickEvent with owned copies of any borrowed frame buffers. Frames that already
        own their data (the common case) are passed through untouched.
        """
        if any(not f.data.flags.owndata for f in event.camera_frames.values()):
            event = replace(
                event,
                camera_frames={
                    name: Frame(data=f.data.copy(), timestamp=f.timestamp, sequence=f.sequence)
                    if not f.data.flags.owndata
                    else f
                    for name, f in event.camera_frames.items()
                },
            )
        self._enqueue("on_tick", event)

    def on_inference(self, event: InferenceEvent) -> None:
        """Enqueue inference event for background processing."""
        self._enqueue("on_inference", event)

    def on_lifecycle(self, event: LifecycleEvent) -> None:
        """Enqueue lifecycle event for background processing."""
        self._enqueue("on_lifecycle", event)

    def close(self) -> None:
        """Stop the worker thread and close the inner callback."""
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
        self._pred_horizon: int = 0
        self._initialized = False
        self._blueprint_updated = False
        self._camera_subscribers: dict[str, Any] = {}
        self._latencies: deque[float] = deque(maxlen=200)

    def on_lifecycle(self, event: LifecycleEvent) -> None:  # noqa: D102
        if event.event == "start" and not self._initialized:
            self._init_rerun(event.session_id, event.metadata)
        self._log_lifecycle_marker(event)

    def on_tick(self, event: TickEvent) -> None:  # noqa: D102
        import rerun as rr  # noqa: PLC0415

        self._last_step = event.step
        rr.set_time("step", sequence=event.step)
        rr.set_time("wall", timestamp=event.timestamp)

        rr.log("robot/joints", rr.Scalars([float(v) for v in event.robot_observation.joint_positions]))

        if event.action_sent is not None:
            rr.log("robot/actions", rr.Scalars([float(v) for v in event.action_sent]))

        rr.log("queue/remaining", rr.Scalars(float(event.queue_remaining)))
        rr.log("queue/inference", rr.Scalars(0.0))
        rr.log("runtime/loop_duration_s", rr.Scalars(event.loop_duration_s))
        rr.log("runtime/sleep_time_s", rr.Scalars(event.sleep_time_s))
        rr.log("runtime/stale_obs", rr.Scalars(float(event.stale_obs)))

        if self._log_images and event.step % self._image_decimation == 0:
            self._log_camera_frames()

    def on_inference(self, event: InferenceEvent) -> None:  # noqa: D102
        import numpy as np  # noqa: PLC0415
        import rerun as rr  # noqa: PLC0415

        horizon = event.chunk.shape[0]
        n_joints = event.chunk.shape[1]
        start_step = self._last_step + 1

        # Clear previous predictions so stale trajectories don't linger.
        if self._pred_horizon > 0:
            rr.set_time("step", sequence=self._last_step)
            rr.log("robot/predicted", rr.Clear(recursive=False))

        self._pred_horizon = horizon

        # Batch-log all prediction steps in one efficient send_columns call.
        # send_columns bypasses the thread-local time context, so the viewer's
        # "latest" cursor is not pushed to the last prediction step.
        steps = np.arange(start_step, start_step + horizon, dtype=np.int64)
        wall_times = event.timestamp + np.arange(horizon, dtype=np.float64) / self._fps

        # Scalars expects one float per row when logging a single series,
        # but for N joints we need N values per row → use partitioned columns.
        flat_scalars = event.chunk.astype(np.float64).ravel()
        rr.send_columns(
            "robot/predicted",
            indexes=[
                rr.TimeColumn("step", sequence=steps),
                rr.TimeColumn("wall", timestamp=wall_times),
            ],
            columns=rr.Scalars.columns(scalars=flat_scalars).partition(lengths=[n_joints] * horizon),
        )

        # Mark the inference event on the queue timeline (shows as a spike/refill).
        rr.set_time("step", sequence=self._last_step)
        rr.set_time("wall", timestamp=event.timestamp)
        rr.log("queue/inference", rr.Scalars(float(horizon)))

        # Inference latency stats as a live-updating table.
        self._latencies.append(event.latency_s)
        self._log_latency_table()

        # Re-send blueprint with correct horizon on first inference.
        if not self._blueprint_updated:
            self._blueprint_updated = True
            self._send_default_blueprint()

    def close(self) -> None:
        """Release independent camera subscribers (SharedCamera only)."""
        from physicalai.capture.transport._shared_camera import SharedCamera  # noqa: PLC0415, PLC2701

        for sub in self._camera_subscribers.values():
            if not isinstance(sub, SharedCamera):
                continue
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
        self._joint_names: list[str] = metadata.get("joint_names", [])
        self._initialized = True

        self._send_series_styles()
        self._open_camera_subscribers()
        self._send_default_blueprint()

    @staticmethod
    def _generate_joint_colors(n: int) -> list[list[int]]:
        """Generate N perceptually distinct colors via evenly-spaced hues.

        Returns:
            List of RGBA colors with values in [0, 255].
        """
        colors = []
        for i in range(n):
            hue = i / n
            r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.85)
            colors.append([int(r * 255), int(g * 255), int(b * 255), 255])
        return colors

    def _send_series_styles(self) -> None:
        """Set static visual style for series: solid lines for actions, dots for predicted."""
        import rerun as rr  # noqa: PLC0415

        names = self._joint_names or None
        n_joints = len(self._joint_names) if self._joint_names else 0
        joint_colors = self._generate_joint_colors(n_joints) if n_joints else None

        # Actions: distinct color per joint so you can identify each line
        action_names = [f"{n} (action)" for n in self._joint_names] if self._joint_names else None
        rr.log("robot/actions", rr.SeriesLines(widths=2.0, colors=joint_colors, names=action_names), static=True)
        # Predicted: same joint colors at lower alpha + cross markers to distinguish from action lines
        pred_names = [f"{n} (pred)" for n in self._joint_names] if self._joint_names else None
        pred_colors = [[r, g, b, 100] for r, g, b, _a in joint_colors] if joint_colors else None
        rr.log(
            "robot/predicted",
            rr.SeriesPoints(marker_sizes=4.0, colors=pred_colors, markers="cross", names=pred_names),
            static=True,
        )
        # Joints: same distinct colors as actions for consistency
        rr.log("robot/joints", rr.SeriesLines(widths=1.5, colors=joint_colors, names=names), static=True)
        # Queue: green line; inference: thin red vertical spikes
        rr.log("queue/remaining", rr.SeriesLines(widths=3.0, colors=[80, 200, 120, 255], names="queue"), static=True)
        rr.log("queue/inference", rr.SeriesLines(widths=1.5, colors=[220, 50, 50, 255], names="inference"), static=True)

    def _send_default_blueprint(self) -> None:
        """Send a default blueprint: actions+predicted overlaid, queue, joints, cameras."""
        try:
            import rerun as rr  # noqa: PLC0415
            import rerun.blueprint as rrb  # noqa: PLC0415
        except ImportError:
            logger.debug("rerun.blueprint not available; skipping default blueprint")
            return

        camera_names = list((self._cameras or {}).keys()) if self._log_images else []
        fps = int(self._fps)
        horizon = self._pred_horizon or int(fps * 1.5)  # best-guess until first inference

        # The viewer's cursor tracks the latest logged "step" value.
        # With send_columns, predictions are at [current+1 … current+horizon],
        # so the cursor sits roughly `horizon` steps ahead of the actual tick.
        # We size the visible window so actions and predictions get equal space:
        # lookback = 2*horizon → horizon steps of history + horizon steps of predictions.
        lookback = horizon * 2
        actions_range = rrb.VisibleTimeRange(
            timeline="step",
            start=rrb.TimeRangeBoundary.cursor_relative(seq=-lookback),
            end=rrb.TimeRangeBoundary.cursor_relative(seq=0),
        )

        latency_view = rrb.TextDocumentView(
            origin="/inference/stats",
            name="Inference Latency",
        )
        if camera_names:
            top_row = rrb.Grid(
                contents=[
                    *[rrb.Spatial2DView(origin=f"/camera/{n}", name=n) for n in camera_names],
                    latency_view,
                ],
            )
        else:
            top_row = latency_view

        views: list[Any] = [
            top_row,
            rrb.Tabs(
                rrb.TimeSeriesView(
                    origin="/robot",
                    contents=["/robot/actions", "/robot/predicted"],
                    name="Actions vs Predicted",
                    time_ranges=actions_range,
                ),
                rrb.TimeSeriesView(
                    origin="/robot/joints",
                    name="Joint State",
                    time_ranges=actions_range,
                ),
                active_tab="Actions vs Predicted",
            ),
            rrb.TimeSeriesView(
                origin="/queue",
                name="Action Queue",
                time_ranges=actions_range,
            ),
        ]

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
                # Direct camera — read from it on tick (no separate subscriber needed)
                self._camera_subscribers[name] = cam

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

    def _log_latency_table(self) -> None:
        import numpy as np  # noqa: PLC0415
        import rerun as rr  # noqa: PLC0415

        arr = np.array(self._latencies)
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        last = float(arr[-1])
        n = len(arr)
        # Headroom = time the queue can sustain minus inference latency.
        # Positive = safe; negative = queue starved before next chunk arrives.
        queue_time = self._pred_horizon / self._fps if self._pred_horizon else 0
        headroom = queue_time - p99

        md = (
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| **Last** | {last * 1000:.1f} ms |\n"
            f"| **p50** | {p50 * 1000:.1f} ms |\n"
            f"| **p95** | {p95 * 1000:.1f} ms |\n"
            f"| **p99** | {p99 * 1000:.1f} ms |\n"
            f"| **Queue headroom** | {headroom * 1000:.0f} ms |\n"
            f"| Samples | {n} |"
        )
        rr.log("inference/stats", rr.TextDocument(md, media_type=rr.MediaType.MARKDOWN))


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
