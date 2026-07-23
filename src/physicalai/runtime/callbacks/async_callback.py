# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Background-thread callback wrapper."""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections import deque
from typing import TYPE_CHECKING, Any

from physicalai.capture.frame import Frame
from physicalai.config import export_config

if TYPE_CHECKING:
    from physicalai.runtime.events import InferenceEvent, LifecycleEvent, MetricsEvent, TickEvent

logger = logging.getLogger(__name__)


@export_config(class_path="physicalai.runtime.AsyncCallback")
class AsyncCallback:
    """Wraps a callback so all hooks run on a dedicated background thread.

    The control loop only pays deque.append per event. On overflow, oldest
    events are dropped.
    """

    _ACTION_HOOKS = ("on_action_ready", "on_action_sent")

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
        this event after the next tick, borrowed frames are replaced with owned copies
        before enqueuing.
        """
        camera_frames = event.camera_frames
        if any(not f.data.flags.owndata for f in camera_frames.values()):
            copied = {
                name: Frame(data=f.data.copy(), timestamp=f.timestamp, sequence=f.sequence)
                if not f.data.flags.owndata
                else f
                for name, f in camera_frames.items()
            }
            event = dataclasses.replace(event, camera_frames=copied)
        self._enqueue("on_tick", event)

    def on_inference(self, event: InferenceEvent) -> None:
        """Enqueue inference event for background processing."""
        self._enqueue("on_inference", event)

    def on_lifecycle(self, event: LifecycleEvent) -> None:
        """Enqueue lifecycle event for background processing."""
        self._enqueue("on_lifecycle", event)

    def on_metrics(self, event: MetricsEvent) -> None:
        """Enqueue metrics event for background processing."""
        self._enqueue("on_metrics", event)

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
