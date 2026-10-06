# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Render simulation cameras on their own thread.

MuJoCo's render call releases the GIL, so rendering on a separate thread takes
the cameras out of the control loop: a tick no longer waits ~8 ms per camera
frame, and the simulation keeps real time. The sim thread publishes a cheap
pose snapshot (``qpos``, mocap poses) every tick; the camera thread rebuilds
kinematics from the latest snapshot in its own ``MjData`` and renders whichever
cameras are due.

The renderers are created, used and closed on the camera thread, so no OpenGL
context ever moves between threads (EGL does not allow that).
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import threading
import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

_IDLE_S = 0.002
"""Longest the camera thread sleeps between checks for due frames."""


class PoseSnapshot:
    """The part of ``MjData`` rendering needs, copied under a lock."""

    def __init__(self, model: object) -> None:
        """Allocate buffers sized for `model`."""
        self._lock = threading.Lock()
        self._qpos = np.zeros(int(model.nq))
        self._mocap_pos = np.zeros((int(model.nmocap), 3))
        self._mocap_quat = np.zeros((int(model.nmocap), 4))
        self._seq = 0

    @property
    def seq(self) -> int:
        """Increases with every publish; lets the reader skip unchanged state."""
        return self._seq

    def publish(self, data: object) -> None:
        """Copy the current poses out of `data` (sim thread)."""
        with self._lock:
            np.copyto(self._qpos, data.qpos)
            np.copyto(self._mocap_pos, data.mocap_pos)
            np.copyto(self._mocap_quat, data.mocap_quat)
            self._seq += 1

    def load_into(self, data: object) -> int:
        """Copy the latest poses into `data` (camera thread).

        Returns:
            The sequence number of the snapshot that was loaded.
        """
        with self._lock:
            np.copyto(data.qpos, self._qpos)
            np.copyto(data.mocap_pos, self._mocap_pos)
            np.copyto(data.mocap_quat, self._mocap_quat)
            return self._seq


class CameraThread:
    """Run camera rendering for one model on a background thread."""

    def __init__(
        self,
        model: object,
        *,
        setup: Callable[[object], None],
        render: Callable[[object], None],
        teardown: Callable[[], None],
    ) -> None:
        """Bind to `model`; `setup`, `render` and `teardown` all run on the camera thread.

        Args:
            model: The MuJoCo model being simulated (read-only here).
            setup: Creates the renderers for the model it is given (this thread's, never a newer one).
            render: Renders the cameras that are due, from the given ``MjData``.
            teardown: Closes the renderers.
        """
        self._model = model
        self._setup = setup
        self._render = render
        self._teardown = teardown
        self.snapshot = PoseSnapshot(model)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the camera thread."""
        self._thread = threading.Thread(target=self._run, name="mujoco-cameras", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> bool:
        """Stop the camera thread and wait for it to close its renderers.

        Returns:
            Whether the thread ended. If not, it still owns its renderers and closes them itself
            when its current render returns.
        """
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is None:
            return True
        thread.join(timeout=timeout_s)
        if thread.is_alive():
            logger.warning("Camera thread did not stop within {:.1f}s", timeout_s)
            return False
        return True

    def publish(self, data: object) -> None:
        """Hand the camera thread the current poses (call once per sim tick)."""
        self.snapshot.publish(data)

    def _run(self) -> None:
        import mujoco  # noqa: PLC0415

        try:
            self._setup(self._model)
            data = mujoco.MjData(self._model)
            loaded = 0  # nothing published yet: the zeroed snapshot is not a pose
            while not self._stop.is_set():
                if self.snapshot.seq == 0:
                    self._stop.wait(_IDLE_S)
                    continue
                if self.snapshot.seq != loaded:
                    loaded = self.snapshot.load_into(data)
                    # Positions only: kinematics, cameras, lights, flex vertices.
                    mujoco.mj_fwdPosition(self._model, data)
                self._render(data)
                self._stop.wait(_IDLE_S)
        except Exception as exc:  # noqa: BLE001 - a camera failure must not take the simulation down
            logger.warning("Camera thread stopped: {}", exc)
        finally:
            try:
                self._teardown()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Camera teardown failed: {}", exc)


class RateMeter:
    """Rates over a sliding window of recent events (thread-safe)."""

    def __init__(self, window_s: float = 2.0) -> None:
        """Keep events from the last `window_s` seconds."""
        self._window_s = window_s
        self._events: deque[tuple[float, float]] = deque()
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Forget all events (for example after the simulation clock restarts)."""
        with self._lock:
            self._events.clear()

    def add(self, value: float = 0.0, now: float | None = None) -> None:
        """Record one event with an optional value (such as the sim time)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            self._events.append((now, value))
            while self._events and now - self._events[0][0] > self._window_s:
                self._events.popleft()

    def rate(self) -> float | None:
        """Events per wall-clock second over the window.

        Returns:
            The rate, or ``None`` with fewer than two events.
        """
        with self._lock:
            if len(self._events) < 2:  # noqa: PLR2004
                return None
            span = self._events[-1][0] - self._events[0][0]
            return (len(self._events) - 1) / span if span > 0 else None

    def value_rate(self) -> float | None:
        """Change in the recorded value per wall-clock second over the window.

        Returns:
            The rate, or ``None`` with fewer than two events.
        """
        with self._lock:
            if len(self._events) < 2:  # noqa: PLR2004
                return None
            span = self._events[-1][0] - self._events[0][0]
            return (self._events[-1][1] - self._events[0][1]) / span if span > 0 else None
