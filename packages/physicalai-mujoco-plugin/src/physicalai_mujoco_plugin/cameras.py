# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Camera service: renders the simulation's cameras into the HTTP frame buffers.

The driver owns one :class:`CameraService` and publishes ``MjData`` to it once per tick; the
service never steps physics. By default it renders on its own thread (``camera_thread``).
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import contextlib
import functools
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from physicalai_mujoco_plugin.camera_thread import CameraThread, RateMeter

if TYPE_CHECKING:
    from physicalai_mujoco_plugin.http_server import FrameBuffer


@dataclass(frozen=True)
class CameraConfig:
    """Output configuration for a virtual camera.

    The camera is served over HTTP (MJPEG stream and JPEG snapshot). Frames
    are streamed as MuJoCo renders them; ``mirror_horizontal`` flips them
    left to right.
    """

    name: str
    width: int = 640
    height: int = 480
    fps: int = 30
    mirror_horizontal: bool = False

    def __post_init__(self) -> None:
        """Reject sizes and frame rates the renderer cannot use.

        Raises:
            ValueError: If the width, height or frame rate is not positive.
        """
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            size = f"{self.width}x{self.height}@{self.fps}"
            msg = f"Camera {self.name!r} needs a positive size and frame rate, got {size}"
            raise ValueError(msg)


class CameraService:
    """Render configured cameras of the current model into one :class:`FrameBuffer` each."""

    def __init__(self, configs: list[CameraConfig], *, render_in_thread: bool = True) -> None:
        """Create a stopped service.

        Args:
            configs: Cameras to stream, by model camera name.
            render_in_thread: Render on a dedicated thread (tests render synchronously instead).
        """
        self.configs = list(configs)
        self.render_in_thread = render_in_thread
        self.frame_buffers: dict[str, FrameBuffer] = {}
        self.meters: dict[str, RateMeter] = {}
        self._lock = threading.RLock()
        self._renderers: dict[str, object] = {}
        self._last_frame_ts: dict[str, float] = {}
        self._thread: CameraThread | None = None
        self._data: object | None = None

    @property
    def renderers(self) -> dict[str, object]:
        """The current renderers, by camera name."""
        return self._renderers

    @property
    def on_thread(self) -> bool:
        """Whether a camera thread is rendering, or has not stopped yet."""
        return self._thread is not None

    def rendering(self) -> set[str]:
        """Return the names of the cameras that have a renderer (any thread)."""
        with self._lock:
            return set(self._renderers)

    def fps(self) -> dict[str, float | None]:
        """Return each camera's frame rate over the last ~2 s."""
        return {name: meter.rate() for name, meter in self.meters.items()}

    def start(self, model: object, data: object) -> None:
        """Create the buffers and renderers for *model* and start rendering *data*."""
        self._data = data
        configured = {config.name for config in self.configs}
        for name in [name for name in self.frame_buffers if name not in configured]:
            del self.frame_buffers[name]  # a camera of the previous model; the HTTP server shares this dict
        if not self.configs:
            return
        for config in self.configs:
            if config.name not in self.frame_buffers:
                from physicalai_mujoco_plugin.http_server import FrameBuffer  # noqa: PLC0415

                self.frame_buffers[config.name] = FrameBuffer(config.name)
            self._last_frame_ts[config.name] = 0.0
            self.meters.setdefault(config.name, RateMeter())
            logger.info(
                "Camera started: {} ({}x{}@{} fps, {})",
                config.name,
                config.width,
                config.height,
                config.fps,
                "own thread" if self.render_in_thread else "control loop",
            )
        # Each camera thread gets its own renderer dict, bound to its own model. A thread that
        # outlives stop() can then only touch its own renderers, never the next scene's.
        renderers: dict[str, object] = {}
        with self._lock:
            self._renderers = renderers
        if self.render_in_thread:
            self._thread = CameraThread(
                model,
                setup=functools.partial(self._create_renderers, renderers),
                render=functools.partial(self.render, renderers=renderers),
                teardown=functools.partial(self._close_renderers, renderers),
            )
            self._thread.publish(data)  # the first frame shows the real scene
            self._thread.start()
        else:
            self._create_renderers(renderers, model)

    def publish(self, data: object) -> None:
        """Hand the tick's state to the camera thread, or render due cameras now (sim thread)."""
        self._data = data
        if self._thread is not None:
            self._thread.publish(data)
        else:
            self.render(data)

    def stop(self, timeout_s: float = 5.0) -> bool:
        """Stop rendering and release the renderers; the frame buffers stay for the next model.

        Args:
            timeout_s: How long to wait for the camera thread to finish its current render.

        Returns:
            Whether rendering stopped. If not, the camera thread is still inside a render of the
            current model and stays `on_thread`; call `stop` again to wait for it again.
        """
        thread = self._thread
        stopped = thread is None or thread.stop(timeout_s)
        if stopped:
            self._thread = None
        with self._lock:
            renderers, self._renderers = self._renderers, {}
        if stopped:
            self._close_renderers(renderers)  # no-op after the thread closed its own
        else:
            logger.warning("Leaving {} camera renderer(s) to the camera thread that did not stop", len(renderers))
        self._last_frame_ts.clear()
        return stopped

    def close(self) -> None:
        """Stop rendering and drop the frame buffers."""
        self.stop()
        self.frame_buffers.clear()

    def render(self, data: object | None = None, renderers: dict[str, object] | None = None) -> None:
        """Render every camera that is due, from `data` with `renderers` (the current ones by default)."""
        data = self._data if data is None else data
        renderers = self._renderers if renderers is None else renderers
        now = time.monotonic()
        for config in self.configs:
            if renderers is not self._renderers:
                return  # a camera thread that outlived stop(): the scene moved on, publish nothing
            renderer = renderers.get(config.name)
            if renderer is None:
                continue

            period = 1.0 / config.fps
            if now - self._last_frame_ts[config.name] < period:
                continue

            try:
                renderer.update_scene(data, camera=config.name)
                # mujoco.Renderer already returns the image upright (it flips the
                # OpenGL read-back itself), so the frame is used as rendered.
                frame = renderer.render()[:, :, :3]
            except (RuntimeError, ValueError) as exc:
                logger.debug("Camera render error for '{}': {}", config.name, exc)
                continue

            if renderers is not self._renderers:
                return  # the scene switched while this frame rendered
            if config.mirror_horizontal:
                frame = frame[:, ::-1, :]
            frame = np.ascontiguousarray(frame)

            buffer = self.frame_buffers.get(config.name)
            if buffer is not None:
                buffer.put(frame)

            # Keep a fixed frame grid so polling jitter does not lower the rate; after a stall, restart it.
            last = self._last_frame_ts[config.name]
            self._last_frame_ts[config.name] = last + period if now - last < 2 * period else now
            meter = self.meters.get(config.name)
            if meter is not None:
                meter.add(now=now)

    def _create_renderers(self, renderers: dict[str, object], model: object) -> None:
        """Fill `renderers` with one renderer per camera, on the thread that will use them."""
        import mujoco  # noqa: PLC0415

        created: dict[str, object] = {}
        for config in self.configs:
            try:
                created[config.name] = mujoco.Renderer(model, config.height, config.width)
            except OSError as exc:
                logger.warning("Camera '{}' renderer unavailable: {}", config.name, exc)
        with self._lock:
            renderers.update(created)

    def _close_renderers(self, renderers: dict[str, object]) -> None:
        with self._lock:
            closing = list(renderers.values())
            renderers.clear()
        for renderer in closing:
            with contextlib.suppress(Exception):
                renderer.close()


__all__ = ["CameraConfig", "CameraService"]
