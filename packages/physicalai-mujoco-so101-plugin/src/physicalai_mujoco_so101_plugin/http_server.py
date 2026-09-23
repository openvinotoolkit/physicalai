# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""HTTP server exposing MuJoCo camera streams and simulation control.

The simulation thread renders camera frames into per-camera
:class:`FrameBuffer` slots and enqueues no work itself; the HTTP thread
reads the latest frame per client and encodes it as JPEG. Streams wait
for new frames on the event loop (:meth:`FrameBuffer.async_waiter`), so a
client costs a task rather than a pooled thread. Control requests (reset,
scene switch, shutdown) are enqueued onto a command queue that the
simulation thread drains, so MuJoCo *stepping* only ever happens on the
simulation thread; the status callback passed to :func:`build_app` runs on
the HTTP thread and is responsible for its own locking.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from loguru import logger

if TYPE_CHECKING:
    import queue
    from collections.abc import AsyncIterator, Callable, Iterator, Mapping

    import numpy as np

_MJPEG_BOUNDARY = "mujoco-frame"
_FRAME_WAIT_TIMEOUT_S = 1.0
_SERVER_START_TIMEOUT_S = 5.0
_SERVER_STOP_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class ResetCommand:
    """Reset/randomize the current scene."""


@dataclass(frozen=True)
class SwitchSceneCommand:
    """Switch to another registered scene."""

    scene_id: str


@dataclass(frozen=True)
class ShutdownCommand:
    """Request a graceful owner shutdown."""


SimCommand = ResetCommand | SwitchSceneCommand | ShutdownCommand


@dataclass(frozen=True)
class FrameSample:
    """One rendered camera frame with sequencing metadata."""

    frame: np.ndarray
    seq: int
    timestamp: float


class FrameBuffer:
    """Thread-safe latest-frame slot shared by the sim and HTTP threads."""

    def __init__(self, name: str) -> None:
        """Initialize an empty frame slot for *name*."""
        self.name = name
        self._cond = threading.Condition()
        self._sample: FrameSample | None = None
        self._seq = 0
        self._async_waiters: set[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = set()

    def put(self, frame: np.ndarray) -> None:
        """Publish a new frame (sim thread)."""
        with self._cond:
            self._seq += 1
            self._sample = FrameSample(frame=frame, seq=self._seq, timestamp=time.time())
            self._cond.notify_all()
            waiters = tuple(self._async_waiters)
        for loop, event in waiters:
            # The loop is gone if the client's task was torn down mid-publish.
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(event.set)

    @contextlib.contextmanager
    def async_waiter(self) -> Iterator[asyncio.Event]:
        """Yield an event set from the sim thread whenever a frame is published.

        Lets an async consumer wait for frames on its own event loop instead of
        parking a worker thread per client in ``wait_newer_than``.

        Yields:
            An :class:`asyncio.Event` signalled on every :meth:`put`.
        """
        entry = (asyncio.get_running_loop(), asyncio.Event())
        with self._cond:
            self._async_waiters.add(entry)
        try:
            yield entry[1]
        finally:
            with self._cond:
                self._async_waiters.discard(entry)

    def snapshot(self) -> FrameSample | None:
        """Return the newest frame without blocking (HTTP thread).

        Returns:
            The newest sample, or ``None`` if no frame was published yet.
        """
        with self._cond:
            return self._sample

    def wait_newer_than(self, seq: int, timeout: float) -> FrameSample | None:
        """Block until a frame newer than *seq* exists or *timeout* elapses.

        On timeout the current (possibly stale) sample is returned so
        stream consumers can re-emit it as a keep-alive.

        Returns:
            The newest sample, or ``None`` when nothing was ever published.
        """
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._sample is None or self._sample.seq <= seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._sample
                self._cond.wait(timeout=remaining)
            return self._sample


def encode_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    """Encode an RGB frame as JPEG bytes.

    Returns:
        The JPEG-encoded byte string.

    Raises:
        RuntimeError: If the encoder fails.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        msg = "JPEG encoding failed"
        raise RuntimeError(msg)
    return buf.tobytes()


async def _await_newer_than(buffer: FrameBuffer, new_frame: asyncio.Event, seq: int) -> FrameSample | None:
    """Wait on the event loop for a frame newer than *seq*.

    On timeout the current (possibly stale) sample is returned so the stream can
    re-emit it as a keep-alive, matching :meth:`FrameBuffer.wait_newer_than`.

    Returns:
        The newest sample, or ``None`` when nothing was ever published.
    """
    new_frame.clear()
    sample = buffer.snapshot()
    if sample is not None and sample.seq > seq:
        return sample
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(new_frame.wait(), _FRAME_WAIT_TIMEOUT_S)
    return buffer.snapshot()


async def _mjpeg_stream(buffer: FrameBuffer, quality: int) -> AsyncIterator[bytes]:
    seq = 0
    while True:
        # The waiter is registered per frame rather than held across the yield:
        # an abandoned generator would otherwise leave it in the buffer, and
        # _await_newer_than re-checks the latest sample before waiting, so a
        # frame published between iterations is never missed.
        with buffer.async_waiter() as new_frame:
            sample = await _await_newer_than(buffer, new_frame, seq)
        if sample is None:
            continue
        seq = sample.seq
        try:
            jpeg = encode_jpeg(sample.frame, quality)
        except (RuntimeError, ValueError) as exc:
            logger.debug("MJPEG encode error for '{}': {}", buffer.name, exc)
            continue
        yield (
            f"--{_MJPEG_BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode()
            + jpeg
            + b"\r\n"
        )


def build_app(
    *,
    service_name: str,
    buffers: Mapping[str, FrameBuffer],
    commands: queue.Queue[SimCommand],
    get_status: Callable[[], dict[str, Any]],
    jpeg_quality: int = 85,
) -> FastAPI:
    """Build the FastAPI application serving camera streams and control.

    Args:
        service_name: Human-readable service identifier reported at ``/``.
        buffers: Latest-frame buffers keyed by camera name (shared with the
            sim thread).
        commands: Queue drained by the sim thread; control endpoints only
            enqueue, never touch MuJoCo state directly.
        get_status: Returns the live status dict: ``connected``, ``scene``,
            ``scenes`` (available ids), and ``cameras`` (per-camera config).
        jpeg_quality: JPEG quality for streams and snapshots.

    Returns:
        The configured FastAPI application.
    """
    app = FastAPI(title=f"{service_name} camera server")

    @app.get("/")
    def root() -> dict[str, Any]:
        status = get_status()
        return {
            "service": service_name,
            "endpoints": {
                "health": "/health",
                "cameras": "/cameras",
                "stream": "/cameras/{name}/mjpeg",
                "snapshot": "/cameras/{name}/frame.jpg",
                "scenes": "/scenes",
                "switch_scene": "POST /scenes/{scene_id}",
                "reset": "POST /reset",
                "shutdown": "POST /shutdown",
            },
            "cameras": [camera["name"] for camera in status["cameras"]],
            "scene": status["scene"],
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        return get_status()

    @app.get("/cameras")
    def cameras() -> list[dict[str, Any]]:
        status = get_status()
        return [
            {
                **camera,
                "stream_url": f"/cameras/{camera['name']}/mjpeg",
                "snapshot_url": f"/cameras/{camera['name']}/frame.jpg",
                "streaming": camera["name"] in buffers,
            }
            for camera in status["cameras"]
        ]

    @app.get("/cameras/{name}/mjpeg")
    def mjpeg(name: str) -> StreamingResponse:
        buffer = buffers.get(name)
        if buffer is None:
            raise HTTPException(status_code=404, detail=f"Unknown camera {name!r}")
        return StreamingResponse(
            _mjpeg_stream(buffer, jpeg_quality),
            media_type=f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY}",
        )

    @app.get("/cameras/{name}/frame.jpg")
    def snapshot(name: str) -> Response:
        buffer = buffers.get(name)
        if buffer is None:
            raise HTTPException(status_code=404, detail=f"Unknown camera {name!r}")
        sample = buffer.snapshot()
        if sample is None:
            raise HTTPException(status_code=503, detail=f"No frame rendered yet for {name!r}")
        return Response(content=encode_jpeg(sample.frame, jpeg_quality), media_type="image/jpeg")

    @app.get("/scenes")
    def scenes() -> dict[str, Any]:
        status = get_status()
        return {"current": status["scene"], "available": status["scenes"]}

    @app.post("/scenes/{scene_id}")
    def switch_scene(scene_id: str) -> dict[str, Any]:
        status = get_status()
        if scene_id not in status["scenes"]:
            raise HTTPException(status_code=404, detail=f"Unknown scene {scene_id!r}")
        commands.put(SwitchSceneCommand(scene_id=scene_id))
        return {"status": "queued", "scene": scene_id}

    @app.post("/reset")
    def reset() -> dict[str, Any]:
        commands.put(ResetCommand())
        return {"status": "queued"}

    @app.post("/shutdown")
    def shutdown() -> dict[str, Any]:
        commands.put(ShutdownCommand())
        return {"status": "shutting_down"}

    return app


class HttpServer:
    """Runs a uvicorn server for a FastAPI app on a daemon thread."""

    def __init__(self, app: FastAPI, host: str, port: int) -> None:
        """Configure a server for *app* on *host*:*port* (not yet started)."""
        self._host = host
        self._port = port
        self._server = uvicorn.Server(
            uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False),
        )
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        """Base URL of the running server."""
        return f"http://{self._host}:{self._port}"

    def start(self) -> None:
        """Start the server thread and wait for the socket to bind.

        Raises:
            RuntimeError: If the port is already in use or the server thread
                exits before becoming ready.
        """
        self._check_port_available()
        self._thread = threading.Thread(target=self._server.run, name="mujoco-http-server", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + _SERVER_START_TIMEOUT_S
        while not self._server.started:
            if not self._thread.is_alive():
                msg = f"HTTP server failed to start on {self._host}:{self._port}"
                raise RuntimeError(msg)
            if time.monotonic() > deadline:
                self.stop()
                msg = f"HTTP server did not become ready within {_SERVER_START_TIMEOUT_S:.1f}s"
                raise RuntimeError(msg)
            time.sleep(0.05)

    def _check_port_available(self) -> None:
        """Fail fast when the port is taken (uvicorn exits its thread otherwise).

        Raises:
            RuntimeError: If the port cannot be bound.
        """
        import socket  # noqa: PLC0415

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((self._host, self._port))
            except OSError as exc:
                msg = f"port {self._port} is not available on {self._host}: {exc}"
                raise RuntimeError(msg) from exc

    def stop(self) -> None:
        """Signal the server to exit and join its thread."""
        self._server.should_exit = True
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=_SERVER_STOP_TIMEOUT_S)
        self._thread = None
