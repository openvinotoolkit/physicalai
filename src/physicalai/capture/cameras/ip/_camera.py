# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import os
import threading
import time
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

import cv2
from loguru import logger

from physicalai.capture.camera import Camera, ColorMode
from physicalai.capture.errors import CaptureError, CaptureTimeoutError, NotConnectedError
from physicalai.capture.frame import Frame
from physicalai.config import export_config

if TYPE_CHECKING:
    import numpy as np

# OpenCV's Python bindings expose no CAP_PROP for RTSP transport selection,
# so this FFmpeg-backend env var (read fresh on each VideoCapture open) is
# the only portable way to force TCP: FFmpeg's UDP default is often silently
# dropped by routers/NAT, leaving the RTSP session "open" but frame-less
# forever. `stimeout` (microseconds) is a session-wide socket timeout — a
# socket-level backstop in addition to the open-only FFmpeg read timeout.
# Guarded by a lock since this is process-global state and each connect() computes its
# own stimeout value from its own `timeout`.
_RTSP_TRANSPORT_LOCK = threading.Lock()


def _rtsp_capture_options(timeout: float) -> str:
    """Build the OPENCV_FFMPEG_CAPTURE_OPTIONS value for an RTSP connect().

    Returns:
        A pipe-separated ``key;value`` option string for FFmpeg's rtsp demuxer.
    """
    stimeout_us = int(timeout * 1_000_000)
    return f"rtsp_transport;tcp|stimeout;{stimeout_us}"


def _redact_url(url: str) -> str:
    """Strip embedded credentials from a stream URL for safe logging/identity.

    ``rtsp://user:pass@host/...`` style URLs are common for IP cameras. The
    credentials must never appear in logs, error messages, or ``device_id``
    (which may be persisted in config or shared-camera service names).

    Returns:
        The URL with any userinfo replaced by ``***``, or unchanged if the
        URL has no credentials.
    """
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    host = parts.netloc.rsplit("@", 1)[-1]
    redacted_netloc = f"***@{host}"
    return urlunsplit((parts.scheme, redacted_netloc, parts.path, parts.query, parts.fragment))


@export_config(class_path="physicalai.capture.IPCamera")
class IPCamera(Camera):
    """Network camera (RTSP/HTTP) read via OpenCV's ``VideoCapture`` (FFmpeg backend).

    No camera controls, no discovery (:meth:`discover` returns ``[]``).
    ``connect()`` forces TCP transport for ``rtsp://`` URLs — plain UDP RTP is
    often silently dropped by routers/NAT. Credentials in the URL are
    redacted from :attr:`device_id`/logs, but TLS isn't verified for
    ``https://`` sources — treat the camera and network as trusted.

    A single reader continuously drains the stream into a latest-frame slot,
    independently of consumer/publisher pacing. Both read methods can skip
    frames; this backend is for live control, not lossless recording.
    Timestamps mark local decode/conversion completion, NOT source capture.
    Sequences count locally decoded frames, not source exposures. Upstream
    latency and sources faster than this host can decode remain unobservable.
    """

    # Sleep between failed read attempts in the retry loops below. Matches
    # the polling interval used by the UVC backend (_omnicamera.py); avoids
    # pinning a CPU core in a busy-spin if the stream drops and cap.read()
    # starts returning immediately instead of blocking on network I/O.
    _POLL_INTERVAL_S = 0.001

    # FFmpeg accepts this only at open time; cap.set() silently fails.
    # Bound native I/O so disconnect can join before releasing the handle.
    _IO_TIMEOUT_S = 1.0

    def __init__(
        self,
        *,
        url: str,
        fps: int = 30,
        width: int | None = None,
        height: int | None = None,
        color_mode: ColorMode = ColorMode.RGB,
    ) -> None:
        super().__init__(color_mode=color_mode)
        self._url = url
        self._fps = fps
        # User-requested dimensions (None = whatever the stream delivers for
        # that axis; the two are resolved independently, like BaslerCamera).
        self._requested_width = width
        self._requested_height = height
        self._width: int = 0
        self._height: int = 0
        self._connected = False
        self._first_timestamp: float | None = None
        self._cap: cv2.VideoCapture | None = None
        self._latest_frame: Frame | None = None
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None
        self._error: CaptureError | None = None
        self._read_sequence = 0

    def connect(self, timeout: float = 5.0) -> None:
        if self.is_connected:
            return
        self._do_disconnect()
        deadline = time.monotonic() + timeout
        timeout_ms = int(timeout * 1000)
        is_rtsp = self._url.lower().startswith("rtsp://")
        try:
            # BUFFERSIZE is best-effort below, not a supported FFmpeg open
            # parameter. READ_TIMEOUT_MSEC, unlike BUFFERSIZE, is open-only.
            with _RTSP_TRANSPORT_LOCK if is_rtsp else contextlib.nullcontext():
                previous_options = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS") if is_rtsp else None
                if is_rtsp:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _rtsp_capture_options(timeout)
                try:
                    cap = cv2.VideoCapture(
                        self._url,
                        cv2.CAP_FFMPEG,
                        [
                            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                            timeout_ms,
                            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                            max(1, min(timeout_ms, int(self._IO_TIMEOUT_S * 1000))),
                        ],
                    )
                finally:
                    # FFmpeg only reads this env var once, at open time (a
                    # capture already past isOpened() doesn't need it) —
                    # verified against a real stream. Restore right away so
                    # it can't leak into an unrelated cv2.VideoCapture() call
                    # elsewhere in the process after this one returns.
                    if is_rtsp:
                        if previous_options is None:
                            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
                        else:
                            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = previous_options
        except Exception:  # noqa: BLE001
            # Native exception text can contain the authenticated URL.
            msg = f"Failed to open IP camera stream {_redact_url(self._url)}"
            raise CaptureError(msg) from None

        if not cap.isOpened():
            with contextlib.suppress(Exception):
                cap.release()
            msg = f"Failed to open IP camera stream {_redact_url(self._url)}"
            raise CaptureError(msg)

        # Best-effort: not honored by every stream/build, so failures are
        # silently ignored (cap.set() returns False rather than raising).
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, self._fps)

        self._cap = cap
        self._stop.clear()
        self._error = None
        self._first_timestamp = None
        self._read_sequence = 0
        self._reader = threading.Thread(target=self._capture_loop, args=(cap,), name="ip-camera-reader", daemon=True)
        try:
            self._reader.start()
            self._wait_first_frame(deadline)
            self._connected = True
        except CaptureError:
            self._do_disconnect()
            raise
        except RuntimeError:
            self._do_disconnect()
            msg = "Failed to start IP camera reader"
            raise CaptureError(msg) from None
        logger.info(f"IP camera {_redact_url(self._url)} connected ({self._width}x{self._height})")

    def _wait_first_frame(self, deadline: float) -> None:
        with self._condition:
            while True:
                if self._error is not None:
                    raise self._error
                if self._first_timestamp is not None and self._first_timestamp <= deadline:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    msg = f"Timed out waiting for first frame from {_redact_url(self._url)}"
                    raise CaptureTimeoutError(msg)
                self._condition.wait(remaining)

    def _capture_loop(self, cap: cv2.VideoCapture) -> None:
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    self._stop.wait(self._POLL_INTERVAL_S)
                    continue
                data = self._convert(frame)
                with self._condition:
                    self._height, self._width = data.shape[:2]
                    sequence = 0 if self._latest_frame is None else self._latest_frame.sequence + 1
                    self._latest_frame = Frame(data=data, timestamp=time.monotonic(), sequence=sequence)
                    if self._first_timestamp is None:
                        self._first_timestamp = self._latest_frame.timestamp
                    self._condition.notify_all()
        except Exception:  # noqa: BLE001
            # Native errors can contain the authenticated URL; do not relay them.
            with self._condition:
                self._error = CaptureError("IP camera stream acquisition failed")
                self._condition.notify_all()
        finally:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                logger.debug("Error releasing IP camera stream")

    def _ensure_connected(self) -> cv2.VideoCapture:
        cap = self._cap
        if not self._connected or cap is None:
            raise NotConnectedError
        return cap

    def _do_disconnect(self) -> None:
        with self._condition:
            self._connected = False
            self._stop.set()
            self._condition.notify_all()
        if self._reader is not None and self._reader.ident is not None:
            self._reader.join(timeout=self._IO_TIMEOUT_S + 1.0)
            if self._reader.is_alive():
                msg = "IP camera backend did not stop within its I/O timeout"
                raise CaptureError(msg)
        elif self._cap is not None:
            with contextlib.suppress(Exception):
                self._cap.release()  # reader never started, so no native read can be in flight
        self._reader = None
        self._cap = None
        self._latest_frame = None
        self._error = None

    def _convert(self, frame: np.ndarray) -> np.ndarray:
        """Convert a raw BGR frame (as decoded by OpenCV) to the requested color mode and size.

        Width and height are resolved independently (like ``BaslerCamera``):
        requesting only one of the two resizes just that axis and leaves the
        other at whatever the stream natively delivers, rather than silently
        ignoring both.

        Returns:
            Converted, possibly resized, image data.
        """
        if self._color_mode == ColorMode.RGB:
            data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self._color_mode == ColorMode.GRAY:
            data = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            data = frame

        native_h, native_w = data.shape[:2]
        target_w = native_w if self._requested_width is None else self._requested_width
        target_h = native_h if self._requested_height is None else self._requested_height
        if (target_w, target_h) != (native_w, native_h):
            return cv2.resize(data, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return data

    @property
    def is_connected(self) -> bool:
        return self._connected and self._cap is not None and self._error is None

    @property
    def device_id(self) -> str:
        return f"ip:{_redact_url(self._url)}"

    def read(self, timeout: float = 2.0) -> Frame:
        """Wait for a newer decoded frame, skipping overwritten frames.

        The timeout bounds the consumer wait, not native network I/O.

        Returns:
            The newest frame decoded since the previous consumer read.

        Raises:
            CaptureTimeoutError: No newer frame arrived within timeout.
        """
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                self._ensure_connected()
                if self._error is not None:
                    raise self._error
                if self._latest_frame is not None and self._latest_frame.sequence > self._read_sequence:
                    return self.read_latest()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    msg = "Timed out waiting for IP camera frame"
                    raise CaptureTimeoutError(msg)
                self._condition.wait(remaining)

    def read_latest(self) -> Frame:
        """Return the latest decoded frame without performing network I/O.

        A stalled source returns the same sequence and decode timestamp.

        Returns:
            The most recently decoded frame, including its original metadata.

        Raises:
            CaptureError: Acquisition failed or no frame is available.
        """
        with self._condition:
            self._ensure_connected()
            if self._error is not None:
                raise self._error
            if self._latest_frame is None:
                msg = "No frame available"
                raise CaptureError(msg)
            self._read_sequence = self._latest_frame.sequence
            return self._latest_frame
