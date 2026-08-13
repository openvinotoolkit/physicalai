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
# real, working backstop against a stalled socket, unlike
# CAP_PROP_READ_TIMEOUT_MSEC (see read()/_wait_first_frame()). Guarded by a
# lock since this is process-global state and each connect() computes its
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
    ``https://`` sources — treat the camera and network as trusted. A single
    read can occasionally take longer than requested (e.g. waiting for the
    next RTSP keyframe); that's normal RTSP behavior, not a bug here.
    """

    # Sleep between failed read attempts in the retry loops below. Matches
    # the polling interval used by the UVC backend (_omnicamera.py); avoids
    # pinning a CPU core in a busy-spin if the stream drops and cap.read()
    # starts returning immediately instead of blocking on network I/O.
    _POLL_INTERVAL_S = 0.001

    # read_latest() must not block for a control-loop tick behind a stalled
    # stream, so it uses a minimal read timeout instead of whatever timeout
    # a prior read()/connect() call left configured on the capture object.
    _LATEST_READ_TIMEOUT_MS = 1

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
        self._sequence = 0
        self._last_timestamp: float = 0.0
        self._cap: cv2.VideoCapture | None = None
        self._last_frame_data: np.ndarray | None = None

    def connect(self, timeout: float = 5.0) -> None:
        # One deadline for the whole call: CAP_PROP_OPEN_TIMEOUT_MSEC bounds
        # only the constructor's handshake, so the first-frame wait below
        # must consume whatever time is left, not a fresh `timeout` budget on
        # top of it. Otherwise connect() could take up to ~2x timeout,
        # violating the base class's "timeout bounds the whole call" contract.
        deadline = time.monotonic() + timeout
        timeout_ms = int(timeout * 1000)
        is_rtsp = self._url.lower().startswith("rtsp://")
        try:
            # Only CAP_PROP_OPEN_TIMEOUT_MSEC goes here — this build rejects
            # the whole params array if CAP_PROP_READ_TIMEOUT_MSEC/BUFFERSIZE
            # are also included (verified against real opencv-python-headless,
            # not reachable via the mocked tests). Both are applied via
            # cap.set() below instead, after a successful open.
            with _RTSP_TRANSPORT_LOCK if is_rtsp else contextlib.nullcontext():
                previous_options = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS") if is_rtsp else None
                if is_rtsp:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _rtsp_capture_options(timeout)
                try:
                    cap = cv2.VideoCapture(
                        self._url,
                        cv2.CAP_FFMPEG,
                        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms],
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
        except Exception as err:
            msg = f"Failed to open IP camera stream {_redact_url(self._url)}"
            raise CaptureError(msg) from err

        if not cap.isOpened():
            cap.release()
            msg = f"Failed to open IP camera stream {_redact_url(self._url)}"
            raise CaptureError(msg)

        # Best-effort: not honored by every stream/build, so failures are
        # silently ignored (cap.set() returns False rather than raising).
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, self._fps)

        self._cap = cap
        self._wait_first_frame(timeout, deadline, cap)

    def _wait_first_frame(self, timeout: float, deadline: float, cap: cv2.VideoCapture) -> None:
        start = deadline - timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                # Best-effort re-clamp per iteration; not honored on every
                # build (verified: cap.set() can return False even on a live
                # capture), so a single cap.read() can still exceed
                # `remaining`. connect()'s `stimeout` is the real backstop
                # against a stalled socket — this doesn't bound normal RTSP
                # keyframe-wait latency, which is expected, not a bug.
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, remaining * 1000)
                ok, frame = cap.read()
                if ok and frame is not None:
                    data = self._convert(frame)
                    self._height, self._width = data.shape[:2]
                    self._last_frame_data = data
                    self._last_timestamp = time.monotonic()
                    self._connected = True
                    self._sequence = 0
                    logger.info(f"IP camera {_redact_url(self._url)} connected ({self._width}x{self._height})")
                    return
                time.sleep(self._POLL_INTERVAL_S)
        except Exception as err:
            # Any unexpected failure while reading/converting the first frame
            # (e.g. an unusual channel layout) must still release the
            # VideoCapture handle — self._cap was already assigned by
            # connect() before this call, so leaving it open here would leak
            # the handle and orphan it on a subsequent connect() retry.
            self._do_disconnect()
            msg = f"Failed to read first frame from IP camera stream {_redact_url(self._url)}"
            raise CaptureError(msg) from err

        elapsed = time.monotonic() - start
        self._do_disconnect()
        msg = (
            f"Timed out waiting for first frame from {_redact_url(self._url)} "
            f"after {elapsed:.1f}s (requested timeout: {timeout}s)"
        )
        raise CaptureTimeoutError(msg)

    def _ensure_connected(self) -> cv2.VideoCapture:
        cap = self._cap
        if not self._connected or cap is None:
            raise NotConnectedError
        return cap

    def _do_disconnect(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                logger.debug(f"Error releasing IP camera stream {_redact_url(self._url)}")
        self._cap = None
        self._last_frame_data = None
        self._connected = False

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
        return self._connected and self._cap is not None

    @property
    def device_id(self) -> str:
        return f"ip:{_redact_url(self._url)}"

    def read(self, timeout: float = 2.0) -> Frame:
        cap = self._ensure_connected()

        start = time.monotonic()
        deadline = start + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # Best-effort; see the identical clamp in _wait_first_frame() for
            # why a single cap.read() isn't actually guaranteed to respect this.
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, remaining * 1000)
            ok, frame = cap.read()
            if ok and frame is not None:
                data = self._convert(frame)
                self._last_frame_data = data
                self._sequence += 1
                self._last_timestamp = time.monotonic()
                return Frame(data=data, timestamp=self._last_timestamp, sequence=self._sequence)
            time.sleep(self._POLL_INTERVAL_S)

        elapsed = time.monotonic() - start
        msg = f"Timed out waiting for frame after {elapsed:.1f}s (requested timeout: {timeout}s)"
        raise CaptureTimeoutError(msg)

    def read_latest(self) -> Frame:
        cap = self._ensure_connected()

        # Best-effort: bound this read to a minimal timeout so a stalled
        # stream falls back to the cached frame quickly instead of blocking
        # for whatever timeout a prior read()/connect() call configured.
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._LATEST_READ_TIMEOUT_MS)

        ok, frame = cap.read()
        if ok and frame is not None:
            data = self._convert(frame)
            self._last_frame_data = data
            self._sequence += 1
            self._last_timestamp = time.monotonic()
            return Frame(data=data, timestamp=self._last_timestamp, sequence=self._sequence)

        if self._last_frame_data is not None:
            return Frame(data=self._last_frame_data, timestamp=self._last_timestamp, sequence=self._sequence)
        msg = "No frame available"
        raise CaptureError(msg)
