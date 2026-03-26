# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenCV camera backend."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from physicalai.capture.camera import Camera, ColorMode
from physicalai.capture.errors import CaptureError, CaptureTimeoutError, MissingDependencyError, NotConnectedError
from physicalai.capture.frame import Frame

if TYPE_CHECKING:
    import numpy as np

    from physicalai.capture.discovery import DeviceInfo

_MISSING_DEP_PKG = "opencv-python"
_MISSING_DEP_EXTRA = "opencv"

try:
    import cv2
except ImportError as err:
    raise MissingDependencyError(_MISSING_DEP_PKG, _MISSING_DEP_EXTRA) from err


class OpenCVCamera(Camera):
    """OpenCV-based camera backend for dev/testing use.

    Uses ``cv2.VideoCapture`` for frame acquisition. Suitable for macOS/Windows
    development. On Linux, prefer V4L2Camera for production use.

    Args:
        device_id: Camera index or device path (default ``0``).
        width: Requested frame width in pixels.
        height: Requested frame height in pixels.
        fps: Requested frames per second.
        color_mode: Pixel format for image reads.

    Raises:
        MissingDependencyError: If opencv-python is not installed.
        CaptureError: If the camera fails to open.
        CaptureTimeoutError: If no frame arrives within timeout.
    """

    def __init__(
        self,
        *,
        device_id: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        color_mode: ColorMode = ColorMode.RGB,
    ) -> None:
        """Initialise OpenCVCamera with capture parameters.

        Args:
            device_id: Camera index or device path.
            width: Requested frame width.
            height: Requested frame height.
            fps: Requested frames per second.
            color_mode: Pixel format for image reads.
        """
        super().__init__(color_mode=color_mode)
        self._device_id_raw = device_id
        self._width = width
        self._height = height
        self._fps = fps
        self._connected: bool = False
        self._sequence: int = 0
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self, timeout: float = 5.0) -> None:
        """Open the camera and wait for the first frame.

        Args:
            timeout: Maximum seconds to wait for first frame.

        Raises:
            CaptureError: If camera fails to open.
            CaptureTimeoutError: If no frame arrives within timeout.
        """
        self._cap = cv2.VideoCapture(self._device_id_raw)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            msg = f"Failed to open camera: {self._device_id_raw!r}"
            raise CaptureError(msg)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ret, _ = self._cap.read()
            if ret:
                self._connected = True
                self._sequence = 0
                return
        msg = f"No frame from camera {self._device_id_raw!r} within {timeout}s"
        raise CaptureTimeoutError(msg)

    def _do_disconnect(self) -> None:
        """Release ``cv2.VideoCapture`` and reset state."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Whether the camera is currently open."""
        return self._connected

    @property
    def device_id(self) -> str:
        """Camera index or path as a string."""
        return str(self._device_id_raw)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def _convert_color(self, frame: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
        """Convert a BGR frame from cv2 to the requested color mode.

        Args:
            frame: Raw BGR image array from ``cv2.VideoCapture.read()``.

        Returns:
            Converted image array in the configured color mode.
        """
        if self._color_mode == ColorMode.RGB:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore[return-value]
        if self._color_mode == ColorMode.GRAY:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # type: ignore[return-value]
        return frame  # BGR - no conversion

    def read(self, timeout: float | None = None) -> Frame:
        """Read the next frame from the camera.

        Args:
            timeout: Maximum seconds to wait. ``None`` waits indefinitely.

        Returns:
            The captured frame.

        Raises:
            NotConnectedError: If not connected.
            CaptureTimeoutError: If no frame arrives within *timeout*.
        """
        if not self._connected or self._cap is None:
            msg = "Cannot read: camera is not connected. Call connect() first."
            raise NotConnectedError(msg)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            ret, raw = self._cap.read()
            if ret:
                converted = self._convert_color(raw)
                result = Frame(data=converted, timestamp=time.monotonic(), sequence=self._sequence)
                self._sequence += 1
                return result
            if deadline is not None and time.monotonic() >= deadline:
                msg = f"No frame from camera {self._device_id_raw!r} within {timeout}s"
                raise CaptureTimeoutError(msg)

    def read_latest(self) -> Frame:
        """Read the most recent frame (non-blocking, drains buffer).

        Returns:
            The latest captured frame.

        Raises:
            NotConnectedError: If not connected.
            CaptureError: If frame acquisition fails.
        """
        if not self._connected or self._cap is None:
            msg = "Cannot read_latest: camera is not connected. Call connect() first."
            raise NotConnectedError(msg)
        ret, raw = self._cap.read()
        if not ret:
            msg = f"Failed to read frame from camera {self._device_id_raw!r}"
            raise CaptureError(msg)
        converted = self._convert_color(raw)
        result = Frame(data=converted, timestamp=time.monotonic(), sequence=self._sequence)
        self._sequence += 1
        return result

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @classmethod
    def discover(cls) -> list[DeviceInfo]:
        """Discover available cameras using cv2_enumerate_cameras or index probing.

        Tries ``cv2_enumerate_cameras`` first for richer metadata; falls back
        to probing indices 0-9 when that package is absent.

        Returns:
            List of discovered camera devices.
        """
        from physicalai.capture.discovery import DeviceInfo  # noqa: PLC0415

        devices: list[DeviceInfo] = []
        try:
            from cv2_enumerate_cameras import enumerate_cameras  # noqa: PLC0415

            devices.extend(
                DeviceInfo(
                    device_id=str(cam.index),
                    name=cam.name,
                    driver="opencv",
                    hardware_id="",
                    manufacturer="",
                    model=cam.name,
                    metadata={"backend": getattr(cam, "backend", "")},
                )
                for cam in enumerate_cameras()
            )
        except ImportError:
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    devices.append(
                        DeviceInfo(
                            device_id=str(i),
                            name=f"Camera {i}",
                            driver="opencv",
                            hardware_id="",
                            manufacturer="",
                            model=f"Camera {i}",
                            metadata={},
                        ),
                    )
                    cap.release()
        return devices
