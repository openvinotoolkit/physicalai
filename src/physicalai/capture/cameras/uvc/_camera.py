# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""UVC camera facade.

This module exposes :class:`~physicalai.capture.cameras.uvc.UVCCamera` as the
user-facing entry point for "standard USB video cameras" (UVC devices).

Internally it delegates to one of:
  - :class:`~physicalai.capture.cameras.uvc._v4l2.V4L2Camera` on Linux
  - :class:`~physicalai.capture.cameras.uvc._omnicamera.OmniCameraBackend` elsewhere
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from physicalai.capture.camera import Camera, ColorMode

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame


class UVCCamera(Camera):
    """Camera facade for UVC devices (USB Video Class).

    Args:
        device: Unified device selector.
            - On Linux (V4L2): ``0`` maps to ``/dev/video0``.
            - On macOS/Windows (OmniCamera): ``0`` maps to OmniCamera index ``0``.
        width: Requested frame width in pixels.
        height: Requested frame height in pixels.
        fps: Requested frames per second.
        color_mode: Pixel format for returned frames.
        backend: ``"auto"``, ``"v4l2"``, or ``"omnicamera"``.
        backend_options: Backend-specific overrides forwarded to the selected
            backend constructor.
    """

    def __init__(
        self,
        *,
        device: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        color_mode: ColorMode = ColorMode.RGB,
        backend: str = "auto",
        backend_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(color_mode=color_mode)

        backend = backend.lower()
        if backend == "auto":
            backend = "v4l2" if sys.platform == "linux" else "omnicamera"

        opts = dict(backend_options or {})

        if backend == "v4l2":
            from ._v4l2 import V4L2Camera  # noqa: PLC0415

            device_path: str
            device_path = f"/dev/video{device}" if isinstance(device, int) else device

            # Forward V4L2-specific overrides (e.g. num_buffers, pixel_format).
            # The facade's ``device`` maps to V4L2's ``device_path``.
            opts.setdefault("device_path", device_path)
            self._inner: Camera = V4L2Camera(
                width=width,
                height=height,
                fps=fps,
                color_mode=color_mode,
                **opts,
            )
        elif backend == "omnicamera":
            from ._omnicamera import OmniCameraBackend  # noqa: PLC0415

            # Forward OmniCamera-specific overrides while mapping facade
            # ``device`` to OmniCameraBackend.device_id.
            opts.setdefault("device_id", device)
            self._inner = OmniCameraBackend(
                width=width,
                height=height,
                fps=fps,
                color_mode=color_mode,
                **opts,
            )
        elif backend == "opencv":
            msg = "The 'opencv' backend has been removed. Use backend='omnicamera' or backend='auto' instead."
            raise ValueError(msg)
        else:
            msg = f"Unknown backend {backend!r}. Use 'auto', 'v4l2', or 'omnicamera'."
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self, timeout: float = 5.0) -> None:
        self._inner.connect(timeout=timeout)

    def _do_disconnect(self) -> None:
        # Ensure we release the underlying hardware resources.
        self._inner.disconnect()

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read(self, timeout: float | None = None) -> Frame:
        return self._inner.read(timeout=timeout)

    def read_latest(self) -> Frame:
        return self._inner.read_latest()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._inner.is_connected

    @property
    def device_id(self) -> str:
        return self._inner.device_id

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @classmethod
    def discover(cls) -> list[Any]:  # pragma: no cover - wrapper uses typed DeviceInfo
        from ._discover import discover_uvc  # noqa: PLC0415

        return discover_uvc()


__all__ = ["UVCCamera"]
