# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Camera capture interfaces.

Public API::

    from physicalai.capture import Camera, ColorMode, Frame
    from physicalai.capture import read_cameras, async_read_cameras, SyncedFrames
    from physicalai.capture import create_camera, discover_all
    from physicalai.capture import DeviceInfo, DepthMixin
    from physicalai.capture import OpenCVCamera  # requires: opencv-python
"""

from physicalai.capture.camera import Camera, ColorMode, Driver
from physicalai.capture.discovery import DeviceInfo, discover_all
from physicalai.capture.errors import (
    CaptureError,
    CaptureTimeoutError,
    MissingDependencyError,
    NotConnectedError,
)
from physicalai.capture.factory import create_camera
from physicalai.capture.frame import Frame
from physicalai.capture.mixins import DepthMixin
from physicalai.capture.multi import SyncedFrames, async_read_cameras, read_cameras

__all__ = [  # noqa: F822, RUF022
    # ABC & types
    "Camera",
    "ColorMode",
    "Driver",
    "Frame",
    "DeviceInfo",
    "SyncedFrames",
    # Mixins
    "DepthMixin",
    # Errors
    "CaptureError",
    "CaptureTimeoutError",
    "MissingDependencyError",
    "NotConnectedError",
    # Functions
    "async_read_cameras",
    "create_camera",
    "discover_all",
    "read_cameras",
    # Concrete cameras (lazy-loaded)
    "OpenCVCamera",
    "RealSenseCamera",
]


def __getattr__(name: str) -> object:
    """Lazy-load concrete camera implementations.

    This avoids pulling in hardware SDKs (e.g. ``opencv-python``,
    ``pyrealsense2``) at package import time.

    Args:
        name: The attribute name being looked up.

    Returns:
        The requested camera class.

    Raises:
        AttributeError: If *name* does not match a known lazy-loaded symbol.
    """
    if name == "OpenCVCamera":
        from physicalai.capture.cameras.opencv import OpenCVCamera  # noqa: PLC0415

        return OpenCVCamera

    if name == "RealSenseCamera":
        from physicalai.capture.cameras.realsense import RealSenseCamera  # noqa: PLC0415

        return RealSenseCamera

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
