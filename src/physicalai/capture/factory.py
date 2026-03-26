# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory convenience function for config-driven camera creation."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

from physicalai.capture.camera import ColorMode, Driver

if TYPE_CHECKING:
    from physicalai.capture.camera import Camera


class _V4L2CameraArgs(TypedDict):
    device_path: str
    width: int
    height: int
    fps: int
    num_buffers: int
    pixel_format: str
    color_mode: ColorMode


class _OpenCVCameraArgs(TypedDict):
    device_id: int | str
    width: int
    height: int
    fps: int
    color_mode: ColorMode


def create_camera(driver: str, **kwargs: object) -> Camera:
    """Create a camera by driver name.

    Convenience function for config-driven instantiation.  Prefer
    dedicated camera classes for direct usage.

    Args:
        driver: Camera type — one of ``"opencv"``, ``"v4l2"``,
            ``"realsense"``, ``"basler"``, ``"genicam"``, ``"ip"``.
            Case-insensitive.
        **kwargs: Forwarded to the camera constructor.

    Returns:
        A new camera instance.

    Raises:
        ValueError: If *driver* is not a recognised name.
    """
    # Import lazily to avoid pulling in optional dependencies at import
    # time.  Each camera module handles its own MissingDependencyError.
    driver = driver.lower()

    if driver == Driver.V4L2:
        from physicalai.capture.cameras.v4l2 import V4L2Camera  # noqa: PLC0415

        v4l2_args: _V4L2CameraArgs = {
            "device_path": cast("str", kwargs.get("device_path", "/dev/video0")),
            "width": cast("int", kwargs.get("width", 640)),
            "height": cast("int", kwargs.get("height", 480)),
            "fps": cast("int", kwargs.get("fps", 30)),
            "num_buffers": cast("int", kwargs.get("num_buffers", 4)),
            "pixel_format": cast("str", kwargs.get("pixel_format", "mjpeg")),
            "color_mode": cast("ColorMode", kwargs.get("color_mode", ColorMode.RGB)),
        }
        return V4L2Camera(**v4l2_args)

    if driver == Driver.OPENCV:
        from physicalai.capture.cameras.opencv import OpenCVCamera  # noqa: PLC0415

        opencv_args: _OpenCVCameraArgs = {
            "device_id": cast("int | str", kwargs.get("device_id", 0)),
            "width": cast("int", kwargs.get("width", 640)),
            "height": cast("int", kwargs.get("height", 480)),
            "fps": cast("int", kwargs.get("fps", 30)),
            "color_mode": cast("ColorMode", kwargs.get("color_mode", ColorMode.RGB)),
        }
        return OpenCVCamera(**opencv_args)

    if driver == Driver.REALSENSE:
        from physicalai.capture.cameras.realsense import RealSenseCamera  # noqa: PLC0415

        return RealSenseCamera(**kwargs)

    if driver == Driver.BASLER:
        from physicalai.capture.cameras.basler import BaslerCamera  # noqa: PLC0415

        return BaslerCamera(**kwargs)

    if driver == Driver.GENICAM:
        from physicalai.capture.cameras.genicam import GenicamCamera  # noqa: PLC0415

        return GenicamCamera(**kwargs)

    if driver == Driver.IP:
        from physicalai.capture.cameras.ip import IPCamera  # noqa: PLC0415

        return IPCamera(**kwargs)

    msg = f"Unknown camera driver {driver!r}. Expected one of: {', '.join(Driver)}"
    raise ValueError(msg)
