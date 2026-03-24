# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory convenience function for config-driven camera creation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.capture.camera import Driver

if TYPE_CHECKING:
    from physicalai.capture.camera import Camera


def create_camera(driver: str, **kwargs: object) -> Camera:
    """Create a camera by driver name.

    Convenience function for config-driven instantiation.  Prefer
    dedicated camera classes for direct usage.

    Args:
        driver: Camera type — one of ``"opencv"``, ``"realsense"``,
            ``"basler"``, ``"genicam"``, ``"ip"``.  Case-insensitive.
        **kwargs: Forwarded to the camera constructor.

    Returns:
        A new camera instance.

    Raises:
        ValueError: If *driver* is not a recognised name.
    """
    # Import lazily to avoid pulling in optional dependencies at import
    # time.  Each camera module handles its own MissingDependencyError.
    driver = driver.lower()

    if driver == Driver.OPENCV:
        from physicalai.capture.cameras.opencv import OpenCVCamera  # noqa: PLC0415

        return OpenCVCamera(**kwargs)

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
