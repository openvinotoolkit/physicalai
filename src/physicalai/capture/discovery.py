# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Device discovery types and utilities."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeviceInfo:
    """Metadata about a discovered camera device.

    Returned by :meth:`~physicalai.capture.camera.Camera.discover` and
    :func:`discover_all`.

    Attributes:
        device_id: Backend-specific identifier (e.g. ``"/dev/video0"``,
            index, IP address).
        name: Human-readable name (``"Logitech C920"``, ``"D435"``).
        driver: Backend that found the device: ``"opencv"``,
            ``"realsense"``, ``"basler"``, ``"genicam"``.
        hardware_id: Stable cross-backend identifier such as a serial
            number or USB bus path.  Enables deduplication when the same
            physical device is discovered by multiple backends.
        manufacturer: Device manufacturer (``"Intel"``, ``"Basler"``).
        model: Device model (``"D435"``, ``"acA1920-40gc"``).
        metadata: Backend-specific extras.
    """

    device_id: str
    name: str = ""
    driver: str = ""
    hardware_id: str = ""
    manufacturer: str = ""
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def discover_all() -> dict[str, list[DeviceInfo]]:
    """Discover available cameras across all supported backends.

    Each backend is tried independently; failures are silently skipped
    so that a missing SDK does not prevent discovery of other camera
    types.

    Returns:
        Dict mapping driver name to list of discovered devices.
        Backends that are not installed or find no devices return
        an empty list.
    """
    results: dict[str, list[DeviceInfo]] = {}

    with contextlib.suppress(Exception):
        from physicalai.capture.cameras.v4l2 import discover_v4l2  # noqa: PLC0415

        results["v4l2"] = discover_v4l2()

    with contextlib.suppress(Exception):
        from physicalai.capture.cameras.opencv import OpenCVCamera  # noqa: PLC0415

        results["opencv"] = OpenCVCamera.discover()

    with contextlib.suppress(Exception):
        from physicalai.capture.cameras.realsense import RealSenseCamera  # noqa: PLC0415

        results["realsense"] = RealSenseCamera.discover()

    with contextlib.suppress(Exception):
        from physicalai.capture.cameras.basler import BaslerCamera  # noqa: PLC0415

        results["basler"] = BaslerCamera.discover()

    with contextlib.suppress(Exception):
        from physicalai.capture.cameras.genicam import GenicamCamera  # noqa: PLC0415

        results["genicam"] = GenicamCamera.discover()

    with contextlib.suppress(Exception):
        from physicalai.capture.cameras.ip import IPCamera  # noqa: PLC0415

        results["ip"] = IPCamera.discover()

    return results
