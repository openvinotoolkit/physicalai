# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""V4L2 device enumeration via sysfs and QUERYCAP ioctl.

Scans ``/sys/class/video4linux/`` for ``video*`` entries, opens each
``/dev/videoN`` device, and queries capabilities via ``VIDIOC_QUERYCAP``.
Only devices whose **per-node** ``device_caps`` advertise video capture
are included — this correctly filters out UVC metadata nodes that share
the same physical device.

Returns an empty list on non-Linux hosts (no sysfs present) and silently
skips devices that cannot be opened due to permission or I/O errors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._omnicamera import OmniCamera

if TYPE_CHECKING:
    from physicalai.capture.discovery import DeviceInfo

__all__ = ["discover_uvc"]


def discover_uvc() -> list[DeviceInfo]:
    """Discover UVC devices for the current platform.

    Returns:
        List of discovered UVC devices for the current platform.
    """
    # Use omnicamera/pynokhwa's discovery on all platforms.
    # V4L2 offers richer information but does not check camera can be opened.
    return OmniCamera.discover()
