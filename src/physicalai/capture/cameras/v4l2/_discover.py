# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""V4L2 device enumeration via sysfs and QUERYCAP ioctl.

Scans ``/sys/class/video4linux/`` for ``video*`` entries, opens each
``/dev/videoN`` device, and queries capabilities via ``VIDIOC_QUERYCAP``.
Devices that do not support video capture are silently skipped.

Returns an empty list on non-Linux hosts (no sysfs present) and silently
skips devices that cannot be opened due to permission or I/O errors.
"""

from __future__ import annotations

import os
import pathlib

from physicalai.capture.discovery import DeviceInfo

from ._ioctl import (
    V4L2_CAP_VIDEO_CAPTURE,
    VIDIOC_QUERYCAP,
    v4l2_capability,
    xioctl,
)

__all__ = ["discover_v4l2"]

_SYSFS_V4L2 = pathlib.Path("/sys/class/video4linux")


def discover_v4l2() -> list[DeviceInfo]:
    """Enumerate V4L2 capture devices via sysfs and QUERYCAP.

    Scans ``/sys/class/video4linux/`` for ``video*`` directory entries,
    opens the corresponding ``/dev/videoN`` character device, and issues
    a ``VIDIOC_QUERYCAP`` ioctl to retrieve hardware metadata.  Only
    devices that advertise ``V4L2_CAP_VIDEO_CAPTURE`` are included in the
    result.

    Returns:
        Sorted list of :class:`~physicalai.capture.discovery.DeviceInfo`
        objects for every accessible V4L2 capture device found on the
        system.  Returns an empty list on non-Linux hosts or when no
        devices are present.
    """
    if not _SYSFS_V4L2.exists():
        return []

    try:
        entries = sorted(_SYSFS_V4L2.iterdir())
    except FileNotFoundError:
        return []

    devices: list[DeviceInfo] = []

    for entry in entries:
        if not entry.name.startswith("video"):
            continue

        device_path = f"/dev/{entry.name}"

        sysfs_name = ""
        try:
            sysfs_name = (entry / "name").read_text().strip()
        except FileNotFoundError:
            pass
        except OSError:
            pass

        fd = -1
        try:
            fd = os.open(device_path, os.O_RDWR | os.O_NONBLOCK)
            cap = v4l2_capability()
            xioctl(fd, VIDIOC_QUERYCAP, cap)

            if not (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE):
                continue

            card_name = cap.card.decode().rstrip("\x00") or sysfs_name
            devices.append(
                DeviceInfo(
                    device_id=device_path,
                    name=card_name,
                    driver="v4l2",
                    hardware_id=cap.bus_info.decode().rstrip("\x00"),
                    manufacturer="",
                    model=card_name,
                    metadata={"capabilities": cap.capabilities},
                ),
            )
        except PermissionError:
            continue
        except OSError:
            continue
        finally:
            if fd >= 0:
                os.close(fd)

    return devices
