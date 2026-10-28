# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""V4L2 queries; explicit stream tests can also restore a saved wire format."""

from __future__ import annotations

import errno
import fcntl
import os
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

from physicalai.capture.cameras.uvc.v4l2._ioctl import (  # noqa: PLC2701
    V4L2_CAP_DEVICE_CAPS,
    V4L2_CAP_VIDEO_CAPTURE,
    VIDIOC_ENUM_FMT,
    VIDIOC_G_FMT,
    VIDIOC_G_PARM,
    VIDIOC_QUERYCAP,
    VIDIOC_S_FMT,
    VIDIOC_S_PARM,
    v4l2_capability,
    v4l2_fmtdesc,
    v4l2_format,
    v4l2_streamparm,
    xioctl,
)
from physicalai.diagnostics.models import VideoMode

if TYPE_CHECKING:
    from collections.abc import Iterator

    from physicalai.diagnostics.models import Endpoint

_ENUM_FRAME_SIZES = 74
_ENUM_FRAME_INTERVALS = 75
_DISCRETE = 1
_ENUM_LIMIT = 128
_MODE_LIMIT = 512


@dataclass
class VideoState:
    """Saved wire format/interval, used only inside explicitly requested tests."""

    image_format: v4l2_format
    parameters: v4l2_streamparm

    @classmethod
    def read(cls, path: str) -> VideoState:
        """Query format and interval without allocating buffers or streaming.

        Returns:
            A saved native state suitable for a later explicit restoration.
        """
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        try:
            image_format = v4l2_format()
            image_format.type = 1
            xioctl(fd, VIDIOC_G_FMT, image_format)
            parameters = v4l2_streamparm()
            parameters.type = 1
            xioctl(fd, VIDIOC_G_PARM, parameters)
            return cls(image_format, parameters)
        finally:
            os.close(fd)

    def restore(self, path: str) -> None:
        """Actively restore a format after an operator-approved streaming test.

        Passive inventory/discovery/watch must never call this method. If another
        application has started streaming, the driver can refuse restoration.
        """
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        try:
            xioctl(fd, VIDIOC_S_FMT, self.image_format)
            xioctl(fd, VIDIOC_S_PARM, self.parameters)
        finally:
            os.close(fd)

    def describe(self) -> dict[str, int | float | str | None]:
        """Return the reported wire format as serializable evidence."""
        rate = self.parameters.parm.capture.timeperframe
        return {
            "width": self.image_format.fmt.pix.width,
            "height": self.image_format.fmt.pix.height,
            "pixel_format": struct
            .pack("<I", self.image_format.fmt.pix.pixelformat)
            .decode("ascii", errors="replace")
            .strip(),
            "fps": rate.denominator / rate.numerator if rate.numerator else None,
        }


def _enumerate(fd: int, number: int, values: tuple[int, ...], count: int) -> tuple[int, ...] | None:
    # These UAPI records contain only uint32 fields, including their unions.
    buffer = bytearray(struct.pack(f"={count}I", *values, *([0] * (count - len(values)))))
    request = (3 << 30) | (len(buffer) << 16) | (ord("V") << 8) | number
    try:
        fcntl.ioctl(fd, request, buffer)
    except OSError as exc:
        if exc.errno == errno.EINVAL:
            return None
        raise
    return struct.unpack(f"={count}I", buffer)


def _format_modes(fd: int, pixel_format: int, endpoint: Endpoint) -> Iterator[VideoMode]:
    fourcc = struct.pack("<I", pixel_format).decode("ascii", errors="replace").strip()
    for size_index in range(_ENUM_LIMIT):
        size = _enumerate(fd, _ENUM_FRAME_SIZES, (size_index, pixel_format), 11)
        if size is None:
            break
        if size[2] != _DISCRETE:
            endpoint.properties["mode_limitations"] = "Continuous/stepwise sizes require manual mode selection."
            break
        width, height = size[3:5]
        for interval_index in range(_ENUM_LIMIT):
            interval = _enumerate(fd, _ENUM_FRAME_INTERVALS, (interval_index, pixel_format, width, height), 13)
            if interval is None or interval[4] != _DISCRETE:
                break
            numerator, denominator = interval[5:7]
            if numerator and denominator:
                yield VideoMode(width=width, height=height, fps=denominator / numerator, pixel_format=fourcc)


def probe_video(endpoint: Endpoint) -> None:
    """Populate capabilities and modes without allocating buffers or streaming.

    Raises:
        OSError: If access or a capability query fails.
    """
    endpoint.modes.clear()
    endpoint.properties.pop("mode_limitations", None)
    fd = os.open(endpoint.path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        cap = v4l2_capability()
        xioctl(fd, VIDIOC_QUERYCAP, cap)
        caps = cap.device_caps if cap.capabilities & V4L2_CAP_DEVICE_CAPS else cap.capabilities
        endpoint.role = "capture" if caps & V4L2_CAP_VIDEO_CAPTURE else "metadata" if caps & 0x00800000 else "unknown"
        endpoint.properties["v4l2_driver"] = cap.driver.decode(errors="replace").rstrip("\x00")
        endpoint.properties["name"] = cap.card.decode(errors="replace").rstrip("\x00")
        if endpoint.role != "capture":
            return
        for index in range(_ENUM_LIMIT):
            fmt = v4l2_fmtdesc()
            fmt.index, fmt.type = index, 1
            try:
                xioctl(fd, VIDIOC_ENUM_FMT, fmt)
            except OSError as exc:
                if exc.errno == errno.EINVAL:
                    break
                raise
            for mode in _format_modes(fd, fmt.pixelformat, endpoint):
                endpoint.modes.append(mode)
                if len(endpoint.modes) >= _MODE_LIMIT:
                    endpoint.properties["mode_limitations"] = "Mode enumeration limited to 512 advertised modes."
                    return
    finally:
        os.close(fd)
