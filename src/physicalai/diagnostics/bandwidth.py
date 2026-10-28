# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Payload accounting, deliberately separate from USB link signaling rates."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physicalai.diagnostics.models import VideoMode

# Only fixed-size formats can be estimated without sampling the encoded payload.
_BITS_PER_PIXEL = {
    "YUYV": 16,
    "UYVY": 16,
    "Z16": 16,
    "Y16": 16,
    "RGB3": 24,
    "BGR3": 24,
    "RGB4": 32,
    "BGR4": 32,
    "GREY": 8,
    "BA81": 8,
    "NV12": 12,
    "YU12": 12,
    "YV12": 12,
}
_MBPS_PER_GBPS = 1000
_HIGH_SPEED_MBPS = 480
_SUPER_SPEED_MBPS = 5000


def frame_payload_bytes(mode: VideoMode) -> float | None:
    """Estimate an uncompressed wire frame, never a decoded/compressed buffer.

    Returns:
        Reported padded frame size or a format-derived minimum; unknown for
        compressed and unrecognized formats even if ``sizeimage`` was reported.
    """
    bits = _BITS_PER_PIXEL.get(mode.pixel_format.strip())
    if bits is None:
        return None
    minimum = math.ceil(mode.width * mode.height * bits / 8)
    return max(mode.frame_bytes or 0, minimum)


def payload_demand(mode: VideoMode) -> float | None:
    """Return estimated payload bytes/second for this configured mode, or unknown."""
    size = frame_payload_bytes(mode)
    return size * mode.fps if size is not None else None


def format_link(mbps: float | None) -> str:
    """Return a labeled SI signaling rate, not a payload measurement."""
    if mbps is None:
        return "link unknown"
    generation = "USB 3.x / " if mbps >= _SUPER_SPEED_MBPS else "USB 2.0 / " if mbps == _HIGH_SPEED_MBPS else ""
    if mbps >= _MBPS_PER_GBPS:
        return f"{generation}{mbps / _MBPS_PER_GBPS:g} Gb/s link"
    return f"{generation}{mbps:g} Mb/s link"


def format_payload(bytes_per_second: float | None) -> str:
    """Return SI payload MB/s or an explicit absence of measurement."""
    if bytes_per_second is None:
        return "not measured"
    return f"{bytes_per_second / 1_000_000:.2f} MB/s"


def format_ceiling(mbps: float | None) -> str:
    """Return the raw signaling ceiling in MB/s, not usable or allocated payload."""
    return f"{mbps / 8:g} MB/s" if mbps is not None else "unknown"


def format_mode(mode: VideoMode | None) -> str:
    """Return a compact configured wire mode, without asserting an active stream."""
    if mode is None:
        return "mode unknown"
    return f"{mode.width}x{mode.height} {mode.pixel_format} @{mode.fps:g} FPS"
