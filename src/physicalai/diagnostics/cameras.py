# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared wire-mode support used by discovery and active camera diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.config import Config

if TYPE_CHECKING:
    from physicalai.diagnostics.models import Endpoint, VideoMode

_V4L2_FORMATS = {"MJPG": "mjpeg", "YUYV": "yuyv"}


def supported_modes(endpoint: Endpoint) -> list[VideoMode]:
    """List modes the current Runtime V4L2 backend can represent exactly.

    Returns:
        MJPEG/YUYV modes with integer FPS; other modes remain in the inventory.
    """
    return [mode for mode in endpoint.modes if mode.pixel_format in _V4L2_FORMATS and mode.fps.is_integer()]


def camera_config(selector: str, mode: VideoMode) -> Config:
    """Export a supported camera mode using the actual backend constructor names.

    Returns:
        A passive UVC camera recipe.

    Raises:
        ValueError: If the backend cannot represent the advertised mode.
    """
    if mode.pixel_format not in _V4L2_FORMATS or not mode.fps.is_integer():
        msg = "The V4L2 Runtime backend requires MJPEG/YUYV and integer FPS."
        raise ValueError(msg)
    return Config(
        "physicalai.capture.UVCCamera",
        {
            "device": selector,
            "backend": "v4l2",
            "width": mode.width,
            "height": mode.height,
            "fps": int(mode.fps),
            "backend_options": {"pixel_format": _V4L2_FORMATS[mode.pixel_format]},
        },
    )
