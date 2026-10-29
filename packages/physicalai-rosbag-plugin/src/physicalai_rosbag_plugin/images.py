# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Decode the supported ROS image encodings with explicit allocation limits."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from physicalai_rosbag_plugin.config import ResourceLimits

_GRAY_DIMENSIONS = 2
_RGB_CHANNELS = 3
_RGBA_CHANNELS = 4
_CHANNELS = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}


def _check_pixels(width: int, height: int, limits: ResourceLimits) -> None:
    if width <= 0 or height <= 0 or width * height > limits.image_pixels:
        msg = f"Image pixels exceed limit or dimensions are invalid: {width}x{height}"
        raise ValueError(msg)


def decode_image(msg: Any, message_type: str, limits: ResourceLimits) -> NDArray[np.uint8]:  # noqa: ANN401
    """Decode a bounded image payload.

    Returns:
        Contiguous RGB uint8 image data.

    Raises:
        ValueError: The encoding, dimensions, or payload are invalid.
    """
    if len(msg.data) > limits.message_bytes:
        msg_0 = "Image payload exceeds message_bytes"
        raise ValueError(msg_0)
    if message_type == "sensor_msgs/msg/Image":
        _check_pixels(msg.width, msg.height, limits)
        if msg.encoding not in _CHANNELS:
            msg_0 = f"Unsupported raw image encoding: {msg.encoding}"
            raise ValueError(msg_0)
        channels = _CHANNELS[msg.encoding]
        row_bytes = msg.width * channels
        if msg.step < row_bytes or len(msg.data) != msg.height * msg.step:
            msg_0 = "Image step/payload length is inconsistent with dimensions"
            raise ValueError(msg_0)
        rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        pixels = rows[:, :row_bytes].reshape(msg.height, msg.width, channels)
        if msg.encoding == "mono8":
            return np.repeat(pixels, 3, axis=2)
        rgb = pixels[:, :, :3]
        return np.ascontiguousarray(rgb[:, :, ::-1] if msg.encoding in {"bgr8", "bgra8"} else rgb)
    if "compressedDepth" in msg.format:
        msg_0 = "compressedDepth is unsupported"
        raise ValueError(msg_0)
    payload = bytes(msg.data)
    with Image.open(BytesIO(payload)) as header:
        if header.format not in {"JPEG", "PNG"} or header.mode not in {"RGB", "RGBA", "L"}:
            msg_0 = "Only 8-bit RGB/gray JPEG and PNG images are supported"
            raise ValueError(msg_0)
        _check_pixels(*header.size, limits)
        expected_size = header.size
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype != np.uint8 or decoded.shape[:2] != expected_size[::-1]:
        msg_0 = "Invalid compressed image payload"
        raise ValueError(msg_0)
    if decoded.ndim == _GRAY_DIMENSIONS:
        return np.asarray(cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB), dtype=np.uint8)
    if decoded.shape[2] == _RGBA_CHANNELS:
        return np.asarray(cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGB), dtype=np.uint8)
    if decoded.shape[2] == _RGB_CHANNELS:
        return np.asarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    msg_0 = "Unsupported compressed image channels"
    raise ValueError(msg_0)
