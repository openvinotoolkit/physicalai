# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from typing import Any, cast

from physicalai.capture.discovery import DeviceInfo

__all__ = ["discover_realsense"]

_context: Any = None
_context_lock = threading.Lock()


def discover_realsense() -> list[DeviceInfo]:
    try:
        import pyrealsense2 as rs  # noqa: PLC0415
    except ImportError:
        return []

    rs_any = cast("Any", rs)  # cast to Any to avoid false positive "missing-attribute"
    global _context  # noqa: PLW0603
    with _context_lock:
        if _context is None:
            _context = rs_any.context()
        ctx = _context
    results: list[DeviceInfo] = []

    for i, dev in enumerate(ctx.query_devices()):
        try:
            serial = dev.get_info(rs_any.camera_info.serial_number)
            name = dev.get_info(rs_any.camera_info.name)
        except RuntimeError:
            continue

        results.append(
            DeviceInfo(
                device_id=serial,
                index=i,
                name=name,
                driver="realsense",
                hardware_payload={"serial": serial},
                manufacturer="RealSense",
                model=name,
            )
        )

    return results
