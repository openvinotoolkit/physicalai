# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Helpers for deriving canonical physical-device identities."""

from __future__ import annotations

from pathlib import Path


def device_id_from_serial_port(port: str) -> str:
    """Return the canonical device identity for a serial port.

    Args:
        port: Serial port path or device name.

    Returns:
        A ``serial:``-qualified identity using the symlink-resolved basename
        for paths under ``/dev``.
    """
    name = Path(port).resolve().name if port.startswith("/dev/") else Path(port).name
    return f"serial:{name}"
