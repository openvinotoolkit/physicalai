# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for physical-device identity helpers."""

from __future__ import annotations

from pathlib import Path

from physicalai.robot import device_id_from_serial_port


def test_device_id_from_serial_port_uses_port_basename() -> None:
    """Non-device paths use their basename as the serial identity."""
    assert device_id_from_serial_port("ttyUSB0") == "serial:ttyUSB0"
    assert device_id_from_serial_port("custom/ttyACM0") == "serial:ttyACM0"


def test_device_id_from_serial_port_resolves_device_alias(monkeypatch: object) -> None:
    """Device paths use the basename of their resolved target."""
    monkeypatch.setattr(Path, "resolve", lambda _path: Path("/dev/ttyUSB0"))  # type: ignore[attr-defined]

    assert device_id_from_serial_port("/dev/serial/by-id/robot-arm") == "serial:ttyUSB0"