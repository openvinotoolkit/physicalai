# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for V4L2 device discovery."""

from __future__ import annotations

import ctypes
from pathlib import Path
from unittest import mock

import pytest

from physicalai.capture.cameras.v4l2._discover import discover_v4l2
from physicalai.capture.cameras.v4l2._ioctl import V4L2_CAP_STREAMING, V4L2_CAP_VIDEO_CAPTURE, v4l2_capability
from physicalai.capture.discovery import DeviceInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sysfs_entry(video_name: str, device_label: str) -> mock.MagicMock:
    """Return a mock sysfs directory entry for *video_name* with *device_label*."""
    entry = mock.MagicMock()
    entry.name = video_name
    # Make entries sortable by name so sorted() works deterministically
    entry.__lt__ = lambda self, other: self.name < other.name
    entry.__le__ = lambda self, other: self.name <= other.name
    entry.__gt__ = lambda self, other: self.name > other.name
    entry.__ge__ = lambda self, other: self.name >= other.name
    name_path = mock.MagicMock()
    name_path.read_text.return_value = device_label + "\n"
    entry.__truediv__ = lambda self, key: name_path  # entry / "name"
    return entry


def _make_ioctl_side_effect(
    card: str = "Test Cam",
    bus_info: str = "usb-0000:00:14.0-1",
    capabilities: int = V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING,
    driver: str = "uvcvideo",
):
    """Return a side_effect function for xioctl that fills *arg* in place."""

    def _ioctl(fd: int, request: int, arg: v4l2_capability) -> int:  # noqa: ARG001
        arg.driver = driver.encode()
        arg.card = card.encode()
        arg.bus_info = bus_info.encode()
        arg.capabilities = capabilities
        return 0

    return _ioctl


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_discover_returns_empty_when_no_sysfs() -> None:
    """discover_v4l2() returns [] when /sys/class/video4linux does not exist."""
    with mock.patch("physicalai.capture.cameras.v4l2._discover._SYSFS_V4L2") as mock_sysfs:
        mock_sysfs.exists.return_value = False
        result = discover_v4l2()

    assert result == []


def test_discover_returns_empty_when_sysfs_iterdir_raises() -> None:
    """discover_v4l2() returns [] when iterdir() raises FileNotFoundError."""
    with mock.patch("physicalai.capture.cameras.v4l2._discover._SYSFS_V4L2") as mock_sysfs:
        mock_sysfs.exists.return_value = True
        mock_sysfs.iterdir.side_effect = FileNotFoundError("gone")
        result = discover_v4l2()

    assert result == []


def test_discover_single_capture_device() -> None:
    """discover_v4l2() returns one DeviceInfo for a single capture-capable device."""
    entry0 = _make_sysfs_entry("video0", "Test Cam")

    with mock.patch("physicalai.capture.cameras.v4l2._discover._SYSFS_V4L2") as mock_sysfs:
        mock_sysfs.exists.return_value = True
        mock_sysfs.iterdir.return_value = [entry0]

        with mock.patch("physicalai.capture.cameras.v4l2._discover.os.open", return_value=5):
            with mock.patch("physicalai.capture.cameras.v4l2._discover.os.close"):
                with mock.patch(
                    "physicalai.capture.cameras.v4l2._discover.xioctl",
                    side_effect=_make_ioctl_side_effect(
                        card="Test Cam",
                        bus_info="usb-0000:00:14.0-1",
                        capabilities=V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING,
                        driver="uvcvideo",
                    ),
                ):
                    result = discover_v4l2()

    assert len(result) == 1
    device = result[0]
    assert isinstance(device, DeviceInfo)
    assert device.device_id == "/dev/video0"
    assert device.name == "Test Cam"
    assert device.driver == "v4l2"
    assert device.hardware_id == "usb-0000:00:14.0-1"


def test_discover_skips_non_capture_device() -> None:
    """discover_v4l2() returns [] when device lacks V4L2_CAP_VIDEO_CAPTURE."""
    entry0 = _make_sysfs_entry("video0", "Radio Device")

    # Capabilities without VIDEO_CAPTURE (e.g., radio only)
    radio_caps = V4L2_CAP_STREAMING  # no V4L2_CAP_VIDEO_CAPTURE

    with mock.patch("physicalai.capture.cameras.v4l2._discover._SYSFS_V4L2") as mock_sysfs:
        mock_sysfs.exists.return_value = True
        mock_sysfs.iterdir.return_value = [entry0]

        with mock.patch("physicalai.capture.cameras.v4l2._discover.os.open", return_value=5):
            with mock.patch("physicalai.capture.cameras.v4l2._discover.os.close"):
                with mock.patch(
                    "physicalai.capture.cameras.v4l2._discover.xioctl",
                    side_effect=_make_ioctl_side_effect(capabilities=radio_caps),
                ):
                    result = discover_v4l2()

    assert result == []


def test_discover_skips_device_on_permission_error() -> None:
    """discover_v4l2() silently skips devices that raise PermissionError on open."""
    entry0 = _make_sysfs_entry("video0", "Locked Cam")

    with mock.patch("physicalai.capture.cameras.v4l2._discover._SYSFS_V4L2") as mock_sysfs:
        mock_sysfs.exists.return_value = True
        mock_sysfs.iterdir.return_value = [entry0]

        with mock.patch(
            "physicalai.capture.cameras.v4l2._discover.os.open",
            side_effect=PermissionError("access denied"),
        ):
            result = discover_v4l2()

    assert result == []


def test_discover_multiple_devices_one_non_capture() -> None:
    """discover_v4l2() returns only capture-capable devices from a mixed list."""
    entry0 = _make_sysfs_entry("video0", "USB Cam A")
    entry1 = _make_sysfs_entry("video1", "Radio")
    entry2 = _make_sysfs_entry("video2", "USB Cam B")

    # Map device path → capabilities
    caps_by_device: dict[str, int] = {
        "/dev/video0": V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING,
        "/dev/video1": V4L2_CAP_STREAMING,  # no capture
        "/dev/video2": V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING,
    }
    # Track which fd corresponds to which device (open returns fd sequentially)
    _open_calls: list[str] = []

    def _fake_open(path: str, flags: int) -> int:  # noqa: ARG001
        _open_calls.append(path)
        return len(_open_calls)  # fd = 1, 2, 3, ...

    def _multi_ioctl(fd: int, request: int, arg: v4l2_capability) -> int:
        device_path = _open_calls[fd - 1]
        caps = caps_by_device[device_path]
        arg.driver = b"uvcvideo"
        arg.card = f"Cam {device_path[-1]}".encode()
        arg.bus_info = f"usb-0000:00:14.0-{fd}".encode()
        arg.capabilities = caps
        return 0

    with mock.patch("physicalai.capture.cameras.v4l2._discover._SYSFS_V4L2") as mock_sysfs:
        mock_sysfs.exists.return_value = True
        mock_sysfs.iterdir.return_value = [entry0, entry1, entry2]

        with mock.patch("physicalai.capture.cameras.v4l2._discover.os.open", side_effect=_fake_open):
            with mock.patch("physicalai.capture.cameras.v4l2._discover.os.close"):
                with mock.patch(
                    "physicalai.capture.cameras.v4l2._discover.xioctl",
                    side_effect=_multi_ioctl,
                ):
                    result = discover_v4l2()

    assert len(result) == 2  # noqa: PLR2004
    device_ids = [d.device_id for d in result]
    assert "/dev/video0" in device_ids
    assert "/dev/video1" not in device_ids
    assert "/dev/video2" in device_ids
    for device in result:
        assert device.driver == "v4l2"
