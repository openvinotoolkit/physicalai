# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Linux filesystem fixtures and read-only capability queries, no real devices."""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from physicalai.diagnostics import assess
from physicalai.diagnostics.linux import LinuxCollector, _speed, _video_role

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux USB collector")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def collector(tmp_path: Path) -> LinuxCollector:
    sysfs, dev, udev = (tmp_path / name for name in ("sys", "dev", "udev"))
    udev.mkdir()
    dev.mkdir()
    bus = sysfs / "bus/usb/devices"
    bus.mkdir(parents=True)
    root = sysfs / "devices/pci0000:00/0000:00:14.0/usb1"
    for name, path, product, speed in (
        ("usb1", root, "Root hub", "480"),
        ("1-1", root / "1-1", "Hub", "480"),
        ("1-1.1", root / "1-1/1-1.1", "Camera", "480"),
        ("1-1.2", root / "1-1/1-1.2", "Serial adapter", "12"),
    ):
        write(path / "idVendor", "1234")
        write(path / "idProduct", "abcd")
        write(path / "product", product)
        write(path / "speed", speed)
        write(path / "bDeviceClass", "09" if name in {"usb1", "1-1"} else "00")
        (bus / name).symlink_to(path, target_is_directory=True)
    camera = root / "1-1/1-1.1/1-1.1:1.0"
    write(camera / "bInterfaceClass", "0e")
    serial = root / "1-1/1-1.2/1-1.2:1.0"
    for cls, name, path, major_minor in (
        ("video4linux", "video0", camera / "video4linux/video0", "81:0"),
        ("video4linux", "video1", camera / "video4linux/video1", "81:1"),
        ("tty", "ttyACM0", serial / "tty/ttyACM0", "166:0"),
        # Non-USB video pipeline: must not probe or report permission failures.
        ("video4linux", "video9", sysfs / "devices/platform/ipu/video4linux/video9", "81:9"),
    ):
        write(path / "dev", major_minor)
        directory = sysfs / "class" / cls
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).symlink_to(path, target_is_directory=True)
        (dev / name).touch()
    role = Mock(side_effect=lambda path: "capture" if path.name == "video0" else "metadata")
    return LinuxCollector(sysfs=sysfs, dev=dev, udev=udev, video_role=role)


def test_groups_endpoints_with_physical_usb_devices(collector: LinuxCollector) -> None:
    inventory = collector.collect()
    assert not inventory.errors
    device, endpoint = inventory.resolve(str(collector.dev / "video0"))
    metadata_device, metadata = inventory.resolve(str(collector.dev / "video1"))
    assert device is metadata_device
    assert endpoint is not None and endpoint.role == "capture"
    assert metadata is not None and metadata.role == "metadata"
    assert device.video_interface
    assert device.speed_mbps == 480
    assert [d.id for d in inventory.ancestors(device)] == ["1-1", "usb1", "pci:0000:00:14.0"]
    assert len([d for d in inventory.devices if d.relevant]) == 2
    assert isinstance(collector.video_role, Mock)
    assert {call.args[0].name for call in collector.video_role.call_args_list} == {"video0", "video1"}


@pytest.mark.parametrize("value", [0, 16, 256, 1024])
def test_usbfs_memory_is_collected_without_changing_setting(collector: LinuxCollector, value: int) -> None:
    path = collector.sysfs / "module/usbcore/parameters/usbfs_memory_mb"
    write(path, f"{value}\n")
    inventory = collector.collect()
    assert inventory.usbfs_memory_mb == value
    assert path.read_text() == f"{value}\n"
    assert not inventory.errors and not inventory.notes


@pytest.mark.parametrize("value", ["", "-1", "nan", "16 MB", "1.5"])
def test_invalid_usbfs_memory_is_a_limitation_not_an_error(collector: LinuxCollector, value: str) -> None:
    write(collector.sysfs / "module/usbcore/parameters/usbfs_memory_mb", value)
    inventory = collector.collect()
    assert inventory.usbfs_memory_mb is None
    assert any("usbfs transfer-memory limit unavailable" in note for note in inventory.notes)
    assert not inventory.errors
    assert inventory.resolve(str(collector.dev / "video0"))[1] is not None


def test_missing_usbfs_memory_is_a_limitation_not_an_error(collector: LinuxCollector) -> None:
    inventory = collector.collect()
    assert inventory.usbfs_memory_mb is None
    assert any("usbfs transfer-memory limit unavailable" in note for note in inventory.notes)
    assert not inventory.errors


def test_unreadable_usbfs_memory_is_a_limitation_not_an_error(
    collector: LinuxCollector, monkeypatch: pytest.MonkeyPatch
) -> None:
    read_text = Path.read_text

    def denied(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        if path.name == "usbfs_memory_mb":
            raise PermissionError("parameter denied")
        return read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", denied)
    inventory = collector.collect()
    assert inventory.usbfs_memory_mb is None
    assert any("parameter denied" in note for note in inventory.notes)
    assert not inventory.errors


def test_serial_enumeration_does_not_open_ports(collector: LinuxCollector, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "open", Mock(side_effect=AssertionError("No device opens allowed")))
    device, endpoint = collector.collect().resolve(str(collector.dev / "ttyACM0"))
    assert endpoint is not None and endpoint.kind == "serial"
    assert device.speed_mbps == 12


def test_alias_claims_include_losers_not_just_symlink_winners(collector: LinuxCollector) -> None:
    alias = collector.dev / "v4l/by-id/shared"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(collector.dev / "video1")
    for major in ("81:0", "81:1"):
        write(collector.udev / f"c{major}", "S:v4l/by-id/shared\n")
    inventory = collector.collect()
    claims = [a for d in inventory.devices for e in d.endpoints for a in e.aliases]
    assert len(claims) == 2
    assert all(a.claimed and a.target == str(collector.dev / "video1") for a in claims)
    with pytest.raises(ValueError, match="matches 2 endpoints"):
        inventory.resolve(str(alias))


def test_symlink_fallback_is_not_treated_as_udev_proof(collector: LinuxCollector) -> None:
    alias = collector.dev / "serial/by-id/arm"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(collector.dev / "ttyACM0")
    inventory = collector.collect()
    device, endpoint = inventory.resolve(str(alias))
    assert endpoint is not None
    assert not endpoint.aliases[0].claimed
    assert inventory.unique_aliases(device, endpoint) == []


def test_alias_failure_keeps_endpoint_visible(collector: LinuxCollector, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collector, "_aliases", Mock(side_effect=PermissionError("udev denied")))
    inventory = collector.collect()
    assert inventory.resolve(str(collector.dev / "video0"))[1] is not None
    assert any("udev denied" in note for note in inventory.notes)


def test_video_query_failure_keeps_unknown_node_visible(collector: LinuxCollector) -> None:
    collector.video_role = Mock(side_effect=OSError("capability query failed"))
    inventory = collector.collect()
    _, endpoint = inventory.resolve(str(collector.dev / "video0"))
    assert endpoint is not None
    assert endpoint.role == "unknown"
    assert endpoint.query_error == "capability query failed"
    assert any(f.code == "video_role_unknown" for f in assess(inventory))


def test_unknown_capability_is_disclosed(collector: LinuxCollector) -> None:
    collector.video_role = Mock(return_value="unknown")
    inventory = collector.collect()
    assert any(f.code == "video_role_unknown" for f in assess(inventory))


def test_endpoint_resolution_failure_is_returned_as_collection_error(
    collector: LinuxCollector,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve = Path.resolve

    def denied(path: Path, strict: bool = False) -> Path:
        if path == collector.sysfs / "class/video4linux/video0":
            raise PermissionError("endpoint sysfs denied")
        return resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", denied)
    inventory = collector.collect()
    assert any("endpoint sysfs denied" in error for error in inventory.errors)
    assert inventory.resolve(str(collector.dev / "ttyACM0"))[1] is not None


def test_missing_usb_root_is_a_collection_error(tmp_path: Path) -> None:
    inventory = LinuxCollector(sysfs=tmp_path).collect()
    assert inventory.devices == []
    assert inventory.errors and "USB topology" in inventory.errors[0]


def test_unreadable_usb_attribute_is_reported(collector: LinuxCollector, monkeypatch: pytest.MonkeyPatch) -> None:
    read_text = Path.read_text

    def denied(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        if path.name == "speed" and path.parent.name == "1-1.1":
            raise PermissionError("speed denied")
        return read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", denied)
    inventory = collector.collect()
    assert any("speed denied" in error for error in inventory.errors)


@pytest.mark.parametrize("value", ["", "nan", "inf", "-1", "0", "n/a"])
def test_invalid_speeds_are_unknown(value: str) -> None:
    assert _speed(value) is None


@pytest.mark.parametrize("cap,role", [(1, "capture"), (0x1000, "capture"), (0x00800000, "metadata"), (0, "unknown")])
def test_capability_query_is_read_only_and_closes_fd(
    cap: int,
    role: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fcntl

    opened, closed = Mock(return_value=42), Mock()
    monkeypatch.setattr(os, "open", opened)
    monkeypatch.setattr(os, "close", closed)

    def ioctl(fd: int, request: int, buffer: bytearray) -> None:
        assert fd == 42
        assert request == 0x80685600  # VIDIOC_QUERYCAP only, never VIDIOC_S_* or STREAMON
        buffer[:] = struct.pack("=16s32s32s6I", b"uvcvideo", b"Camera", b"usb", 0, 0x80000000, cap, 0, 0, 0)

    monkeypatch.setattr(fcntl, "ioctl", ioctl)
    assert _video_role(Path("/dev/video0")) == role
    opened.assert_called_once_with(Path("/dev/video0"), os.O_RDONLY | os.O_NONBLOCK)
    closed.assert_called_once_with(42)


def test_failed_capability_query_closes_fd(monkeypatch: pytest.MonkeyPatch) -> None:
    import fcntl

    closed = Mock()
    monkeypatch.setattr(os, "open", Mock(return_value=42))
    monkeypatch.setattr(os, "close", closed)
    monkeypatch.setattr(fcntl, "ioctl", Mock(side_effect=OSError("gone")))
    with pytest.raises(OSError, match="gone"):
        _video_role(Path("/dev/video0"))
    closed.assert_called_once_with(42)
