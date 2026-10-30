# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Synthetic reproductions of the attached USB hardware, without hardware I/O."""

import pytest

from physicalai.diagnostics.checks import assess
from physicalai.diagnostics.identity import binding_paths, resolve_endpoint
from physicalai.diagnostics.models import Alias, CollectionError, Device, Endpoint, Inventory


def serial_device(index: int, *, serial: str | None = None) -> Device:
    path = f"/dev/ttyUSB{index}"
    return Device(
        id=f"usb:3-1.{index + 1}", name="USB Serial", vendor_id="1a86", product_id="7523", serial=serial,
        endpoints=[Endpoint(
            id=f"serial:{index}", path=path, kind="serial", readable=True, writable=True,
            aliases=[
                Alias(path="/dev/serial/by-id/usb-1a86_USB_Serial", binding="device", target="/dev/ttyUSB2"),
                Alias(path=f"/dev/serial/by-path/pci-port{index}", binding="port", target=path),
            ],
        )],
    )


def test_three_serialless_adapters_are_not_deduplicated() -> None:
    inventory = Inventory(platform="linux", devices=[serial_device(i) for i in range(3)])
    report = assess(inventory)
    collisions = [f for f in report.findings if f.check_id == "identity.alias_collision"]
    assert len(collisions) == 1
    assert len(collisions[0].evidence) == 3
    for device in inventory.devices:
        endpoint = device.endpoints[0]
        assert binding_paths(inventory, endpoint, "device") == []
        assert len(binding_paths(inventory, endpoint, "port")) == 1
        assert resolve_endpoint(inventory, endpoint.path) == (device, endpoint)
    with pytest.raises(ValueError, match="3 endpoints"):
        resolve_endpoint(inventory, "/dev/serial/by-id/usb-1a86_USB_Serial")


def test_orbbec_serial_does_not_make_stream_alias_unique() -> None:
    device = Device(id="orbbec", name="Gemini", serial="SERIAL", vendor_id="2bc5", product_id="0840")
    device.endpoints = [Endpoint(
        id=f"video:{index}", path=f"/dev/video{index}", kind="video", role="capture",
        aliases=[Alias(path="/dev/v4l/by-id/camera-index0", binding="device", target="/dev/video38")],
    ) for index in (32, 38, 40)]
    inventory = Inventory(platform="linux", devices=[device])
    assert binding_paths(inventory, device.endpoints[1], "device") == []
    assert any(f.check_id == "identity.alias_collision" for f in assess(inventory).findings)


def test_missing_zed_video_is_not_a_healthy_camera() -> None:
    inventory = Inventory(platform="linux", devices=[Device(
        id="zed", name="ZED-M HID", vendor_id="2b03", product_id="f681", serial="123",
    )], settings={"usbfs_memory_mb": "16"})
    report = assess(inventory)
    assert report.exit_code() == 1
    assert next(f for f in report.findings if f.check_id == "usb.zed_video_missing").severity == "FAIL"
    assert next(f for f in report.findings if f.check_id == "usb.usbfs_memory").severity == "INFO"


def test_partial_is_distinct_from_failed_or_warning() -> None:
    inventory = Inventory(platform="linux", devices=[serial_device(0)])
    report = assess(inventory)
    assert report.exit_code() == 0
    assert report.exit_code(strict=True) == 1
    inventory.errors.append(CollectionError(source="sysfs", message="Denied", essential=True))
    assert report.exit_code() == 3
    inventory.devices[0].endpoints[0].readable = False
    assert assess(inventory).exit_code() == 1


def test_root_hub_is_not_a_shared_external_480_mbps_uplink() -> None:
    root = Device(id="root", name="USB root", kind="hub", speed_mbps=480, properties={"root_hub": "true"})
    cameras = [Device(id=f"camera:{i}", name="camera", parent_id="root", serial=f"serial{i}", endpoints=[
        Endpoint(id=f"video:{i}", path=f"/dev/video{i}", kind="video", role="capture"),
    ]) for i in range(2)]
    inventory = Inventory(platform="linux", devices=[root, *cameras])
    assert not any(f.check_id == "usb.shared_uplink" for f in assess(inventory).findings)
    hub = Device(id="external", name="hub", kind="hub", parent_id="root", speed_mbps=480)
    inventory.devices.append(hub)
    for camera in cameras:
        camera.parent_id = "external"
    shared = [f for f in assess(inventory).findings if f.check_id == "usb.shared_uplink"]
    assert len(shared) == 1 and shared[0].severity == "WARN" and shared[0].subject == "external"


def test_another_zed_video_does_not_hide_unpaired_hid() -> None:
    hid = Device(id="hid", name="ZED HID", vendor_id="2b03", product_id="f681", serial="ONE")
    video = Device(id="video", name="ZED video", vendor_id="2b03", product_id="video", serial="TWO", endpoints=[
        Endpoint(id="capture", path="/dev/video40", kind="video", role="capture"),
    ])
    inventory = Inventory(platform="linux", devices=[hid, video])
    assert any(f.check_id == "usb.zed_video_unpaired" for f in assess(inventory).findings)
    video.serial = "ONE"
    assert not any(f.check_id.startswith("usb.zed_video") for f in assess(inventory).findings)
