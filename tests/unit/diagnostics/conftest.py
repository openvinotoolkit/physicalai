# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Snapshots matching a pair of cameras and two serial adapters on nested hubs."""

import pytest

from physicalai.diagnostics import Alias, Device, Endpoint, Inventory


@pytest.fixture
def inventory() -> Inventory:
    camera1 = Endpoint("/dev/video0", "video", "capture", True, True)
    camera2 = Endpoint("/dev/video2", "video", "capture", True, True)
    shared = "/dev/v4l/by-id/usb-camera-SN0001-video-index0"
    for endpoint, port in ((camera1, "1.1"), (camera2, "1.2")):
        endpoint.aliases = [
            Alias(shared, camera2.path, claimed=True),
            Alias(f"/dev/v4l/by-path/port-{port}", endpoint.path, claimed=True),
        ]
    return Inventory(
        devices=[
            Device("pci:0", "Controller", "controller"),
            Device("usb1", "Root hub", "hub", "pci:0", 480, root_hub=True),
            Device("1-1", "Outer hub", "hub", "usb1", 480),
            Device("1-1.1", "Inner hub", "hub", "1-1", 480),
            Device(
                "1-1.1.1",
                "Camera",
                parent_id="1-1.1",
                speed_mbps=480,
                vendor_id="1234",
                product_id="abcd",
                serial="SN0001",
                video_interface=True,
                endpoints=[camera1, Endpoint("/dev/video1", "video", "metadata", True, True)],
            ),
            Device(
                "1-1.1.2",
                "Camera",
                parent_id="1-1.1",
                speed_mbps=480,
                vendor_id="1234",
                product_id="abcd",
                serial="SN0001",
                video_interface=True,
                endpoints=[camera2, Endpoint("/dev/video3", "video", "metadata", True, True)],
            ),
            Device(
                "1-1.1.3",
                "Serial A",
                parent_id="1-1.1",
                speed_mbps=12,
                serial="arm-a",
                endpoints=[Endpoint("/dev/ttyACM0", "serial", readable=True, writable=True)],
            ),
            Device(
                "1-1.4",
                "Serial B",
                parent_id="1-1",
                speed_mbps=12,
                serial="arm-b",
                endpoints=[Endpoint("/dev/ttyACM1", "serial", readable=True, writable=True)],
            ),
            Device("1-2", "Keyboard", parent_id="usb1", speed_mbps=12),
        ]
    )
