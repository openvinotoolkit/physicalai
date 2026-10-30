# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

from physicalai.diagnostics.models import Device, Endpoint, Inventory
from physicalai.diagnostics.platforms.processes import collect_holders

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc symlinks")


def test_visible_holders_are_deduplicated_without_reading_command_lines(tmp_path: Path) -> None:
    process = tmp_path / "42"
    (process / "fd").mkdir(parents=True)
    (process / "comm").write_text("camera-owner\n")
    (process / "fd/3").symlink_to("/dev/video40")
    (process / "fd/4").symlink_to("/dev/video40")
    (process / "fd/5").symlink_to("/some/unrelated-file")
    inventory = Inventory(platform="linux", devices=[Device(id="camera", name="camera", endpoints=[
        Endpoint(id="video", path="/dev/video40", kind="video", role="capture"),
    ])])
    collect_holders(inventory, tmp_path)
    holders = inventory.devices[0].endpoints[0].holders
    assert len(holders) == 1 and holders[0].pid == 42
    assert holders[0].name == "camera-owner"
    assert inventory.capabilities["process_ownership"].status == "available"


def test_missing_proc_does_not_claim_devices_are_idle(tmp_path: Path) -> None:
    inventory = Inventory(platform="linux", devices=[Device(id="arm", name="serial", endpoints=[
        Endpoint(id="serial", path="/dev/ttyUSB0", kind="serial"),
    ])])
    collect_holders(inventory, tmp_path / "absent")
    assert inventory.capabilities["process_ownership"].status == "unavailable"
