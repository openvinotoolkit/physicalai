# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from physicalai.diagnostics.checks import assess
from physicalai.diagnostics.models import Alias, Device, Endpoint, Inventory, WatchEvent
from physicalai.diagnostics.redact import Redactor


def test_redaction_preserves_alias_collisions_and_leaves_original_untouched() -> None:
    inventory = Inventory(platform="linux", devices=[Device(
        id="camera", name="camera", serial="SERIAL-123", endpoints=[Endpoint(
            id=f"video:{i}", path=f"/dev/video{i}", kind="video", aliases=[Alias(
                path="/dev/v4l/by-id/camera-SERIAL-123", binding="device", target="/dev/video0",
            )], properties={"ID_SERIAL_SHORT": "SERIAL-123"},
        ) for i in range(2)],
    )])
    original = assess(inventory)
    redacted = Redactor().model(original)
    assert "SERIAL-123" not in redacted.model_dump_json()
    assert "SERIAL-123" in original.model_dump_json()
    assert redacted.inventory.redacted is True
    serial = redacted.inventory.devices[0].serial
    assert serial is not None and serial in redacted.findings[0].subject
    assert redacted.findings[0].check_id == "identity.alias_collision"


def test_same_watch_serial_gets_same_pseudonym_and_schema_is_unchanged() -> None:
    redactor = Redactor()
    first = redactor.model(WatchEvent(sequence=1, kind="device_added", data={"serial": "SN", "kind": "device"}))
    second = redactor.model(WatchEvent(sequence=2, kind="device_removed", data={"serial": "SN", "kind": "device"}))
    assert first.data["serial"] == second.data["serial"] != "SN"
    assert first.kind == "device_added"
    assert first.data["kind"] == "device"
