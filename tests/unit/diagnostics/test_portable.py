# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import pytest

from physicalai.diagnostics.platforms import create_collector
from physicalai.diagnostics.platforms.portable import PortableCollector


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_portable_enumeration_does_not_open_ports_or_cameras(platform: str, monkeypatch: pytest.MonkeyPatch) -> None:
    port = SimpleNamespace(device="COM4", description="Serial", serial_number=None, vid=0x1A86, pid=0x7523, location="1-2")
    def comports():
        return [port]
    monkeypatch.setitem(sys.modules, "serial.tools.list_ports", SimpleNamespace(comports=comports))
    camera = SimpleNamespace(index=4, name="Camera", device_meta_json={"serial": "SN"})
    def query(*, only_usable):
        assert only_usable is False
        return [camera]
    monkeypatch.setitem(sys.modules, "pynokhwa", SimpleNamespace(query=query))
    snapshot = PortableCollector(platform).collect()
    assert not snapshot.errors
    assert len(snapshot.devices) == 2
    assert snapshot.devices[0].endpoints[0].path == "COM4"
    assert snapshot.devices[1].endpoints[0].path == "index:4"
    assert snapshot.capabilities["usb_topology"].status == "unsupported"
    assert isinstance(create_collector(platform), PortableCollector)


def test_backend_failure_is_not_empty_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def query(**kwargs):
        raise PermissionError("camera permission denied")
    monkeypatch.setitem(sys.modules, "pynokhwa", SimpleNamespace(query=query))
    snapshot = PortableCollector("darwin").collect()
    assert any(error.source == "cameras" and error.essential for error in snapshot.errors)


def test_unknown_platform_is_explained() -> None:
    with pytest.raises(ValueError, match="do not support platform"):
        create_collector("unsupported-os")
