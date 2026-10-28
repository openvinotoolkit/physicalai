# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fake sysfs/udev collection, including alias losers and disappearing devices."""

from pathlib import Path
import sys

import pytest

from physicalai.diagnostics.platforms.linux import LinuxCollector

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux sysfs/udev fixture paths")


def write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def test_udev_alias_claims_survive_symlink_collision(tmp_path: Path) -> None:
    sysfs, dev, udev = (tmp_path / name for name in ("sys", "dev", "udev"))
    real = sysfs / "devices/pci0000:00/0000:00:14.0/usb3/3-1/3-1.2"
    for name, value in {"idVendor": "1a86", "idProduct": "7523", "product": "Serial", "devpath": "1.2", "speed": "12"}.items():
        write(real / name, value)
    bus = sysfs / "bus/usb/devices"
    bus.mkdir(parents=True)
    (bus / "3-1.2").symlink_to(real)
    tty = real / "3-1.2:1.0/ttyUSB0/tty/ttyUSB0"
    write(tty / "dev", "188:0")
    write(real / "3-1.2:1.0/bInterfaceNumber", "00")
    class_dir = sysfs / "class/tty"
    class_dir.mkdir(parents=True)
    (class_dir / "ttyUSB0").symlink_to(tty)
    write(dev / "ttyUSB0", "")
    write(udev / "c188:0", "E:ID_SERIAL=1a86_USB_Serial\nS:serial/by-id/collision\nS:serial/by-path/unique\n")
    (dev / "serial/by-id").mkdir(parents=True)
    write(dev / "ttyUSB2", "")
    (dev / "serial/by-id/collision").symlink_to(dev / "ttyUSB2")
    (dev / "serial/by-path").mkdir(parents=True)
    (dev / "serial/by-path/unique").symlink_to(dev / "ttyUSB0")
    snapshot = LinuxCollector(sysfs=sysfs, dev=dev, udev=udev, logs=False, proc=None).collect()
    device = next(device for device in snapshot.devices if device.endpoints)
    assert device.serial is None  # ID_SERIAL is not a manufacturer serial.
    assert "0000:00:14.0" in device.location
    endpoint = device.endpoints[0]
    assert endpoint.interface == "1.0"
    claims = {alias.path: alias for alias in endpoint.aliases}
    assert claims[str(dev / "serial/by-id/collision")].target == str(dev / "ttyUSB2")
    assert claims[str(dev / "serial/by-path/unique")].target == str(dev / "ttyUSB0")


def test_missing_sysfs_is_partial_not_empty_success(tmp_path: Path) -> None:
    result = LinuxCollector(sysfs=tmp_path / "absent", dev=tmp_path, udev=tmp_path, logs=False, proc=None).collect()
    assert not result.devices
    assert any(error.essential for error in result.errors)
    assert result.capabilities["usb_topology"].status == "unavailable"


def test_log_history_cursors_and_recovery_do_not_replay_old_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from physicalai.diagnostics.models import Inventory
    from physicalai.diagnostics.platforms import linux

    old = {"__CURSOR": "old", "__REALTIME_TIMESTAMP": "1000000", "MESSAGE": "usb 3-1: old failure"}
    new = {"__CURSOR": "new", "__REALTIME_TIMESTAMP": "2000000", "MESSAGE": "xhci_hcd: new controller event"}
    recovered = {"__CURSOR": "recovered", "__REALTIME_TIMESTAMP": "3000000", "MESSAGE": "cdc_acm: recovered view"}
    responses = iter([json.dumps(old), json.dumps(old) + "\n" + json.dumps(new), PermissionError("restricted"),
                      "\n".join(json.dumps(value) for value in (old, new, recovered))])
    commands = []
    def query(command):
        commands.append(command)
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response
    monkeypatch.setattr(linux, "_journal_output", query)
    collector = LinuxCollector(proc=None)
    first, second = Inventory(platform="linux"), Inventory(platform="linux")
    collector._logs(first)
    collector._logs(second)
    assert first.events[0].historical is True
    assert [event.id for event in second.events] == ["new"]
    assert second.events[0].historical is False
    assert "--after-cursor" in commands[1] and commands[1][-1] == "old"
    denied, restored = Inventory(platform="linux"), Inventory(platform="linux")
    collector._logs(denied)
    collector._logs(restored)
    assert denied.capabilities["system_logs"].status == "unavailable"
    assert [event.id for event in restored.events] == ["recovered"]
    assert restored.events[0].historical is True
