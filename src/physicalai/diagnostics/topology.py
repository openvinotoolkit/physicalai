# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared USB ancestry and contention domains, independent of OS and terminal."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physicalai.diagnostics.models import Device, Inventory


def port_number(device: Device) -> str:
    """Return the downstream port number, when the collector reported a chain."""
    path = device.properties.get("usb_path", "")
    if path.startswith("usb") or "-" not in path:
        return ""
    return path.split("-", 1)[1].rsplit(".", 1)[-1]


def _order(device: Device) -> tuple[tuple[int, ...], str]:
    path = device.properties.get("usb_path", "") or device.id
    return tuple(int(part) for part in re.findall(r"\d+", path)), device.id


class USBTopology:
    """Index authoritative parent links; never group devices by serial/name."""

    def __init__(self, inventory: Inventory) -> None:
        """Keep the snapshot's evidence without mutating its device records."""
        self.devices = {device.id: device for device in inventory.devices}
        self.children: dict[str, list[Device]] = defaultdict(list)
        for device in inventory.devices:
            if device.parent_id in self.devices:
                self.children[device.parent_id].append(device)
        for children in self.children.values():
            children.sort(key=_order)

    def route(self, device_id: str) -> list[Device]:
        """Return the controller-to-device path, stopping on missing/cyclic data."""
        path: list[Device] = []
        seen: set[str] = set()
        current = self.devices.get(device_id)
        while current is not None and current.id not in seen:
            seen.add(current.id)
            path.append(current)
            current = self.devices.get(current.parent_id) if current.parent_id else None
        return list(reversed(path))

    def descendants(self, device_id: str) -> list[Device]:
        """Return physical descendants once, even with malformed parent cycles."""
        found: list[Device] = []
        seen = {device_id}
        pending = list(reversed(self.children.get(device_id, [])))
        while pending:
            device = pending.pop()
            if device.id in seen:
                continue
            seen.add(device.id)
            found.append(device)
            pending.extend(reversed(self.children.get(device.id, [])))
        return found

    def shared_devices(self, hub_id: str, *, excluding: str | None = None) -> list[Device]:
        """Return physical peers under a hub, not its interfaces or metadata nodes."""
        return [device for device in self.descendants(hub_id) if device.kind == "device" and device.id != excluding]

    def external_hubs(self, device_id: str) -> list[Device]:
        """Return upstream shared uplinks, excluding independent root-port domains."""
        return [
            device
            for device in self.route(device_id)[:-1]
            if device.kind == "hub" and device.properties.get("root_hub") != "true"
        ]

    def roots(self) -> list[Device]:
        """Return deterministically ordered known roots/orphans of the snapshot."""
        return sorted((device for device in self.devices.values() if device.parent_id not in self.devices), key=_order)
