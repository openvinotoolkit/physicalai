# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""USB evidence from Linux sysfs/udev, with only read-only video queries."""

from __future__ import annotations

import math
import os
import re
import struct
from pathlib import Path
from typing import TYPE_CHECKING

from physicalai.diagnostics.models import Alias, Device, Endpoint, Inventory

if TYPE_CHECKING:
    from collections.abc import Callable

# Linux videodev2.h: struct v4l2_capability and _IOR('V', 0, ...).
_VIDEO_CAP = struct.Struct("=16s32s32s6I")
_QUERY_CAP = (2 << 30) | (_VIDEO_CAP.size << 16) | (ord("V") << 8)
_CAP_DEVICE_CAPS = 0x80000000
_CAP_CAPTURE = 0x00000001 | 0x00001000  # single- and multi-planar capture
_CAP_METADATA = 0x00800000


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except FileNotFoundError:
        return ""  # Optional attributes and disappearing unplugged devices are normal.


def _speed(value: str) -> float | None:
    try:
        speed = float(value)
    except ValueError:
        return None
    return speed if math.isfinite(speed) and speed > 0 else None


def _video_role(path: Path) -> str:
    import fcntl  # noqa: PLC0415 - Linux-only module, imported only during collection

    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        data = bytearray(_VIDEO_CAP.size)
        fcntl.ioctl(fd, _QUERY_CAP, data)
        fields = _VIDEO_CAP.unpack(data)
        caps = fields[5] if fields[4] & _CAP_DEVICE_CAPS else fields[4]
        return "capture" if caps & _CAP_CAPTURE else "metadata" if caps & _CAP_METADATA else "unknown"
    finally:
        os.close(fd)


class LinuxCollector:
    """Read USB cameras, serial endpoints, and their ancestry without drivers.

    The injectable filesystem roots and video query keep tests hardware-free.
    Serial ports are never opened; video queries never set formats or stream.
    """

    def __init__(
        self,
        *,
        sysfs: Path = Path("/sys"),
        dev: Path = Path("/dev"),
        udev: Path = Path("/run/udev/data"),
        video_role: Callable[[Path], str] = _video_role,
    ) -> None:
        """Set filesystem roots and the read-only video capability query."""
        self.sysfs = sysfs
        self.dev = dev
        self.udev = udev
        self.video_role = video_role

    def collect(self) -> Inventory:
        """Return a snapshot; report collection failures rather than hiding them."""
        inventory = Inventory()
        try:
            devices = self._usb_devices(inventory)
        except OSError as exc:
            inventory.errors.append(f"USB topology: {exc}")
            return inventory
        if not self.udev.is_dir() or not os.access(self.udev, os.R_OK):
            inventory.notes.append("udev alias claims unavailable; by-id uniqueness cannot be verified.")
        for class_name, kind in (("video4linux", "video"), ("tty", "serial")):
            directory = self.sysfs / "class" / class_name
            try:
                entries = sorted(directory.iterdir())
            except FileNotFoundError:
                continue
            except OSError as exc:
                inventory.errors.append(f"{class_name} enumeration: {exc}")
                continue
            for entry in entries:
                if kind == "video" and not re.fullmatch(r"video\d+", entry.name):
                    continue
                if kind == "serial" and not re.fullmatch(r"tty(?:USB|ACM)\d+", entry.name):
                    continue
                try:
                    real = entry.resolve()
                    device = next((devices[p] for p in real.parents if p in devices), None)
                    if device is None:
                        continue  # Internal/platform camera pipelines are outside this USB diagnostic.
                    endpoint = self._endpoint(entry, serial=kind == "serial", inventory=inventory)
                    device.endpoints.append(endpoint)
                    if device.name == device.id:
                        device.name = _read(entry / "name") or device.name
                except OSError as exc:
                    inventory.errors.append(f"{entry.name}: {exc}")
        return inventory

    def _usb_devices(self, inventory: Inventory) -> dict[Path, Device]:
        devices: dict[Path, Device] = {}
        for entry in sorted((self.sysfs / "bus/usb/devices").iterdir()):
            try:
                if ":" in entry.name or not (entry / "idVendor").exists():
                    continue
                real = entry.resolve()
                pci = next(
                    (p for p in real.parents if re.fullmatch(r"[\da-f]{4}:[\da-f]{2}:[\da-f]{2}\.[\da-f]", p.name)),
                    None,
                )
                if pci is not None and pci not in devices:
                    controller = Device(id=f"pci:{pci.name}", name=f"USB controller {pci.name}", kind="controller")
                    devices[pci] = controller
                    inventory.devices.append(controller)
                device = self._usb_device(entry)
                devices[real] = device
                inventory.devices.append(device)
            except OSError as exc:
                inventory.errors.append(f"USB {entry.name}: {exc}")
        for real, device in devices.items():
            parent = next((devices[p] for p in real.parents if p in devices), None)
            device.parent_id = parent.id if parent else None
        return devices

    @staticmethod
    def _usb_device(entry: Path) -> Device:
        root = entry.name.startswith("usb")
        interfaces = [p for p in entry.glob("*:*") if p.is_dir()]
        return Device(
            id=entry.name,
            name=f"USB root hub {entry.name}" if root else _read(entry / "product") or entry.name,
            kind="hub" if root or _read(entry / "bDeviceClass") == "09" else "device",
            speed_mbps=_speed(_read(entry / "speed")),
            vendor_id=_read(entry / "idVendor"),
            product_id=_read(entry / "idProduct"),
            serial="" if root else _read(entry / "serial"),
            root_hub=root,
            video_interface=any(_read(p / "bInterfaceClass") == "0e" for p in interfaces),
        )

    def _endpoint(self, entry: Path, *, serial: bool, inventory: Inventory) -> Endpoint:
        node = self.dev / entry.name
        endpoint = Endpoint(
            path=str(node),
            kind="serial" if serial else "video",
            readable=os.access(node, os.R_OK),
            writable=os.access(node, os.W_OK),
        )
        try:
            endpoint.aliases = self._aliases(entry, serial=serial)
        except OSError as exc:
            inventory.notes.append(f"Aliases for {node} could not be collected: {exc}")
        if not serial:
            try:
                role = self.video_role(node)
                if role in {"capture", "metadata"}:
                    endpoint.role = role
                else:
                    endpoint.query_error = "The device did not report an image or metadata capture capability."
            except OSError as exc:
                endpoint.query_error = str(exc)
        return endpoint

    def _aliases(self, entry: Path, *, serial: bool) -> list[Alias]:
        node = self.dev / entry.name
        claims = []
        properties: dict[str, str] = {}
        for line in _read(self.udev / f"c{_read(entry / 'dev')}").splitlines():
            if line.startswith("S:"):
                claims.append(str(self.dev / line[2:]))
            elif line.startswith("E:") and "=" in line:
                key, value = line[2:].split("=", 1)
                properties[key] = value
        claims.extend(properties.get("DEVLINKS", "").split())
        base = self.dev / ("serial" if serial else "v4l")
        paths = set(claims)
        for directory in (base / "by-id", base / "by-path"):
            if directory.is_dir():
                paths.update(str(p) for p in directory.iterdir() if p.resolve() == node)
        aliases = []
        for name in sorted(paths):
            path = Path(name)
            if path.parent not in {base / "by-id", base / "by-path"}:
                continue
            target = str(path.resolve()) if path.exists() else None
            aliases.append(Alias(path=name, target=target, claimed=name in claims))
        return aliases
