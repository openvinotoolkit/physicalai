# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Linux USB/endpoint inventory from sysfs and udev, without vendor SDKs."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess  # noqa: S404 - fixed read-only argv; no shell or device-supplied commands
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from physicalai.diagnostics.models import (
    Alias,
    Capability,
    CollectionError,
    Device,
    Endpoint,
    Inventory,
    SystemEvent,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_USB_LOG = re.compile(
    r"\b(?:usb|uvcvideo|xhci(?:_hcd)?|ch341|usbfs|urb|cdc_acm|cp210x|ftdi_sio|usbserial)\b",
    re.IGNORECASE,
)


def read_attribute(path: Path) -> str:
    """Read an optional sysfs attribute; disappearance during unplug is normal.

    Returns:
        Attribute contents, or an empty string when unavailable.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _number(value: str) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except ValueError:
        return None


def _pci_parent(path: Path) -> Path | None:
    return next(
        (parent for parent in path.parents if re.fullmatch(r"[\da-f]{4}:[\da-f]{2}:[\da-f]{2}\.[\da-f]", parent.name)),
        None,
    )


def _controller_device(path: Path, *, pci: bool) -> Device:
    driver = path / "driver"
    return Device(
        id=f"pci:{path.name}" if pci else f"platform-controller:{path}",
        name=f"USB host controller {path.name}",
        kind="controller",
        location=f"pci-{path.name}" if pci else str(path),
        vendor_id=read_attribute(path / "vendor").removeprefix("0x"),
        product_id=read_attribute(path / "device").removeprefix("0x"),
        driver=driver.resolve().name if driver.exists() else "",
        properties={"sysfs": str(path)},
    )


def _journal_output(command: list[str]) -> str:
    completed = subprocess.run(  # noqa: S603 - fixed journal argv, no shell or device-supplied commands
        command,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=3,
        check=False,
        env={**os.environ, "LC_ALL": "C"},
    )
    if completed.returncode:
        msg = completed.stderr.strip()[:500] or "Kernel journal is not accessible."
        raise OSError(msg)
    if "permission" in completed.stderr.lower() or "not seeing" in completed.stderr.lower():
        msg = "Kernel journal visibility is restricted for this user."
        raise PermissionError(msg)
    return completed.stdout


class LinuxCollector:
    """Collect USB topology and every serial/video/HID/media endpoint.

    Paths and video probing are injectable so tests need no hardware or root.
    """

    def __init__(
        self,
        *,
        sysfs: Path = Path("/sys"),
        dev: Path = Path("/dev"),
        udev: Path = Path("/run/udev/data"),
        logs: bool = True,
        probe_video: Callable[[Endpoint], None] | None = None,
        proc: Path | None = Path("/proc"),
    ) -> None:
        """Set filesystem roots and optional read-only video query injection."""
        self.sysfs = sysfs
        self.dev = dev
        self.udev = udev
        self.logs = logs
        self.probe_video = probe_video
        self.proc = proc
        self._cursor: str | None = None
        self._seen_events: deque[str] = deque(maxlen=512)

    def collect(self) -> Inventory:
        """Return a snapshot including collection failures, never hidden devices."""
        result = Inventory(platform="linux")
        result.capabilities = {
            "usb_topology": Capability(status="available", detail="sysfs"),
            "device_events": Capability(status="available", detail="Polling; brief unplug/replug can be missed."),
            "video_formats": Capability(status="available", detail="Read-only V4L2 queries; no streams."),
        }
        if not self.udev.is_dir() or not os.access(self.udev, os.R_OK):
            result.capabilities["alias_claims"] = Capability(
                status="unavailable",
                detail="Full udev alias claims are unavailable; symlink winners can hide collisions.",
            )
        usb_dir = self.sysfs / "bus/usb/devices"
        devices: dict[Path, Device] = {}
        try:
            usb_entries = sorted(usb_dir.iterdir())
        except OSError as exc:
            usb_entries = []
            result.errors.append(CollectionError(source="sysfs.usb", message=str(exc), essential=True))
            result.capabilities["usb_topology"] = Capability(status="unavailable", detail=str(exc))
        for entry in usb_entries:
            if ":" in entry.name or not (entry / "idVendor").exists():
                continue
            real = entry.resolve()
            pci_parent = _pci_parent(real)
            controller_path = pci_parent or (real.parent if entry.name.startswith("usb") else None)
            if controller_path is not None and controller_path not in devices:
                controller = _controller_device(controller_path, pci=pci_parent is not None)
                devices[controller_path] = controller
                result.devices.append(controller)
            device = self._usb_device(entry)
            devices[real] = device
            result.devices.append(device)
        for real, device in devices.items():
            parent = next((devices[p] for p in real.parents if p in devices), None)
            device.parent_id = parent.id if parent else None
        self._endpoints(result, devices)
        if self.proc is not None:
            from physicalai.diagnostics.platforms.processes import collect_holders  # noqa: PLC0415

            collect_holders(result, self.proc)
        memory = read_attribute(self.sysfs / "module/usbcore/parameters/usbfs_memory_mb")
        if memory:
            result.settings["usbfs_memory_mb"] = memory
        result.settings["kernel"] = os.uname().release
        if self.logs:
            self._logs(result)
        else:
            result.capabilities["system_logs"] = Capability(status="unavailable", detail="Log collection disabled.")
        return result

    def _usb_device(self, entry: Path) -> Device:
        real = entry.resolve()
        parent = _pci_parent(real)
        pci = parent.name if parent else ""
        root = entry.name.startswith("usb")
        port = read_attribute(entry / "devpath")
        driver_link = entry / "driver"
        interfaces = [path for path in entry.glob("*:*") if path.is_dir()]
        classes = sorted({read_attribute(path / "bInterfaceClass") for path in interfaces} - {""})
        drivers = sorted({(path / "driver").resolve().name for path in interfaces if (path / "driver").exists()})
        device = Device(
            id=f"usb:{entry.name}",
            name=f"USB root hub {entry.name}" if root else read_attribute(entry / "product") or entry.name,
            kind="hub" if root or read_attribute(entry / "bDeviceClass") == "09" else "device",
            location=f"pci-{pci}-usb-port-{port}" if pci else entry.name,
            vendor_id=read_attribute(entry / "idVendor"),
            product_id=read_attribute(entry / "idProduct"),
            serial=None if root else read_attribute(entry / "serial") or None,
            speed_mbps=_number(read_attribute(entry / "speed")),
            driver=driver_link.resolve().name if driver_link.exists() else "",
            properties={
                key: value
                for key, value in {
                    "sysfs": str(real),
                    "root_hub": "true" if root else "false",
                    "usb_path": entry.name,
                    "bus_number": read_attribute(entry / "busnum"),
                    "device_number": read_attribute(entry / "devnum"),
                    "controller": pci,
                    "interface_classes": ",".join(classes),
                    "interface_drivers": ",".join(drivers),
                    "authorized": read_attribute(entry / "authorized"),
                    "manufacturer": read_attribute(entry / "manufacturer"),
                    "power_control": read_attribute(entry / "power/control"),
                    "runtime_status": read_attribute(entry / "power/runtime_status"),
                    "declared_max_power": read_attribute(entry / "bMaxPower"),
                }.items()
                if value
            },
        )
        bus, number = read_attribute(entry / "busnum"), read_attribute(entry / "devnum")
        if bus.isdecimal() and number.isdecimal():
            path = self.dev / "bus/usb" / f"{int(bus):03}" / f"{int(number):03}"
            device.endpoints.append(
                Endpoint(
                    id=f"{device.id}:usb",
                    path=str(path),
                    kind="usb",
                    readable=os.access(path, os.R_OK),
                    writable=os.access(path, os.W_OK),
                )
            )
        return device

    def _properties(self, entry: Path) -> tuple[dict[str, str], list[str]]:
        major_minor = read_attribute(entry / "dev")
        properties: dict[str, str] = {}
        claims: list[str] = []
        for line in read_attribute(self.udev / f"c{major_minor}").splitlines():
            if line.startswith("E:") and "=" in line:
                key, value = line[2:].split("=", 1)
                properties[key] = value
            elif line.startswith("S:"):
                claims.append(str(self.dev / line[2:]))
        claims.extend(properties.get("DEVLINKS", "").split())
        return properties, claims

    def _aliases(self, node: Path, claims: list[str], kind: str) -> list[Alias]:
        udev_claims = set(claims)
        base = self.dev / ("serial" if kind == "serial" else "v4l" if kind == "video" else "input")
        for directory in (base / "by-id", base / "by-path"):
            try:
                claims.extend(str(link) for link in directory.iterdir() if link.resolve() == node.resolve())
            except OSError:
                continue
        aliases = []
        for name in sorted(set(claims)):
            if "/by-id/" not in name and "/by-path/" not in name:
                continue
            path = Path(name)
            aliases.append(
                Alias(
                    path=name,
                    binding="port" if "/by-path/" in name else "device",
                    target=str(path.resolve()) if path.exists() else None,
                    source="udev" if name in udev_claims else "symlink-target",
                )
            )
        return aliases

    def _endpoints(self, result: Inventory, devices: dict[Path, Device]) -> None:
        for class_name, kind in (("video4linux", "video"), ("tty", "serial"), ("hidraw", "hid"), ("media", "media")):
            directory = self.sysfs / "bus/media/devices" if kind == "media" else self.sysfs / "class" / class_name
            try:
                entries = sorted(directory.iterdir())
            except FileNotFoundError:
                continue
            except OSError as exc:
                result.errors.append(CollectionError(source=class_name, message=str(exc)))
                continue
            for entry in entries:
                if kind == "serial" and not entry.name.startswith(("ttyUSB", "ttyACM")):
                    continue
                real = entry.resolve()
                parent_path = next((p for p in (real, *real.parents) if p in devices), None)
                device = devices.get(parent_path) if parent_path else None
                if device is None:
                    # Preserve platform/virtual cameras; do not pretend they are USB.
                    if kind not in {"video", "media"}:
                        continue
                    link = entry / "device"
                    parent_path = link.resolve() if link.exists() else real if kind == "video" else real.parent
                    if parent_path not in devices:
                        devices[parent_path] = Device(
                            id=f"platform:{parent_path}",
                            name=read_attribute(entry / "name") or parent_path.name,
                            kind="virtual" if "/devices/virtual/" in str(parent_path) else "platform",
                            location=str(parent_path),
                        )
                        result.devices.append(devices[parent_path])
                    device = devices[parent_path]
                properties, claims = self._properties(entry)
                node = self.dev / entry.name
                interface = next(
                    (
                        p.name.split(":", 1)[1]
                        for p in real.parents
                        if ":" in p.name and (p / "bInterfaceNumber").exists()
                    ),
                    "",
                )
                index = read_attribute(entry / "index") or entry.name
                endpoint = Endpoint(
                    id=f"{device.id}:{kind}:{interface}:{index}",
                    path=str(node),
                    kind=kind,
                    role="capture" if ":capture:" in properties.get("ID_V4L_CAPABILITIES", "") else "unknown",
                    interface=interface,
                    aliases=self._aliases(node, claims, kind),
                    readable=os.access(node, os.R_OK),
                    writable=os.access(node, os.W_OK),
                    properties={k: v for k, v in properties.items() if k.startswith("ID_")},
                )
                if kind == "video" and entry.name.startswith("video"):
                    try:
                        if self.probe_video:
                            self.probe_video(endpoint)
                        else:
                            from physicalai.diagnostics.platforms.video import probe_video  # noqa: PLC0415

                            probe_video(endpoint)
                    except (OSError, ValueError) as exc:
                        result.errors.append(CollectionError(source=str(node), message=str(exc)))
                device.endpoints.append(endpoint)

    def _logs(self, result: Inventory) -> None:
        command = ["journalctl", "-k", "-b", "--no-pager", "-o", "json", "-n", "100"]
        historical = self._cursor is None
        if self._cursor:
            command.extend(["--after-cursor", self._cursor])
        try:
            for line in _journal_output(command).splitlines():
                record = json.loads(line)
                cursor = record.get("__CURSOR", "")
                if cursor:
                    self._cursor = cursor
                message = record.get("MESSAGE", "")
                event_id = cursor or line
                if isinstance(message, str) and _USB_LOG.search(message) and event_id not in self._seen_events:
                    self._seen_events.append(event_id)
                    at = record.get("__REALTIME_TIMESTAMP")
                    recorded_at = datetime.fromtimestamp(int(at) / 1_000_000, UTC).isoformat() if at else "unknown"
                    result.events.append(
                        SystemEvent(
                            id=event_id,
                            source="journal",
                            timestamp=recorded_at,
                            message=message[:2000],
                            historical=historical,
                        )
                    )
            result.capabilities["system_logs"] = Capability(
                status="available", detail="Latest 100 kernel records at baseline; cursor-based changes afterward."
            )
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            self._cursor = None  # Recover after rotation/permission changes; replay is deduplicated and historical.
            result.capabilities["system_logs"] = Capability(status="unavailable", detail=str(exc))
            result.errors.append(CollectionError(source="journal", message=str(exc)))
