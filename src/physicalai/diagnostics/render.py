# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Terminal-only views; application callers consume the typed report instead."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import TYPE_CHECKING

from physicalai.diagnostics.constants import USB2_HIGH_SPEED_MBPS, USB3_SUPER_SPEED_MBPS

if TYPE_CHECKING:
    from physicalai.diagnostics.models import Device, Endpoint, Inventory, Report

_LINK_RATE = re.compile(r"\b\d+(?:\.\d+)? Mb/s(?: \(USB (?:2|3\.x)\))?")
_HEADINGS = frozenset({
    "USB connections",
    "Connection notes",
    "Collection limitations",
    "Other devices on shared upstream hubs:",
})


def _style(text: str) -> str:
    # Style only after escaping device text and measuring the plain table widths.
    def link(match: re.Match[str]) -> str:
        value = match.group()
        speed = float(value.split()[0])
        code = "1;33" if speed == USB2_HIGH_SPEED_MBPS else "1;36" if speed >= USB3_SUPER_SPEED_MBPS else "1"
        return f"\x1b[{code}m{value}\x1b[0m"

    lines = []
    for raw_line in text.split("\n"):
        if raw_line in _HEADINGS or raw_line.startswith(("Device ", "USB connection diagnostics (")):
            lines.append(f"\x1b[1m{raw_line}\x1b[0m")
            continue
        line = _LINK_RATE.sub(link, raw_line)
        for severity, code in (("ERROR", "1;31"), ("WARNING", "1;33"), ("INFO", "1;34")):
            if line.startswith(f"  {severity}:"):
                line = line.replace(f"{severity}:", f"\x1b[{code}m{severity}:\x1b[0m", 1)
                break
        lines.append(line)
    return "\n".join(lines)


def _safe(value: str) -> str:
    return "".join(f"\\u{ord(c):04x}" if unicodedata.category(c).startswith("C") else c for c in value)


def _speed(device: Device) -> str:
    if device.speed_mbps is None:
        return "unknown"
    suffix = (
        " (USB 2)"
        if device.speed_mbps == USB2_HIGH_SPEED_MBPS
        else " (USB 3.x)"
        if device.speed_mbps >= USB3_SUPER_SPEED_MBPS
        else ""
    )
    return f"{device.speed_mbps:g} Mb/s{suffix}"


def _nodes(device: Device) -> str:
    return ", ".join(ep.path for ep in device.endpoints if ep.relevant) or "no endpoint"


def _tree(inventory: Inventory, devices: list[Device]) -> list[str]:
    visible = {d.id for d in devices}
    for device in devices:
        visible.update(p.id for p in inventory.ancestors(device))
    children: dict[str | None, list[Device]] = defaultdict(list)
    for device in inventory.devices:
        if device.id in visible:
            parent = device.parent_id if device.parent_id in visible else None
            children[parent].append(device)
    lines = ["USB connections"]
    seen: set[str] = set()

    def append(device: Device, prefix: str, *, last: bool) -> None:
        if device.id in seen:
            return
        seen.add(device.id)
        branch, continuation = ("└── ", "    ") if last else ("├── ", "│   ")
        label = f"{device.name} [{device.id}]"
        if device.kind != "controller":
            label += " — " + _speed(device)
        if device.root_hub:
            label += "; root ports are separate links"
        elif device.kind == "device":
            label += " — " + _nodes(device)
        lines.append(prefix + branch + _safe(label))
        group = children.get(device.id, [])
        for index, child in enumerate(group):
            append(child, prefix + continuation, last=index == len(group) - 1)

    roots = children.get(None, [])
    for index, root in enumerate(roots):
        append(root, "", last=index == len(roots) - 1)
    # Keep malformed/cyclic ancestry visible without recursing forever.
    for device in inventory.devices:
        if device.id in visible and device.id not in seen:
            append(device, "  [unresolved ancestry] ", last=True)
    return lines


def _endpoint_lines(inventory: Inventory, device: Device, endpoint: Endpoint) -> list[str]:
    role = endpoint.role if endpoint.kind == "video" else "serial adapter"
    access = "yes" if endpoint.readable and endpoint.writable else "no"
    lines = [f"  {_safe(endpoint.path)} ({role}); read/write access: {access}"]
    aliases = inventory.unique_aliases(device, endpoint)
    if endpoint.relevant and aliases:
        preferred = aliases[0]
        binding = "device" if "/by-id/" in preferred.path else "port"
        lines.append(f"    Unique {binding}-bound path: {_safe(preferred.path)}")
    elif endpoint.relevant:
        lines.append("    No verified unique alias; this endpoint path may change after reconnect/reboot.")
    usable = {a.path for a in aliases}
    lines.extend(
        f"    Unverified/ambiguous alias: {_safe(alias.path)}" for alias in endpoint.aliases if alias.path not in usable
    )
    if endpoint.query_error:
        lines.append("    Capability query: " + _safe(endpoint.query_error))
    if endpoint.kind == "video" and endpoint.role == "metadata":
        lines.append("    Metadata only; select an image endpoint, not this node, for camera capture.")
    return lines


def _details(report: Report, device: Device) -> list[str]:
    inventory = report.inventory
    lines = [
        f"{_safe(device.name)} [{_safe(device.id)}] — {_speed(device)}",
        (
            f"USB ID: {_safe(device.vendor_id)}:{_safe(device.product_id)}"
            f" · Reported serial: {_safe(device.serial) or 'none'}"
        ),
    ]
    route = [*reversed(inventory.ancestors(device)), device]
    lines.append("USB path: " + " → ".join(_safe(d.id) for d in route))
    hubs = {d.id for d in route if d.kind == "hub" and not d.root_hub}
    peers = [
        d
        for d in inventory.devices
        if d.kind == "device" and d.id != device.id and hubs.intersection(p.id for p in inventory.ancestors(d))
    ]
    if peers:
        lines.append("Other devices on shared upstream hubs:")
        lines.extend(f"  {_safe(d.name)} [{_safe(d.id)}] — {_safe(_nodes(d))}" for d in peers)
    for endpoint in device.endpoints:
        if report.selected_endpoint is None or endpoint.path == report.selected_endpoint:
            lines.extend(_endpoint_lines(inventory, device, endpoint))
    lines.append("Port-bound paths follow the slot, not the physical unit. Backend selector support can differ.")
    return lines


def render_report(report: Report, *, tree: bool = False, ansi: bool = False) -> str:
    """Render a compact overview or focused inspection, with terminal-safe text.

    Args:
        report: The same data returned to application/API callers.
        tree: Show relevant USB branches instead of the default device table.
        ansi: Add terminal colors and bold text. Defaults to plain text; the
            caller decides terminal support. Does not change the report data.

    Returns:
        Terminal-safe text with actionable findings and collection limitations.
    """
    inventory = report.inventory
    devices = [d for d in inventory.devices if d.relevant]
    selected = next((d for d in inventory.devices if d.id == report.selected_device_id), None)
    platform = {"linux": "Linux", "darwin": "macOS", "win32": "Windows"}.get(report.platform, report.platform)
    lines = [f"USB connection diagnostics ({_safe(platform)})", ""]
    if selected is not None:
        lines.extend(_details(report, selected))
    if tree:
        lines.extend(_tree(inventory, [selected] if selected else devices))
    elif selected is None and devices:
        rows = [[d.name, _nodes(d), _speed(d)] for d in devices]
        rows = [[_safe(cell) for cell in row] for row in [["Device", "Endpoint", "USB link"], *rows]]
        widths = [max(len(row[i]) for row in rows) for i in range(3)]
        lines.extend("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows)
    if not devices and selected is None:
        lines.append("No USB camera/serial endpoints found. Check collection limitations and device connections.")
    if report.platform == "linux":
        memory = inventory.usbfs_memory_mb
        limit = "unavailable" if memory is None else "unlimited (0)" if memory == 0 else f"{memory} MB"
        lines.extend([
            "",
            f"Host USB transfer memory: usbcore.usbfs_memory_mb = {limit} (system-wide)",
            "  Applies to usbfs/libusb transfers, not USB bandwidth; camera backend use is unknown.",
            "  Troubleshooting: https://github.com/openvinotoolkit/physicalai/blob/main/docs/how-to/cli/doctor.md#usb-transfer-memory",
        ])
    if report.findings:
        lines.extend(["", "Connection notes"])
        for finding in report.findings:
            lines.append(f"  {finding.severity.upper()}: {_safe(finding.message)}")
            if finding.hint:
                lines.append("    " + _safe(finding.hint))
    if inventory.errors or inventory.notes:
        lines.extend(["", "Collection limitations"])
        lines.extend("  " + _safe(error) for error in [*inventory.errors, *inventory.notes])
    lines.extend([
        "",
        "Passive inspection only; streaming and robot operation were not tested.",
        "USB link speed is not measured throughput; USB 2 is not necessarily a fault.",
    ])
    if selected is None:
        lines.append("Details: pai doctor <endpoint-or-device-id> · Topology: pai doctor --tree")
    text = "\n".join(lines)
    return _style(text) if ansi else text
