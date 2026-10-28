# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Terminal-safe plain-text rendering; JSON consumers use the models directly."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from physicalai.diagnostics.identity import alias_claims

if TYPE_CHECKING:
    from physicalai.diagnostics.models import Device, Endpoint, Inventory, Report, WatchEvent


def safe_text(value: object) -> str:
    """Escape terminal controls from device strings, log messages, and paths.

    Returns:
        Text that cannot inject terminal control sequences.
    """
    return "".join(char if char.isprintable() else repr(char)[1:-1] for char in str(value))


def _label(device: Device) -> str:
    speed = f" · {device.speed_mbps:g} Mb/s" if device.speed_mbps is not None else ""
    vid_pid = f" · {device.vendor_id}:{device.product_id}" if device.vendor_id else ""
    serial = f" · serial={device.serial}" if device.serial else ""
    return safe_text(f"{device.name} [{device.id}]{vid_pid}{speed}{serial}")


def _device_lines(device: Device, depth: int, *, details: bool) -> list[str]:
    prefix = "  " * depth
    lines = [prefix + _label(device)]
    if device.location:
        lines.append(prefix + "  location: " + safe_text(device.location))
    if device.kind in {"virtual", "platform"} and not details:
        lines.append(prefix + f"  {len(device.endpoints)} platform/virtual endpoints (use inspect for details)")
    else:
        for endpoint in device.endpoints:
            if endpoint.kind == "usb" and len(device.endpoints) > 1 and not details:
                continue
            lines.append(prefix + f"  {safe_text(endpoint.path)} ({endpoint.kind}/{endpoint.role})")
            if details:
                lines.extend(prefix + f"    {alias.binding}: {safe_text(alias.path)}" for alias in endpoint.aliases)
    return lines


def render_inventory(inventory: Inventory, *, tree: bool = False, details: bool = False) -> str:
    """Render grouped physical devices, not one pretend camera per video node.

    Returns:
        Plain-text inventory with collection limitations.
    """
    lines = [f"Hardware inventory · {inventory.platform} · {inventory.collected_at}"]
    if inventory.redacted:
        lines.append("REDACTED support report: displayed paths are not deployable configuration selectors.")
    children: dict[str | None, list[Device]] = defaultdict(list)
    for device in inventory.devices:
        children[device.parent_id].append(device)

    def append(device: Device, depth: int) -> None:
        lines.extend(_device_lines(device, depth, details=details))
        if tree:
            for child in children[device.id]:
                append(child, depth + 1)

    if tree:
        known_ids = {device.id for device in inventory.devices}
        for device in inventory.devices:
            if device.parent_id not in known_ids:
                append(device, 0)
    else:
        for device in inventory.devices:
            if details or device.kind not in {"hub", "controller"}:
                append(device, 0)
    if not inventory.devices:
        lines.append("No devices enumerated. Inspect collection errors before concluding hardware is absent.")
    lines.append("Use 'pai devices inspect <device-id-or-endpoint>' for formats, access, and alias claims.")
    if inventory.errors:
        counts = Counter(
            error.source.split("/", 2)[1] if error.source.startswith("/") else error.source
            for error in inventory.errors
        )
        lines.append(
            f"Collection limitations: {len(inventory.errors)} ({safe_text(dict(counts))}); see 'pai doctor' or --json."
        )
    return "\n".join(lines)


def _inspect_endpoint(endpoint: Endpoint, claims: dict[str, list[Endpoint]]) -> list[str]:
    lines = [
        f"\n{safe_text(endpoint.path)} · {endpoint.kind}/{endpoint.role}",
        f"  Readable: {endpoint.readable} · Writable: {endpoint.writable}",
    ]
    if endpoint.interface:
        lines.append("  USB interface: " + safe_text(endpoint.interface))
    if endpoint.holders:
        lines.extend(f"  Open by pid={holder.pid} {safe_text(holder.name)}" for holder in endpoint.holders)
    else:
        lines.append("  No opener observed (visibility is limited; not proof of idleness).")
    for alias in endpoint.aliases:
        count = len(claims.get(alias.path, []))
        warning = f" · COLLISION: {count} claims" if count > 1 else ""
        lines.extend([
            f"  {alias.binding}-bound{warning}: {safe_text(alias.path)}",
            f"    target={safe_text(alias.target)} · evidence={safe_text(alias.source)}",
        ])
    modes: dict[tuple[str, int, int], set[float]] = defaultdict(set)
    for mode in endpoint.modes:
        modes[mode.pixel_format, mode.width, mode.height].add(mode.fps)
    if modes:
        lines.append("  Advertised wire modes (not measured throughput):")
        for (pixel_format, width, height), rates in sorted(modes.items()):
            fps = ", ".join(f"{rate:g}" for rate in sorted(rates, reverse=True))
            lines.append(f"    {safe_text(pixel_format)} {width}x{height} @ {fps} FPS")
    return lines


def render_inspection(inventory: Inventory, device: Device, endpoint: Endpoint | None = None) -> str:
    """Explain one device/endpoint without duplicating its entire JSON payload.

    Returns:
        Terminal-safe identities, aliases, access evidence, and grouped wire modes.
    """
    lines = [
        _label(device),
        "Location: " + safe_text(device.location),
        "Reported serial: " + safe_text(device.serial or "none; physical unit is not identified"),
    ]
    if inventory.redacted:
        lines.append("REDACTED support output; do not use displayed paths as configuration.")
    lines.extend(
        f"{name}: {safe_text(device.properties[name])}"
        for name in ("controller", "usb_path", "bus_number", "device_number", "interface_drivers", "authorized")
        if device.properties.get(name)
    )
    claims = alias_claims(inventory)
    for selected in [endpoint] if endpoint is not None else device.endpoints:
        lines.extend(_inspect_endpoint(selected, claims))
    paths = {endpoint.path} if endpoint is not None else {ep.path for ep in device.endpoints}
    lines.extend(
        f"Query limitation: {safe_text(error.source)}: {safe_text(error.message)}"
        for error in inventory.errors
        if error.essential or error.source in paths
    )
    lines.append("\nPort binding follows a slot, not a physical unit. Backend format support can differ.")
    return "\n".join(lines)


def render_report(report: Report) -> str:
    """Render findings, remedies, and an honest summary of incomplete checks.

    Returns:
        A complete plain-text assessment.
    """
    scope = (
        "passive; no streams or robot connections"
        if report.mode == "passive"
        else ("includes explicitly requested camera streaming tests; no robot connections")
    )
    lines = [render_inventory(report.inventory), "", f"Health assessment ({scope})"]
    collection = [f for f in report.findings if f.check_id == "collection.error"]
    for finding in report.findings:
        if finding.check_id == "collection.error":
            continue
        lines.append(f"{finding.severity:4}  {safe_text(finding.summary)} [{safe_text(finding.subject)}]")
        lines.extend(f"      {safe_text(evidence)}" for evidence in finding.evidence)
        if finding.action:
            lines.append(f"      Next: {safe_text(finding.action)}")
        if finding.verification:
            lines.append(f"      Verify: {safe_text(finding.verification)}")
    if collection:
        lines.append(f"SKIP  {len(collection)} collection queries were incomplete (details in --json).")
        lines.extend(f"      {safe_text(finding.subject)}: {safe_text(finding.summary)}" for finding in collection[:3])
    if report.inventory.events:
        lines.append("\nUSB-related system messages (temporal evidence, not proof of causation):")
        lines.extend(
            f"      {safe_text(event.timestamp)} {'[history]' if event.historical else '[new]'} "
            f"{safe_text(event.message)}"
            for event in report.inventory.events[-5:]
        )
    counts = Counter(f.severity for f in report.findings)
    lines.append("\n" + " · ".join(f"{count} {severity}" for severity, count in counts.items()))
    return "\n".join(lines)


def render_event(event: WatchEvent) -> str:
    """Render watch changes without dumping an entire inventory each second.

    Returns:
        A compact event description.
    """
    if event.kind == "snapshot":
        return (
            f"Watching for changes; see device_events capability for notification coverage. Baseline: {event.timestamp}"
        )
    summary = event.data.get("summary", event.data.get("message", event.data.get("name", "")))
    if event.kind == "finding_resolved":
        summary = f"No longer observed ({event.data.get('resolution', 'evidence changed')}): {summary}"
    if event.kind == "final":
        summary = f"exit={event.data['exit_code']}"
    text = f"{event.timestamp} {event.kind}: {safe_text(event.subject)} {safe_text(summary)}".rstrip()
    if event.kind == "finding_opened" and event.data.get("action"):
        text += "\n  Next: " + safe_text(event.data["action"])
    return text
