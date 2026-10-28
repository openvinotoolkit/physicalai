# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Terminal-safe plain-text rendering; JSON consumers use the models directly."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from physicalai.diagnostics.identity import alias_claims
from physicalai.diagnostics.overview import device_nodes, render_payloads, render_shared_links, render_tree
from physicalai.diagnostics.text import safe_text, table

__all__ = ["render_event", "render_inspection", "render_inventory", "render_report", "safe_text"]

if TYPE_CHECKING:
    from physicalai.diagnostics.models import Device, Endpoint, Finding, Inventory, Report, WatchEvent


def _label(device: Device) -> str:
    speed = f" · {device.speed_mbps:g} Mb/s" if device.speed_mbps is not None else ""
    vid_pid = f" · {device.vendor_id}:{device.product_id}" if device.vendor_id else ""
    serial = f" · serial={device.serial}" if device.serial else ""
    return safe_text(f"{device.name} [{device.id}]{vid_pid}{speed}{serial}")


def render_inventory(inventory: Inventory, *, tree: bool = True, details: bool = False) -> str:
    """Render grouped physical devices, not one pretend camera per video node.

    Returns:
        Plain-text inventory with collection limitations.
    """
    lines = [f"Hardware inventory · {inventory.platform} · {inventory.collected_at}"]
    if inventory.redacted:
        lines.append("REDACTED support report: displayed paths are not deployable configuration selectors.")
    lines.extend(["", render_payloads(inventory), ""])
    if tree:
        lines.append(render_tree(inventory, details=details))
    else:
        rows = [[d.name, device_nodes(d), d.location] for d in inventory.devices if details or d.kind == "device"]
        lines.append(table(["Device", "Nodes", "Location"], rows))
    if not inventory.devices:
        lines.append("No devices enumerated. Inspect collection errors before concluding hardware is absent.")
    lines.append("Use 'pai devices inspect <device-id-or-endpoint>' for formats, access, and alias claims.")
    if inventory.errors:
        lines.append(
            f"Collection limitations: {len(inventory.errors)} queries incomplete; see doctor --verbose or --json."
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
    lines.extend([
        "",
        render_payloads(inventory, endpoint=endpoint)
        if endpoint
        else render_payloads(inventory.model_copy(update={"devices": [device]})),
        "",
        render_tree(inventory, focus_id=device.id),
        "",
        render_shared_links(inventory, device.id),
        "",
        "Identity and endpoint details",
    ])
    lines.extend(
        f"{name}: {safe_text(device.properties[name])}"
        for name in (
            "controller",
            "usb_path",
            "bus_number",
            "device_number",
            "usb_version",
            "interface_drivers",
            "authorized",
        )
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


def _affected(report: Report, finding: Finding) -> list[str]:
    if finding.check_id == "identity.alias_collision":
        return finding.evidence
    for device in report.inventory.devices:
        if finding.subject == device.id:
            return [device_nodes(device) or device.id]
        for endpoint in device.endpoints:
            if finding.subject == endpoint.id:
                return [endpoint.path]
    return [finding.subject]


def _show_finding(finding: Finding, comparisons: set[str], *, verbose: bool) -> bool:
    if finding.severity == "SKIP" or finding.check_id == "collection.error":
        return False
    if verbose:
        return True
    if finding.check_id in {"access.available", "usb.usb2_camera", "usb.usbfs_memory"} or finding.severity == "PASS":
        return False
    if finding.severity == "INFO" and finding.check_id != "stream.comparison":
        return False
    return not (
        finding.check_id in {"stream.individual", "stream.concurrent"}
        and finding.severity != "FAIL"
        and finding.subject in comparisons
    )


def _group_lines(report: Report, severity: str, check_id: str, findings: list[Finding], *, verbose: bool) -> list[str]:
    lines: list[str] = []
    affected = sorted({subject for finding in findings for subject in _affected(report, finding)})
    summary = findings[0].summary
    if check_id == "identity.alias_collision":
        summary = f"{len(findings)} ambiguous alias name(s); do not use them to select a device."
    elif check_id == "identity.no_serial":
        summary = f"{len(findings)} device(s) have no reported serial; use port-bound selectors."
    lines.extend([f"  {severity}: {safe_text(summary)}", "    Affects: " + safe_text(", ".join(affected))])
    if not verbose and findings[0].action and check_id not in {"identity.alias_collision", "identity.no_serial"}:
        lines.append("    Next: " + safe_text(findings[0].action))
    if verbose:
        for finding in findings:
            lines.extend(f"    {safe_text(value)}" for value in finding.evidence)
            if finding.action:
                lines.append("    Next: " + safe_text(finding.action))
            if finding.verification:
                lines.append("    Verify: " + safe_text(finding.verification))
    return lines


def _attention(report: Report, *, verbose: bool) -> list[str]:
    groups: dict[tuple[str, str], list[Finding]] = defaultdict(list)
    comparisons = {f.subject for f in report.findings if f.check_id == "stream.comparison"}
    for finding in report.findings:
        if _show_finding(finding, comparisons, verbose=verbose):
            groups[finding.severity, finding.check_id].append(finding)
    lines = ["Needs attention" if not verbose else "All checks"]
    order = {"FAIL": 0, "WARN": 1, "INFO": 2, "PASS": 3}
    for (severity, check_id), findings in sorted(groups.items(), key=lambda item: (order[item[0][0]], item[0][1])):
        lines.extend(_group_lines(report, severity, check_id, findings, verbose=verbose))
    if len(lines) == 1:
        lines.append("  No additional failed/warning checks; unavailable evidence is listed below.")
    return lines


def render_report(report: Report, *, verbose: bool = False) -> str:
    """Render findings, remedies, and an honest summary of incomplete checks.

    Returns:
        A complete plain-text assessment.
    """
    scope = (
        "passive; no streams or robot connections"
        if report.mode == "passive"
        else ("includes explicitly requested camera streaming tests; no robot connections")
    )
    counts = Counter(f.severity for f in report.findings)
    headline = " · ".join(f"{counts[severity]} {severity}" for severity in ("FAIL", "WARN", "PASS", "SKIP"))
    lines = [
        f"Doctor: {headline}",
        f"Scope: {scope}",
        "",
        render_payloads(report.inventory, report.findings),
        "",
        render_tree(report.inventory, details=verbose, findings=report.findings),
        "",
        *_attention(report, verbose=verbose),
    ]
    if report.inventory.redacted:
        lines.insert(1, "REDACTED support report; displayed selectors are not deployable configuration.")
    skipped = [finding for finding in report.findings if finding.severity == "SKIP"]
    if skipped:
        lines.append(
            f"\nVisibility: {len(skipped)} checks/queries incomplete or unsupported; "
            "use --verbose or --json for details."
        )
        if verbose:
            lines.extend(f"  {safe_text(f.subject)}: {safe_text(f.summary)}" for f in skipped)
    if verbose and report.inventory.events:
        lines.append("\nUSB-related system messages (temporal evidence, not proof of causation):")
        lines.extend(
            f"      {safe_text(event.timestamp)} {'[history]' if event.historical else '[new]'} "
            f"{safe_text(event.message)}"
            for event in report.inventory.events[-5:]
        )
    if "usbfs_memory_mb" in report.inventory.settings:
        lines.append(
            f"\nHost usbfs buffer cap: {safe_text(report.inventory.settings['usbfs_memory_mb'])} MiB; "
            "not USB link capacity."
        )
    lines.append("Details: pai devices inspect <image/serial-node> · Full findings: pai doctor --verbose")
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
