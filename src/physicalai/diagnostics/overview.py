# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Question-oriented topology and throughput views shared by CLI commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.diagnostics.bandwidth import format_ceiling, format_link, format_mode, format_payload, payload_demand
from physicalai.diagnostics.identity import resolve_endpoint
from physicalai.diagnostics.text import safe_text, table
from physicalai.diagnostics.topology import USBTopology, port_number

if TYPE_CHECKING:
    from physicalai.diagnostics.models import Device, Endpoint, Finding, Inventory

_NODE_PREVIEW = 2


def device_nodes(device: Device) -> str:
    """Return concise useful endpoint names, not a dump of metadata/raw nodes."""
    serial = [ep.path for ep in device.endpoints if ep.kind == "serial"]
    captures = [ep.path for ep in device.endpoints if ep.role == "capture"]
    if serial:
        return ", ".join(serial)
    if captures:
        shown = ", ".join(captures[:_NODE_PREVIEW])
        return shown + (f" +{len(captures) - _NODE_PREVIEW} image nodes" if len(captures) > _NODE_PREVIEW else "")
    hid = [ep.path for ep in device.endpoints if ep.kind == "hid"]
    return ", ".join(hid[:_NODE_PREVIEW])


def _tree_label(device: Device, topology: USBTopology, *, selected: bool) -> str:
    name = safe_text(device.name)
    marker = "  <selected>" if selected else ""
    if device.kind == "controller":
        return name + marker
    port = port_number(device)
    prefix = f"Port {port}: " if port else ""
    rate = format_link(device.speed_mbps)
    if device.properties.get("root_hub") == "true":
        maximum = rate.removesuffix(" link")
        return f"{name} — {maximum} root capability; root ports independent"
    if device.kind == "hub":
        count = len(topology.shared_devices(device.id))
        sharing = "1 device on uplink" if count == 1 else f"{count} devices share uplink" if count else "empty"
        return f"{prefix}{name} [{safe_text(device.id)}] — {rate}; {sharing}" + marker
    nodes = device_nodes(device)
    return f"{prefix}{name} — {safe_text(nodes) + '; ' if nodes else ''}{rate}" + marker


def _focus(topology: USBTopology, device_id: str) -> tuple[set[str], set[str]]:
    path = topology.route(device_id)
    expanded = {device.id for device in path}
    for hub in topology.external_hubs(device_id):
        expanded.update(device.id for device in topology.descendants(hub.id))
    visible = set(expanded)
    for device in path:
        visible.update(child.id for child in topology.children.get(device.id, []))
    return visible, expanded


def _sum_payload(findings: list[Finding], *, received: bool) -> float | None:
    total = 0.0
    for finding in findings:
        value = finding.details.get("payload_demand_bytes_per_second")
        if received:
            metrics = finding.details.get("metrics")
            value = metrics.get("payload_bytes_per_second") if isinstance(metrics, dict) else None
        if not isinstance(value, (int, float)):
            return None
        total += value
    return total


def _hub_payloads(inventory: Inventory, topology: USBTopology, findings: list[Finding]) -> dict[str, str]:
    samples = [f for f in findings if f.check_id == "stream.concurrent"]
    if not samples:
        samples = [f for f in findings if f.check_id == "stream.individual"]
        if len(samples) != 1:
            return {}  # Individual trials occur at different times; never sum them.
    if any(f.severity == "FAIL" for f in samples):
        return {}
    grouped: dict[str, list[Finding]] = {}
    for finding in samples:
        try:
            device, _ = resolve_endpoint(inventory, finding.subject)
        except ValueError:
            continue
        for hub in topology.external_hubs(device.id):
            grouped.setdefault(hub.id, []).append(finding)
    return {
        hub_id: f"{len(group)} tested stream(s): need {format_payload(_sum_payload(group, received=False))} est.; "
        f"getting {format_payload(_sum_payload(group, received=True))}; other traffic unmeasured"
        for hub_id, group in grouped.items()
    }


def render_tree(
    inventory: Inventory, *, details: bool = False, focus_id: str | None = None, findings: list[Finding] | None = None
) -> str:
    """Render actual branches and shared external uplinks, with cycle protection.

    Returns:
        A controller/hub/port tree, or an explicit native-topology limitation.
    """
    topology = USBTopology(inventory)
    payloads = _hub_payloads(inventory, topology, findings or [])
    visible, expanded = _focus(topology, focus_id) if focus_id else (set(topology.devices), set(topology.devices))
    platform = [device for device in inventory.devices if device.kind in {"platform", "virtual"}]
    roots = [
        device
        for device in topology.roots()
        if device.id in visible and (details or device not in platform or bool(focus_id and device.id in expanded))
    ]
    lines = ["USB topology" + (" — selected path and shared branches" if focus_id else "")]
    capability = inventory.capabilities.get("usb_topology")
    if capability and capability.status != "available":
        lines.extend([
            "  Topology unavailable: " + safe_text(capability.detail),
            "  Devices below are ungrouped; no shared-hub inference is possible.",
        ])
    visited: set[str] = set()

    def append(device: Device, prefix: str, *, last: bool) -> None:
        if device.id in visited:
            return
        visited.add(device.id)
        branch, continuation = ("└── ", "    ") if last else ("├── ", "│   ")
        label = _tree_label(device, topology, selected=device.id == focus_id)
        lines.append(prefix + branch + label)
        next_prefix = prefix + continuation
        if device.id in payloads:
            lines.append(next_prefix + "    " + payloads[device.id])
        children = [
            child for child in topology.children.get(device.id, []) if child.id in visible and child.id not in visited
        ]
        if device.id not in expanded:
            count = len(topology.shared_devices(device.id))
            if count:
                lines.append(next_prefix + f"└── {count} devices on this other branch (use devices --tree --all)")
            return
        if details:
            for ep in device.endpoints:
                lines.append(next_prefix + f"    {ep.kind}/{ep.role}: {safe_text(ep.path)}")
                lines.extend(next_prefix + f"      {alias.binding}: {safe_text(alias.path)}" for alias in ep.aliases)
        for index, child in enumerate(children):
            append(child, next_prefix, last=index == len(children) - 1)

    for index, root in enumerate(roots):
        append(root, "", last=index == len(roots) - 1)
    # Malformed strategy data must not silently erase cyclic/disconnected devices.
    remaining = [
        d
        for d in inventory.devices
        if d.id in visible and d.id not in visited and d.kind not in {"platform", "virtual"}
    ]
    for device in remaining:
        lines.append("  [unresolved ancestry]")
        append(device, "  ", last=True)
    if platform and not details and not focus_id:
        nodes = sum(len(device.endpoints) for device in platform)
        lines.append(f"  Other: {len(platform)} platform/virtual groups ({nodes} nodes), hidden; use --all.")
    if not inventory.devices:
        lines.append("  No devices enumerated; check collection limitations before concluding hardware is absent.")
    return "\n".join(lines)


def render_shared_links(inventory: Inventory, device_id: str) -> str:
    """Return the hubs and physical peers sharing a selected device's uplinks."""
    topology = USBTopology(inventory)
    lines = ["Shared upstream links"]
    for hub in reversed(topology.external_hubs(device_id)):
        peers = topology.shared_devices(hub.id, excluding=device_id)
        lines.append(f"  {safe_text(hub.name)} [{safe_text(hub.id)}] — {format_link(hub.speed_mbps)}")
        if peers:
            lines.extend(
                f"    Shares with: {safe_text(peer.name)} ({safe_text(device_nodes(peer) or peer.id)})"
                for peer in peers
            )
        else:
            lines.append("    No other physical device on this external-hub branch.")
    if len(lines) == 1:
        lines.append("  No shared external hub is visible on this path (or topology is unavailable).")
    lines.append(
        "  All peers share the named uplink, including devices behind nested hubs. Root ports are separate links."
    )
    return "\n".join(lines)


def _passive_rows(inventory: Inventory, endpoint: Endpoint | None) -> list[list[str]]:
    rows = []
    for device in inventory.devices:
        if device.kind != "device" and endpoint is None:
            continue
        for ep in device.endpoints:
            if (endpoint is not None and ep.id != endpoint.id) or ep.role != "capture":
                continue
            demand = payload_demand(ep.current_mode) if ep.current_mode else None
            need = (
                format_payload(demand) + " est."
                if demand is not None
                else ("unknown/variable" if ep.current_mode else "unknown")
            )
            rows.append([
                ep.path,
                format_mode(ep.current_mode),
                need,
                "not measured",
                format_ceiling(device.speed_mbps),
            ])
    return rows


def _stream_row(inventory: Inventory, finding: Finding) -> list[str] | None:
    data = finding.details
    request, metrics = data.get("requested"), data.get("metrics")
    if not isinstance(request, dict) or not isinstance(metrics, dict):
        return None
    try:
        device, ep = resolve_endpoint(inventory, finding.subject)
        path, link = ep.path, format_ceiling(device.speed_mbps)
    except ValueError:
        path, link = finding.subject, "unknown"
    rate = metrics.get("payload_bytes_per_second")
    demand = data.get("payload_demand_bytes_per_second")
    measured = format_payload(rate if isinstance(rate, (int, float)) else None)
    need = format_payload(demand) + " est." if isinstance(demand, (int, float)) else "unknown"
    fps = metrics.get("received_fps")
    got = f"{fps:.1f} FPS; {measured}" if isinstance(fps, (int, float)) else measured
    mode = f"{request.get('width')}x{request.get('height')} {request.get('pixel_format')} @{request.get('fps')} FPS"
    trial = "alone" if finding.check_id == "stream.individual" else "together"
    return [f"{path} ({trial})", mode, need, got, link]


def _stream_rows(inventory: Inventory, findings: list[Finding]) -> list[list[str]]:
    rows = []
    for finding in findings:
        if finding.check_id not in {"stream.individual", "stream.concurrent"}:
            continue
        row = _stream_row(inventory, finding)
        if row is not None:
            rows.append(row)
    return rows


def render_payloads(
    inventory: Inventory, findings: list[Finding] | None = None, *, endpoint: Endpoint | None = None
) -> str:
    """Answer required versus delivered payload without inventing bus utilization.

    Returns:
        A compact payload table with explicit units and measurement limitations.
    """
    if endpoint is not None and endpoint.kind == "serial":
        return (
            "Serial throughput\n  Needs: protocol/workload dependent; Getting: not measured.\n"
            "  USB signaling is not UART baud or robot traffic. Passive inspection does not open serial ports."
        )
    active = findings is not None and any(f.check_id.startswith("stream.") for f in findings)
    rows = _stream_rows(inventory, findings or []) if active else _passive_rows(inventory, endpoint)
    lines = [
        "Camera throughput" + (" — measured test window" if active else " — configured modes; streams may be idle")
    ]
    if rows:
        lines.append(table(["Stream", "Mode / target", "Needs (payload)", "Getting (payload)", "USB ceiling*"], rows))
    else:
        lines.append("  No camera payload rates available for this selection.")
    lines.append(
        "  Rates use MB/s (1,000,000 bytes/s). *USB ceiling = signaling Mb/s ÷ 8, NOT usable/allocated payload."
    )
    if active:
        lines.extend([
            "  MJPEG need uses sampled frame size at target FPS; scene-dependent compression makes it an estimate.",
            "  Getting counts encoded image bytes, not RGB size, USB headers/retries, or traffic lost before dequeue.",
        ])
    else:
        lines.extend([
            "  Needs = uncompressed configured-mode estimate (may be idle). Do not sum alternative endpoints.",
            "  Measure MJPEG/YUYV: pai doctor --stream --camera <image-node> [--camera <image-node>] --duration 5",
        ])
    if endpoint is None and any(ep.kind == "serial" for device in inventory.devices for ep in device.endpoints):
        lines.append("  Robot/UART traffic needs protocol/workload measurements; USB speed is not UART baud.")
    return "\n".join(lines)
