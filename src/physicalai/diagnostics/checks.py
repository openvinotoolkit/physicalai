# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""A few evidence-based connection checks; never measure or infer bus traffic."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

from physicalai.diagnostics.models import Finding

if TYPE_CHECKING:
    from physicalai.diagnostics.models import Device, Inventory

_USB2_HIGH_SPEED = 480


def assess(inventory: Inventory) -> list[Finding]:
    """Check access, ambiguous identities, missing video nodes, and shared hubs.

    USB 2 alone is not a fault. A serial adapter's link rate does not identify
    its robot model, UART baud, calibration, or operational state.

    Returns:
        Observed problems and shared-link context, without a health scoreboard.
    """
    findings = []
    relevant = [d for d in inventory.devices if d.relevant]
    for device in relevant:
        inaccessible = [ep.path for ep in device.endpoints if ep.relevant and not (ep.readable and ep.writable)]
        if inaccessible:
            findings.append(
                Finding(
                    "access",
                    "error",
                    [device.id],
                    f"Cannot read/write {', '.join(inaccessible)} as the current user.",
                    "Review device ACLs/group membership and vendor rules; do not run Runtime as root.",
                )
            )
        if device.video_interface and not any(ep.kind == "video" and ep.role != "metadata" for ep in device.endpoints):
            findings.append(
                Finding(
                    "video_endpoint_missing",
                    "warning",
                    [device.id],
                    f"{device.name} exposes a USB video interface but no image endpoint was found.",
                    "Check driver binding and container device access; a vendor SDK may own the interface instead.",
                )
            )
        unknown = [ep.path for ep in device.endpoints if ep.query_error and ep.readable and ep.writable]
        if unknown:
            findings.append(
                Finding(
                    "video_role_unknown",
                    "warning",
                    [device.id],
                    f"Could not determine image/metadata roles for {', '.join(unknown)}.",
                    "Inspect the selected endpoint; unknown nodes are not verified camera streams.",
                )
            )
    findings.extend(_identities(relevant))
    findings.extend(_shared_links(inventory, relevant))
    return findings


def _identities(relevant: list[Device]) -> list[Finding]:
    aliases: dict[str, set[tuple[str, str]]] = defaultdict(set)
    serials: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for device in relevant:
        if device.serial:
            serials[device.vendor_id, device.product_id, device.serial].add(device.id)
        for endpoint in device.endpoints:
            for alias in endpoint.aliases:
                aliases[alias.path].add((device.id, endpoint.path))
    collisions = {frozenset(device_id for device_id, _ in claims) for claims in aliases.values() if len(claims) > 1}
    duplicates = {frozenset(ids) for ids in serials.values() if len(ids) > 1}
    findings = []
    for group in sorted(collisions | duplicates, key=sorted):
        paths = [ep.path for d in relevant if d.id in group for ep in d.endpoints if ep.relevant]
        if group in collisions and group in duplicates:
            reason = "report the same model/serial and share ambiguous aliases"
        else:
            reason = "share ambiguous alias paths" if group in collisions else "report the same model and serial"
        findings.append(
            Finding(
                "ambiguous_identity",
                "warning",
                sorted(group),
                f"{', '.join(paths)} {reason}.",
                "Use distinct by-path selectors; inspect each endpoint for a unique port-bound path.",
            )
        )
    return findings


def _shared_links(inventory: Inventory, relevant: list[Device]) -> list[Finding]:
    hubs: dict[str, Device] = {}
    cameras: dict[str, set[str]] = defaultdict(set)
    for device in relevant:
        if not (device.video_interface or any(ep.kind == "video" and ep.role == "capture" for ep in device.endpoints)):
            continue
        for parent in inventory.ancestors(device):
            if parent.kind == "hub" and not parent.root_hub:
                hubs[parent.id] = parent
                cameras[parent.id].add(device.id)
    # Nested hubs carrying the same cameras are one finding, not repeated risks.
    groups: dict[frozenset[str], Device] = {}
    for key, ids in cameras.items():
        if len(ids) <= 1:
            continue
        hub = hubs[key]
        group = frozenset(ids)
        previous = groups.get(group)
        if previous is None or (hub.speed_mbps or math.inf) < (previous.speed_mbps or math.inf):
            groups[group] = hub
    return [
        Finding(
            "shared_usb_link",
            "warning" if hub.speed_mbps is not None and hub.speed_mbps <= _USB2_HIGH_SPEED else "info",
            sorted(ids),
            f"{len(ids)} cameras share a {hub.speed_mbps:g} Mb/s hub uplink ({hub.id})."
            if hub.speed_mbps is not None
            else f"{len(ids)} cameras share hub {hub.id}; uplink speed is unknown.",
            "If concurrent capture fails, try separate upstream connections. Sharing alone does not prove congestion.",
            hub_id=hub.id,
        )
        for ids, hub in groups.items()
    ]
