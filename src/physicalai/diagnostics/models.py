# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Small snapshots of USB devices and their camera/serial endpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Literal


@dataclass
class Alias:
    """A udev path claim and its current symlink target, which can disagree."""

    path: str
    target: str | None
    claimed: bool = False


@dataclass
class Endpoint:
    """An OS endpoint; metadata nodes are not separate camera streams."""

    path: str
    kind: Literal["video", "serial"]
    role: Literal["capture", "metadata", "unknown"] = "unknown"
    readable: bool = False
    writable: bool = False
    aliases: list[Alias] = field(default_factory=list)
    query_error: str = ""

    @property
    def relevant(self) -> bool:
        """Whether this endpoint can be a camera stream or serial connection."""
        return self.kind == "serial" or self.role != "metadata"


@dataclass
class Device:
    """One physical USB device, hub, or controller, never one per video node."""

    id: str
    name: str
    kind: Literal["controller", "hub", "device"] = "device"
    parent_id: str | None = None
    speed_mbps: float | None = None
    vendor_id: str = ""
    product_id: str = ""
    serial: str = ""
    root_hub: bool = False
    video_interface: bool = False
    endpoints: list[Endpoint] = field(default_factory=list)

    @property
    def relevant(self) -> bool:
        """Whether the device exposes camera/serial evidence worth diagnosing."""
        return self.kind == "device" and (self.video_interface or any(ep.relevant for ep in self.endpoints))


@dataclass
class Finding:
    """An issue or context with a machine-readable code and affected device IDs.

    ``hub_id`` identifies the shared uplink for ``shared_usb_link`` findings, so
    applications need not parse messages to highlight connections in a UI.
    """

    code: Literal["access", "video_endpoint_missing", "video_role_unknown", "ambiguous_identity", "shared_usb_link"]
    severity: Literal["error", "warning", "info"]
    devices: list[str]
    message: str
    hint: str = ""
    hub_id: str | None = None


@dataclass
class Inventory:
    """A read-only snapshot, including failures that limit its completeness."""

    devices: list[Device] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def ancestors(self, device: Device) -> list[Device]:
        """Return the parent-to-root route, stopping on missing or cyclic data."""
        by_id = {d.id: d for d in self.devices}
        result: list[Device] = []
        seen = {device.id}
        current = by_id.get(device.parent_id or "")
        while current is not None and current.id not in seen:
            seen.add(current.id)
            result.append(current)
            current = by_id.get(current.parent_id or "")
        return result

    def resolve(self, selector: str) -> tuple[Device, Endpoint | None]:
        """Select a device ID, endpoint, or unambiguous current alias.

        Returns:
            The physical device and, when selected, its endpoint.

        Raises:
            ValueError: If the selector is missing, ambiguous, or stale.
        """
        for device in self.devices:
            if device.id == selector:
                return device, None
        matches = [
            (d, ep)
            for d in self.devices
            for ep in d.endpoints
            if selector == ep.path or any(a.path == selector for a in ep.aliases)
        ]
        if len(matches) != 1:
            msg = f"Selector {selector!r} matches {len(matches)} endpoints; use a device ID or unique endpoint path."
            raise ValueError(msg)
        device, endpoint = matches[0]
        if selector != endpoint.path and not any(
            a.path == selector and a.target == endpoint.path for a in endpoint.aliases
        ):
            msg = f"Alias {selector!r} no longer points to {endpoint.path}; inspect the endpoint directly."
            raise ValueError(msg)
        return device, endpoint

    def unique_aliases(self, device: Device, endpoint: Endpoint) -> list[Alias]:
        """Return usable aliases, preferring unique device identity over a port.

        A symlink winner alone cannot establish by-id uniqueness. Port paths bind
        a slot, not a physical unit; neither kind of alias authenticates a device.
        """
        same_serial = sum(
            (d.vendor_id, d.product_id, d.serial) == (device.vendor_id, device.product_id, device.serial)
            for d in self.devices
            if d.kind == "device"
        )
        result = []
        for alias in endpoint.aliases:
            claimants = sum(alias.path == a.path for d in self.devices for ep in d.endpoints for a in ep.aliases)
            device_bound = "/by-id/" in alias.path
            if alias.target != endpoint.path or claimants != 1:
                continue
            if device_bound and (not alias.claimed or not device.serial or same_serial != 1):
                continue
            result.append(alias)
        return sorted(result, key=lambda a: ("/by-id/" not in a.path, a.path))


@dataclass
class Report:
    """UI-independent diagnostics, suitable for a CLI or a Studio response.

    Inventory retains USB ancestry and peers even for a selected device. Findings
    are scoped to that device when a selector was supplied. IDs describe this
    snapshot's USB locations, not persistent physical identities.
    """

    inventory: Inventory
    findings: list[Finding]
    selected_device_id: str | None = None
    selected_endpoint: str | None = None
    platform: Literal["linux"] = field(default="linux", init=False)
    collected_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: Literal[1] = field(default=1, init=False)

    @property
    def has_errors(self) -> bool:
        """Whether device access failed or essential collection was incomplete."""
        return bool(self.inventory.errors) or any(f.severity == "error" for f in self.findings)

    def to_dict(self) -> dict[str, object]:
        """Return a detached, JSON-serializable document without terminal styling."""
        return asdict(self)
