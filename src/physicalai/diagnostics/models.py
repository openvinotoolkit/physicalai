# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Versioned, serializable hardware evidence shared by CLI and API consumers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


def timestamp() -> str:
    """Return a UTC collection timestamp."""
    return datetime.now(UTC).isoformat()


class EvidenceModel(BaseModel):
    """Reject unknown fields rather than silently dropping diagnostic evidence."""

    model_config = ConfigDict(extra="forbid")


class Capability(EvidenceModel):
    """Availability of an OS feature, not the health of connected hardware."""

    status: Literal["available", "unavailable", "unsupported"]
    detail: str = ""


class CollectionError(EvidenceModel):
    """A check that could not be completed."""

    source: str
    message: str
    essential: bool = False


class Alias(EvidenceModel):
    """An endpoint's claim to a path; multiple endpoints can claim one alias."""

    path: str
    binding: Literal["device", "port"]
    source: str = "udev"
    target: str | None = None


class VideoMode(EvidenceModel):
    """An advertised wire format; this does not assert streaming success."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0, allow_inf_nan=False)
    pixel_format: str
    frame_bytes: int | None = Field(
        default=None,
        gt=0,
        description="Reported uncompressed frame size including padding; never a compressed payload measurement.",
    )


class ProcessInfo(EvidenceModel):
    """An observed device opener, without command-line arguments or environment."""

    pid: int
    name: str


class Endpoint(EvidenceModel):
    """One OS-accessible function of a physical device."""

    id: str
    path: str
    kind: Literal["video", "serial", "hid", "media", "usb"]
    role: Literal["capture", "metadata", "unknown"] = "unknown"
    interface: str = ""
    aliases: list[Alias] = Field(default_factory=list)
    readable: bool | None = None
    writable: bool | None = None
    modes: list[VideoMode] = Field(default_factory=list)
    current_mode: VideoMode | None = None
    holders: list[ProcessInfo] = Field(default_factory=list)
    properties: dict[str, str] = Field(default_factory=dict)


class Device(EvidenceModel):
    """Physical device or hub; ``id`` is an inventory key, not a stored selector."""

    id: str
    name: str
    kind: Literal["controller", "hub", "device", "platform", "virtual"] = "device"
    parent_id: str | None = None
    location: str = ""
    vendor_id: str = ""
    product_id: str = ""
    serial: str | None = None
    speed_mbps: float | None = None
    driver: str = ""
    endpoints: list[Endpoint] = Field(default_factory=list)
    properties: dict[str, str] = Field(default_factory=dict)


class SystemEvent(EvidenceModel):
    """Bounded system-log evidence with a source cursor for deduplication."""

    id: str
    source: str
    timestamp: str
    message: str
    historical: bool = True


class Inventory(EvidenceModel):
    """A hardware snapshot; no objects in this model hold hardware connections."""

    schema_version: Literal[1] = 1
    redacted: bool = False
    collected_at: str = Field(default_factory=timestamp)
    platform: str
    capabilities: dict[str, Capability] = Field(default_factory=dict)
    devices: list[Device] = Field(default_factory=list)
    settings: dict[str, str] = Field(default_factory=dict)
    events: list[SystemEvent] = Field(default_factory=list)
    errors: list[CollectionError] = Field(default_factory=list)


class Finding(EvidenceModel):
    """An assessment with enough evidence to reproduce and verify it."""

    check_id: str
    severity: Literal["PASS", "INFO", "WARN", "FAIL", "SKIP"]
    subject: str
    summary: str
    evidence: list[str] = Field(default_factory=list)
    details: dict[str, JsonValue] = Field(default_factory=dict)
    action: str = ""
    verification: str = ""

    @property
    def key(self) -> tuple[str, str]:
        """Return a finding's identity across watch assessments."""
        return self.check_id, self.subject


class Report(EvidenceModel):
    """An inventory and its pure health assessment."""

    schema_version: Literal[1] = 1
    mode: Literal["passive", "camera_streams"] = "passive"
    inventory: Inventory
    findings: list[Finding] = Field(default_factory=list)

    def exit_code(self, *, strict: bool = False) -> int:
        """Return failure, partial, or success in that precedence order."""
        if any(f.severity == "FAIL" or (strict and f.severity == "WARN") for f in self.findings):
            return 1
        return 3 if any(error.essential for error in self.inventory.errors) else 0


class WatchEvent(EvidenceModel):
    """One independently parseable NDJSON watch record."""

    schema_version: Literal[1] = 1
    sequence: int
    timestamp: str = Field(default_factory=timestamp)
    kind: str
    subject: str = "host"
    data: dict[str, JsonValue] = Field(default_factory=dict)
