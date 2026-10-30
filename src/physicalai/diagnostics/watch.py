# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Bounded polling watcher with inventory diffs and finding lifecycles."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from physicalai.diagnostics.checks import assess
from physicalai.diagnostics.models import WatchEvent

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from physicalai.diagnostics.models import Device, Inventory, Report
    from physicalai.diagnostics.platforms.base import Collector
    from physicalai.diagnostics.platforms.events import ChangeSource


def _describe_change(previous: Device, current: Device) -> str:
    details = []
    if previous.properties.get("device_number") != current.properties.get("device_number"):
        details.append("USB address changed (device re-enumerated at this location)")
    if previous.speed_mbps != current.speed_mbps:
        details.append(f"link speed {previous.speed_mbps} -> {current.speed_mbps} Mb/s")
    before_paths = sorted(ep.path for ep in previous.endpoints)
    after_paths = sorted(ep.path for ep in current.endpoints)
    if before_paths != after_paths:
        details.append(f"endpoints {before_paths} -> {after_paths}")
    if (previous.vendor_id, previous.product_id, previous.serial) != (
        current.vendor_id,
        current.product_id,
        current.serial,
    ):
        details.append("reported device identity changed; review the connection binding")
    return "; ".join(details) or "endpoint/alias, access, or device metadata changed"


def changes(previous: Report, current: Report) -> list[WatchEvent]:
    """Compare evidence, without asserting continuity of anonymous devices.

    Returns:
        Ordered changes; the watch session supplies final sequence numbers.
    """
    old = {device.id: device for device in previous.inventory.devices}
    new = {device.id: device for device in current.inventory.devices}
    events = [
        WatchEvent(sequence=0, kind="device_removed", subject=key, data=old[key].model_dump(mode="json"))
        for key in sorted(old.keys() - new.keys())
    ]
    events.extend(
        WatchEvent(sequence=0, kind="device_added", subject=key, data=new[key].model_dump(mode="json"))
        for key in sorted(new.keys() - old.keys())
    )
    events.extend(
        WatchEvent(
            sequence=0,
            kind="device_changed",
            subject=key,
            data={
                "before": old[key].model_dump(mode="json"),
                "after": new[key].model_dump(mode="json"),
                "summary": _describe_change(old[key], new[key]),
                "identity_verified": False,
                "note": "Same inventory location; compare serial/endpoint evidence before inferring identity.",
            },
        )
        for key in sorted(old.keys() & new.keys())
        if old[key] != new[key]
    )
    old_findings = {finding.key: finding for finding in previous.findings}
    new_findings = {finding.key: finding for finding in current.findings}
    for key, finding in old_findings.items():
        if key not in new_findings or new_findings[key] != finding:
            data = finding.model_dump(mode="json")
            data["resolution"] = (
                "device removed" if finding.subject in old and finding.subject not in new else "evidence changed"
            )
            events.append(WatchEvent(sequence=0, kind="finding_resolved", subject=finding.subject, data=data))
    for key, finding in new_findings.items():
        if old_findings.get(key) != finding:
            events.append(
                WatchEvent(
                    sequence=0, kind="finding_opened", subject=finding.subject, data=finding.model_dump(mode="json")
                )
            )
    events.extend(
        WatchEvent(sequence=0, kind="system_log", data=event.model_dump(mode="json"))
        for event in current.inventory.events
    )
    return events


def watch(
    collector: Collector,
    *,
    interval: float = 1.0,
    duration: float | None = None,
    strict: bool = False,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    assessment: Callable[[Inventory], Report] = assess,
    source: ChangeSource | None = None,
) -> Iterator[WatchEvent]:
    """Yield a baseline, observed changes, and final assessment.

    Polling cannot observe devices disconnected and reconnected between samples.
    Log cursors are maintained by the collector, not by message-text matching.

    Raises:
        ValueError: If interval or duration is not finite and positive.
    """
    import math  # noqa: PLC0415

    if (
        not math.isfinite(interval)
        or interval <= 0
        or (duration is not None and (not math.isfinite(duration) or duration <= 0))
    ):
        msg = "Watch interval and duration must be finite and positive."
        raise ValueError(msg)
    started = clock()
    sequence = 1
    baseline = collector.collect()
    _notification_capability(baseline, source)
    report = assessment(baseline)
    yield WatchEvent(sequence=sequence, kind="snapshot", data=report.model_dump(mode="json"))
    while duration is None or clock() - started < duration:
        delay = interval if duration is None else min(interval, max(0, duration - (clock() - started)))
        if source is not None:
            source.wait(delay)
        else:
            sleep(delay)
        snapshot = collector.collect()
        _notification_capability(snapshot, source)
        current = assessment(snapshot)
        for event in changes(report, current):
            sequence += 1
            yield event.model_copy(update={"sequence": sequence})
        report = current
    yield WatchEvent(
        sequence=sequence + 1,
        kind="final",
        data={"exit_code": report.exit_code(strict=strict), "report": report.model_dump(mode="json")},
    )


def _notification_capability(inventory: Inventory, source: ChangeSource | None) -> None:
    if source is not None:
        from physicalai.diagnostics.models import Capability  # noqa: PLC0415

        inventory.capabilities["device_events"] = Capability(status="available", detail=source.detail)
