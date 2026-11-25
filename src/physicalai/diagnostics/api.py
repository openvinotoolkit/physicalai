# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Application-facing collection and assessment, independent of CLI output."""

from __future__ import annotations

import sys

from physicalai.diagnostics.checks import assess
from physicalai.diagnostics.models import Inventory, Report


def collect_inventory() -> Inventory:
    """Collect Linux USB topology and camera/serial endpoints without drivers.

    Serial ports are never opened. Video nodes may be opened read-only to
    distinguish image and metadata endpoints, but never configured or streamed.
    Non-USB cameras and network robots are outside this collector's scope.

    Returns:
        Device evidence, collection errors, and limitations; no live handles.

    Raises:
        NotImplementedError: If Linux USB diagnostics are unavailable on this OS.
    """
    if not sys.platform.startswith("linux"):
        msg = "USB connection diagnostics currently support Linux only."
        raise NotImplementedError(msg)
    from physicalai.diagnostics.linux import LinuxCollector  # noqa: PLC0415 - load Linux code only on Linux

    return LinuxCollector().collect()


def diagnose(selector: str | None = None) -> Report:
    """Collect and assess USB connections for CLI or application use.

    Args:
        selector: Optional USB device ID, endpoint path, or unique alias. Scope
            findings to this device while retaining topology/peers as context.

    Returns:
        Typed evidence and findings; use ``to_dict()`` for JSON serialization.
        This function does not print, change hardware, or load robot plugins.
        Unsupported hosts raise ``NotImplementedError``; missing, ambiguous, or
        stale selectors raise ``ValueError``.
    """
    platform = sys.platform
    inventory = collect_inventory()
    findings = assess(inventory)
    if selector is None:
        return Report(inventory, findings, platform=platform)
    device, endpoint = inventory.resolve(selector)
    return Report(
        inventory,
        [f for f in findings if device.id in f.devices or f.hub_id == device.id],
        selected_device_id=device.id,
        selected_endpoint=endpoint.path if endpoint else None,
        platform=platform,
    )
