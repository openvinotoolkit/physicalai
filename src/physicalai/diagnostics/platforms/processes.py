# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Bounded /proc inspection; never open device nodes or collect command lines."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from physicalai.diagnostics.models import Capability, ProcessInfo

if TYPE_CHECKING:
    from physicalai.diagnostics.models import Endpoint, Inventory

_SCAN_SECONDS = 0.25
_LINK_LIMIT = 10000


def _targets(inventory: Inventory) -> dict[str, Endpoint]:
    targets = {
        ep.path: ep
        for device in inventory.devices
        for ep in device.endpoints
        if ep.kind in {"serial", "usb"} or ep.role == "capture"
    }
    for endpoint in targets.values():
        endpoint.holders.clear()
    return targets


def collect_holders(inventory: Inventory, proc: Path = Path("/proc")) -> None:
    """Record visible endpoint users while explicitly bounding work and visibility."""
    targets = _targets(inventory)
    if not targets:
        return
    started = time.monotonic()
    denied = 0
    scanned = 0
    limited = False
    try:
        processes = [path for path in proc.iterdir() if path.name.isdecimal()]
    except OSError as exc:
        inventory.capabilities["process_ownership"] = Capability(status="unavailable", detail=str(exc))
        return
    for process in processes:
        if time.monotonic() - started > _SCAN_SECONDS or scanned >= _LINK_LIMIT:
            limited = True
            break
        try:
            descriptors = list((process / "fd").iterdir())
            name = (process / "comm").read_text(encoding="utf-8", errors="replace").strip()
        except PermissionError:
            denied += 1
            continue
        except OSError:
            continue  # Processes can exit during inspection.
        observed: set[str] = set()
        for descriptor in descriptors:
            if time.monotonic() - started > _SCAN_SECONDS or scanned >= _LINK_LIMIT:
                limited = True
                break
            scanned += 1
            try:
                target = str(descriptor.readlink())
            except PermissionError:
                denied += 1
                break
            except OSError:
                continue
            if target in targets and target not in observed:
                targets[target].holders.append(ProcessInfo(pid=int(process.name), name=name))
                observed.add(target)
    detail = "Visible /proc file descriptors only; absence is not proof of system-wide idleness."
    if denied or limited:
        detail += " Some processes were not inspectable or the bounded scan was incomplete."
    inventory.capabilities["process_ownership"] = Capability(
        status="unavailable" if denied or limited else "available",
        detail=detail,
    )
