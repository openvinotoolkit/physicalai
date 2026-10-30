# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inspect local config selectors without resolving class paths or constructors."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import yaml

from physicalai.diagnostics.identity import resolve_endpoint
from physicalai.diagnostics.models import Finding

if TYPE_CHECKING:
    from pathlib import Path

    from physicalai.diagnostics.models import Inventory

_MAX_FILE_BYTES = 2 * 1024 * 1024
_MAX_DEPTH = 64
_MAX_VISITS = 10000
_CAPTURE_CLASSES = {
    "physicalai.capture.UVCCamera",
    "physicalai.capture.cameras.uvc.UVCCamera",
    "physicalai.capture.cameras.uvc._camera.UVCCamera",
    "physicalai.capture.cameras.uvc.v4l2.V4L2Camera",
    "physicalai.capture.cameras.uvc.v4l2._camera.V4L2Camera",
}


def inspect_config(path: Path, inventory: Inventory) -> list[Finding]:
    """Check explicit connection selectors in a bounded YAML/JSON document.

    This is not a full Runtime/policy compatibility test and never imports the
    configured classes, even if a document contains an arbitrary ``class_path``.

    Returns:
        Selector findings and an inspection-scope notice.

    Raises:
        ValueError: If the document is oversized, cyclic, too deep, or malformed.
        TypeError: If the root is not a mapping.
    """
    with path.open("rb") as stream:
        content = stream.read(_MAX_FILE_BYTES + 1)
    if len(content) > _MAX_FILE_BYTES:
        msg = "Config exceeds the 2 MiB passive inspection limit."
        raise ValueError(msg)
    try:
        document = yaml.safe_load(content)
    except (yaml.YAMLError, RecursionError) as exc:
        msg = f"Invalid YAML/JSON configuration: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(document, dict):
        msg = "Config must contain a mapping."
        raise TypeError(msg)
    findings: list[Finding] = []
    visits = 0

    def visit(value: object, location: str, ancestors: frozenset[int], *, capture: bool = False) -> None:
        nonlocal visits
        visits += 1
        if visits > _MAX_VISITS:
            msg = "Config exceeds the bounded inspection node limit (including expanded YAML aliases)."
            raise ValueError(msg)
        if len(ancestors) > _MAX_DEPTH or id(value) in ancestors:
            msg = "Config is cyclic or exceeds the passive inspection depth limit."
            raise ValueError(msg)
        if isinstance(value, dict):
            if "class_path" in value:
                capture = value["class_path"] in _CAPTURE_CLASSES if isinstance(value["class_path"], str) else False
            for key, child in value.items():
                current = f"{location}.{key}" if location else str(key)
                if key in {"device", "device_path", "port", "connection_string"} and isinstance(child, (str, int)):
                    selector = str(child)
                    if key == "device" and selector.isdecimal():
                        selector = f"/dev/video{selector}" if inventory.platform == "linux" else f"index:{selector}"
                    if selector.startswith(("/dev/", "index:")) or re.fullmatch(r"COM\d+", selector, re.IGNORECASE):
                        findings.append(_selector_finding(inventory, selector, current, capture=capture))
                visit(child, current, ancestors | {id(value)}, capture=capture)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{location}[{index}]", ancestors | {id(value)}, capture=capture)

    visit(document, "", frozenset())
    findings.append(
        Finding(
            check_id="config.inspection_scope",
            severity="INFO",
            subject=str(path),
            summary="Checked explicit endpoint selectors only; no classes imported and no hardware connected.",
            action="Review calibration, policy features, joint order, units, and startup behavior before running.",
        )
    )
    return findings


def _selector_finding(inventory: Inventory, selector: str, location: str, *, capture: bool = False) -> Finding:
    try:
        _, endpoint = resolve_endpoint(inventory, selector)
    except ValueError as exc:
        return Finding(
            check_id="config.selector",
            severity="FAIL",
            subject=location,
            summary=str(exc),
            evidence=[selector],
            action="Reconnect the intended device or select an unambiguous binding.",
            verification="Rerun doctor with the same config after reconnecting.",
        )
    ephemeral = selector == endpoint.path
    if capture and (endpoint.kind != "video" or endpoint.role == "metadata"):
        return Finding(
            check_id="config.selector",
            severity="FAIL",
            subject=location,
            summary="The UVC camera recipe selects a non-image/metadata endpoint, not a capture stream.",
            evidence=[selector, f"{endpoint.kind}/{endpoint.role}"],
            action="Select an image capture endpoint with a supported wire format.",
            verification="Inspect per-node capabilities with pai devices inspect.",
        )
    if endpoint.readable is False or endpoint.writable is False:
        return Finding(
            check_id="config.selector",
            severity="FAIL",
            subject=location,
            summary="Configured endpoint is present but not readable/writable by the current user.",
            evidence=[selector, endpoint.path],
            action="Review device access rules before running this configuration.",
            verification="Rerun doctor as the same user that runs Runtime.",
        )
    return Finding(
        check_id="config.selector",
        severity="WARN" if ephemeral else "PASS",
        subject=location,
        summary="Ephemeral/unverified connection path in config."
        if ephemeral
        else "Connection alias selects one current endpoint.",
        evidence=[selector, endpoint.path],
        action="Prefer a validated device/port binding; native indices and COM numbers can change."
        if ephemeral
        else "",
        verification="Recheck the selector after reconnecting or rebooting.",
    )
