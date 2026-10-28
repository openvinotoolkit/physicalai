# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Passive hardware inventory CLI; heavy/platform imports stay in dispatch."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from jsonargparse import ArgumentParser, Namespace

from physicalai.cli._spec import SubcommandSpec  # noqa: PLC2701

if TYPE_CHECKING:
    from physicalai.diagnostics.models import Inventory
    from physicalai.diagnostics.redact import Redactor

HELP = "Inspect connected cameras, serial candidates, and USB topology without connecting drivers."


def build_parser() -> ArgumentParser:
    """Build the inventory parser without enumerating hardware.

    Returns:
        The device listing/inspection parser.
    """
    parser = ArgumentParser(prog="pai devices", description=HELP)
    parser.add_argument("operation", nargs="?", choices=["list", "inspect"], default="list")
    parser.add_argument("selector", nargs="?", help="Device ID, endpoint path, or unique alias for inspect.")
    parser.add_argument("--tree", action="store_true", help="Show controllers, hubs, and shared upstream links.")
    parser.add_argument("--all", action="store_true", help="Include empty hubs and detailed endpoint aliases.")
    parser.add_argument("--json", action="store_true", help="Print a versioned JSON inventory.")
    parser.add_argument(
        "--redact",
        action="store_true",
        help="Pseudonymize known device/host identifiers; output is not a deployable config.",
    )
    return parser


def _inspect(inventory: Inventory, selector: str, *, redactor: Redactor | None, json_output: bool) -> None:
    from physicalai.diagnostics.identity import alias_claims, resolve_endpoint  # noqa: PLC0415
    from physicalai.diagnostics.render import render_inspection  # noqa: PLC0415

    matched = next((device for device in inventory.devices if device.id == selector), None)
    endpoint = None
    if matched is None:
        matched, endpoint = resolve_endpoint(inventory, selector)
    index = inventory.devices.index(matched)
    endpoint_index = matched.endpoints.index(endpoint) if endpoint is not None else None
    display = redactor.model(inventory) if redactor else inventory
    device = display.devices[index]
    selected_endpoint = device.endpoints[endpoint_index] if endpoint_index is not None else None
    if json_output:
        endpoints = [selected_endpoint] if selected_endpoint is not None else device.endpoints
        paths = {alias.path for ep in endpoints for alias in ep.aliases}
        claims = alias_claims(display)
        data = {
            "device": device.model_dump(mode="json"),
            "endpoint": selected_endpoint.model_dump(mode="json") if selected_endpoint else None,
        }
        print(  # noqa: T201 - CLI JSON output
            json.dumps(
                {
                    "schema_version": 1,
                    "collected_at": inventory.collected_at,
                    "platform": inventory.platform,
                    "redacted": bool(redactor),
                    "selection": data,
                    "alias_claims": [
                        {"path": path, "claimants": [ep.path for ep in claims.get(path, [])]} for path in sorted(paths)
                    ],
                    "capabilities": {key: value.model_dump(mode="json") for key, value in display.capabilities.items()},
                    "collection_errors": [
                        error.model_dump(mode="json")
                        for error in display.errors
                        if error.essential or error.source in {ep.path for ep in endpoints}
                    ],
                },
                indent=2,
            )
        )
    else:
        print(render_inspection(display, device, selected_endpoint))  # noqa: T201


def run(parser: ArgumentParser, cfg: Namespace) -> int:
    """Collect once and display physical devices or one selected endpoint.

    Returns:
        Zero on success, or three if essential collection is incomplete.
    """
    from physicalai.diagnostics import collect_inventory  # noqa: PLC0415
    from physicalai.diagnostics.render import render_inventory  # noqa: PLC0415

    if (cfg.operation == "inspect") != bool(cfg.selector):
        parser.error("Use 'pai devices inspect SELECTOR', or 'pai devices' without a selector.")
    try:
        inventory = collect_inventory()
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if cfg.redact:
        from physicalai.diagnostics.redact import Redactor  # noqa: PLC0415

        # Resolve against original identifiers first; redact only display data.
        redactor = Redactor()
    else:
        redactor = None
    if cfg.operation == "inspect":
        try:
            _inspect(inventory, cfg.selector, redactor=redactor, json_output=cfg.json)
        except ValueError as exc:
            parser.error(str(exc))
    elif cfg.json:
        print((redactor.model(inventory) if redactor else inventory).model_dump_json(indent=2))  # noqa: T201
    else:
        display = redactor.model(inventory) if redactor else inventory
        print(render_inventory(display, tree=cfg.tree, details=cfg.all))  # noqa: T201
    return 3 if any(error.essential for error in inventory.errors) else 0


def register() -> SubcommandSpec:
    """Register the hardware inventory command.

    Returns:
        The host CLI registration.
    """
    return SubcommandSpec(name="devices", parser=build_parser(), dispatch=run, help=HELP)
