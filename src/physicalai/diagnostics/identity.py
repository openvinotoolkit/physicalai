# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Resolve inventory handles and assess path bindings without opening devices."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from physicalai.diagnostics.models import Device, Endpoint, Inventory


def alias_claims(inventory: Inventory) -> dict[str, list[Endpoint]]:
    """Return every claimant, not just the current winner of an OS symlink."""
    claims: dict[str, list[Endpoint]] = defaultdict(list)
    for device in inventory.devices:
        for endpoint in device.endpoints:
            for alias in endpoint.aliases:
                claims[alias.path].append(endpoint)
    return dict(claims)


def resolve_endpoint(inventory: Inventory, selector: str) -> tuple[Device, Endpoint]:
    """Resolve an exact endpoint handle, native path, or unique alias claim.

    Returns:
        The physical device and its uniquely selected endpoint.

    Raises:
        ValueError: If no endpoint or more than one endpoint matches.
    """
    matches = [
        (device, endpoint)
        for device in inventory.devices
        for endpoint in device.endpoints
        if selector in {endpoint.id, endpoint.path, *(alias.path for alias in endpoint.aliases)}
    ]
    if len(matches) != 1:
        msg = f"Selector {selector!r} matches {len(matches)} endpoints; select an unambiguous endpoint path."
        raise ValueError(msg)
    device, endpoint = matches[0]
    if selector not in {endpoint.id, endpoint.path}:
        alias = next(alias for alias in endpoint.aliases if alias.path == selector)
        if alias.target != endpoint.path:
            msg = f"Alias {selector!r} does not currently resolve to its claimed endpoint {endpoint.path!r}."
            raise ValueError(msg)
    return device, endpoint


def binding_paths(inventory: Inventory, endpoint: Endpoint, binding: Literal["device", "port"]) -> list[str]:
    """Return currently valid, unambiguous aliases with the requested scope."""
    claims = alias_claims(inventory)
    device = next(device for device in inventory.devices if endpoint in device.endpoints)
    if binding == "device":
        if not device.serial:
            return []
        peers = [
            d
            for d in inventory.devices
            if (d.vendor_id, d.product_id, d.serial) == (device.vendor_id, device.product_id, device.serial)
        ]
        if len(peers) != 1:
            return []
    return sorted(
        alias.path
        for alias in endpoint.aliases
        if alias.binding == binding
        and alias.target == endpoint.path
        and len(claims[alias.path]) == 1
        and (binding != "device" or alias.source != "symlink-target")
    )
