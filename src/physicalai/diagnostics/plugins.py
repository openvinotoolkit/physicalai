# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Adapter for existing robot catalog registrations, without a Studio server."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from physicalai.diagnostics.identity import resolve_endpoint
from physicalai.diagnostics.models import CollectionError

if TYPE_CHECKING:
    from physicalai_studio_plugin import RobotCatalogDefinition, SerialPortInfo

    from physicalai.diagnostics.models import Endpoint, Inventory


@dataclass
class Catalog:
    """Operator-installed robot types; duplicate registrations are rejected."""

    definitions: dict[str, RobotCatalogDefinition] = field(default_factory=dict)
    errors: list[CollectionError] = field(default_factory=list)

    def register_robot(self, definition: RobotCatalogDefinition) -> None:
        """Accept the existing structural registry contract.

        Raises:
            ValueError: If a robot type is already registered.
        """
        if definition.type in self.definitions:
            msg = f"Duplicate robot type {definition.type!r}; remove the conflicting plugin."
            raise ValueError(msg)
        self.definitions[definition.type] = definition


def load_catalog() -> Catalog:
    """Load installed, trusted plugin entry points, isolating registration failures.

    Returns:
        Registered types and structured plugin failures.
    """
    catalog = Catalog()
    for entry in sorted(entry_points(group="physicalai.studio.catalog_plugins"), key=lambda ep: ep.name):
        staged = Catalog()
        try:
            entry.load()(staged)
            _check_overlap(catalog, staged)
            catalog.definitions.update(staged.definitions)
        except Exception as exc:  # noqa: BLE001 - optional plugin boundary; preserve failure as evidence
            catalog.errors.append(CollectionError(source=f"plugin:{entry.name}", message=str(exc)))
    return catalog


def _check_overlap(catalog: Catalog, staged: Catalog) -> None:
    overlap = catalog.definitions.keys() & staged.definitions.keys()
    if overlap:
        msg = f"Duplicate robot types: {', '.join(sorted(overlap))}"
        raise ValueError(msg)


class InventoryPortScanner:
    """Cached, enumeration-only adapter for the existing ``PortScanner`` protocol."""

    def __init__(self, inventory: Inventory) -> None:
        """Adapt this snapshot using the installed plugin-contract schema."""
        from physicalai_studio_plugin import SerialPortInfo  # noqa: PLC0415

        self._robots = [
            SerialPortInfo(connection_string=ep.path, serial_number=device.serial)
            for device in inventory.devices
            for ep in device.endpoints
            if ep.kind == "serial"
        ]

    @property
    def robots(self) -> list[SerialPortInfo]:
        """Return the same enumeration for every probe in this refresh."""
        return [port.model_copy(deep=True) for port in self._robots]

    async def find_robots(self) -> None:
        """Satisfy the scanner contract without rescanning or opening ports."""


def _candidate_endpoint(port: SerialPortInfo, inventory: Inventory) -> Endpoint:
    if port.connection_string:
        _, endpoint = resolve_endpoint(inventory, port.connection_string)
    else:
        matches = [
            ep
            for device in inventory.devices
            if port.serial_number and device.serial == port.serial_number
            for ep in device.endpoints
            if ep.kind == "serial"
        ]
        if len(matches) != 1:
            msg = "Probe serial identity does not select exactly one enumerated serial endpoint."
            raise ValueError(msg)
        endpoint = matches[0]
    if endpoint.kind != "serial":
        msg = "This USB discovery adapter only maps enumerated serial endpoints."
        raise ValueError(msg)
    return endpoint


def _candidate_paths(ports: list[SerialPortInfo], inventory: Inventory, catalog: Catalog, name: str) -> list[str]:
    paths: set[str] = set()
    for port in ports:
        try:
            paths.add(_candidate_endpoint(port, inventory).path)
        except ValueError as exc:
            catalog.errors.append(CollectionError(source=f"probe:{name}", message=str(exc)))
    return sorted(paths)


async def discover_candidates(
    catalog: Catalog, inventory: Inventory, *, timeout_s: float = 2.0
) -> dict[str, list[str]]:
    """Call only explicitly passive probes; returned candidates do not prove model.

    Each probe gets the same cached inventory. Never call identify/is_online here.

    Returns:
        Candidate endpoint paths per compatible robot type, not verified models.
    """
    if not any(getattr(definition, "probe", None) is not None for definition in catalog.definitions.values()):
        return {}
    try:
        scanner = InventoryPortScanner(inventory)
    except ImportError as exc:
        catalog.errors.append(CollectionError(source="plugin-contracts", message=str(exc)))
        return {}
    results: dict[str, list[str]] = {}
    cached: dict[int, list[str]] = {}
    failed: set[int] = set()
    for name, definition in catalog.definitions.items():
        probe = getattr(definition, "probe", None)
        if probe is None:
            continue
        key = id(probe)
        if key in failed:
            continue
        try:
            parameter = inspect.signature(probe.discover).parameters.get("passive")
            if parameter is None or parameter.kind not in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }:
                catalog.errors.append(
                    CollectionError(
                        source=f"probe:{name}",
                        message="Legacy probe has no passive keyword argument; skipped safely.",
                    )
                )
                failed.add(key)
                continue
            if key not in cached:
                ports = await asyncio.wait_for(probe.discover(scanner, passive=True), timeout_s)
                cached[key] = _candidate_paths(ports, inventory, catalog, name)
            results[name] = cached[key]
        except Exception as exc:  # noqa: BLE001 - one optional probe must not abort discovery
            failed.add(key)
            catalog.errors.append(CollectionError(source=f"probe:{name}", message=str(exc) or type(exc).__name__))
    return results
