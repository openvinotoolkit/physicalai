# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from physicalai.diagnostics.models import Device, Endpoint, Inventory
from physicalai.diagnostics.plugins import Catalog, discover_candidates


def test_passive_argument_is_explicit_and_legacy_probes_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    class Port(BaseModel):
        connection_string: str | None = None
        serial_number: str | None = None

    monkeypatch.setitem(sys.modules, "physicalai_studio_plugin", SimpleNamespace(SerialPortInfo=Port))
    calls = []

    class PassiveProbe:
        async def discover(self, scanner, *, passive=False):
            calls.append(passive)
            await scanner.find_robots()
            return scanner.robots

        async def identify(self, *args, **kwargs):
            pytest.fail("must never identify")

        async def is_online(self, *args, **kwargs):
            pytest.fail("candidate discovery must not check online")

    class LegacyProbe:
        async def discover(self, scanner):
            pytest.fail("must not call an unaudited legacy probe")

    probe = PassiveProbe()
    catalog = Catalog(definitions={
        "passive": cast(Any, SimpleNamespace(probe=probe)),
        "same_probe": cast(Any, SimpleNamespace(probe=probe)),
        "legacy": cast(Any, SimpleNamespace(probe=LegacyProbe())),
    })
    inventory = Inventory(platform="linux", devices=[Device(id="arm", name="serial", endpoints=[Endpoint(id="serial", path="/dev/ttyUSB0", kind="serial")])])
    result = asyncio.run(discover_candidates(catalog, inventory))
    assert result == {"passive": ["/dev/ttyUSB0"], "same_probe": ["/dev/ttyUSB0"]}
    assert calls == [True]


def test_duplicate_registration_is_rejected() -> None:
    catalog = Catalog()
    definition = cast(Any, SimpleNamespace(type="arm"))
    catalog.register_robot(definition)
    with pytest.raises(ValueError, match="Duplicate"):
        catalog.register_robot(definition)


def test_passive_probe_aliases_resolve_and_unknown_candidates_are_reported() -> None:
    from physicalai.diagnostics.models import Alias
    from physicalai.diagnostics.plugins import _candidate_paths
    endpoint = Endpoint(id="serial", path="/dev/ttyUSB0", kind="serial", aliases=[
        Alias(path="/dev/serial/by-path/arm", binding="port", target="/dev/ttyUSB0"),
    ])
    inventory = Inventory(platform="linux", devices=[Device(id="arm", name="serial", endpoints=[endpoint])])
    ports = [cast(Any, SimpleNamespace(connection_string="/dev/serial/by-path/arm", serial_number=None)),
             cast(Any, SimpleNamespace(connection_string="unsupported-network-endpoint", serial_number=None))]
    catalog = Catalog()
    assert _candidate_paths(ports, inventory, catalog, "example") == ["/dev/ttyUSB0"]
    assert len(catalog.errors) == 1 and catalog.errors[0].source == "probe:example"
