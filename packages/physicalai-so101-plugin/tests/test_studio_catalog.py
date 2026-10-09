# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the single-arm SO-101 catalog plugin."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError


class _FakeRegistry:
    def __init__(self) -> None:
        self.definitions: list[object] = []

    def register_robot(self, definition: object) -> None:
        self.definitions.append(definition)


class _StubFactory:
    def __init__(self, port: str | None) -> None:
        self.port = port

    async def find_port(self, _port_info: object) -> str | None:
        return self.port


class _StubRobot:
    def __init__(self, robot_type: str, payload: object) -> None:
        self.type = robot_type
        self.payload = payload


def test_registers_so101_definitions() -> None:
    from physicalai_so101_plugin.studio_catalog import register_physicalai_studio_plugin

    registry = _FakeRegistry()
    register_physicalai_studio_plugin(registry)
    assert {definition.type for definition in registry.definitions} == {"SO101_Follower", "SO101_Leader"}


def test_catalog_asset_resolves_to_bundled_urdf() -> None:
    from physicalai_so101_plugin.studio_catalog import _definitions

    definition = next(definition for definition in _definitions() if definition.type == "SO101_Follower")
    assert definition.asset is not None
    assert definition.asset.urdf_relative_path == Path("so101_new_calib.urdf")
    assert definition.asset.root_resolver is not None
    assert (definition.asset.root_resolver() / definition.asset.urdf_relative_path).is_file()


def test_payload_requires_connection_identifier() -> None:
    from physicalai_so101_plugin.studio_catalog import SO101RobotPayload

    with pytest.raises(ValidationError, match="Either serial_number or connection_string"):
        SO101RobotPayload()


@pytest.mark.anyio
async def test_builder_rejects_unresolvable_port() -> None:
    from physicalai_so101_plugin.studio_catalog import SO101RobotPayload, _build_so101_driver

    robot = _StubRobot("SO101_Follower", SO101RobotPayload(connection_string="/dev/ttyUSB0"))
    with pytest.raises(ValueError, match="Could not resolve a serial port"):
        await _build_so101_driver(robot, _StubFactory(port=None))  # type: ignore[arg-type]
