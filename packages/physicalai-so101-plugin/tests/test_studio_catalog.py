# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the single-arm and bimanual SO-101 catalog plugin."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from xml.etree import ElementTree

import numpy as np
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


def test_identify_disconnects_when_torque_off_acknowledgment_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from physicalai_so101_plugin import studio_catalog

    factory = MagicMock()
    factory.JOINT_ORDER = studio_catalog.SO101.JOINT_ORDER
    driver = factory.uncalibrated.return_value
    driver.get_observation.return_value.joint_positions = np.full(6, 2048.0)
    driver.set_torque.side_effect = ConnectionError("torque-off failed")
    monkeypatch.setattr(studio_catalog, "SO101", factory)
    monkeypatch.setattr(studio_catalog.time, "sleep", lambda _seconds: None)

    with pytest.raises(ConnectionError, match="torque-off failed"):
        studio_catalog.identify_so101_robot_visually("unused")
    driver.disconnect.assert_called_once()


def test_registers_so101_definitions() -> None:
    from physicalai_so101_plugin.studio_catalog import register_physicalai_studio_plugin

    registry = _FakeRegistry()
    register_physicalai_studio_plugin(registry)
    assert {definition.type for definition in registry.definitions} == {
        "SO101_Follower",
        "SO101_Leader",
        "BimanualSO101_Follower",
        "BimanualSO101_Leader",
    }


def test_catalog_assets_resolve_to_bundled_role_specific_urdfs() -> None:
    from physicalai_so101_plugin.studio_catalog import _definitions

    expected = {
        "SO101_Follower": (Path("so101-follower/so101_follower.urdf"), Path("so101-follower/assets")),
        "SO101_Leader": (Path("so101-leader/urdf/so101_leader.urdf"), Path("so101-leader/urdf/meshes")),
        "BimanualSO101_Follower": (
            Path("so101-follower/so101_bimanual_follower.urdf"),
            Path("so101-follower/assets"),
        ),
        "BimanualSO101_Leader": (
            Path("so101-leader/urdf/so101_bimanual_leader.urdf"),
            Path("so101-leader/urdf/meshes"),
        ),
    }
    definitions = {definition.type: definition for definition in _definitions()}

    for robot_type, (urdf_path, mesh_root) in expected.items():
        asset = definitions[robot_type].asset
        assert asset is not None
        assert asset.urdf_relative_path == urdf_path
        assert asset.root_resolver is not None
        root = asset.root_resolver()
        assert (root / asset.urdf_relative_path).is_file()
        assert (root / mesh_root).is_dir()


def test_catalog_urdf_meshes_resolve_from_their_own_directories() -> None:
    from physicalai_so101_plugin.studio_catalog import _definitions

    for definition in _definitions():
        asset = definition.asset
        assert asset is not None
        assert asset.root_resolver is not None
        urdf_path = asset.root_resolver() / asset.urdf_relative_path
        mesh_paths = [Path(mesh.attrib["filename"]) for mesh in ElementTree.parse(urdf_path).findall(".//mesh")]
        assert all((urdf_path.parent / mesh_path).is_file() for mesh_path in mesh_paths)


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
