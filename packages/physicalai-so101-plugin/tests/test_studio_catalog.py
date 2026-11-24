# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the single-arm and bimanual SO-101 catalog plugin."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import pytest
from pydantic import ValidationError

from physicalai_so101_plugin import SO101Calibration
from physicalai_so101_plugin.constants import SO101_JOINT_ORDER


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


def _calibration() -> SO101Calibration:
    return SO101Calibration.from_dict({
        name: {"id": servo_id, "drive_mode": 0, "homing_offset": 0, "range_min": 100, "range_max": 3100}
        for servo_id, name in enumerate(SO101_JOINT_ORDER, 1)
    })


def test_unit_selector_schema_and_help() -> None:
    from physicalai_studio_plugin import validate_robot_payload_ui
    from physicalai_so101_plugin.studio_catalog import SO101RobotPayload

    validate_robot_payload_ui(SO101RobotPayload)
    SO101RobotPayload.model_rebuild(raise_errors=True)
    schema = SO101RobotPayload.model_json_schema()
    field = schema["properties"]["unit"]
    assert field["enum"] == ["normalized", "degrees"]
    assert field["default"] == "normalized"
    assert field["title"] == "Joint units"
    assert {"kind": "field", "name": "unit"} in schema["x-physicalai-ui"]
    help_text = field["x-physicalai-ui"]["info"]["description"]
    for text in ("[-100, 100]", "gripper remains [0, 100]", "midpoint", "matching units", "requires calibration"):
        assert text in help_text


def test_legacy_payload_defaults_to_normalized() -> None:
    from physicalai_so101_plugin.studio_catalog import SO101RobotPayload

    payload = SO101RobotPayload.model_validate({"connection_string": "fake"})
    assert payload.unit == "normalized"
    assert payload.model_dump(mode="json")["unit"] == "normalized"


@pytest.mark.parametrize("unit", ["ticks", "radians", "use_degrees", ""])
def test_catalog_rejects_unsupported_unit(unit: str) -> None:
    from physicalai_so101_plugin.studio_catalog import SO101RobotPayload

    with pytest.raises(ValidationError):
        SO101RobotPayload(connection_string="fake", unit=unit)  # type: ignore[arg-type]


@pytest.mark.anyio
@pytest.mark.parametrize("robot_type,role", [("SO101_Follower", "follower"), ("SO101_Leader", "leader")])
@pytest.mark.parametrize("unit", ["normalized", "degrees"])
async def test_builder_passes_selected_unit_and_role(robot_type: str, role: str, unit: str) -> None:
    from physicalai_so101_plugin.studio_catalog import SO101RobotPayload, _build_so101_driver

    payload = SO101RobotPayload(
        connection_string="fake", calibration=_calibration().joints, unit=unit,  # type: ignore[arg-type]
    )
    restored = SO101RobotPayload.model_validate(payload.model_dump(mode="json"))
    arm = await _build_so101_driver(_StubRobot(robot_type, restored), _StubFactory(port="resolved"))  # type: ignore[arg-type]
    assert arm.unit == unit
    assert arm.role == role
    assert arm.port == "resolved"
    assert not arm.is_connected()


@pytest.mark.anyio
async def test_degree_selection_without_calibration_does_not_silently_use_ticks() -> None:
    from physicalai_so101_plugin.studio_catalog import SO101RobotPayload, _build_so101_driver

    payload = SO101RobotPayload(connection_string="fake", unit="degrees")
    with pytest.raises(ValueError, match="degree units require calibration"):
        await _build_so101_driver(_StubRobot("SO101_Follower", payload), _StubFactory(port="resolved"))  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_default_uncalibrated_bringup_is_unchanged() -> None:
    from physicalai_so101_plugin.studio_catalog import SO101RobotPayload, _build_so101_driver

    payload = SO101RobotPayload(connection_string="fake")
    arm = await _build_so101_driver(_StubRobot("SO101_Leader", payload), _StubFactory(port="resolved"))  # type: ignore[arg-type]
    assert arm.unit == "ticks"
    assert arm.role == "leader"
    assert not arm.is_connected()
