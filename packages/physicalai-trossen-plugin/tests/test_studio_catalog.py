# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Trossen Studio catalog registration and bundled URDF assets."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from xml.etree import ElementTree

import pytest
from pydantic import ValidationError

from physicalai.config import Config
from physicalai.robot import Robot


class _FakeRegistry:
    def __init__(self) -> None:
        self.definitions: list[object] = []

    def register_robot(self, definition: object) -> None:
        self.definitions.append(definition)


def test_registers_all_trossen_catalog_definitions() -> None:
    from physicalai_trossen_plugin.studio_catalog import register_physicalai_studio_plugin

    registry = _FakeRegistry()
    register_physicalai_studio_plugin(registry)

    assert {definition.type for definition in registry.definitions} == {
        "Trossen_WidowXAI_Follower",
        "Trossen_WidowXAI_Leader",
        "Trossen_Bimanual_WidowXAI_Follower",
        "Trossen_Bimanual_WidowXAI_Leader",
    }


def test_catalog_assets_resolve_to_bundled_role_specific_urdfs() -> None:
    from physicalai_trossen_plugin.studio_catalog import _definitions

    expected_paths = {
        "Trossen_WidowXAI_Follower": Path("urdf/generated/wxai/wxai_follower.urdf"),
        "Trossen_WidowXAI_Leader": Path("urdf/generated/wxai/wxai_leader_left.urdf"),
        "Trossen_Bimanual_WidowXAI_Follower": Path("urdf/generated/stationary_ai.urdf"),
        "Trossen_Bimanual_WidowXAI_Leader": Path("urdf/generated/stationary_ai_leader.urdf"),
    }
    definitions = {definition.type: definition for definition in _definitions()}

    for robot_type, expected_path in expected_paths.items():
        asset = definitions[robot_type].asset
        assert asset is not None
        assert asset.urdf_relative_path == expected_path
        assert asset.root_resolver is not None
        assert (asset.root_resolver() / asset.urdf_relative_path).is_file()


def test_leader_assets_map_actions_to_leader_joints() -> None:
    from physicalai_trossen_plugin.studio_catalog import _definitions

    definitions = {definition.type: definition for definition in _definitions()}
    single_arm_asset = definitions["Trossen_WidowXAI_Leader"].asset
    bimanual_asset = definitions["Trossen_Bimanual_WidowXAI_Leader"].asset

    assert single_arm_asset is not None
    assert bimanual_asset is not None
    assert single_arm_asset.joint_map["gripper.pos"] == ["left_carriage_joint", "right_carriage_joint"]
    assert bimanual_asset.joint_map["left_gripper.pos"] == [
        "leader_left_left_carriage_joint",
        "leader_left_right_carriage_joint",
    ]
    assert bimanual_asset.joint_map["right_gripper.pos"] == [
        "leader_right_left_carriage_joint",
        "leader_right_right_carriage_joint",
    ]


def test_catalog_asset_meshes_and_joint_maps_resolve() -> None:
    from physicalai_trossen_plugin.studio_catalog import _definitions

    package_uri_prefix = "package://trossen_arm_description/"
    for definition in _definitions():
        asset = definition.asset
        assert asset is not None
        assert asset.root_resolver is not None
        urdf_path = asset.root_resolver() / asset.urdf_relative_path
        root = ElementTree.parse(urdf_path).getroot()
        joint_names = {joint.get("name") for joint in root.findall("joint")}

        assert all(joint_name in joint_names for joints in asset.joint_map.values() for joint_name in joints)
        for asset_element in root.findall(".//*[@filename]"):
            filename = asset_element.get("filename")
            assert filename is not None
            assert filename.startswith(package_uri_prefix)
            assert (asset.root_resolver() / filename.removeprefix(package_uri_prefix)).is_file()


@pytest.mark.anyio
@pytest.mark.parametrize(("platform", "count_flag"), [("linux", "-c"), ("win32", "-n")])
async def test_ping_uses_only_cross_platform_arguments(
    monkeypatch: pytest.MonkeyPatch, platform: str, count_flag: str
) -> None:
    import physicalai_trossen_plugin.studio_catalog as catalog

    proc = MagicMock()
    proc.wait = AsyncMock(return_value=0)
    create_subprocess_exec = AsyncMock(return_value=proc)
    monkeypatch.setattr(catalog.asyncio, "create_subprocess_exec", create_subprocess_exec)
    monkeypatch.setattr(catalog.sys, "platform", platform)

    assert await catalog._ping("192.168.1.100")
    assert create_subprocess_exec.await_args.args == ("ping", count_flag, "1", "192.168.1.100")


def test_drivers_export_plugin_construction_recipes() -> None:
    from physicalai_trossen_plugin import BimanualWidowXAI, WidowXAI

    left = WidowXAI(ip="192.168.1.10")
    right = WidowXAI(ip="192.168.1.11")
    bimanual = BimanualWidowXAI(left=left, right=right)

    assert isinstance(left, Robot)
    assert Config.from_instance(left).to_dict() == {
        "class_path": "physicalai_trossen_plugin.WidowXAI",
        "init_args": {"ip": "192.168.1.10"},
    }
    assert Config.from_instance(bimanual).to_dict()["class_path"] == "physicalai_trossen_plugin.BimanualWidowXAI"


@pytest.mark.parametrize("bimanual", [False, True])
def test_payload_goal_time_defaults_and_advanced_schema(bimanual: bool) -> None:
    from physicalai_studio_plugin import validate_robot_payload_ui
    from physicalai_trossen_plugin.studio_catalog import TrossenBimanualPayload, TrossenSingleArmPayload

    model = TrossenBimanualPayload if bimanual else TrossenSingleArmPayload
    connections = (
        {"connection_string_left": "192.168.1.10", "connection_string_right": "192.168.1.11"}
        if bimanual
        else {"connection_string": "192.168.1.10"}
    )
    assert model.model_validate(connections).goal_time_scale == 3.0
    model.model_rebuild(raise_errors=True)
    validate_robot_payload_ui(model)
    field_schema = model.model_json_schema()["properties"]["goal_time_scale"]
    assert field_schema["default"] == 3.0
    assert field_schema["exclusiveMinimum"] == 0
    assert field_schema["x-physicalai-ui"]["advanced_configuration"] is True
    assert "goal_time_scale" not in model.model_json_schema()["required"]


@pytest.mark.parametrize("bimanual", [False, True])
@pytest.mark.parametrize("goal_time_scale", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_payload_rejects_invalid_goal_time_scale(bimanual: bool, goal_time_scale: float) -> None:
    from physicalai_trossen_plugin.studio_catalog import TrossenBimanualPayload, TrossenSingleArmPayload

    model = TrossenBimanualPayload if bimanual else TrossenSingleArmPayload
    connections = (
        {"connection_string_left": "192.168.1.10", "connection_string_right": "192.168.1.11"}
        if bimanual
        else {"connection_string": "192.168.1.10"}
    )
    with pytest.raises(ValidationError, match="goal_time_scale"):
        model.model_validate({**connections, "goal_time_scale": goal_time_scale})


@pytest.mark.anyio
@pytest.mark.parametrize("goal_time_scale", [None, 1.0, 5.0])
async def test_builders_preserve_timing_in_exported_config(goal_time_scale: float | None) -> None:
    from physicalai_trossen_plugin.studio_catalog import _definitions

    for definition in _definitions():
        connections = (
            {"connection_string_left": "192.168.1.10", "connection_string_right": "192.168.1.11"}
            if "Bimanual" in definition.type
            else {"connection_string": "192.168.1.10"}
        )
        values = connections if goal_time_scale is None else {**connections, "goal_time_scale": goal_time_scale}
        payload = definition.robot_payload.model_validate(values)
        driver = await definition.robot_builder(SimpleNamespace(type=definition.type, payload=payload), MagicMock())
        expected_scale = 3.0 if goal_time_scale is None else goal_time_scale
        assert definition.adapter_options.goal_time_scale == 1.0
        restored = Config.from_instance(driver).instantiate()
        arms = (restored.left, restored.right) if "Bimanual" in definition.type else (restored,)
        for arm in arms:
            assert arm.goal_time_scale == expected_scale
            assert arm.role == definition.role
            assert not arm.is_connected()
