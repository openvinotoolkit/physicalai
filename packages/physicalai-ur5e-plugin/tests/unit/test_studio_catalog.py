# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from importlib.metadata import distribution
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest
from physicalai_studio_plugin import validate_robot_payload_ui
from pydantic import ValidationError

from physicalai.config import Config
from physicalai_ur5e_plugin import JOINT_NAMES, SUPPORTED_MODELS, UniversalRobot, get_asset_root
from physicalai_ur5e_plugin.profiles import joint_limits
from physicalai_ur5e_plugin.settings import RobotSettings
from physicalai_ur5e_plugin.studio_catalog import URPayload, register_physicalai_studio_plugin


def definitions():
    result = []
    register_physicalai_studio_plugin(SimpleNamespace(register_robot=result.append))
    return result


def test_entry_point():
    entry_points = distribution("physicalai-ur5e-plugin").entry_points
    entries = [entry for entry in entry_points if entry.group == "physicalai.studio.catalog_plugins"]
    assert len(entries) == 1
    assert entries[0].load() is register_physicalai_studio_plugin


def test_catalog_contract():
    entries = definitions()
    assert {entry.type for entry in entries} == {
        "UniversalRobots_UR3e_Follower",
        "UniversalRobots_UR5e_Follower",
        "UniversalRobots_UR7e_Follower",
        "UniversalRobots_UR10e_Follower",
        "UniversalRobots_UR12e_Follower",
        "UniversalRobots_UR16e_Follower",
    }
    for entry in entries:
        assert entry.role == "follower"
        assert entry.robot_payload is URPayload
        assert entry.probe is entry.zero_calibration is entry.simulation is None
        assert entry.asset.joint_map == {f"{name}.pos": [name] for name in JOINT_NAMES}


def test_payload_validation_and_defaults():
    URPayload.model_rebuild(raise_errors=True)
    validate_robot_payload_ui(URPayload)
    payload = URPayload(connection_string="192.0.2.10")
    assert payload.driver_args() == RobotSettings(ip="192.0.2.10").model_dump(exclude={"model"})
    assert "model" not in URPayload.model_json_schema()["properties"]
    assert URPayload.model_json_schema()["required"] == ["connection_string"]


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"connection_string": "robot.local"},
        {"connection_string": "192.0.2.10", "model": "ur20"},
        {"connection_string": "192.0.2.10", "gain": float("inf")},
        {"connection_string": "192.0.2.10", "external_control_port": 0},
    ],
)
def test_payload_rejects_invalid_values(values):
    with pytest.raises(ValidationError):
        URPayload(**values)


def test_builders_are_disconnected_and_exportable():
    for entry, model in zip(definitions(), SUPPORTED_MODELS, strict=True):
        robot = SimpleNamespace(type=entry.type, payload=URPayload(connection_string="192.0.2.10"))
        driver = asyncio.run(entry.robot_builder(robot, None))
        assert isinstance(driver, UniversalRobot)
        assert driver.model == model
        assert not driver.is_connected()
        assert Config.from_instance(driver).instantiate().model == model


def test_builder_rejects_unknown_type():
    entry = definitions()[0]
    robot = SimpleNamespace(type="UniversalRobots_UR20_Follower", payload=URPayload(connection_string="192.0.2.10"))
    with pytest.raises(ValueError, match="Unknown"):
        asyncio.run(entry.robot_builder(robot, None))


def test_builder_rejects_untyped_payload():
    entry = definitions()[0]
    with pytest.raises(TypeError):
        asyncio.run(entry.robot_builder(SimpleNamespace(type=entry.type, payload={}), None))


@pytest.mark.parametrize("entry", definitions(), ids=lambda entry: entry.type)
def test_assets_and_limits(entry):
    root = entry.asset.root_resolver()
    assert root == get_asset_root()
    assert entry.asset.packages.keys() == {"ur_description"}
    path = root / entry.asset.urdf_relative_path
    text = path.read_text()
    assert "$(" not in text and "xacro:" not in text and "/tmp/" not in text
    robot = ElementTree.fromstring(text)
    actuated = {joint.attrib["name"]: joint for joint in robot.findall("joint") if joint.attrib["type"] != "fixed"}
    assert set(actuated) == set(JOINT_NAMES)
    limits = joint_limits(path.stem)
    for index, name in enumerate(JOINT_NAMES):
        limit = actuated[name].find("limit")
        for key, attribute in (("lower", "lower"), ("upper", "upper"), ("velocity", "velocity")):
            value = float(limit.attrib[attribute]) if attribute in limit.attrib else None
            assert limits[key][index] == value
    for mesh in robot.findall(".//mesh"):
        prefix = "package://ur_description/"
        filename = mesh.attrib["filename"]
        assert filename.startswith(prefix)
        file = root / filename.removeprefix(prefix)
        assert file.is_file()
        assert file.stat().st_size > 100
        assert not file.read_bytes().startswith(b"version https://git-lfs.github.com/spec/")
        if file.suffix == ".dae":
            document = ElementTree.fromstring(file.read_bytes())
            for image in document.findall(".//{*}image/{*}init_from"):
                assert (file.parent / image.text).is_file()


def test_asset_provenance_and_licenses():
    root = get_asset_root()
    assert "81f7c9fe003c2f0e02eec5a7253f1b262ad38b61" in (root / "SOURCE").read_text()
    assert "Redistribution and use" in (root / "LICENSE-BSD-3-Clause").read_text()
    assert {path.name for path in (root / "meshes").iterdir()} == {"ur3e", "ur5e", "ur10e", "ur16e"}
    assert set(json.loads((root / "limits.json").read_text())) == set(SUPPORTED_MODELS)
