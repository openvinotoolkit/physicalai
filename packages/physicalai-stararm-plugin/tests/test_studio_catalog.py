from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

_RETIRED_UI_KEYS = {"groups", "group", "widget", "connection_key", "serial_number_key"}
_CONNECTION_UI = [
    {
        "kind": "connection",
        "label": "Select robot",
        "device_discovery": True,
        "bind": {"connection": "connection_string", "serial_number": "serial_number"},
    },
]


def _assert_no_retired_ui_keys(value: object) -> None:
    if isinstance(value, dict):
        assert not _RETIRED_UI_KEYS.intersection(value)
        for nested_value in value.values():
            _assert_no_retired_ui_keys(nested_value)
    elif isinstance(value, list):
        for nested_value in value:
            _assert_no_retired_ui_keys(nested_value)


class _StubFactory:
    def __init__(self, port: str | None = "/dev/ttyUSB0") -> None:
        self._port = port

    async def find_port(self, serial_info: object) -> str | None:
        _ = serial_info
        return self._port


class _StubRobot:
    def __init__(self, payload: object) -> None:
        self.payload = payload


class _FakeRegistry:
    def __init__(self) -> None:
        self.definitions: list = []

    def register_robot(self, definition: object) -> None:
        self.definitions.append(definition)


def test_definitions_count() -> None:
    from physicalai_stararm_plugin.studio_catalog import _definitions

    defs = _definitions()
    assert len(defs) == 3


def test_definitions_have_expected_types() -> None:
    from physicalai_stararm_plugin.studio_catalog import _definitions

    types = {d.type for d in _definitions()}
    assert types == {"StarArm_102_LD_Leader", "StarArm_102_HD_Leader", "StarArm_102_FL_Follower"}


def test_register_physicalai_studio_plugin() -> None:
    from physicalai_stararm_plugin.studio_catalog import register_physicalai_studio_plugin

    registry = _FakeRegistry()
    register_physicalai_studio_plugin(registry)
    assert len(registry.definitions) == 3


def test_ld_leader_structure() -> None:
    from physicalai_stararm_plugin.studio_catalog import _definitions

    ld = next(d for d in _definitions() if d.type == "StarArm_102_LD_Leader")

    assert ld.display_name == "Star Arm 102-LD Leader"
    assert ld.role == "leader"
    assert ld.asset is not None
    assert ld.asset.urdf_relative_path == Path("stararm102/urdf/stararm102_ld_description.urdf")
    assert ld.asset.packages == {"stararm102": Path("stararm102")}
    assert ld.asset.root_resolver is not None
    assert (ld.asset.root_resolver() / ld.asset.urdf_relative_path).is_file()


def test_hd_leader_structure() -> None:
    from physicalai_stararm_plugin.studio_catalog import _definitions

    hd = next(d for d in _definitions() if d.type == "StarArm_102_HD_Leader")

    assert hd.display_name == "Star Arm 102-HD Leader"
    assert hd.role == "leader"
    assert hd.asset is not None
    assert hd.asset.urdf_relative_path == Path("stararm102/urdf/stararm102_hd_description.urdf")
    assert hd.asset.packages == {"stararm102": Path("stararm102")}
    assert hd.asset.root_resolver is not None
    assert (hd.asset.root_resolver() / hd.asset.urdf_relative_path).is_file()


def test_fl_follower_structure() -> None:
    from physicalai_stararm_plugin.studio_catalog import _definitions

    fl = next(d for d in _definitions() if d.type == "StarArm_102_FL_Follower")

    assert fl.display_name == "Star Arm 102-FL Follower"
    assert fl.role == "follower"
    assert fl.asset is not None
    assert fl.asset.urdf_relative_path == Path("stararm102/urdf/stararm102_fl_description.urdf")
    assert fl.asset.packages == {"stararm102": Path("stararm102")}
    assert fl.asset.root_resolver is not None
    assert (fl.asset.root_resolver() / fl.asset.urdf_relative_path).is_file()


def test_hd_leader_has_robot_builder() -> None:
    from physicalai_stararm_plugin.studio_catalog import StarArm102HDPayload, _definitions

    hd = next(d for d in _definitions() if d.type == "StarArm_102_HD_Leader")
    assert callable(hd.robot_builder)
    assert hd.robot_payload is StarArm102HDPayload
    assert hd.adapter_options.include_velocities is False
    assert hd.adapter_options.external_effort_gain is None
    assert hd.adapter_options.goal_time_scale == 1.0


def test_ld_leader_has_robot_builder() -> None:
    from physicalai_stararm_plugin.studio_catalog import StarArm102LDPayload, _definitions

    ld = next(d for d in _definitions() if d.type == "StarArm_102_LD_Leader")
    assert callable(ld.robot_builder)
    assert ld.robot_payload is StarArm102LDPayload
    assert ld.adapter_options.include_velocities is False
    assert ld.adapter_options.external_effort_gain is None
    assert ld.adapter_options.goal_time_scale == 1.0


def test_fl_follower_has_robot_builder() -> None:
    from physicalai_stararm_plugin.studio_catalog import StarArm102FLPayload, _definitions

    fl = next(d for d in _definitions() if d.type == "StarArm_102_FL_Follower")
    assert callable(fl.robot_builder)
    assert fl.robot_payload is StarArm102FLPayload
    assert fl.adapter_options.include_velocities is True
    assert fl.adapter_options.external_effort_gain is None
    assert fl.adapter_options.goal_time_scale == 1.0


def test_stararm_102_ld_payload_defaults() -> None:
    from physicalai_stararm_plugin.studio_catalog import StarArm102LDPayload

    payload = StarArm102LDPayload(serial_number="SN-LD")
    assert payload.connection_string == ""
    assert payload.serial_number == "SN-LD"
    assert payload.baudrate == 1_000_000
    assert payload.unlock_on_connect is True
    assert payload.reset_multi_turn_on_connect is True
    assert payload.zero_on_connect is False


def test_stararm_102_hd_payload_defaults() -> None:
    from physicalai_stararm_plugin.studio_catalog import StarArm102HDPayload

    payload = StarArm102HDPayload(serial_number="SN-HD")
    assert payload.connection_string == ""
    assert payload.serial_number == "SN-HD"
    assert payload.baudrate == 1_000_000
    assert payload.unlock_on_connect is True
    assert payload.reset_multi_turn_on_connect is True
    assert payload.zero_on_connect is False
    assert payload.control_mode == "passive"
    assert payload.command_interval_ms == 10


def test_stararm_102_fl_payload_defaults() -> None:
    from physicalai_stararm_plugin.studio_catalog import StarArm102FLPayload

    payload = StarArm102FLPayload(serial_number="SN-FL")
    assert payload.connection_string == ""
    assert payload.serial_number == "SN-FL"
    assert payload.baudrate == 1_000_000
    assert payload.unlock_on_connect is True
    assert payload.reset_multi_turn_on_connect is True
    assert payload.zero_on_connect is False
    assert payload.command_interval_ms == 10


@pytest.mark.parametrize("payload_name", ["StarArm102LDPayload", "StarArm102HDPayload", "StarArm102FLPayload"])
def test_payload_schemas_configure_serial_connection_picker(payload_name: str) -> None:
    from physicalai_studio_plugin import validate_robot_payload_ui

    from physicalai_stararm_plugin.studio_catalog import StarArm102FLPayload, StarArm102HDPayload, StarArm102LDPayload

    payload_models = {
        "StarArm102LDPayload": StarArm102LDPayload,
        "StarArm102HDPayload": StarArm102HDPayload,
        "StarArm102FLPayload": StarArm102FLPayload,
    }
    payload_model = payload_models[payload_name]

    validate_robot_payload_ui(payload_model)
    schema = payload_model.model_json_schema()

    assert schema["x-physicalai-ui"] == _CONNECTION_UI
    _assert_no_retired_ui_keys(schema)


@pytest.mark.anyio
async def test_build_stararm_102_hd_from_pydantic_payload() -> None:
    from physicalai_stararm_plugin.studio_catalog import StarArm102HDPayload, _build_stararm_102_hd_driver

    payload = StarArm102HDPayload(serial_number="HD-001", baudrate=115200, control_mode="assist")
    robot = _StubRobot(payload)
    factory = _StubFactory(port="/dev/ttyUSB0")
    driver = await _build_stararm_102_hd_driver(robot, cast(Any, factory))
    assert driver is not None
    assert cast(Any, driver).control_mode == "assist"


@pytest.mark.anyio
async def test_build_stararm_102_ld_from_pydantic_payload() -> None:
    from physicalai_stararm_plugin.studio_catalog import StarArm102LDPayload, _build_stararm_102_ld_driver

    payload = StarArm102LDPayload(serial_number="LD-001", baudrate=115200)
    robot = _StubRobot(payload)
    factory = _StubFactory(port="/dev/ttyUSB2")
    driver = await _build_stararm_102_ld_driver(robot, cast(Any, factory))
    assert driver is not None


@pytest.mark.anyio
async def test_build_stararm_102_fl_from_dict_payload() -> None:
    from physicalai_stararm_plugin.studio_catalog import _build_stararm_102_fl_driver

    payload: dict[str, object] = {
        "serial_number": "FL-002",
        "baudrate": 1000000,
        "unlock_on_connect": False,
        "command_interval_ms": 30,
    }
    robot = _StubRobot(payload)
    factory = _StubFactory(port="/dev/ttyUSB1")
    driver = await _build_stararm_102_fl_driver(robot, cast(Any, factory))
    assert driver is not None


@pytest.mark.anyio
async def test_build_stararm_102_hd_port_not_found() -> None:
    from physicalai_stararm_plugin.studio_catalog import StarArm102HDPayload, _build_stararm_102_hd_driver

    payload = StarArm102HDPayload(serial_number="HD-MISSING")
    robot = _StubRobot(payload)
    factory = _StubFactory(port=None)
    with pytest.raises(RuntimeError, match="Robot not found"):
        await _build_stararm_102_hd_driver(robot, cast(Any, factory))


@pytest.mark.anyio
async def test_build_stararm_102_ld_port_not_found() -> None:
    from physicalai_stararm_plugin.studio_catalog import StarArm102LDPayload, _build_stararm_102_ld_driver

    payload = StarArm102LDPayload(serial_number="LD-MISSING")
    robot = _StubRobot(payload)
    factory = _StubFactory(port=None)
    with pytest.raises(RuntimeError, match="Robot not found"):
        await _build_stararm_102_ld_driver(robot, cast(Any, factory))


@pytest.mark.anyio
async def test_build_stararm_102_fl_port_not_found() -> None:
    from physicalai_stararm_plugin.studio_catalog import StarArm102FLPayload, _build_stararm_102_fl_driver

    payload = StarArm102FLPayload(serial_number="FL-MISSING")
    robot = _StubRobot(payload)
    factory = _StubFactory(port=None)
    with pytest.raises(RuntimeError, match="Robot not found"):
        await _build_stararm_102_fl_driver(robot, cast(Any, factory))
