from __future__ import annotations

from pathlib import Path

import pytest

_RETIRED_UI_KEYS = {"groups", "group", "widget", "connection_key", "serial_number_key"}

_CONNECTION_UI = [
    {
        "kind": "connection",
        "label": "Robot control unit",
        "description": "Enter the FCI hostname or IP address of the control unit.",
        "device_discovery": False,
        "manual_entry": True,
        "bind": {"connection": "connection_string"},
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


class _StubRobot:
    def __init__(self, payload: object) -> None:
        self.payload = payload


class _StubFactory:
    def __init__(self, port: str | None = None) -> None:
        self._port = port

    async def find_port(self, serial_info: object) -> str | None:
        _ = serial_info
        return self._port


class _FakeRegistry:
    def __init__(self) -> None:
        self.definitions: list = []

    def register_robot(self, definition: object) -> None:
        self.definitions.append(definition)


def test_definitions_count() -> None:
    from physicalai_franka_plugin.studio_catalog import _definitions

    assert len(_definitions()) == 3


def test_register_physicalai_studio_plugin() -> None:
    from physicalai_franka_plugin.studio_catalog import register_physicalai_studio_plugin

    registry = _FakeRegistry()
    register_physicalai_studio_plugin(registry)
    assert len(registry.definitions) == 3


def test_fr3_follower_structure() -> None:
    from physicalai_franka_plugin.studio_catalog import _definitions

    fr3 = next(d for d in _definitions() if d.type == "Franka_FR3_Follower")

    assert fr3.display_name == "Franka FR3 Follower"
    assert fr3.role == "follower"
    assert fr3.asset is not None
    assert fr3.asset.urdf_relative_path == Path("franka_description/urdf/fr3.urdf")
    assert fr3.asset.packages == {"franka_description": Path("franka_description")}
    assert fr3.asset.root_resolver is not None
    assert (fr3.asset.root_resolver() / fr3.asset.urdf_relative_path).is_file()


def test_fr3_follower_has_robot_builder() -> None:
    from physicalai_franka_plugin.studio_catalog import FrankaFR3Payload, _definitions

    fr3 = next(d for d in _definitions() if d.type == "Franka_FR3_Follower")

    assert callable(fr3.robot_builder)
    assert fr3.robot_payload is FrankaFR3Payload
    assert fr3.adapter_options.include_velocities is True
    assert fr3.adapter_options.external_effort_gain is None
    assert fr3.adapter_options.goal_time_scale == 1.0


def test_bimanual_fr3_follower_structure() -> None:
    from physicalai_franka_plugin.studio_catalog import _definitions

    fr3 = next(d for d in _definitions() if d.type == "Bimanual_Franka_FR3_Follower")
    assert fr3.display_name == "Bimanual Franka FR3 Follower"
    assert fr3.role == "follower"
    assert fr3.asset is not None
    assert fr3.asset.urdf_relative_path == Path("franka_description/urdf/fr3_bimanual.urdf")
    assert fr3.asset.root_resolver is not None
    assert (fr3.asset.root_resolver() / fr3.asset.urdf_relative_path).is_file()


def test_bimanual_gello_leader_structure() -> None:
    from physicalai_franka_plugin.studio_catalog import _definitions

    gello = next(d for d in _definitions() if d.type == "Bimanual_GELLO_Leader")
    assert gello.display_name == "Bimanual GELLO Leader"
    assert gello.role == "leader"
    assert gello.asset is None
    assert gello.adapter_options.include_velocities is False


def test_fr3_payload_defaults() -> None:
    from physicalai_franka_plugin.studio_catalog import FrankaFR3Payload

    payload = FrankaFR3Payload(connection_string="172.16.0.2")
    assert payload.connection_string == "172.16.0.2"
    assert payload.gripper_enabled is True
    assert payload.realtime_config == "kEnforce"
    assert payload.goal_time == 0.1
    assert payload.auto_recover_on_error is True
    assert payload.gripper_speed_m_s == 0.05


def test_payload_requires_host() -> None:
    from pydantic import ValidationError

    from physicalai_franka_plugin.studio_catalog import FrankaFR3Payload

    with pytest.raises(ValidationError, match="robot host"):
        FrankaFR3Payload()


def test_payload_schema_configures_manual_network_connection() -> None:
    from physicalai_studio_plugin import validate_robot_payload_ui

    from physicalai_franka_plugin.studio_catalog import FrankaFR3Payload

    validate_robot_payload_ui(FrankaFR3Payload)
    schema = FrankaFR3Payload.model_json_schema()

    assert schema["x-physicalai-ui"] == _CONNECTION_UI
    _assert_no_retired_ui_keys(schema)


def test_payload_models_rebuild() -> None:
    from physicalai_franka_plugin.studio_catalog import FrankaFR3Payload

    FrankaFR3Payload.model_rebuild(raise_errors=True)


@pytest.mark.anyio
async def test_build_fr3_from_pydantic_payload() -> None:
    from physicalai_franka_plugin.studio_catalog import FrankaFR3Payload, _build_franka_fr3_driver

    payload = FrankaFR3Payload(connection_string="172.16.0.2", gripper_enabled=False)
    driver = await _build_franka_fr3_driver(_StubRobot(payload), _StubFactory(port="172.16.0.2"))
    assert driver is not None
    assert driver.host == "172.16.0.2"
    assert driver.gripper_enabled is False


@pytest.mark.anyio
async def test_build_fr3_from_dict_payload() -> None:
    from physicalai_franka_plugin.studio_catalog import _build_franka_fr3_driver

    payload: dict[str, object] = {
        "connection_string": "172.16.0.2",
        "gripper_enabled": True,
        "goal_time": 0.25,
    }
    driver = await _build_franka_fr3_driver(_StubRobot(payload), _StubFactory())
    assert driver is not None
    assert driver.host == "172.16.0.2"
    assert driver.goal_time == pytest.approx(0.25)


@pytest.mark.anyio
async def test_build_bimanual_fr3_from_payload() -> None:
    from physicalai_franka_plugin.studio_catalog import BimanualFrankaFR3Payload, _build_bimanual_franka_fr3_driver

    payload = BimanualFrankaFR3Payload(left_host="172.16.0.2", right_host="172.16.0.3", gripper_enabled=False)
    driver = await _build_bimanual_franka_fr3_driver(_StubRobot(payload), _StubFactory())
    assert driver.left.host == "172.16.0.2"
    assert driver.right.host == "172.16.0.3"
    assert driver.joint_names[0] == "left_fr3_joint1"


def test_get_franka_urdf_root_returns_path() -> None:
    from physicalai_franka_plugin.studio_catalog import _get_franka_urdf_root

    root = _get_franka_urdf_root()
    assert isinstance(root, Path)
    assert root.exists()
