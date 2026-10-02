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

_REBOT_B601_DM_UI = [
    *_CONNECTION_UI,
    {
        "kind": "section",
        "id": "control",
        "title": "Control mode",
        "description": (
            "MIT mode drives joints with torque/stiffness gains for fast, responsive motion; "
            "POS_VEL / FORCE_POS use velocity-capped position control."
        ),
        "items": [
            {"kind": "field", "name": "control_mode"},
            {"kind": "field", "name": "gripper_control_mode"},
        ],
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
    def __init__(self, port: str | None = "/dev/ttyACM0") -> None:
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
    from physicalai_rebot_b601_plugin.studio_catalog import _definitions

    defs = _definitions()
    assert len(defs) == 2


def test_definitions_have_expected_types() -> None:
    from physicalai_rebot_b601_plugin.studio_catalog import _definitions

    types = {d.type for d in _definitions()}
    assert types == {"ReBot_B601_DM_Follower", "ReBot_B601_RS_Follower"}


def test_register_physicalai_studio_plugin() -> None:
    from physicalai_rebot_b601_plugin.studio_catalog import register_physicalai_studio_plugin

    registry = _FakeRegistry()
    register_physicalai_studio_plugin(registry)
    assert len(registry.definitions) == 2


def test_dm_follower_structure() -> None:
    from physicalai_rebot_b601_plugin.studio_catalog import _definitions

    dm = next(d for d in _definitions() if d.type == "ReBot_B601_DM_Follower")

    assert dm.display_name == "ReBot B601 DM Follower"
    assert dm.role == "follower"
    assert dm.asset is not None
    assert dm.asset.urdf_relative_path == Path("rebot-b601-dm/urdf/reBot-DevArm_fixend.urdf")
    assert dm.asset.packages == {"rebot-b601-dm": Path("rebot-b601-dm")}
    assert dm.asset.root_resolver is not None
    assert (dm.asset.root_resolver() / dm.asset.urdf_relative_path).is_file()


def test_definition_robot_type_property() -> None:
    from physicalai_rebot_b601_plugin.studio_catalog import _definitions

    for d in _definitions():
        assert d.type in {"ReBot_B601_DM_Follower", "ReBot_B601_RS_Follower"}


def test_dm_follower_has_robot_builder() -> None:
    from physicalai_rebot_b601_plugin.studio_catalog import ReBotB601DMPayload, _definitions

    dm = next(d for d in _definitions() if d.type == "ReBot_B601_DM_Follower")
    assert callable(dm.robot_builder)
    assert dm.robot_payload is ReBotB601DMPayload
    assert dm.adapter_options.include_velocities is True
    assert dm.adapter_options.external_effort_gain is None
    assert dm.adapter_options.goal_time_scale == 1.0


def test_rebot_b601_dm_payload_defaults() -> None:
    from physicalai_rebot_b601_plugin.studio_catalog import ReBotB601DMPayload

    payload = ReBotB601DMPayload(serial_number="SN-001")
    assert payload.connection_string == ""
    assert payload.serial_number == "SN-001"
    assert payload.can_adapter == "damiao"
    assert payload.dm_serial_baud == 921600
    assert payload.disable_torque_on_disconnect is True
    assert payload.force_pos_torque_ratio == 0.1
    assert payload.control_mode == "pos_vel"
    assert payload.gripper_control_mode == "force_pos"


def test_payload_models_rebuild() -> None:
    from physicalai_rebot_b601_plugin.studio_catalog import ReBotB601DMPayload

    ReBotB601DMPayload.model_rebuild(raise_errors=True)


def test_payload_requires_connection_identifier() -> None:
    from pydantic import ValidationError

    from physicalai_rebot_b601_plugin.studio_catalog import ReBotB601DMPayload

    with pytest.raises(ValidationError, match="connection_string or serial_number"):
        ReBotB601DMPayload()


def test_payload_schemas_configure_serial_connection_picker() -> None:
    from physicalai_studio_plugin import validate_robot_payload_ui

    from physicalai_rebot_b601_plugin.studio_catalog import ReBotB601DMPayload

    validate_robot_payload_ui(ReBotB601DMPayload)
    schema = ReBotB601DMPayload.model_json_schema()

    assert schema["x-physicalai-ui"] == _REBOT_B601_DM_UI
    _assert_no_retired_ui_keys(schema)


@pytest.mark.anyio
async def test_build_rebot_b601_dm_from_pydantic_payload() -> None:
    from physicalai_rebot_b601_plugin.studio_catalog import ReBotB601DMPayload, _build_rebot_b601_dm_driver

    payload = ReBotB601DMPayload(serial_number="DM-001", can_adapter="socketcan")
    robot = _StubRobot(payload)
    factory = _StubFactory(port="/dev/ttyACM0")
    driver = await _build_rebot_b601_dm_driver(robot, cast(Any, factory))
    assert driver is not None


@pytest.mark.anyio
async def test_build_rebot_b601_dm_from_dict_payload() -> None:
    from physicalai_rebot_b601_plugin.studio_catalog import _build_rebot_b601_dm_driver

    payload: dict[str, object] = {
        "serial_number": "DM-002",
        "can_adapter": "damiao",
        "force_pos_torque_ratio": 0.2,
    }
    robot = _StubRobot(payload)
    factory = _StubFactory(port="/dev/ttyACM1")
    driver = await _build_rebot_b601_dm_driver(robot, cast(Any, factory))
    assert driver is not None


@pytest.mark.anyio
async def test_build_rebot_b601_dm_port_not_found() -> None:
    from physicalai_rebot_b601_plugin.studio_catalog import ReBotB601DMPayload, _build_rebot_b601_dm_driver

    payload = ReBotB601DMPayload(serial_number="DM-MISSING")
    robot = _StubRobot(payload)
    factory = _StubFactory(port=None)
    with pytest.raises(RuntimeError, match="Robot not found"):
        await _build_rebot_b601_dm_driver(robot, cast(Any, factory))


def test_get_rebot_urdf_root_returns_path() -> None:
    from physicalai_rebot_b601_plugin.studio_catalog import _get_rebot_urdf_root

    root = _get_rebot_urdf_root()
    assert isinstance(root, Path)
    assert root.exists()


def test_rs_follower_structure() -> None:
    from physicalai_rebot_b601_plugin.studio_catalog import ReBotB601RSPayload, _definitions

    rs = next(d for d in _definitions() if d.type == "ReBot_B601_RS_Follower")

    assert rs.display_name == "ReBot B601 RS Follower"
    assert rs.role == "follower"
    assert callable(rs.robot_builder)
    assert rs.robot_payload is ReBotB601RSPayload
    assert rs.asset is not None
    assert rs.asset.urdf_relative_path == Path("rebot-b601-rs/urdf/00-arm-rs_asm-v3_joint_frame.urdf")
    assert rs.asset.packages == {"rebot-b601-rs": Path("rebot-b601-rs")}
    assert rs.asset.root_resolver is not None
    assert (rs.asset.root_resolver() / rs.asset.urdf_relative_path).is_file()
    assert rs.probe is not None
    assert rs.adapter_options.include_velocities is True
    assert rs.adapter_options.external_effort_gain is None


def test_rebot_b601_rs_payload_defaults_and_ui_schema() -> None:
    from physicalai_studio_plugin import validate_robot_payload_ui

    from physicalai_rebot_b601_plugin.studio_catalog import ReBotB601RSPayload

    from physicalai_rebot_b601_plugin.constants import REBOT_B601_RS_MIT_KD, REBOT_B601_RS_MIT_KP

    payload = ReBotB601RSPayload()
    assert payload.connection_string == "can0"
    assert payload.max_relative_target == 10.0
    assert payload.mit_kp.model_dump() == REBOT_B601_RS_MIT_KP
    assert payload.mit_kd.model_dump() == REBOT_B601_RS_MIT_KD
    assert (payload.gripper_mit_torque_limit, payload.gripper_mit_hold_torque_limit) == (3.5, 1.0)
    # Robots saved before these settings existed only stored the CAN interface.
    assert ReBotB601RSPayload.model_validate({"connection_string": "can1"}).max_relative_target == 10.0
    ReBotB601RSPayload.model_rebuild(raise_errors=True)
    validate_robot_payload_ui(ReBotB601RSPayload)
    _assert_no_retired_ui_keys(ReBotB601RSPayload.model_json_schema())


@pytest.mark.parametrize("interface", ["", ".", "..", "can0/../x", "can0;ls", "a" * 16])
def test_rebot_b601_rs_payload_rejects_invalid_interface_names(interface: str) -> None:
    from pydantic import ValidationError

    from physicalai_rebot_b601_plugin.studio_catalog import ReBotB601RSPayload

    with pytest.raises(ValidationError):
        ReBotB601RSPayload(connection_string=interface)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_relative_target": 0.0},
        {"max_relative_target": float("inf")},
        {"gripper_mit_torque_limit": -1.0},
        {"gripper_mit_kp": float("nan")},
        {"mit_kp": {"shoulder_pan": -1.0, "shoulder_lift": 1, "elbow_flex": 1, "wrist_flex": 1, "wrist_yaw": 1, "wrist_roll": 1}},
    ],
)
def test_rebot_b601_rs_payload_rejects_invalid_settings(overrides: dict[str, Any]) -> None:
    from pydantic import ValidationError

    from physicalai_rebot_b601_plugin.studio_catalog import ReBotB601RSPayload

    with pytest.raises(ValidationError):
        ReBotB601RSPayload.model_validate(overrides)


def test_rs_joint_frame_urdf_reverses_negative_direction_joints() -> None:
    import xml.etree.ElementTree as ET

    from physicalai_rebot_b601_plugin.constants import REBOT_B601_RS_JOINT_DIRECTIONS
    from physicalai_rebot_b601_plugin.studio_catalog import _REBOT_B601_DM_TO_URDF, _get_rebot_urdf_root

    urdf_dir = _get_rebot_urdf_root() / "rebot-b601-rs" / "urdf"

    def axes(name: str) -> dict[str, list[float]]:
        root = ET.parse(urdf_dir / name).getroot()  # noqa: S314 - bundled, trusted URDF
        return {j.get("name"): [float(v) for v in j.find("axis").get("xyz").split()] for j in root.iter("joint") if j.find("axis") is not None}

    motor_frame = axes("00-arm-rs_asm-v3.urdf")
    joint_frame = axes("00-arm-rs_asm-v3_joint_frame.urdf")
    for key, (urdf_joint,) in ((k, v) for k, v in _REBOT_B601_DM_TO_URDF.items() if v):
        direction = REBOT_B601_RS_JOINT_DIRECTIONS[key.removesuffix(".pos")]
        assert joint_frame[urdf_joint] == [direction * v + 0.0 for v in motor_frame[urdf_joint]]


@pytest.mark.anyio
@pytest.mark.parametrize("as_model", [True, False])
async def test_build_rebot_b601_rs_returns_exportable_socketcan_driver(as_model: bool) -> None:
    from physicalai.config import Config

    from physicalai_rebot_b601_plugin.studio_catalog import ReBotB601RSPayload, _build_rebot_b601_rs_driver

    payload = ReBotB601RSPayload(connection_string="can1") if as_model else {"connection_string": "can1"}
    driver = await _build_rebot_b601_rs_driver(_StubRobot(payload), cast(Any, _StubFactory(port=None)))

    assert driver.device_ids == ("rebot-rs:socketcan:can1",)
    exported = Config.from_instance(driver)
    assert exported["class_path"].endswith("ReBotB601RS")
    assert exported["init_args"]["port"] == "can1"
    assert exported["init_args"]["max_relative_target"] == 10.0
    assert exported["init_args"]["gripper_mit_hold_torque_limit"] == 1.0


@pytest.mark.anyio
async def test_build_rebot_b601_rs_passes_payload_settings_to_driver() -> None:
    from physicalai.config import Config

    from physicalai_rebot_b601_plugin.studio_catalog import ReBotB601RSPayload, _build_rebot_b601_rs_driver

    payload = ReBotB601RSPayload.model_validate(
        {
            "max_relative_target": 5.0,
            "mit_kp": {k: 40.0 for k in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")},
            "gripper_mit_torque_limit": 2.0,
        }
    )
    driver = await _build_rebot_b601_rs_driver(_StubRobot(payload), cast(Any, _StubFactory(port=None)))

    init_args = Config.from_instance(driver)["init_args"]
    assert driver.max_relative_target == 5.0
    assert init_args["mit_kp"]["shoulder_lift"] == 40.0
    assert init_args["mit_kd"] == payload.mit_kd.model_dump()
    assert init_args["gripper_mit_torque_limit"] == 2.0


@pytest.mark.anyio
async def test_rs_probe_reports_socketcan_interface_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from physicalai_rebot_b601_plugin import studio_catalog

    for interface, state in (("can0", "up"), ("can1", "down")):
        (tmp_path / interface).mkdir()
        (tmp_path / interface / "operstate").write_text(f"{state}\n", encoding="utf-8")
    monkeypatch.setattr(studio_catalog, "_SYSFS_NET_ROOT", tmp_path)
    probe = studio_catalog.ReBotRSProbe()

    assert await probe.is_online(studio_catalog.ReBotB601RSPayload(connection_string="can0")) is True
    assert await probe.is_online(studio_catalog.ReBotB601RSPayload(connection_string="can1")) is False
    assert await probe.is_online(studio_catalog.ReBotB601RSPayload(connection_string="can9")) is False
    assert await probe.discover(cast(Any, None)) == []


@pytest.mark.anyio
async def test_rs_calibration_releases_torque_then_sets_zero() -> None:
    from unittest.mock import MagicMock

    from physicalai_rebot_b601_plugin import ReBotB601RS
    from physicalai_rebot_b601_plugin.studio_catalog import _definitions

    calibration = next(d for d in _definitions() if d.type == "ReBot_B601_RS_Follower").zero_calibration
    assert calibration is not None
    assert calibration.release is not None
    robot = MagicMock(spec=ReBotB601RS)

    await calibration.release(robot)
    await calibration.set_zero(robot)

    robot.disable_torque.assert_called_once_with()
    robot.set_zero_position.assert_called_once_with()



def test_dm_follower_has_no_zero_calibration() -> None:
    from physicalai_rebot_b601_plugin.studio_catalog import _definitions

    assert next(d for d in _definitions() if d.type == "ReBot_B601_DM_Follower").zero_calibration is None
