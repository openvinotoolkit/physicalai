from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from physicalai_studio_plugin import RobotCatalogDefinition, SerialPortInfo

import pytest
from pydantic import BaseModel, ValidationError


def _ensure_optional_sdk_spec() -> None:
    sdk = sys.modules.get("scservo_sdk")
    if sdk is not None and getattr(sdk, "__spec__", None) is None:
        sdk.__spec__ = ModuleSpec("scservo_sdk", None)


@dataclass
class _FakeRegistry:
    definitions: list[RobotCatalogDefinition[Any]] | None = None

    def register_robot(self, definition: RobotCatalogDefinition[Any]) -> None:
        if self.definitions is None:
            self.definitions = []
        self.definitions.append(definition)

    def register_many(self, definitions: Sequence[RobotCatalogDefinition[Any]]) -> None:
        if self.definitions is None:
            self.definitions = []
        self.definitions.extend(definitions)


class _StubFactory:
    def __init__(self) -> None:
        self.calls: list[SerialPortInfo] = []

    async def find_port(self, serial_info: SerialPortInfo) -> str | None:
        self.calls.append(serial_info)
        return getattr(serial_info, "connection_string", None)


class _StubRobot:
    def __init__(self, payload: object) -> None:
        self.payload = payload


# ── Registration tests ─────────────────────────────────────────────────────


def test_register_plugin_includes_all_registered_robot_and_teleoperator_types() -> None:
    _ensure_optional_sdk_spec()
    from lerobot.robots.config import RobotConfig
    from lerobot.teleoperators.config import TeleoperatorConfig

    from physicalai_lerobot_plugin.studio_catalog import register_physicalai_studio_plugin

    registry = _FakeRegistry()
    register_physicalai_studio_plugin(registry)

    assert registry.definitions is not None
    definitions_by_type = {definition.type: definition for definition in registry.definitions}
    expected_follower_types = {
        f"LeRobot_{type_str}_Follower" for type_str in RobotConfig.get_known_choices() if type_str != "mock_robot"
    }
    expected_leader_types = {f"LeRobot_{type_str}_Leader" for type_str in TeleoperatorConfig.get_known_choices()}

    assert set(definitions_by_type) == expected_follower_types | expected_leader_types
    assert len(definitions_by_type) == len(registry.definitions)
    for type_str in expected_follower_types:
        assert definitions_by_type[type_str].role == "follower"
    for type_str in expected_leader_types:
        assert definitions_by_type[type_str].role == "leader"


def test_skip_list_excludes_test_only_robots() -> None:
    from physicalai_lerobot_plugin.studio_catalog import register_physicalai_studio_plugin

    registry = _FakeRegistry()
    register_physicalai_studio_plugin(registry)

    assert registry.definitions is not None
    types = {d.type for d in registry.definitions}
    for skipped in ("LeRobot_mock_robot_Follower",):
        assert skipped not in types


def test_unverified_plugins_are_not_discovered_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot.utils import import_utils

    import physicalai_lerobot_plugin.studio_catalog as catalog

    discovered: list[bool] = []
    monkeypatch.setattr(import_utils, "register_third_party_plugins", lambda: discovered.append(True))
    monkeypatch.setattr(catalog, "_LEROBOT_THIRD_PARTY_PLUGINS_IMPORTED", False)
    monkeypatch.delenv("TRUST_UNVERIFIED_PLUGIN", raising=False)

    catalog._ensure_lerobot_third_party_plugins_imported()  # noqa: SLF001

    assert discovered == []


def test_trusted_unverified_plugins_are_discovered_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot.utils import import_utils

    import physicalai_lerobot_plugin.studio_catalog as catalog

    discovered: list[bool] = []
    monkeypatch.setattr(import_utils, "register_third_party_plugins", lambda: discovered.append(True))
    monkeypatch.setattr(catalog, "_LEROBOT_THIRD_PARTY_PLUGINS_IMPORTED", False)
    monkeypatch.setenv("TRUST_UNVERIFIED_PLUGIN", "1")

    catalog._ensure_lerobot_third_party_plugins_imported()  # noqa: SLF001
    catalog._ensure_lerobot_third_party_plugins_imported()  # noqa: SLF001

    assert discovered == [True]


def test_dynamic_import_guard_allows_only_trusted_prefixes() -> None:
    from physicalai_lerobot_plugin.studio_catalog import _is_allowed_dynamic_import

    assert _is_allowed_dynamic_import("lerobot.robots.so_follower")
    assert _is_allowed_dynamic_import("lerobot.teleoperators.so_leader")
    assert not _is_allowed_dynamic_import("os")
    assert not _is_allowed_dynamic_import("subprocess")
    assert not _is_allowed_dynamic_import("numpy")


def test_dynamic_import_guard_allows_third_party_prefixes_only_when_trusted(monkeypatch: pytest.MonkeyPatch) -> None:
    from physicalai_lerobot_plugin.studio_catalog import _is_allowed_dynamic_import

    monkeypatch.setenv("TRUST_UNVERIFIED_PLUGIN", "true")

    assert _is_allowed_dynamic_import("lerobot_robot_example")
    assert _is_allowed_dynamic_import("lerobot_teleoperator_example")


@pytest.mark.anyio
async def test_probe_accepts_non_serial_endpoints() -> None:
    from physicalai_lerobot_plugin.studio_catalog import LeRobotProbe

    class _NetworkPayload(BaseModel):
        host: str = "robot.local"

    assert await LeRobotProbe().is_online(_NetworkPayload())


def test_payload_types_namespace_rejects_untrusted_module() -> None:
    import dataclasses

    from physicalai_lerobot_plugin.studio_catalog import _payload_types_namespace

    @dataclasses.dataclass
    class _FakeConfig:
        port: str = ""

    _FakeConfig.__module__ = "subprocess"

    assert _payload_types_namespace(_FakeConfig) == {}


def test_register_plugin_skips_invalid_schema_and_registers_later_definitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physicalai_lerobot_plugin.studio_catalog import (
        _definitions,
        register_physicalai_studio_plugin,
    )

    definitions = _definitions()
    assert len(definitions) >= 2
    invalid_definition, valid_definition = definitions[:2]

    def reject_invalid_schema(model: type[BaseModel]) -> None:
        if model is invalid_definition.robot_payload:
            msg = "invalid third-party schema"
            raise ValueError(msg)

    monkeypatch.setattr(
        "physicalai_lerobot_plugin.studio_catalog._definitions",
        lambda: [invalid_definition, valid_definition],
    )
    monkeypatch.setattr(
        "physicalai_lerobot_plugin.studio_catalog._assert_payload_model_resolvable",
        reject_invalid_schema,
    )

    registry = _FakeRegistry()
    register_physicalai_studio_plugin(registry)

    assert registry.definitions == [valid_definition]


def test_each_definition_has_correct_role() -> None:
    from physicalai_lerobot_plugin.studio_catalog import register_physicalai_studio_plugin

    registry = _FakeRegistry()
    register_physicalai_studio_plugin(registry)

    assert registry.definitions is not None
    for d in registry.definitions:
        role = d.role
        assert role in {"follower", "leader"}
        assert d.type.startswith("LeRobot_")
        assert d.type.endswith("_Follower" if role == "follower" else "_Leader")


# ── Dynamic payload model tests ────────────────────────────────────────────


def test_make_payload_model_for_so100() -> None:
    from lerobot.robots import so_follower  # noqa: F401
    from lerobot.robots.config import RobotConfig

    from physicalai_lerobot_plugin.studio_catalog import _make_payload_model

    config_cls = RobotConfig.get_known_choices()["so100_follower"]
    model = _make_payload_model(config_cls)

    assert issubclass(model, BaseModel)
    assert model.__name__ == "SOFollowerRobotConfigPayload"

    payload = model(port="/dev/ttyACM0")
    assert payload.port == "/dev/ttyACM0"
    assert payload.disable_torque_on_disconnect is True
    assert payload.use_degrees is True


def test_payload_schema_marks_id_as_required() -> None:
    import importlib
    import pkgutil

    import lerobot.teleoperators

    for _importer, modname, is_pkg in pkgutil.walk_packages(
        lerobot.teleoperators.__path__,
        prefix="lerobot.teleoperators.",
    ):
        if "config" in modname and not is_pkg:
            importlib.import_module(modname)

    from lerobot.teleoperators.config import TeleoperatorConfig

    from physicalai_lerobot_plugin.studio_catalog import _make_payload_model

    model = _make_payload_model(TeleoperatorConfig.get_known_choices()["so101_leader"])

    assert model.model_json_schema()["properties"]["id"]["x-physicalai-ui"] == {"required": True}


def test_make_payload_model_for_hope_jr_hand() -> None:
    from lerobot.robots import hope_jr  # noqa: F401
    from lerobot.robots.config import RobotConfig

    from physicalai_lerobot_plugin.studio_catalog import _make_payload_model

    config_cls = RobotConfig.get_known_choices()["hope_jr_hand"]
    model = _make_payload_model(config_cls)

    payload = model(port="/dev/ttyACM1", side="left")
    assert payload.port == "/dev/ttyACM1"
    assert payload.side == "left"


def test_make_payload_model_resolves_nested_dataclass_forward_references() -> None:
    from lerobot.motors.motors_bus import Motor

    from physicalai_lerobot_plugin.studio_catalog import _assert_payload_model_resolvable, _make_payload_model

    model = _make_payload_model(Motor)

    _assert_payload_model_resolvable(model)
    assert model.model_fields["norm_mode"].annotation is not None


def test_nested_payload_models_are_rebuilt_with_their_own_namespace() -> None:
    from lerobot.motors.motors_bus import Motor

    from physicalai_lerobot_plugin.studio_catalog import (
        _assert_payload_model_resolvable,
        _make_payload_model,
    )

    motor_model = _make_payload_model(Motor)

    _assert_payload_model_resolvable(motor_model)

    assert motor_model.__pydantic_complete__
    motor_model.model_json_schema()


def test_make_payload_model_requires_native_required_fields() -> None:
    from lerobot.robots import so_follower  # noqa: F401
    from lerobot.robots.config import RobotConfig

    from physicalai_lerobot_plugin.studio_catalog import _make_payload_model

    config_cls = RobotConfig.get_known_choices()["so100_follower"]
    model = _make_payload_model(config_cls)

    with pytest.raises(ValidationError):
        model()


def test_make_payload_model_includes_complex_fields_for_bimanual() -> None:
    from lerobot.robots import bi_so_follower  # noqa: F401
    from lerobot.robots.config import RobotConfig

    from physicalai_lerobot_plugin.studio_catalog import _make_payload_model

    config_cls = RobotConfig.get_known_choices()["bi_so_follower"]
    model = _make_payload_model(config_cls)

    assert "left_arm_config" in model.model_fields
    assert "right_arm_config" in model.model_fields
    assert "cameras" in model.model_fields


def test_make_payload_model_for_bimanual_so100() -> None:
    from lerobot.robots import bi_so_follower  # noqa: F401
    from lerobot.robots.config import RobotConfig

    from physicalai_lerobot_plugin.studio_catalog import _make_payload_model

    config_cls = RobotConfig.get_known_choices()["bi_so_follower"]
    model = _make_payload_model(config_cls)

    payload = model(
        left_arm_config={"port": "/dev/ttyACM0"},
        right_arm_config={"port": "/dev/ttyACM1"},
    )
    assert payload.left_arm_config.port == "/dev/ttyACM0"
    assert payload.right_arm_config.port == "/dev/ttyACM1"


def test_make_payload_model_for_reachy2() -> None:
    from lerobot.robots import reachy2  # noqa: F401
    from lerobot.robots.config import RobotConfig

    from physicalai_lerobot_plugin.studio_catalog import _make_payload_model

    config_cls = RobotConfig.get_known_choices()["reachy2"]
    model = _make_payload_model(config_cls)

    assert "ip_address" in model.model_fields
    assert "with_mobile_base" in model.model_fields
    assert "cameras" in model.model_fields


def test_payload_schema_marks_only_serial_ports_as_connection_fields() -> None:
    from lerobot.robots import bi_rebot_b601_follower, reachy2, rebot_b601_follower  # noqa: F401
    from lerobot.robots.config import RobotConfig

    from physicalai_lerobot_plugin.studio_catalog import _make_payload_model

    serial_model = _make_payload_model(RobotConfig.get_known_choices()["rebot_b601_follower"])
    serial_schema = serial_model.model_json_schema()
    assert serial_schema["x-physicalai-ui"] == [
        {
            "kind": "section",
            "id": "connection",
            "title": "Select robot",
            "items": [
                {
                    "kind": "connection",
                    "label": "Select robot",
                    "device_discovery": True,
                    "bind": {"connection": "port"},
                },
            ],
        },
    ]
    assert "x-physicalai-ui" not in serial_schema["properties"]["port"]

    bimanual_model = _make_payload_model(RobotConfig.get_known_choices()["bi_rebot_b601_follower"])
    bimanual_schema = bimanual_model.model_json_schema()
    for arm in ("left_arm_config", "right_arm_config"):
        definition_name = bimanual_schema["properties"][arm]["$ref"].removeprefix("#/$defs/")
        port_schema = bimanual_schema["$defs"][definition_name]["properties"]["port"]
        assert "x-physicalai-ui" not in port_schema

    reachy_model = _make_payload_model(RobotConfig.get_known_choices()["reachy2"])
    assert "x-physicalai-ui" not in reachy_model.model_json_schema()["properties"]["port"]


@pytest.mark.anyio
async def test_builder_resolves_nested_ports_for_bimanual_config() -> None:
    from unittest.mock import MagicMock, patch

    from lerobot.robots import bi_so_follower  # noqa: F401
    from lerobot.robots.config import RobotConfig

    from physicalai_lerobot_plugin.studio_catalog import _make_builder, _make_payload_model

    config_cls = RobotConfig.get_known_choices()["bi_so_follower"]
    payload_cls = _make_payload_model(config_cls)
    builder = _make_builder("bi_so_follower", config_cls, payload_cls, role="follower")

    payload = payload_cls(
        left_arm_config={"port": "/dev/ttyACM0"},
        right_arm_config={"port": "/dev/ttyACM1"},
    )

    factory = _StubFactory()
    robot = _StubRobot(payload)
    fake_lerobot = MagicMock()
    with patch("lerobot.robots.make_robot_from_config", return_value=fake_lerobot):
        built = await builder(robot, cast(Any, factory))

    assert callable(builder)
    assert len(factory.calls) == 2
    assert built is not None


# ── Builder tests ──────────────────────────────────────────────────────────


def test_make_builder_config_kwargs() -> None:
    from lerobot.robots import so_follower  # noqa: F401
    from lerobot.robots.config import RobotConfig

    from physicalai_lerobot_plugin.studio_catalog import _make_builder, _make_payload_model

    config_cls = RobotConfig.get_known_choices()["so100_follower"]
    payload_cls = _make_payload_model(config_cls)
    follower_builder = _make_builder("so100_follower", config_cls, payload_cls, role="follower")
    assert callable(follower_builder)


def test_make_teleop_builder_config_kwargs() -> None:
    import importlib
    import pkgutil

    import lerobot.teleoperators

    for _importer, modname, is_pkg in pkgutil.walk_packages(
        lerobot.teleoperators.__path__,
        prefix="lerobot.teleoperators.",
    ):
        if "config" in modname and not is_pkg:
            importlib.import_module(modname)

    from lerobot.teleoperators.config import TeleoperatorConfig

    from physicalai_lerobot_plugin.studio_catalog import _make_payload_model, _make_teleop_builder

    teleop_config_cls = TeleoperatorConfig.get_known_choices()["so100_leader"]
    payload_cls = _make_payload_model(teleop_config_cls)
    builder = _make_teleop_builder("so100_leader", teleop_config_cls, payload_cls, role="leader")
    assert callable(builder)


def test_coerce_union_prefers_matching_shape() -> None:
    from physicalai_lerobot_plugin.studio_catalog import _coerce_to_annotation

    assert _coerce_to_annotation(list[float] | float, 1.5) == 1.5
    assert _coerce_to_annotation(list[float] | float, [1.5, 2.5]) == [1.5, 2.5]
    assert _coerce_to_annotation(dict[str, int] | str, "hello") == "hello"
    assert _coerce_to_annotation(dict[str, int] | str, {"a": 1}) == {"a": 1}


# ── URDF path test ─────────────────────────────────────────────────────────


def test_urdf_path_exists() -> None:
    from physicalai_lerobot_plugin import get_urdf_path

    path = get_urdf_path()
    assert path.exists()
    urdf_file = path / "lerobot" / "urdf" / "lerobot.urdf"
    assert urdf_file.exists(), f"URDF file not found at {urdf_file}"
    assert urdf_file.stat().st_size > 0
