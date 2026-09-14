# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: PLC0415

"""Studio catalog plugin for LeRobot adapters.

Dynamically registers one ``RobotCatalogDefinition`` per LeRobot robot type,
creating a typed Pydantic payload model for each from the lerobot
``RobotConfig`` dataclass.
"""

from __future__ import annotations

import dataclasses
import importlib.machinery
import sys
import types
from itertools import starmap
from typing import TYPE_CHECKING, Any, Literal, TypeGuard, Union, cast, get_args, get_origin

from loguru import logger
from physicalai_studio_plugin import (
    CatalogRobotFactory,
    PayloadContainer,
    PortScanner,
    RobotAdapterOptions,
    RobotCatalogDefinition,
    RobotFieldUiOptions,
    RobotProbe,
    SerialPortInfo,
    robot_field_ui,
    robot_payload_ui,
)
from pydantic import BaseModel, ConfigDict, Field, create_model
from serial.tools import list_ports

from physicalai_lerobot_plugin.constants import trust_unverified_plugins

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Protocol

    from physicalai.robot.interface import Robot as PhysicalAIRobot

    class _RobotCatalogRegistry(Protocol):
        def register_robot(self, definition: RobotCatalogDefinition) -> None: ...


# ── Robots to exclude from dynamic registration ────────────────────────────

_ROBOTS_TO_SKIP: frozenset[str] = frozenset({
    "mock_robot",  # test-only
})


# ── Lerobot config importing ───────────────────────────────────────────────

_LEROBOT_CONFIGS_IMPORTED: bool = False
_LEROBOT_THIRD_PARTY_PLUGINS_IMPORTED: bool = False


def _ensure_optional_dependency_module_specs() -> None:
    """Normalize optional dependency stubs in ``sys.modules`` for ``find_spec``.

    Some tests inject ``MagicMock`` placeholders for optional hardware SDKs.
    ``importlib.util.find_spec()`` raises ``ValueError`` when a loaded module's
    ``__spec__`` is missing, so we attach a minimal ``ModuleSpec`` when needed.
    """
    for module_name in ("scservo_sdk", "motorbridge", "motorbridge_smart_servo"):
        loaded = sys.modules.get(module_name)
        if loaded is None:
            continue
        if not isinstance(getattr(loaded, "__spec__", None), importlib.machinery.ModuleSpec):
            loaded.__spec__ = importlib.machinery.ModuleSpec(module_name, loader=None)


def _ensure_lerobot_configs_imported() -> None:
    """Walk the ``lerobot.robots`` package and import every ``config_*`` module.

    This triggers the ``@RobotConfig.register_subclass(...)`` decorators so
    that ``RobotConfig.get_known_choices()`` returns all available types.
    """
    global _LEROBOT_CONFIGS_IMPORTED  # noqa: PLW0603
    if _LEROBOT_CONFIGS_IMPORTED:
        return
    _ensure_optional_dependency_module_specs()
    import importlib
    import pkgutil

    import lerobot.robots

    for _importer, modname, is_pkg in pkgutil.walk_packages(lerobot.robots.__path__, prefix="lerobot.robots."):
        if "config" in modname and not is_pkg and _is_allowed_dynamic_import(modname):
            # Modules are enumerated below the allowlisted lerobot.robots package.
            # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
            importlib.import_module(modname)
    _LEROBOT_CONFIGS_IMPORTED = True


def _ensure_lerobot_third_party_plugins_imported() -> None:
    """Load explicitly trusted third-party extensions through LeRobot discovery.

    LeRobot scans installed distributions whose import names begin with
    ``lerobot_robot_`` or ``lerobot_teleoperator_`` and imports their package
    roots. Compatible extensions register config subclasses during that import.
    LeRobot handles individual optional-plugin import failures without
    preventing other installed extensions from loading.
    """
    global _LEROBOT_THIRD_PARTY_PLUGINS_IMPORTED  # noqa: PLW0603
    if _LEROBOT_THIRD_PARTY_PLUGINS_IMPORTED or not trust_unverified_plugins():
        return

    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    _LEROBOT_THIRD_PARTY_PLUGINS_IMPORTED = True


# ── Payload model factory ──────────────────────────────────────────────────

_REQUIRED_SENTINEL: Any = object()
_PAYLOAD_MODEL_CACHE: dict[type, type[BaseModel]] = {}
_PAIR_LEN: int = 2
_NON_SERIAL_PORT_MODULES: frozenset[str] = frozenset({"reachy2", "yam_follower"})
_ADVANCED_CONFIGURATION_NAME_MARKERS: frozenset[str] = frozenset({
    "calibration",
    "baudrate",
    "offset",
    "can_adapter",
})


def _is_allowed_dynamic_import(module_name: str) -> bool:
    """Return whether a module name is trusted for dynamic importing."""
    return module_name.startswith("lerobot.") or (
        trust_unverified_plugins() and module_name.startswith(("lerobot_robot_", "lerobot_teleoperator_"))
    )


def _is_dataclass_type(annotation: object) -> TypeGuard[type[Any]]:
    return isinstance(annotation, type) and dataclasses.is_dataclass(annotation)


def _payload_types_namespace(config_cls: type[Any], visited: set[type[Any]] | None = None) -> dict[str, object]:
    """Collect namespaces needed by dataclasses nested in a config schema.

    Returns:
        Combined module namespaces for the config and nested dataclasses.
    """
    visited = set() if visited is None else visited
    if config_cls in visited:
        return {}
    visited.add(config_cls)

    module_name = config_cls.__module__
    if not _is_allowed_dynamic_import(module_name):
        return {}

    module = sys.modules.get(module_name)
    if module is None:
        return {}

    namespace = vars(module).copy()
    for field in dataclasses.fields(config_cls):
        namespace.update(_annotation_types_namespace(field.type, visited))
    return namespace


def _annotation_types_namespace(annotation: object, visited: set[type[Any]]) -> dict[str, object]:
    if _is_dataclass_type(annotation):
        return _payload_types_namespace(annotation, visited)

    namespace: dict[str, object] = {}
    for argument in get_args(annotation):
        namespace.update(_annotation_types_namespace(argument, visited))
    return namespace


def _annotation_has_str(annotation: object) -> bool:
    if annotation is str:
        return True

    origin = get_origin(annotation)
    if origin in {Union, types.UnionType}:
        return any(_annotation_has_str(arg) for arg in get_args(annotation) if arg is not type(None))

    return False


def _is_serial_port_field(config_cls: type, field: dataclasses.Field[Any]) -> bool:
    """Return whether a config field represents a discoverable serial path."""
    return (
        field.name == "port"
        and not any(f".{name}." in config_cls.__module__ for name in _NON_SERIAL_PORT_MODULES)
        and _annotation_has_str(field.type)
    )


def _to_payload_annotation(annotation: object) -> object:  # noqa: PLR0911
    if annotation is Any:
        return Any

    if _is_dataclass_type(annotation):
        return _make_payload_model(annotation)

    origin = get_origin(annotation)
    if origin is None:
        return annotation

    args = get_args(annotation)
    converted_args = tuple(_to_payload_annotation(arg) for arg in args)

    if origin in {Union, types.UnionType}:
        result: Any = converted_args[0]
        for arg in converted_args[1:]:
            result |= cast("Any", arg)
        return result

    if origin is Literal:
        return annotation

    if origin in {list, set, frozenset, tuple, dict}:
        if not converted_args:
            return origin
        return origin[converted_args]  # type: ignore[index]

    return annotation


def _to_plain_data(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, dict):
        return {k: _to_plain_data(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_data(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_plain_data(v) for v in value)
    if isinstance(value, set):
        return {_to_plain_data(v) for v in value}
    return value


def _coerce_to_annotation(  # noqa: C901, PLR0911, PLR0912
    annotation: object,
    value: object,
) -> object:
    if value is None:
        return None

    if annotation is Any:
        return value

    if _is_dataclass_type(annotation):
        if isinstance(annotation, type) and isinstance(value, annotation):
            return value
        if isinstance(value, dict):
            return _materialize_dataclass(annotation, value)
        return value

    origin = get_origin(annotation)
    if origin is None:
        return value

    args = get_args(annotation)
    if origin in {Union, types.UnionType}:
        non_none_args = [arg for arg in args if arg is not type(None)]
        for arg in non_none_args:
            converted = _coerce_to_annotation(arg, value)
            expected = get_origin(arg)
            if expected is None and isinstance(arg, type):
                if isinstance(converted, arg):
                    return converted
            elif expected is not None:
                if expected is list and not isinstance(value, list):
                    continue
                if expected is dict and not isinstance(value, dict):
                    continue
                if expected is tuple and not isinstance(value, tuple | list):
                    continue
                if expected is set and not isinstance(value, set | list | tuple):
                    continue
                return converted
        return value

    if origin is list and args and isinstance(value, list):
        return [_coerce_to_annotation(args[0], item) for item in value]
    if origin is tuple and args and isinstance(value, tuple | list):
        if len(args) == _PAIR_LEN and args[1] is Ellipsis:
            return tuple(_coerce_to_annotation(args[0], item) for item in value)
        return tuple(starmap(_coerce_to_annotation, zip(args, value, strict=False)))
    if origin is dict and len(args) == _PAIR_LEN and isinstance(value, dict):
        key_type, val_type = args
        return {
            _coerce_to_annotation(key_type, key): _coerce_to_annotation(val_type, val) for key, val in value.items()
        }
    if origin is set and args and isinstance(value, set | list | tuple):
        return {_coerce_to_annotation(args[0], item) for item in value}

    return value


def _materialize_dataclass(config_cls: type[object], payload_data: dict[str, object]) -> object:
    kwargs: dict[str, object] = {}
    for f in dataclasses.fields(config_cls):
        if f.name not in payload_data:
            continue
        kwargs[f.name] = _coerce_to_annotation(f.type, payload_data[f.name])
    return config_cls(**kwargs)


def _config_to_kwargs(config: object) -> dict[str, object]:
    if not dataclasses.is_dataclass(config):
        msg = f"Expected dataclass instance, got {type(config)!r}"
        raise TypeError(msg)
    return {field.name: getattr(config, field.name) for field in dataclasses.fields(config)}


async def _resolve_ports_in_dataclass(config: object, factory: CatalogRobotFactory, path: str = "") -> None:
    if not dataclasses.is_dataclass(config):
        msg = f"Expected dataclass instance, got {type(config)!r}"
        raise TypeError(msg)

    config_type = type(config)
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        field_path = f"{path}.{field.name}" if path else field.name

        if _is_dataclass_type(field.type) and value is not None:
            await _resolve_ports_in_dataclass(value, factory, field_path)
            continue

        if _is_serial_port_field(config_type, field) and isinstance(value, str):
            if not value:
                msg = f"Missing required port at {field_path}."
                raise ValueError(msg)
            resolved = await factory.find_port(SerialPortInfo(connection_string=value, serial_number=None))
            if resolved is not None:
                setattr(config, field.name, resolved)


def _iter_port_values(payload: object) -> list[str]:
    values: list[str] = []
    if isinstance(payload, BaseModel):
        return _iter_port_values(payload.model_dump(mode="python"))
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "port" and isinstance(value, str) and value:
                values.append(value)
            values.extend(_iter_port_values(value))
        return values
    if isinstance(payload, list | tuple | set):
        for item in payload:
            values.extend(_iter_port_values(item))
    return values


def _resolve_field_type(
    f: dataclasses.Field[Any],
) -> tuple[Any, Any]:
    """Resolve a dataclass field into a (pydantic_type, default) pair.

    The original annotation and default are preserved, and nested dataclasses
    and container types are converted recursively via ``_to_payload_annotation``.

    Returns:
        A tuple of (pydantic_type, default_value).
    """
    pydantic_type = _to_payload_annotation(f.type)
    if f.default is not dataclasses.MISSING:
        default_val = f.default
    elif f.default_factory is not dataclasses.MISSING:
        default_val = f.default_factory()
    else:
        default_val = _REQUIRED_SENTINEL

    return pydantic_type, default_val


def _make_payload_model(config_cls: type[Any]) -> type[BaseModel]:
    """Dynamically create a Pydantic payload model from a lerobot ``RobotConfig``.

    The model adds one ``Field`` per dataclass attribute, with nested
    dataclasses and container types converted recursively into Pydantic types
    via ``_to_payload_annotation``.

    Args:
        config_cls: A lerobot ``RobotConfig`` subclass.

    Returns:
        A new Pydantic ``BaseModel`` subclass.
    """
    cached = _PAYLOAD_MODEL_CACHE.get(config_cls)
    if cached is not None:
        return cached

    field_defs: dict[str, tuple[Any, Any]] = {}
    for f in dataclasses.fields(config_cls):
        pydantic_type, default_val = _resolve_field_type(f)
        ui_options: RobotFieldUiOptions = {}
        if f.name == "id":
            ui_options["required"] = True
        if any(marker in f.name for marker in _ADVANCED_CONFIGURATION_NAME_MARKERS):
            ui_options["advanced_configuration"] = True
        extra = robot_field_ui(ui_options) if ui_options else None
        if default_val is _REQUIRED_SENTINEL:
            field_defs[f.name] = (pydantic_type, Field(..., description=f.name, json_schema_extra=extra))
        else:
            field_defs[f.name] = (
                pydantic_type,
                Field(default=default_val, description=f.name, json_schema_extra=extra),
            )

    payload_model = create_model(
        f"{config_cls.__name__}Payload",
        __config__=ConfigDict(
            json_schema_extra=robot_payload_ui(
                [
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
                ],
            ),
        )
        if any(_is_serial_port_field(config_cls, f) for f in dataclasses.fields(config_cls))
        else None,
        **field_defs,
    )
    payload_model.__pydantic_parent_namespace__ = _payload_types_namespace(config_cls)
    _PAYLOAD_MODEL_CACHE[config_cls] = payload_model
    return payload_model


_LEROBOT_TELEOPERATORS_IMPORTED: bool = False


def _ensure_lerobot_teleoperators_imported() -> None:
    """Walk the ``lerobot.teleoperators`` package and import every ``config_*`` module.

    This triggers the ``@TeleoperatorConfig.register_subclass(...)`` decorators
    so that ``TeleoperatorConfig.get_known_choices()`` returns all types.
    """
    global _LEROBOT_TELEOPERATORS_IMPORTED  # noqa: PLW0603
    if _LEROBOT_TELEOPERATORS_IMPORTED:
        return
    _ensure_optional_dependency_module_specs()
    import importlib
    import pkgutil

    import lerobot.teleoperators

    for _importer, modname, is_pkg in pkgutil.walk_packages(
        lerobot.teleoperators.__path__,
        prefix="lerobot.teleoperators.",
    ):
        if "config" in modname and not is_pkg and _is_allowed_dynamic_import(modname):
            # Modules are enumerated below the allowlisted lerobot.teleoperators package.
            # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
            importlib.import_module(modname)
    _LEROBOT_TELEOPERATORS_IMPORTED = True


# ── Builder factory ────────────────────────────────────────────────────────


def _make_builder(
    config_type: str,
    config_cls: type[Any],
    payload_cls: type[BaseModel],
    role: Literal["follower", "leader"],
) -> Callable[[PayloadContainer[Any], CatalogRobotFactory], Any]:
    """Create an async builder function for a given lerobot robot type.

    Args:
        config_type: The lerobot ``RobotConfig`` type name.
        config_cls: The lerobot ``RobotConfig`` dataclass to instantiate.
        payload_cls: The corresponding Pydantic payload model.
        role: ``"follower"`` or ``"leader"``.

    Returns:
        An async callable ``(robot, factory) -> PhysicalAIRobot``.
    """

    async def _build(
        robot: PayloadContainer[Any],
        factory: CatalogRobotFactory,
    ) -> PhysicalAIRobot:
        from lerobot.robots import make_robot_from_config

        from physicalai_lerobot_plugin.lerobot_adapter import LeRobotAdapter

        raw = robot.payload
        if isinstance(raw, BaseModel) and type(raw) is not payload_cls:
            raw = raw.model_dump()
        validated = raw if isinstance(raw, payload_cls) else payload_cls.model_validate(raw)

        payload_data = _to_plain_data(validated)
        if not isinstance(payload_data, dict):
            msg = "Payload validation did not produce an object"
            raise TypeError(msg)
        lerobot_config = _materialize_dataclass(config_cls, payload_data)
        await _resolve_ports_in_dataclass(lerobot_config, factory)
        config_kwargs = _config_to_kwargs(lerobot_config)
        lerobot_robot = make_robot_from_config(lerobot_config)
        return LeRobotAdapter(
            config_type,
            config_kwargs,
            role=role,
            _robot=lerobot_robot,
        )

    return _build


def _make_teleop_builder(
    config_type: str,
    config_cls: type[Any],
    payload_cls: type[BaseModel],
    role: Literal["follower", "leader"],
) -> Callable[[PayloadContainer[Any], CatalogRobotFactory], Any]:
    """Create an async builder for a lerobot teleoperator.

    Args:
        config_type: The lerobot ``TeleoperatorConfig`` type name.
        config_cls: The lerobot ``TeleoperatorConfig`` dataclass to instantiate.
        payload_cls: The corresponding Pydantic payload model.
        role: ``"leader"`` or ``"follower"``.

    Returns:
        An async callable ``(robot, factory) -> PhysicalAIRobot``.
    """

    async def _build(
        robot: PayloadContainer[Any],
        factory: CatalogRobotFactory,
    ) -> PhysicalAIRobot:
        from lerobot.teleoperators import make_teleoperator_from_config

        from physicalai_lerobot_plugin.lerobot_adapter import (
            LeRobotTeleoperatorAdapter,
        )

        raw = robot.payload
        if isinstance(raw, BaseModel) and type(raw) is not payload_cls:
            raw = raw.model_dump()
        validated = raw if isinstance(raw, payload_cls) else payload_cls.model_validate(raw)

        payload_data = _to_plain_data(validated)
        if not isinstance(payload_data, dict):
            msg = "Payload validation did not produce an object"
            raise TypeError(msg)
        teleop_config = _materialize_dataclass(config_cls, payload_data)
        await _resolve_ports_in_dataclass(teleop_config, factory)
        config_kwargs = _config_to_kwargs(teleop_config)
        teleoperator = make_teleoperator_from_config(teleop_config)
        return LeRobotTeleoperatorAdapter(
            config_type,
            config_kwargs,
            role=role,
            _teleoperator=teleoperator,
        )

    return _build


# ── Probe ──────────────────────────────────────────────────────────────────


class LeRobotProbe(RobotProbe[BaseModel]):
    """Probe implementation for LeRobot-adapted serial devices."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:
        """Discover available serial devices.

        Returns:
            list[SerialPortInfo]: Detected serial devices.
        """
        _ = self
        await manager.find_robots()
        return manager.robots

    async def identify(
        self,
        payload: BaseModel,
        manager: PortScanner | None = None,
        joint: str | None = None,
    ) -> None:
        """Request a visual identify action, if supported."""
        _ = self, payload, manager, joint

    async def is_online(self, payload: BaseModel, manager: PortScanner | None = None) -> bool:
        """Check whether the configured robot endpoint is online.

        Returns:
            bool: ``True`` when any configured serial port is present. Endpoints
            without serial ports cannot be probed and are treated as online.
        """
        _ = self
        configured_ports = set(_iter_port_values(payload))
        if not configured_ports:
            return True

        if manager is not None:
            discovered_ports = {p.connection_string for p in manager.robots}
            return any(port in discovered_ports for port in configured_ports)

        all_ports = list_ports.comports()
        discovered_ports = {p.device for p in all_ports}
        return any(port in discovered_ports for port in configured_ports)


_LEROBOT_PROBE = LeRobotProbe()


# ── Registration ───────────────────────────────────────────────────────────


def _definitions() -> list[RobotCatalogDefinition]:
    """Build catalog definitions for every registered LeRobot endpoint type.

    Followers are built from ``RobotConfig`` + ``make_robot_from_config``.
    Leaders are built from ``TeleoperatorConfig`` +
    ``make_teleoperator_from_config``. Every type has a role suffix in its
    Studio ID so robot and teleoperator choice names cannot collide.

    Returns:
        A list of ``RobotCatalogDefinition`` instances.
    """
    _ensure_lerobot_configs_imported()
    _ensure_lerobot_teleoperators_imported()
    _ensure_lerobot_third_party_plugins_imported()
    from lerobot.robots.config import RobotConfig
    from lerobot.teleoperators.config import TeleoperatorConfig

    defs: list[RobotCatalogDefinition] = []
    for type_str, config_cls in RobotConfig.get_known_choices().items():
        if type_str in _ROBOTS_TO_SKIP:
            continue
        try:
            payload_cls = _make_payload_model(config_cls)
            defs.append(
                RobotCatalogDefinition(
                    type=f"LeRobot_{type_str}_Follower",
                    display_name=f"LeRobot {type_str} Follower",
                    role="follower",
                    robot_builder=_make_builder(type_str, config_cls, payload_cls, "follower"),
                    robot_payload=payload_cls,
                    asset=None,
                    adapter_options=RobotAdapterOptions(include_velocities=False, external_effort_gain=None),
                    probe=_LEROBOT_PROBE,
                ),
            )
        except Exception:  # noqa: BLE001
            logger.exception("Skipping LeRobot follower type '{}' because its schema is incompatible", type_str)

    for type_str, config_cls in TeleoperatorConfig.get_known_choices().items():
        try:
            payload_cls = _make_payload_model(config_cls)
            defs.append(
                RobotCatalogDefinition(
                    type=f"LeRobot_{type_str}_Leader",
                    display_name=f"LeRobot {type_str} Leader",
                    role="leader",
                    robot_builder=_make_teleop_builder(type_str, config_cls, payload_cls, "leader"),
                    robot_payload=payload_cls,
                    asset=None,
                    adapter_options=RobotAdapterOptions(include_velocities=False, external_effort_gain=None),
                    probe=_LEROBOT_PROBE,
                ),
            )
        except Exception:  # noqa: BLE001
            logger.exception("Skipping LeRobot leader type '{}' because its schema is incompatible", type_str)
    return defs


def _rebuild_payload_models(annotation: object, visited: set[type[BaseModel]] | None = None) -> None:
    """Rebuild generated nested payload models before their parent model.

    Pydantic does not rebuild nested dynamically-created models when rebuilding
    a parent. FastAPI later traverses those nested models while generating
    OpenAPI, so they must be complete independently.
    """
    visited = set() if visited is None else visited
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if annotation in visited:
            return
        visited.add(annotation)
        for field in annotation.model_fields.values():
            _rebuild_payload_models(field.annotation, visited)
        annotation.model_rebuild(
            _types_namespace={**globals(), **annotation.__pydantic_parent_namespace__},
            raise_errors=True,
        )
        return

    for argument in get_args(annotation):
        _rebuild_payload_models(argument, visited)


def _assert_payload_model_resolvable(model: type[BaseModel]) -> None:
    _rebuild_payload_models(model)


def register_physicalai_studio_plugin(registry: _RobotCatalogRegistry) -> None:
    """Register LeRobot catalog entries with the Physical AI Studio registry."""
    for definition in _definitions():
        payload_model = definition.robot_payload
        try:
            if isinstance(payload_model, type) and issubclass(payload_model, BaseModel):
                _assert_payload_model_resolvable(payload_model)
            registry.register_robot(definition)
        except Exception:  # noqa: BLE001
            # Third-party config schemas can contain annotations that cannot be
            # represented in Studio. Do not let one extension hide all later types.
            logger.exception("Skipping LeRobot catalog type '{}' because its schema is incompatible", definition.type)
