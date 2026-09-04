# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Instantiate trusted Config trees."""

from __future__ import annotations

import dataclasses
import inspect
import math
import types
from collections.abc import Mapping
from enum import Enum
from typing import TypeVar, Union, cast, get_args, get_origin, get_type_hints

from ._errors import ConfigError, ConfigImportError
from ._export import declared_config_args
from ._normalize import validate_config
from ._path import format_path
from ._typed_wire import enum_from_wire
from ._types import _MAX_CONFIG_DEPTH, JsonValue
from .base import Config, parse_class_config
from .importing import import_dotted_path

_T = TypeVar("_T")


def _resolved_type_satisfies_expected(resolved: type[object], expected_type: type[object]) -> bool | None:
    """Return whether *resolved* is a subclass of *expected_type*, if checkable."""
    try:
        return issubclass(resolved, expected_type)
    except TypeError:
        return None


def _is_nested_config(value: object) -> bool:
    return isinstance(value, Mapping) and "class_path" in value


def _check_preflight_depth(path: str, depth: int) -> None:
    if depth > _MAX_CONFIG_DEPTH:
        msg = f"{format_path(path)}: nesting depth exceeds {_MAX_CONFIG_DEPTH}"
        raise ConfigError(msg)


def _preflight_config(
    config: object,
    *,
    path: str,
    depth: int,
    seen: set[int],
) -> None:
    """Validate a complete component subtree before any class import.

    Raises:
        ConfigError: If the config tree is malformed.
    """
    _check_preflight_depth(path, depth)

    validated = validate_config(config, path=path)
    config_id = id(config)
    if config_id in seen:
        msg = f"{format_path(path)}: cyclic config is not instantiable"
        raise ConfigError(msg)
    seen.add(config_id)
    try:
        for key, item in validated["init_args"].items():
            child_path = f"{path}.init_args.{key}" if path else f"{validated['class_path']}.init_args.{key}"
            _preflight_value(item, path=child_path, depth=depth + 1, seen=seen)
    finally:
        seen.discard(config_id)


def _preflight_mapping(
    value: dict[object, object],
    *,
    path: str,
    depth: int,
    seen: set[int],
) -> None:
    if "class_path" in value:
        _preflight_config(value, path=path, depth=depth, seen=seen)
        return

    value_id = id(value)
    if value_id in seen:
        msg = f"{format_path(path)}: cyclic mapping is not instantiable"
        raise ConfigError(msg)
    seen.add(value_id)
    try:
        for key, item in value.items():
            if not isinstance(key, str):
                msg = f"{format_path(path)}: mapping keys must be strings, got {type(key).__name__}"
                raise ConfigError(msg)
            child_path = f"{path}.{key}" if path else key
            _preflight_value(item, path=child_path, depth=depth + 1, seen=seen)
    finally:
        seen.discard(value_id)


def _preflight_list(
    value: list[object],
    *,
    path: str,
    depth: int,
    seen: set[int],
) -> None:
    value_id = id(value)
    if value_id in seen:
        msg = f"{format_path(path)}: cyclic sequence is not instantiable"
        raise ConfigError(msg)
    seen.add(value_id)
    try:
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            _preflight_value(item, path=child_path, depth=depth + 1, seen=seen)
    finally:
        seen.discard(value_id)


def _preflight_value(
    value: object,
    *,
    path: str,
    depth: int,
    seen: set[int],
) -> None:
    """Validate one constructor value against the recursive JSON model.

    Raises:
        ConfigError: If the value is outside the JSON model.
    """
    _check_preflight_depth(path, depth)

    if isinstance(value, Enum):
        msg = f"{format_path(path)}: Enum is not a JSON-compatible config value; pass its value instead"
        raise ConfigError(msg)
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            msg = f"{format_path(path)}: non-finite float {value!r} is not JSON-portable"
            raise ConfigError(msg)
        return

    if isinstance(value, dict):
        _preflight_mapping(value, path=path, depth=depth, seen=seen)
        return

    if isinstance(value, list):
        _preflight_list(value, path=path, depth=depth, seen=seen)
        return

    msg = f"{format_path(path)}: {type(value).__name__} is not a JSON-compatible config value"
    raise ConfigError(msg)


def _decode_value(
    value: object,
    *,
    path: str,
    depth: int,
    seen: set[int],
) -> object:
    if depth > _MAX_CONFIG_DEPTH:
        msg = f"{format_path(path)}: nesting depth exceeds {_MAX_CONFIG_DEPTH}"
        raise ConfigError(msg)

    if _is_nested_config(value):
        return _instantiate_impl(
            cast("Config | Mapping[str, JsonValue]", value),
            path=path,
            depth=depth,
            seen=seen,
        )

    if isinstance(value, Mapping):
        obj_id = id(value)
        if obj_id in seen:
            msg = f"{format_path(path)}: cyclic mapping is not instantiable"
            raise ConfigError(msg)
        seen.add(obj_id)
        try:
            result: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    msg = f"{format_path(path)}: mapping keys must be strings, got {type(key).__name__}"
                    raise ConfigError(msg)
                child_path = f"{path}.{key}" if path else key
                result[key] = _decode_value(item, path=child_path, depth=depth + 1, seen=seen)
            return result
        finally:
            seen.discard(obj_id)

    if isinstance(value, list):
        obj_id = id(value)
        if obj_id in seen:
            msg = f"{format_path(path)}: cyclic sequence is not instantiable"
            raise ConfigError(msg)
        seen.add(obj_id)
        try:
            items: list[object] = []
            for index, item in enumerate(value):
                child_path = f"{path}[{index}]" if path else f"[{index}]"
                items.append(_decode_value(item, path=child_path, depth=depth + 1, seen=seen))
            return items
        finally:
            seen.discard(obj_id)

    return value


def _resolve_class(class_path: str, *, path: str) -> type:
    if "<locals>" in class_path:
        msg = f"{format_path(path)}: local class {class_path!r} cannot be imported"
        raise ConfigError(msg)
    try:
        obj = import_dotted_path(class_path)
    except (ValueError, ImportError, AttributeError) as exc:
        msg = f"{format_path(path)}: cannot import class_path {class_path!r}: {exc}"
        raise ConfigImportError(msg) from exc
    if not isinstance(obj, type):
        msg = f"{format_path(path)}: {class_path!r} does not resolve to a class (got {type(obj).__name__})"
        raise ConfigImportError(msg)
    if "<locals>" in obj.__qualname__:
        msg = f"{format_path(path)}: local class {class_path!r} cannot be instantiated"
        raise ConfigError(msg)
    return obj


def _enum_type_from_hint(hint: object) -> type[Enum] | None:
    if isinstance(hint, type) and issubclass(hint, Enum):
        return hint
    origin = get_origin(hint)
    if origin in {types.UnionType, Union}:
        for arg in get_args(hint):
            if isinstance(arg, type) and issubclass(arg, Enum):
                return arg
    return None


def _coerce_constructor_args(cls: type, args: Mapping[str, object]) -> dict[str, object]:
    """Coerce JSON wire values (e.g. enum names) before ``cls(**args)`` construction.

    Returns:
        Constructor kwargs with enum-typed parameters resolved from wire strings.
    """
    try:
        signature = inspect.signature(cls.__init__)
        module = inspect.getmodule(cls)
        globalns = vars(module) if module is not None else {}
        hints = get_type_hints(cls.__init__, globalns=globalns, localns=globalns)
    except (TypeError, ValueError, NameError):
        return dict(args)
    coerced = dict(args)
    for name, param in signature.parameters.items():
        if name in {"self", "args", "kwargs"} or name not in coerced:
            continue
        value = coerced[name]
        if isinstance(value, Enum):
            continue
        hint = hints.get(name, param.annotation)
        if hint is inspect.Parameter.empty:
            continue
        enum_cls = _enum_type_from_hint(hint)
        if enum_cls is not None and isinstance(value, str):
            coerced[name] = enum_from_wire(enum_cls, value)
    return coerced


def _instantiate_impl(
    config: Config | Mapping[str, JsonValue],
    *,
    path: str,
    depth: int,
    seen: set[int],
) -> object:
    if depth > _MAX_CONFIG_DEPTH:
        msg = f"{format_path(path)}: nesting depth exceeds {_MAX_CONFIG_DEPTH}"
        raise ConfigError(msg)

    validated = validate_config(config, path=path)
    cls = _resolve_class(validated["class_path"], path=path)

    if cls is Config:
        inner = validated["init_args"]
        if not isinstance(inner, dict):
            loc = path or validated["class_path"]
            msg = f"{format_path(loc)}: Config init_args must be a mapping"
            raise ConfigError(msg)
        return cls.from_dict(inner)

    config_args = declared_config_args(cls)

    decoded_args: dict[str, object] = {}
    for key, item in validated["init_args"].items():
        if key in config_args:
            # Declared Config data: hand the recipe over untouched so
            # nothing nested is constructed in this process.
            decoded_args[key] = item
            continue
        child_path = f"{path}.init_args.{key}" if path else f"{validated['class_path']}.init_args.{key}"
        decoded_args[key] = _decode_value(item, path=child_path, depth=depth + 1, seen=seen)

    try:
        if dataclasses.is_dataclass(cls) and issubclass(cls, Config):
            return cls.from_dict(decoded_args)
        return cls(**_coerce_constructor_args(cls, decoded_args))
    except Exception as exc:
        loc = path or validated["class_path"]
        exc.add_note(f"{format_path(loc)}: constructor failed while instantiating config")
        raise


def instantiate(config: Config | Mapping[str, JsonValue], *, expected_type: type[_T] | None = None) -> _T | object:
    """Build a fresh component from a trusted :class:`Config`.

    Validates *config* before importing. Recursively instantiates nested
    configs in ``init_args``, then calls the class with keyword
    arguments. Init args a class declares via
    ``@export_config(config_args=...)`` are passed through as plain mappings
    instead of being constructed. Does not invoke lifecycle methods beyond the
    constructor.

    Trusted local application and parent→child startup configs only. Never
    pass network metadata or untrusted peer payloads.

    Malformed configs raise :class:`ConfigError` (including
    :class:`ConfigImportError` for unresolved ``class_path``). Constructor
    failures propagate as their original exception type with path context
    attached via :meth:`BaseException.add_note`.

    Args:
        config: Trusted ``class_path`` + ``init_args`` mapping.
        expected_type: Optional known base class or protocol. When supplied,
            jsonargparse performs typed construction and validation.

    Returns:
        The constructed object, optionally constrained to *expected_type*.

    Raises:
        ConfigError: If the recipe is malformed or fails portable preflight.
        ConfigImportError: If the configured class cannot be imported.
    """
    raw: Config | Mapping[str, JsonValue]
    raw = {"class_path": config.class_path, "init_args": config.init_args} if type(config) is Config else config
    _preflight_config(raw, path="", depth=0, seen=set())
    if expected_type is not None:
        class_path = cast("str", raw["class_path"])
        try:
            target = import_dotted_path(class_path)
        except (ValueError, ImportError, AttributeError) as exc:
            msg = f"cannot import class_path {class_path!r}: {exc}"
            raise ConfigImportError(msg) from exc
        if not isinstance(target, type):
            msg = f"{class_path!r} does not resolve to a class"
            raise ConfigImportError(msg)
        satisfies = _resolved_type_satisfies_expected(target, expected_type)
        if satisfies is False:
            msg = f"{class_path!r} does not satisfy {expected_type.__name__}"
            raise ConfigError(msg)
        result = parse_class_config(target, cast("Mapping[str, object]", raw["init_args"]))
        if satisfies is None and not isinstance(result, expected_type):
            msg = f"{class_path!r} does not satisfy {expected_type.__name__}"
            raise ConfigError(msg)
        return result
    return _instantiate_impl(raw, path="", depth=0, seen=set())
