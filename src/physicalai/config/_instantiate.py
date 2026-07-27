# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Instantiate trusted ComponentConfig trees."""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import Enum
from typing import cast

from ._errors import ComponentConfigError, ComponentImportError
from ._export import declared_config_args
from ._normalize import validate_component_config
from ._path import format_path
from ._types import _MAX_CONFIG_DEPTH, ComponentConfig, JsonValue
from .importing import import_dotted_path


def _is_nested_config(value: object) -> bool:
    return isinstance(value, Mapping) and "class_path" in value


def _check_preflight_depth(path: str, depth: int) -> None:
    if depth > _MAX_CONFIG_DEPTH:
        msg = f"{format_path(path)}: nesting depth exceeds {_MAX_CONFIG_DEPTH}"
        raise ComponentConfigError(msg)


def _preflight_config(
    config: object,
    *,
    path: str,
    depth: int,
    seen: set[int],
) -> None:
    """Validate a complete component subtree before any class import.

    Raises:
        ComponentConfigError: If the config tree is malformed.
    """
    _check_preflight_depth(path, depth)

    validated = validate_component_config(config, path=path)
    config_id = id(config)
    if config_id in seen:
        msg = f"{format_path(path)}: cyclic component config is not instantiable"
        raise ComponentConfigError(msg)
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
        raise ComponentConfigError(msg)
    seen.add(value_id)
    try:
        for key, item in value.items():
            if not isinstance(key, str):
                msg = f"{format_path(path)}: mapping keys must be strings, got {type(key).__name__}"
                raise ComponentConfigError(msg)
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
        raise ComponentConfigError(msg)
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
        ComponentConfigError: If the value is outside the JSON model.
    """
    _check_preflight_depth(path, depth)

    if isinstance(value, Enum):
        msg = f"{format_path(path)}: Enum is not a JSON-compatible component config value; pass its value instead"
        raise ComponentConfigError(msg)
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            msg = f"{format_path(path)}: non-finite float {value!r} is not JSON-portable"
            raise ComponentConfigError(msg)
        return

    if isinstance(value, dict):
        _preflight_mapping(value, path=path, depth=depth, seen=seen)
        return

    if isinstance(value, list):
        _preflight_list(value, path=path, depth=depth, seen=seen)
        return

    msg = f"{format_path(path)}: {type(value).__name__} is not a JSON-compatible component config value"
    raise ComponentConfigError(msg)


def _decode_value(
    value: object,
    *,
    path: str,
    depth: int,
    seen: set[int],
) -> object:
    if depth > _MAX_CONFIG_DEPTH:
        msg = f"{format_path(path)}: nesting depth exceeds {_MAX_CONFIG_DEPTH}"
        raise ComponentConfigError(msg)

    if _is_nested_config(value):
        return _instantiate_impl(
            cast("ComponentConfig | Mapping[str, JsonValue]", value),
            path=path,
            depth=depth,
            seen=seen,
        )

    if isinstance(value, Mapping):
        obj_id = id(value)
        if obj_id in seen:
            msg = f"{format_path(path)}: cyclic mapping is not instantiable"
            raise ComponentConfigError(msg)
        seen.add(obj_id)
        try:
            result: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    msg = f"{format_path(path)}: mapping keys must be strings, got {type(key).__name__}"
                    raise ComponentConfigError(msg)
                child_path = f"{path}.{key}" if path else key
                result[key] = _decode_value(item, path=child_path, depth=depth + 1, seen=seen)
            return result
        finally:
            seen.discard(obj_id)

    if isinstance(value, list):
        obj_id = id(value)
        if obj_id in seen:
            msg = f"{format_path(path)}: cyclic sequence is not instantiable"
            raise ComponentConfigError(msg)
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
        raise ComponentConfigError(msg)
    try:
        obj = import_dotted_path(class_path)
    except (ValueError, ImportError, AttributeError) as exc:
        msg = f"{format_path(path)}: cannot import class_path {class_path!r}: {exc}"
        raise ComponentImportError(msg) from exc
    if not isinstance(obj, type):
        msg = f"{format_path(path)}: {class_path!r} does not resolve to a class (got {type(obj).__name__})"
        raise ComponentImportError(msg)
    if "<locals>" in obj.__qualname__:
        msg = f"{format_path(path)}: local class {class_path!r} cannot be instantiated"
        raise ComponentConfigError(msg)
    return obj


def _instantiate_impl(
    config: ComponentConfig | Mapping[str, JsonValue],
    *,
    path: str,
    depth: int,
    seen: set[int],
) -> object:
    if depth > _MAX_CONFIG_DEPTH:
        msg = f"{format_path(path)}: nesting depth exceeds {_MAX_CONFIG_DEPTH}"
        raise ComponentConfigError(msg)

    validated = validate_component_config(config, path=path)
    cls = _resolve_class(validated["class_path"], path=path)
    config_args = declared_config_args(cls)

    decoded_args: dict[str, object] = {}
    for key, item in validated["init_args"].items():
        if key in config_args:
            # Declared ComponentConfig data: hand the recipe over untouched so
            # nothing nested is constructed in this process.
            decoded_args[key] = item
            continue
        child_path = f"{path}.init_args.{key}" if path else f"{validated['class_path']}.init_args.{key}"
        decoded_args[key] = _decode_value(item, path=child_path, depth=depth + 1, seen=seen)

    try:
        return cls(**decoded_args)
    except Exception as exc:
        loc = path or validated["class_path"]
        exc.add_note(f"{format_path(loc)}: constructor failed while instantiating component config")
        raise


def instantiate(config: ComponentConfig | Mapping[str, JsonValue]) -> object:
    """Build a fresh component from a trusted :class:`ComponentConfig`.

    Validates *config* before importing. Recursively instantiates nested
    component configs in ``init_args``, then calls the class with keyword
    arguments. Init args a class declares via
    ``@export_config(config_args=...)`` are passed through as plain mappings
    instead of being constructed. Does not invoke lifecycle methods beyond the
    constructor.

    Trusted local application and parent→child startup configs only. Never
    pass network metadata or untrusted peer payloads.

    Malformed configs raise :class:`ComponentConfigError` (including
    :class:`ComponentImportError` for unresolved ``class_path``). Constructor
    failures propagate as their original exception type with path context
    attached via :meth:`BaseException.add_note`.

    Args:
        config: Trusted ``class_path`` + ``init_args`` mapping.

    Returns:
        A new instance of the configured class.
    """
    _preflight_config(config, path="", depth=0, seen=set())
    return _instantiate_impl(config, path="", depth=0, seen=set())
