# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Normalize live values into JSON-safe component configuration fragments."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from enum import Enum
from pathlib import Path
from typing import cast

from ._errors import ComponentConfigError
from ._path import format_path
from ._types import (
    _MAX_CONFIG_DEPTH,
    _REPR_LIMIT,
    ComponentConfig,
    JsonValue,
)

ToConfigFn = Callable[..., ComponentConfig]
IsExportableFn = Callable[[object], bool]
DomainCodecFn = Callable[[object], JsonValue | None]


def _is_component_config_mapping(value: Mapping[object, object]) -> bool:
    return "class_path" in value


def validate_component_config(config: object, *, path: str = "") -> ComponentConfig:
    """Validate a mapping as a :class:`ComponentConfig` without importing.

    Args:
        config: Candidate config mapping.
        path: Argument path prefix for error messages.

    Returns:
        The validated config (``init_args`` defaulted to ``{}`` when omitted).

    Raises:
        ComponentConfigError: If the mapping is malformed.
    """
    loc = format_path(path)
    if not isinstance(config, dict):
        msg = f"{loc}: component config must be a mapping, got {type(config).__name__}"
        raise ComponentConfigError(msg)

    keys = set(config)
    allowed = {"class_path", "init_args"}
    extra = keys - allowed
    if extra:
        extras = ", ".join(sorted(repr(k) for k in extra))
        msg = f"{loc}: nested component config may only contain 'class_path' and 'init_args'; unexpected keys: {extras}"
        raise ComponentConfigError(msg)
    if "class_path" not in config:
        msg = f"{loc}: component config missing required 'class_path'"
        raise ComponentConfigError(msg)

    class_path = config["class_path"]
    if not isinstance(class_path, str) or not class_path:
        msg = f"{loc}: 'class_path' must be a non-empty string"
        raise ComponentConfigError(msg)

    if "init_args" not in config:
        init_args: object = {}
    else:
        init_args = config["init_args"]
        if init_args is None:
            msg = f"{loc}: 'init_args' must be a mapping, got None"
            raise ComponentConfigError(msg)
    if not isinstance(init_args, dict):
        msg = f"{loc}: 'init_args' must be a mapping, got {type(init_args).__name__}"
        raise ComponentConfigError(msg)
    for key in init_args:
        if not isinstance(key, str):
            msg = f"{loc}.init_args: keys must be strings, got {type(key).__name__}"
            raise ComponentConfigError(msg)

    return {"class_path": class_path, "init_args": dict(init_args)}


def _check_depth(path: str, depth: int) -> None:
    if depth > _MAX_CONFIG_DEPTH:
        msg = f"{format_path(path)}: nesting depth exceeds {_MAX_CONFIG_DEPTH}"
        raise ComponentConfigError(msg)


def _try_normalize_scalar(value: object, *, path: str) -> tuple[bool, JsonValue]:
    """Return ``(True, json_value)`` for scalars/Path, else ``(False, None)``.

    Raises:
        ComponentConfigError: If *value* is a non-finite float.
    """
    if value is None or isinstance(value, (bool, str)):
        return True, value
    if isinstance(value, int) and not isinstance(value, bool):
        return True, value
    if isinstance(value, float):
        if not math.isfinite(value):
            msg = f"{format_path(path)}: non-finite float {value!r} is not JSON-portable"
            raise ComponentConfigError(msg)
        return True, value
    if isinstance(value, Path):
        return True, str(value)
    return False, None


def _normalize_enum(value: Enum, *, path: str) -> JsonValue:
    enum_value = value.value
    if isinstance(enum_value, (bool, str)) or (
        isinstance(enum_value, (int, float)) and not isinstance(enum_value, bool)
    ):
        if isinstance(enum_value, float) and not math.isfinite(enum_value):
            msg = f"{format_path(path)}: non-finite enum value {enum_value!r}"
            raise ComponentConfigError(msg)
        return enum_value  # type: ignore[return-value]
    msg = f"{format_path(path)}: enum {type(value).__name__} value {type(enum_value).__name__} is not JSON-safe"
    raise ComponentConfigError(msg)


def _normalize_exportable(
    value: object,
    *,
    path: str,
    depth: int,
    seen: set[int],
    to_config: ToConfigFn,
) -> JsonValue:
    # Match instantiate: a nested component config found at *depth* is itself at
    # *depth* (its init_args then advance to depth + 1 inside to_config).
    nested = to_config(value, _path=path, _depth=depth, _seen=seen)
    return cast("JsonValue", dict(nested))


def _normalize_mapping(
    value: Mapping[object, object],
    *,
    path: str,
    depth: int,
    seen: set[int],
    to_config: ToConfigFn | None,
    is_exportable: IsExportableFn | None,
    domain_codec: DomainCodecFn | None,
) -> JsonValue:
    obj_id = id(value)
    if obj_id in seen:
        msg = f"{format_path(path)}: cyclic mapping is not serializable"
        raise ComponentConfigError(msg)

    if _is_component_config_mapping(value):
        validated = validate_component_config(value, path=path)
        nested_args: dict[str, JsonValue] = {}
        seen.add(obj_id)
        try:
            for key, item in validated["init_args"].items():
                child_path = f"{path}.init_args.{key}" if path else f"init_args.{key}"
                nested_args[key] = normalize_value(
                    item,
                    path=child_path,
                    depth=depth + 1,
                    seen=seen,
                    to_config=to_config,
                    is_exportable=is_exportable,
                    domain_codec=domain_codec,
                )
        finally:
            seen.discard(obj_id)
        return {"class_path": validated["class_path"], "init_args": nested_args}

    seen.add(obj_id)
    try:
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                msg = f"{format_path(path)}: mapping keys must be strings, got {type(key).__name__}"
                raise ComponentConfigError(msg)
            child_path = f"{path}.{key}" if path else key
            result[key] = normalize_value(
                item,
                path=child_path,
                depth=depth + 1,
                seen=seen,
                to_config=to_config,
                is_exportable=is_exportable,
                domain_codec=domain_codec,
            )
        return result
    finally:
        seen.discard(obj_id)


def _normalize_sequence(
    value: list[object] | tuple[object, ...],
    *,
    path: str,
    depth: int,
    seen: set[int],
    to_config: ToConfigFn | None,
    is_exportable: IsExportableFn | None,
    domain_codec: DomainCodecFn | None,
) -> JsonValue:
    obj_id = id(value)
    if obj_id in seen:
        msg = f"{format_path(path)}: cyclic sequence is not serializable"
        raise ComponentConfigError(msg)
    seen.add(obj_id)
    try:
        items: list[JsonValue] = []
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            items.append(
                normalize_value(
                    item,
                    path=child_path,
                    depth=depth + 1,
                    seen=seen,
                    to_config=to_config,
                    is_exportable=is_exportable,
                    domain_codec=domain_codec,
                )
            )
        return items
    finally:
        seen.discard(obj_id)


def _unsupported_message(value: object, *, path: str) -> str:
    type_name = type(value).__name__
    repr_value = repr(value)
    if len(repr_value) > _REPR_LIMIT:
        repr_value = repr_value[: _REPR_LIMIT - 3] + "..."
    return f"{format_path(path)}: cannot encode {type_name} {repr_value}; omit it or use a supported component value"


def normalize_value(
    value: object,
    *,
    path: str = "",
    depth: int = 0,
    seen: set[int] | None = None,
    to_config: ToConfigFn | None = None,
    is_exportable: IsExportableFn | None = None,
    domain_codec: DomainCodecFn | None = None,
) -> JsonValue:
    """Recursively normalize *value* to a JSON-safe representation.

    Args:
        value: Captured constructor argument or nested structure.
        path: Argument path for error messages.
        depth: Current nesting depth (configs, lists, and mappings).
        seen: Container identities already visited (cycle detection).
        to_config: Callback for nested exportable components.
        is_exportable: Predicate for nested exportable components.
        domain_codec: Optional codec returning a JSON value or ``None``.

    Returns:
        A JSON-safe value.

    Raises:
        ComponentConfigError: On unsupported values, cycles, or depth overflow.
    """
    _check_depth(path, depth)
    if seen is None:
        seen = set()

    handled, scalar = _try_normalize_scalar(value, path=path)
    if handled:
        return scalar

    if isinstance(value, Enum):
        return _normalize_enum(value, path=path)

    if is_exportable is not None and to_config is not None and is_exportable(value):
        return _normalize_exportable(value, path=path, depth=depth, seen=seen, to_config=to_config)

    if domain_codec is not None:
        encoded = domain_codec(value)
        if encoded is not None:
            return encoded

    if isinstance(value, Mapping):
        return _normalize_mapping(
            value,
            path=path,
            depth=depth,
            seen=seen,
            to_config=to_config,
            is_exportable=is_exportable,
            domain_codec=domain_codec,
        )

    if isinstance(value, (list, tuple)):
        return _normalize_sequence(
            value,
            path=path,
            depth=depth,
            seen=seen,
            to_config=to_config,
            is_exportable=is_exportable,
            domain_codec=domain_codec,
        )

    msg = _unsupported_message(value, path=path)
    raise ComponentConfigError(msg)


def snapshot_captured_value(
    value: object,
    *,
    keep_by_reference: Callable[[object], bool] | None = None,
    memo: dict[int, object] | None = None,
) -> object:
    """Deep-snapshot built-in mutable containers; keep selected values by reference.

    Nested exportable components (and other values matching
    *keep_by_reference*) are retained by identity so their own conversion
    remains authoritative. Cyclic containers are preserved via *memo* so
    :func:`~physicalai.config.to_config` can reject them during normalization.

    Returns:
        A snapshot of *value* suitable for later normalization.
    """
    if memo is None:
        memo = {}

    if isinstance(value, (dict, list, tuple)):
        obj_id = id(value)
        if obj_id in memo:
            return memo[obj_id]
        if isinstance(value, dict):
            result: dict[object, object] = {}
            memo[obj_id] = result
            for key, item in value.items():
                result[key] = snapshot_captured_value(item, keep_by_reference=keep_by_reference, memo=memo)
            snap: object = result
        elif isinstance(value, list):
            result_list: list[object] = []
            memo[obj_id] = result_list
            result_list.extend(
                snapshot_captured_value(item, keep_by_reference=keep_by_reference, memo=memo) for item in value
            )
            snap = result_list
        else:
            memo[obj_id] = value
            snap = tuple(
                snapshot_captured_value(item, keep_by_reference=keep_by_reference, memo=memo) for item in value
            )
            memo[obj_id] = snap
        return snap

    if keep_by_reference is not None and keep_by_reference(value):
        return value
    return value
