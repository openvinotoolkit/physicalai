# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Coerce typed dataclass wire values for jsonargparse parsing."""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Mapping
from enum import Enum
from typing import Union, get_args, get_origin, get_type_hints


def enum_from_wire(enum_cls: type[Enum], wire: str) -> Enum:
    """Resolve an enum from a persisted wire string (value first, then member name).

    Returns:
        The matching enum member.
    """
    try:
        return enum_cls(wire)
    except ValueError:
        return enum_cls[wire]


def _resolve_field_type(field_type: object) -> object:
    origin = get_origin(field_type)
    if origin in {types.UnionType, Union}:
        args = [arg for arg in get_args(field_type) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return field_type


def _coerce_field_wire(field_type: object, value: object) -> object:
    if value is None:
        return value
    resolved = _resolve_field_type(field_type)
    if isinstance(resolved, type) and dataclasses.is_dataclass(resolved) and isinstance(value, Mapping):
        return coerce_typed_config_wire(resolved, value)
    if isinstance(resolved, type) and issubclass(resolved, Enum) and isinstance(value, str):
        return enum_from_wire(resolved, value)
    return value


def coerce_typed_config_wire(cls: type[object], data: Mapping[str, object]) -> dict[str, object]:
    """Normalize legacy wire shapes (e.g. enum values) before jsonargparse parses *cls*.

    Returns:
        A mapping safe to pass to jsonargparse ``parse_object`` for *cls*.
    """
    if not dataclasses.is_dataclass(cls):
        return dict(data)
    try:
        hints = get_type_hints(cls)
    except (TypeError, ValueError, NameError):
        hints = {field.name: field.type for field in dataclasses.fields(cls)}
    coerced: dict[str, object] = {}
    for name, value in data.items():
        hint = hints.get(name)
        if hint is None:
            coerced[name] = value
            continue
        coerced[name] = _coerce_field_wire(hint, value)
    return coerced
