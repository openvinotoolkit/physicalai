# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Restore dataclass field types from persisted LeRobot configuration data."""

from __future__ import annotations

import dataclasses
import types
from itertools import starmap
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

_Config = TypeVar("_Config")
_PAIR_LEN = 2


def coerce_to_annotation(annotation: object, value: object) -> object:  # noqa: C901, PLR0911, PLR0912
    """Restore paths, nested dataclasses, and containers to their annotated types.

    Returns:
        The converted value, or the original value for unsupported annotations.
    """
    if value is None or annotation is Any:
        return value

    if annotation is Path and isinstance(value, str):
        return Path(value)

    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        if isinstance(value, annotation):
            return value
        if isinstance(value, dict):
            return materialize_config(annotation, value)
        return value

    origin = get_origin(annotation)
    if origin is None:
        return value

    args = get_args(annotation)
    if origin in {Union, types.UnionType}:
        non_none_args = [arg for arg in args if arg is not type(None)]
        for arg in non_none_args:
            converted = coerce_to_annotation(arg, value)
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
        return [coerce_to_annotation(args[0], item) for item in value]
    if origin is tuple and args and isinstance(value, tuple | list):
        if len(args) == _PAIR_LEN and args[1] is Ellipsis:
            return tuple(coerce_to_annotation(args[0], item) for item in value)
        return tuple(starmap(coerce_to_annotation, zip(args, value, strict=False)))
    if origin is dict and len(args) == _PAIR_LEN and isinstance(value, dict):
        key_type, val_type = args
        return {coerce_to_annotation(key_type, key): coerce_to_annotation(val_type, val) for key, val in value.items()}
    if origin is set and args and isinstance(value, set | list | tuple):
        return {coerce_to_annotation(args[0], item) for item in value}

    return value


def materialize_config(config_cls: type[_Config], payload_data: dict[str, Any]) -> _Config:
    """Build a typed config after JSON storage or local worker serialization.

    Config classes have already passed the adapter's registered-type/import
    trust checks. Resolve their postponed annotations as well as inherited ones.

    Returns:
        The config with paths and nested dataclasses restored.
    """
    hints = get_type_hints(config_cls)
    kwargs = {
        field.name: coerce_to_annotation(hints.get(field.name, field.type), payload_data[field.name])
        for field in dataclasses.fields(config_cls)
        if field.init and field.name in payload_data
    }
    return config_cls(**kwargs)
