# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Serialization utilities for typed dataclass configs.

.. deprecated::
    Prefer :meth:`~physicalai.config.Config.to_dict` and
    :meth:`~physicalai.config.Config.from_dict` (or jsonargparse). Emits
    :class:`DeprecationWarning` at runtime.
"""

from __future__ import annotations

import dataclasses
import operator
import types
from enum import Enum
from functools import reduce
from itertools import starmap
from pathlib import PurePath
from typing import TYPE_CHECKING, TypeVar, Union, get_args, get_origin, get_type_hints

from ._deprecate import deprecate

if TYPE_CHECKING:
    from collections.abc import Mapping

_MIN_DICT_TYPE_ARGS = 2
_VAR_TUPLE_ARG_COUNT = 2
_T = TypeVar("_T")

__all__ = ["dataclass_to_dict", "dict_to_dataclass"]


def dataclass_to_dict(obj: object, *, recursive: bool = True) -> object:  # ruff: ignore[PLR0911]
    """Convert a dataclass or nested structure to plain Python data.

    .. deprecated::
        Use :meth:`~physicalai.config.Config.to_dict` for typed configs.

    Returns:
        Plain dicts, lists, and scalars suitable for ``torch.save(weights_only=True)``.
    """
    deprecate(
        "physicalai.config.serializable.dataclass_to_dict",
        "Config.to_dict() for typed configs or jsonargparse parser.dump",
    )
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        if not recursive:
            return {field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)}
        return {field.name: dataclass_to_dict(getattr(obj, field.name)) for field in dataclasses.fields(obj)}
    if not recursive:
        return obj
    if isinstance(obj, dict):
        return {(key.value if isinstance(key, Enum) else key): dataclass_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(item) for item in obj]
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, PurePath):
        return str(obj)
    if hasattr(obj, "tolist") and hasattr(obj, "ndim"):
        return obj.tolist()  # type: ignore[union-attr]
    return obj


def dict_to_dataclass(cls: type[_T], data: Mapping[str, object], *, strict: bool = True) -> _T:
    """Reconstruct a dataclass from a mapping using its type hints.

    .. deprecated::
        Use :meth:`~physicalai.config.Config.from_dict` or
        :func:`~physicalai.config.base.parse_class_config`.

    Args:
        cls: Dataclass type to construct.
        data: Field values (typically from YAML or a checkpoint).
        strict: When ``True``, reject keys that are not dataclass fields.

    Returns:
        An instance of ``cls``.

    Raises:
        TypeError: If ``cls`` is not a dataclass or ``strict`` rejects extra keys.
    """
    deprecate(
        "physicalai.config.serializable.dict_to_dataclass",
        "Config.from_dict() for Config subclasses or parse_class_config",
    )
    if not dataclasses.is_dataclass(cls):
        msg = f"Expected dataclass, got {cls}"
        raise TypeError(msg)
    if strict:
        field_names = {field.name for field in dataclasses.fields(cls)}
        extras = set(data.keys()) - field_names
        if extras:
            msg = f"Unexpected keys for {cls.__name__}: {sorted(extras)}"
            raise TypeError(msg)
    try:
        hints = get_type_hints(cls)
    except (NameError, TypeError, AttributeError, KeyError):
        hints = {}
    kwargs = {}
    for field in dataclasses.fields(cls):
        if field.name in data:
            kwargs[field.name] = _reconstruct_value(data[field.name], hints.get(field.name, field.type))
    return cls(**kwargs)  # type: ignore[return-value]


def _reconstruct_value(value: object, field_type: object) -> object:  # ruff: ignore[PLR0911]
    if value is None:
        return None
    origin = get_origin(field_type)
    args = get_args(field_type)
    if _is_optional_type(field_type):
        return _reconstruct_value(value, _get_optional_inner_type(field_type))
    if origin is dict and isinstance(value, dict):
        if len(args) >= _MIN_DICT_TYPE_ARGS:
            return {key: _reconstruct_value(item, args[1]) for key, item in value.items()}
        return value
    if origin is list and isinstance(value, list):
        return [_reconstruct_value(item, args[0]) for item in value] if args else value
    if origin is tuple and isinstance(value, list):
        if not args:
            return tuple(value)
        if len(args) == _VAR_TUPLE_ARG_COUNT and args[1] is ...:
            return tuple(_reconstruct_value(item, args[0]) for item in value)
        return tuple(starmap(_reconstruct_value, zip(value, args, strict=False)))
    actual_type = origin or field_type
    if isinstance(actual_type, type) and dataclasses.is_dataclass(actual_type) and isinstance(value, dict):
        return dict_to_dataclass(actual_type, value)
    if isinstance(actual_type, type) and issubclass(actual_type, PurePath) and isinstance(value, str):
        return actual_type(value)
    if isinstance(actual_type, type) and issubclass(actual_type, Enum) and not isinstance(value, Enum):
        return actual_type(value)
    return value


def _is_optional_type(field_type: object) -> bool:
    origin = get_origin(field_type)
    return origin in {types.UnionType, Union} and type(None) in get_args(field_type)


def _get_optional_inner_type(field_type: object) -> object:
    non_none_args = [arg for arg in get_args(field_type) if arg is not type(None)]
    if len(non_none_args) == 1:
        return non_none_args[0]
    return reduce(operator.or_, non_none_args)
