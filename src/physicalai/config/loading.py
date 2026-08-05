# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""General configuration instantiation helpers."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path

import yaml
from pydantic import BaseModel

from ._errors import ConfigError
from ._instantiate import instantiate
from ._types import _MAX_CONFIG_DEPTH
from .base import Config
from .importing import import_dotted_path

ConfigMapping = Mapping[str, object]

__all__ = [
    "import_class",
    "instantiate_obj",
    "instantiate_obj_from_dataclass",
    "instantiate_obj_from_dict",
    "instantiate_obj_from_file",
    "instantiate_obj_from_pydantic",
]


def import_class(class_path: str) -> type:
    """Import and validate a class from a dotted path.

    Returns:
        The imported class.

    Raises:
        ImportError: If the module or attribute cannot be imported.
        TypeError: If the path resolves to a non-class object.
    """
    try:
        value = import_dotted_path(class_path)
    except (ValueError, ImportError, AttributeError) as exc:
        msg = f"Cannot import {class_path!r}: {exc}"
        raise ImportError(msg) from exc
    if not isinstance(value, type):
        msg = f"{class_path!r} does not resolve to a class"
        raise TypeError(msg)
    return value


def _instantiate_recursive(value: object, *, depth: int = 0) -> object:
    if depth > _MAX_CONFIG_DEPTH:
        msg = f"Configuration nesting depth exceeds {_MAX_CONFIG_DEPTH}"
        raise ConfigError(msg)
    if isinstance(value, dict):
        if "class_path" in value:
            return Config.from_dict(value).instantiate()
        return {key: _instantiate_recursive(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return [_instantiate_recursive(item, depth=depth + 1) for item in value]
    if isinstance(value, tuple):
        return tuple(_instantiate_recursive(item, depth=depth + 1) for item in value)
    return value


def instantiate_obj_from_dict(
    config: ConfigMapping,
    *,
    key: str | None = None,
    target_cls: type | None = None,
) -> object:
    """Instantiate an object from a configuration mapping.

    When ``target_cls`` is set and the selected mapping has no ``class_path``,
    entries are passed as keyword arguments after recursive instantiation.
    The reserved key ``args`` supplies positional constructor arguments (a
    sequence); it is removed from ``init_args`` before ``target_cls`` is called.

    Returns:
        The constructed object.

    Raises:
        ValueError: If ``key`` is missing or no ``class_path``/``target_cls`` is available.
        TypeError: If a selected sub-config is not a mapping.
    """
    selected: object = config
    if key is not None:
        if key not in config:
            msg = f"Configuration must contain {key!r} key. Got keys: {list(config.keys())}"
            raise ValueError(msg)
        selected = config[key]
        if not isinstance(selected, Mapping):
            msg = f"Configuration at key {key!r} must be a mapping, got {type(selected).__name__}"
            raise TypeError(msg)
    if not isinstance(selected, Mapping):
        msg = f"Configuration must be a mapping, got {type(selected).__name__}"
        raise TypeError(msg)
    if "class_path" in selected:
        return instantiate(Config.from_dict(selected))
    if target_cls is None:
        msg = (
            "Configuration must contain 'class_path' for instantiation, "
            f"or pass target_cls explicitly. Got keys: {list(selected.keys())}"
        )
        raise ValueError(msg)
    init_args = {name: _instantiate_recursive(value) for name, value in selected.items()}
    args = init_args.pop("args", ())
    return target_cls(*args, **init_args)


def instantiate_obj_from_pydantic(
    config: BaseModel,
    *,
    key: str | None = None,
    target_cls: type | None = None,
) -> object:
    """Instantiate from a Pydantic model.

    Returns:
        The constructed object.
    """
    return instantiate_obj_from_dict(config.model_dump(), key=key, target_cls=target_cls)


def instantiate_obj_from_dataclass(
    config: object,
    *,
    key: str | None = None,
    target_cls: type | None = None,
) -> object:
    """Instantiate from a dataclass instance.

    Returns:
        The constructed object.

    Raises:
        TypeError: If ``config`` is not a dataclass instance.
    """
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        msg = f"Expected dataclass instance, got {type(config)}"
        raise TypeError(msg)
    return instantiate_obj_from_dict(dataclasses.asdict(config), key=key, target_cls=target_cls)


def instantiate_obj_from_file(
    file_path: str | Path,
    *,
    key: str | None = None,
    target_cls: type | None = None,
) -> object:
    """Instantiate from a YAML or JSON file.

    Returns:
        The constructed object.

    Raises:
        TypeError: If the file root is not a mapping.
    """
    config = yaml.safe_load(Path(file_path).read_text(encoding="utf-8"))
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        msg = f"Expected YAML root to be a mapping, got {type(config).__name__}"
        raise TypeError(msg)
    return instantiate_obj_from_dict(config, key=key, target_cls=target_cls)


def instantiate_obj(
    config: ConfigMapping | Config | BaseModel | object | str | Path,
    *,
    key: str | None = None,
    target_cls: type | None = None,
) -> object:
    """Instantiate from a recipe, mapping, model, dataclass, or file.

    Returns:
        The constructed object.

    Raises:
        TypeError: If ``config`` has an unsupported type.
    """
    if type(config) is Config:
        return config.instantiate()
    if isinstance(config, (str, Path)):
        return instantiate_obj_from_file(config, key=key, target_cls=target_cls)
    if isinstance(config, BaseModel):
        return instantiate_obj_from_pydantic(config, key=key, target_cls=target_cls)
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return instantiate_obj_from_dataclass(config, key=key, target_cls=target_cls)
    if isinstance(config, Mapping):
        return instantiate_obj_from_dict(config, key=key, target_cls=target_cls)
    msg = f"Unsupported configuration type: {type(config)}. Expected dict, file path, Pydantic model, or dataclass."
    raise TypeError(msg)
