# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415

"""Mixins for configuration-based construction."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self, TypeVar, cast

from jsonargparse import FromConfigMixin
from pydantic import BaseModel

from .serializable import dataclass_to_dict

__all__ = ["FromConfig", "from_config"]

_T = TypeVar("_T", bound=type)


class FromConfig(FromConfigMixin):
    """Mixin adding constructors for mapping, YAML, Pydantic, and dataclass configs."""

    @classmethod
    def from_yaml(cls, file_path: str | Path, *, key: str | None = None) -> Self:
        """Load configuration from YAML and instantiate the class.

        Returns:
            An instance of ``cls``.

        Raises:
            TypeError: If ``key`` is supplied for a file input.
        """
        if key is not None:
            msg = "key= is supported only for mapping compatibility inputs"
            raise TypeError(msg)
        from .base import parse_class_config_file

        return parse_class_config_file(cls, file_path)

    @classmethod
    def from_dict(cls, config: Mapping[str, Any], *, key: str | None = None) -> Self:
        """Instantiate the class from a mapping.

        Returns:
            An instance of ``cls``.

        Raises:
            TypeError: If ``key`` or a class recipe envelope is malformed.
        """
        values: Mapping[str, Any] = config
        if key is not None:
            selected = values.get(key)
            if not isinstance(selected, Mapping):
                msg = f"Configuration at key {key!r} must be a mapping"
                raise TypeError(msg)
            values = selected
        if "class_path" in values and "init_args" in values:
            values = values["init_args"]
            if not isinstance(values, Mapping):
                msg = "Expected 'init_args' to be a mapping"
                raise TypeError(msg)
        from .base import parse_class_config

        return parse_class_config(cls, values)

    @classmethod
    def from_pydantic(cls, config: BaseModel, *, key: str | None = None, recursive: bool = False) -> Self:
        """Instantiate the class from a Pydantic model.

        Returns:
            An instance of ``cls``.
        """
        values = (
            config.model_dump()
            if recursive
            else {name: getattr(config, name) for name in config.__class__.model_fields}
        )
        return cls.from_dict(values, key=key)

    @classmethod
    def from_dataclass(cls, config: object, *, key: str | None = None, recursive: bool = False) -> Self:
        """Instantiate the class from a dataclass instance.

        Returns:
            An instance of ``cls``.

        Raises:
            TypeError: If ``config`` is not a dataclass instance.
        """
        if not dataclasses.is_dataclass(config) or isinstance(config, type):
            msg = f"Expected dataclass instance, got {type(config)}"
            raise TypeError(msg)
        values = cast("dict[str, Any]", dataclass_to_dict(config, recursive=recursive))
        if key is not None:
            nested = values.get(key)
            if not isinstance(nested, Mapping):
                msg = f"Configuration at key {key!r} must be a mapping"
                raise TypeError(msg)
            values = nested
        return cls(**values)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | BaseModel | object | str | Path,
        *,
        key: str | None = None,
        recursive: bool = False,
    ) -> Self:
        """Dispatch to the matching configuration constructor.

        Returns:
            An instance of ``cls``.

        Raises:
            TypeError: If ``config`` has an unsupported type.
        """
        if isinstance(config, (str, Path)):
            return cls.from_yaml(config, key=key)
        if isinstance(config, BaseModel):
            return cls.from_pydantic(config, key=key, recursive=recursive)
        if dataclasses.is_dataclass(config) and not isinstance(config, type):
            return cls.from_dataclass(config, key=key, recursive=recursive)
        if isinstance(config, Mapping):
            return cls.from_dict(config, key=key)
        msg = f"Unsupported configuration type: {type(config)}. Expected dict, file path, Pydantic model, or dataclass."
        raise TypeError(msg)


def from_config(cls: _T) -> _T:
    """Decorate a class with the constructors provided by :class:`FromConfig`.

    Returns:
        The same class with ``from_*`` constructors attached.
    """
    for name in ("from_yaml", "from_dict", "from_pydantic", "from_dataclass", "from_config"):
        setattr(cls, name, FromConfig.__dict__[name])
    return cls
