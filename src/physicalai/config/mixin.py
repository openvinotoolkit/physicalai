# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mixins for configuration-based construction.

.. deprecated::
    Prefer :class:`jsonargparse.FromConfigMixin` and
    :func:`~physicalai.config.base.parse_class_config`. Emits
    :class:`DeprecationWarning` at runtime.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self, TypeVar, cast

from jsonargparse import FromConfigMixin
from pydantic import BaseModel

from ._deprecate import deprecate
from .loading import instantiate_obj_from_dict, instantiate_obj_from_file
from .serializable import dataclass_to_dict

__all__ = ["FromConfig", "from_config"]

_T = TypeVar("_T", bound=type)

_REPLACEMENT = "jsonargparse.FromConfigMixin.from_config or parse_class_config via physicalai.config.Config"


class FromConfig(FromConfigMixin):
    """Mixin adding constructors for mapping, YAML, Pydantic, and dataclass configs.

    .. deprecated::
        Inherit :class:`jsonargparse.FromConfigMixin` instead.
    """

    @classmethod
    def from_yaml(cls, file_path: str | Path, *, key: str | None = None) -> Self:
        """Load configuration from YAML and instantiate the class.

        .. deprecated::
            Use :meth:`jsonargparse.FromConfigMixin.from_config` with a file path.

        Returns:
            An instance of ``cls``.
        """
        deprecate(f"{cls.__name__}.from_yaml", _REPLACEMENT)
        return cast("Self", instantiate_obj_from_file(file_path, key=key, target_cls=cls))

    @classmethod
    def from_dict(cls, config: Mapping[str, Any], *, key: str | None = None) -> Self:
        """Instantiate the class from a mapping.

        .. deprecated::
            Use :meth:`jsonargparse.FromConfigMixin.from_config` with a dict.

        Returns:
            An instance of ``cls``.
        """
        deprecate(f"{cls.__name__}.from_dict", _REPLACEMENT)
        return cast("Self", instantiate_obj_from_dict(config, key=key, target_cls=cls))

    @classmethod
    def from_pydantic(cls, config: BaseModel, *, key: str | None = None, recursive: bool = False) -> Self:
        """Instantiate the class from a Pydantic model.

        .. deprecated::
            Use :meth:`jsonargparse.FromConfigMixin.from_config` after ``model_dump()``.

        Returns:
            An instance of ``cls``.
        """
        deprecate(f"{cls.__name__}.from_pydantic", _REPLACEMENT)
        values = (
            config.model_dump()
            if recursive
            else {name: getattr(config, name) for name in config.__class__.model_fields}
        )
        return cls.from_dict(values, key=key)

    @classmethod
    def from_dataclass(cls, config: object, *, key: str | None = None, recursive: bool = False) -> Self:
        """Instantiate the class from a dataclass instance.

        .. deprecated::
            Use :meth:`jsonargparse.FromConfigMixin.from_config` with a dict.

        Returns:
            An instance of ``cls``.

        Raises:
            TypeError: If ``config`` is not a dataclass instance.
        """
        deprecate(f"{cls.__name__}.from_dataclass", _REPLACEMENT)
        if not dataclasses.is_dataclass(config) or isinstance(config, type):
            msg = f"Expected dataclass instance, got {type(config)}"
            raise TypeError(msg)
        values = cast("dict[str, Any]", dataclass_to_dict(config, recursive=recursive))
        return cls.from_dict(values, key=key)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | BaseModel | object | str | Path,
        *,
        key: str | None = None,
        recursive: bool = False,
    ) -> Self:
        """Dispatch to the matching configuration constructor.

        .. deprecated::
            Use :meth:`jsonargparse.FromConfigMixin.from_config` instead.

        Returns:
            An instance of ``cls``.

        Raises:
            TypeError: If ``config`` has an unsupported type.
        """
        deprecate(f"{cls.__name__}.from_config", "jsonargparse.FromConfigMixin.from_config")
        if isinstance(config, (str, Path)):
            return cls.from_yaml(config, key=key)
        if isinstance(config, BaseModel):
            return cls.from_pydantic(config, key=key, recursive=recursive)
        if dataclasses.is_dataclass(config) and not isinstance(config, type):
            return cls.from_dataclass(config, key=key, recursive=recursive)
        if isinstance(config, Mapping):
            if "class_path" in config and key is None:
                return cast("Self", instantiate_obj_from_dict(config, target_cls=cls))
            return cls.from_dict(config, key=key)
        msg = f"Unsupported configuration type: {type(config)}. Expected dict, file path, Pydantic model, or dataclass."
        raise TypeError(msg)


def from_config(cls: _T) -> _T:
    """Decorate a class with the constructors provided by :class:`FromConfig`.

    .. deprecated::
        Inherit :class:`jsonargparse.FromConfigMixin` instead.

    Returns:
        The same class with ``from_*`` constructors attached.
    """
    deprecate("@physicalai.config.from_config", "inherit jsonargparse.FromConfigMixin instead")
    for name in ("from_yaml", "from_dict", "from_pydantic", "from_dataclass", "from_config"):
        setattr(cls, name, FromConfig.__dict__[name])
    return cls
