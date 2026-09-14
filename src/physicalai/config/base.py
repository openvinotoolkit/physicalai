# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unified construction-recipe and typed-dataclass configuration class."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, TypeVar, cast, overload

import yaml
from jsonargparse import ArgumentParser

if TYPE_CHECKING:
    from ._types import JsonArgparseEnvelope, JsonValue

from ._errors import ConfigImportError

__all__ = ["Config"]

_T = TypeVar("_T")


def parse_class_config(target: type[_T], data: Mapping[str, object], *, defaults: bool = False) -> _T:
    """Parse and instantiate a known class through jsonargparse.

    Returns:
        An instance of *target*.
    """
    from ._typed_wire import coerce_typed_config_wire  # noqa: PLC0415

    parser = ArgumentParser(exit_on_error=False)
    parser.add_class_arguments(target, "object")
    wire = coerce_typed_config_wire(target, data)
    namespace = parser.parse_object({"object": wire}, defaults=defaults)
    return cast("_T", parser.instantiate(namespace).object)


def parse_class_config_file(target: type[_T], path: str | Path, *, defaults: bool = False) -> _T:
    """Parse a typed config file while preserving the existing bare YAML shape.

    Returns:
        An instance of *target*.

    Raises:
        TypeError: If the configuration root is not a mapping.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        msg = f"Expected configuration root to be a mapping, got {type(data).__name__}"
        raise TypeError(msg)
    values = data.get("object", data)
    if not isinstance(values, Mapping):
        msg = "Expected configuration values to be a mapping"
        raise TypeError(msg)
    if "init_args" in values:
        inner = values["init_args"]
        if inner is None or not isinstance(inner, Mapping):
            msg = "Expected 'init_args' to be a mapping"
            raise TypeError(msg)
        values = inner
    return parse_class_config(target, values, defaults=defaults)


def save_class_config(value: _T, path: str | Path) -> None:
    """Serialize a typed config as a ``class_path``/``init_args`` YAML envelope.

    Raises:
        TypeError: If *value* is not a typed :class:`Config` dataclass instance.
    """
    if not isinstance(value, Config):
        msg = "save_class_config expects a typed Config dataclass instance"
        raise TypeError(msg)
    envelope = value.to_jsonargparse()
    Path(path).write_text(yaml.safe_dump(envelope, sort_keys=False), encoding="utf-8")


def _plain_value(value: object) -> object:  # noqa: PLR0911
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain_value(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {key: _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist") and hasattr(value, "ndim"):
        return value.tolist()  # type: ignore[union-attr]
    return value


class Config:
    """A construction recipe, or the base for a typed dataclass config."""

    __hash__ = None

    def __init__(self, class_path: str, init_args: Mapping[str, object] | None = None) -> None:
        """Create a direct construction recipe.

        Raises:
            TypeError: If called on a dataclass ``Config`` subclass.
        """
        if type(self) is not Config:
            msg = "Config subclasses must be dataclasses"
            raise TypeError(msg)
        from ._normalize import normalize_config  # noqa: PLC0415

        validated = normalize_config({"class_path": class_path, "init_args": dict(init_args or {})})
        self.class_path: str = validated["class_path"]
        self.init_args: dict[str, JsonValue] = validated["init_args"]

    @classmethod
    def from_instance(cls, instance: object) -> Config:
        """Capture an ``@export_config`` instance as a construction recipe.

        Returns:
            A direct :class:`Config` recipe.

        Raises:
            TypeError: If called on a dataclass ``Config`` subclass.
        """
        if cls is not Config:
            msg = "from_instance() constructs the direct Config recipe type"
            raise TypeError(msg)
        from ._export import _export_instance  # noqa: PLC0415

        recipe = _export_instance(instance)
        return cls(recipe["class_path"], recipe["init_args"])

    @staticmethod
    def is_exportable(instance: object) -> bool:
        """Return whether *instance* opted into constructor capture."""
        from ._export import is_config_exportable  # noqa: PLC0415

        return is_config_exportable(instance)

    def resolve_type(self) -> type:
        """Import and return the configured class.

        Returns:
            The configured class.

        Raises:
            ConfigImportError: If the path does not resolve to a class.
        """
        from .importing import import_dotted_path  # noqa: PLC0415

        resolved = import_dotted_path(self.class_path)
        if not isinstance(resolved, type):
            msg = f"{self.class_path!r} does not resolve to a class"
            raise ConfigImportError(msg)
        return resolved

    def to_dict(self) -> dict[str, object]:
        """Convert this config to its plain serialization form.

        Returns:
            A mapping safe for checkpoints and YAML export.

        Raises:
            TypeError: If this instance is not a dataclass subclass or direct recipe.
        """
        if dataclasses.is_dataclass(self):
            result = _plain_value(self)
            if not isinstance(result, dict):
                msg = f"Expected dict from dataclass_to_dict, got {type(result)}"
                raise TypeError(msg)
            return result
        if type(self) is not Config:
            msg = f"{type(self).__name__} must be a dataclass to use Config"
            raise TypeError(msg)
        return {"class_path": self.class_path, "init_args": _plain_value(self.init_args)}

    @classmethod
    def from_dict(cls, data: Mapping[str, object], *, strict: bool = True) -> Self:
        """Parse a direct recipe or reconstruct a typed dataclass config.

        Args:
            data: Recipe envelope or dataclass field mapping.
            strict: When reconstructing a typed dataclass, reject unknown
                top-level keys. When ``False``, keys with no matching field on
                ``cls`` are silently dropped instead of being forwarded to the
                underlying parser (which would otherwise reject them
                regardless of ``strict``). This only applies at the top
                level: unknown keys nested inside a field's own value are not
                filtered and are still rejected, matching prior behavior.

        Returns:
            A :class:`Config` recipe or dataclass instance.

        Raises:
            TypeError: If ``cls`` is not a dataclass when reconstructing nested fields.
        """
        if cls is Config:
            from ._normalize import normalize_config  # noqa: PLC0415

            validated = normalize_config(data)
            return cls(validated["class_path"], validated["init_args"])
        if not dataclasses.is_dataclass(cls):
            msg = f"{cls.__name__} must be a dataclass to use Config"
            raise TypeError(msg)
        field_names = {field.name for field in dataclasses.fields(cls)}
        if strict:
            extras = set(data) - field_names
            if extras:
                msg = f"Unexpected keys for {cls.__name__}: {sorted(extras)}"
                raise TypeError(msg)
        else:
            data = {key: value for key, value in data.items() if key in field_names}
        return parse_class_config(cls, data)

    def instantiate(self, *, expected_type: type[_T] | None = None) -> _T | object:
        """Instantiate this direct construction recipe through the strict core.

        Returns:
            The constructed object.

        Raises:
            TypeError: If called on a dataclass ``Config`` subclass.
        """
        if type(self) is not Config:
            msg = "instantiate() is only available on a direct Config recipe"
            raise TypeError(msg)
        from ._instantiate import instantiate  # noqa: PLC0415

        return instantiate(self, expected_type=expected_type)

    def to_jsonargparse(self) -> JsonArgparseEnvelope:
        """Convert this config to a ``class_path``/``init_args`` envelope.

        Returns:
            A jsonargparse-compatible mapping.
        """
        if type(self) is Config:
            return {
                "class_path": self.class_path,
                "init_args": self.init_args,
            }
        return {
            "class_path": f"{type(self).__module__}.{type(self).__qualname__}",
            "init_args": cast("dict[str, JsonValue]", _plain_value(self)),
        }

    def save(
        self,
        path: str | Path,
        *,
        format: Literal["jsonargparse", "dict"] = "jsonargparse",  # noqa: A002
    ) -> None:
        """Save this config to YAML.

        Raises:
            ValueError: If the path extension is not ``.yaml`` or ``.yml``.
        """
        target = Path(path)
        if target.suffix not in {".yaml", ".yml"}:
            msg = f"Unsupported file extension: {target.suffix}. Use .yaml or .yml"
            raise ValueError(msg)
        if format == "dict":
            target.write_text(
                yaml.safe_dump(self.to_dict(), default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )
            return
        if type(self) is Config:
            target.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
            return
        save_class_config(self, target)

    @classmethod
    def load(cls, source: Mapping[str, object] | str | Path) -> Self:
        """Load a direct recipe or typed config from a mapping or YAML file.

        Returns:
            A reconstructed config instance.

        Raises:
            ValueError: If the path extension is not ``.yaml`` or ``.yml``.
            TypeError: If the YAML root or ``init_args`` is not a mapping.
        """
        if cls is Config:
            if isinstance(source, Mapping):
                return cls.from_dict(source)
            source_path = Path(source)
            if source_path.suffix not in {".yaml", ".yml"}:
                msg = f"Unsupported file extension: {source_path.suffix}. Use .yaml or .yml"
                raise ValueError(msg)
            loaded = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, Mapping):
                msg = f"Expected YAML root to be a mapping, got {type(loaded).__name__}"
                raise TypeError(msg)
            return cls.from_dict(loaded)

        if isinstance(source, Mapping):
            values: Mapping[str, object] = source
            if "init_args" in values:
                inner = values["init_args"]
                if inner is None or not isinstance(inner, Mapping):
                    msg = "Expected 'init_args' to be a mapping"
                    raise TypeError(msg)
                values = inner
            return parse_class_config(cls, values)

        source_path = Path(source)
        if source_path.suffix not in {".yaml", ".yml"}:
            msg = f"Unsupported file extension: {source_path.suffix}. Use .yaml or .yml"
            raise ValueError(msg)
        return parse_class_config_file(cls, source_path)

    @overload
    def __getitem__(self, key: Literal["class_path"]) -> str: ...

    @overload
    def __getitem__(self, key: Literal["init_args"]) -> dict[str, JsonValue]: ...

    @overload
    def __getitem__(self, key: str) -> JsonValue: ...

    def __getitem__(self, key: str) -> JsonValue:
        """Expose direct recipe keys for gradual internal migration.

        Returns:
            The value for ``key`` in :meth:`to_dict`.
        """
        return cast("JsonValue", self.to_dict()[key])

    def get(self, key: str, default: object | None = None) -> object | None:
        """Return a direct recipe value, or *default* when absent.

        Returns:
            The value for ``key``, or *default* if missing.
        """
        return self.to_dict().get(key, default)

    def __eq__(self, other: object) -> bool:
        """Compare configs by their plain serialization form.

        Returns:
            ``NotImplemented`` for unsupported comparison types, otherwise equality.
        """
        if isinstance(other, Config):
            return self.to_dict() == other.to_dict()
        if isinstance(other, Mapping):
            return self.to_dict() == dict(other)
        return NotImplemented

    def __repr__(self) -> str:
        """Return a direct recipe representation."""
        if type(self) is Config:
            return f"Config(class_path={self.class_path!r}, init_args={self.init_args!r})"
        return object.__repr__(self)
