# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unified construction-recipe and typed-dataclass configuration class."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, cast, overload

import yaml

from .serializable import dataclass_to_dict, dict_to_dataclass

if TYPE_CHECKING:
    from ._types import JsonArgparseEnvelope, JsonValue

__all__ = ["Config"]


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
        from ._normalize import normalize_config  # ruff: ignore[PLC0415]

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
        from ._export import _export_instance  # ruff: ignore[PLC0415]

        recipe = _export_instance(instance)
        return cls(recipe["class_path"], recipe["init_args"])

    def to_dict(self) -> dict[str, object]:
        """Convert this config to its plain serialization form.

        Returns:
            A mapping safe for checkpoints and YAML export.

        Raises:
            TypeError: If this instance is not a dataclass subclass or direct recipe.
        """
        if dataclasses.is_dataclass(self):
            result = dataclass_to_dict(self)
            if not isinstance(result, dict):
                msg = f"Expected dict from dataclass_to_dict, got {type(result)}"
                raise TypeError(msg)
            return result
        if type(self) is not Config:
            msg = f"{type(self).__name__} must be a dataclass to use Config"
            raise TypeError(msg)
        return {"class_path": self.class_path, "init_args": dataclass_to_dict(self.init_args)}

    @classmethod
    def from_dict(cls, data: Mapping[str, object], *, strict: bool = True) -> Self:
        """Parse a direct recipe or reconstruct a typed dataclass config.

        Args:
            data: Recipe envelope or dataclass field mapping.
            strict: When reconstructing a typed dataclass, reject unknown keys.

        Returns:
            A :class:`Config` recipe or dataclass instance.

        Raises:
            TypeError: If ``cls`` is not a dataclass when reconstructing nested fields.
        """
        if cls is Config:
            from ._normalize import normalize_config  # ruff: ignore[PLC0415]

            validated = normalize_config(data)
            return cls(validated["class_path"], validated["init_args"])
        if not dataclasses.is_dataclass(cls):
            msg = f"{cls.__name__} must be a dataclass to use Config"
            raise TypeError(msg)
        return dict_to_dataclass(cls, data, strict=strict)

    def instantiate(self) -> object:
        """Instantiate this direct construction recipe through the strict core.

        Returns:
            The constructed object.

        Raises:
            TypeError: If called on a dataclass ``Config`` subclass.
        """
        if type(self) is not Config:
            msg = "instantiate() is only available on a direct Config recipe"
            raise TypeError(msg)
        from ._instantiate import instantiate  # ruff: ignore[PLC0415]

        return instantiate(self)

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
            "init_args": cast("dict[str, JsonValue]", dataclass_to_dict(self)),
        }

    def save(
        self,
        path: str | Path,
        *,
        format: Literal["jsonargparse", "dict"] = "jsonargparse",  # ruff: ignore[A002]
    ) -> None:
        """Save this config to YAML.

        Raises:
            ValueError: If the path extension is not ``.yaml`` or ``.yml``.
        """
        target = Path(path)
        if target.suffix not in {".yaml", ".yml"}:
            msg = f"Unsupported file extension: {target.suffix}. Use .yaml or .yml"
            raise ValueError(msg)
        data = self.to_dict() if format == "dict" else self.to_jsonargparse()
        target.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a direct recipe or typed config from YAML.

        Returns:
            A reconstructed config instance.

        Raises:
            ValueError: If the path extension is not ``.yaml`` or ``.yml``.
            TypeError: If the YAML root or ``init_args`` is not a mapping.
        """
        source = Path(path)
        if source.suffix not in {".yaml", ".yml"}:
            msg = f"Unsupported file extension: {source.suffix}. Use .yaml or .yml"
            raise ValueError(msg)
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            msg = f"Expected YAML root to be a mapping, got {type(data).__name__}"
            raise TypeError(msg)
        if cls is not Config and "init_args" in data:
            data = data["init_args"]
            if data is None:
                data = {}
            if not isinstance(data, Mapping):
                msg = f"Expected 'init_args' to be a mapping, got {type(data).__name__}"
                raise TypeError(msg)
        return cls.from_dict(data)

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
