# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Types for JSON-safe configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from .base import Config

JsonScalar = None | bool | int | float | str
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class ValidatedConfigDict(TypedDict):
    class_path: str
    init_args: dict[str, JsonValue]


class JsonArgparseEnvelope(TypedDict):
    class_path: str
    init_args: dict[str, JsonValue]


@runtime_checkable
class ConfigValue(Protocol):
    """Domain value that encodes to constructor-compatible JSON for export."""

    def to_config_value(self) -> JsonValue:
        """Return ctor-compatible JSON for config export."""
        ...


@runtime_checkable
class ConfigExportable(Protocol):
    """Internal typing protocol for ``@export_config`` instances.

    Instances gain :meth:`supports_config_export` and :meth:`as_config` when
    their class is decorated; subclasses inherit them when they reuse a
    decorated constructor without overriding ``__init__``.
    """

    def supports_config_export(self) -> bool:
        """Return whether this instance can export a portable construction recipe."""
        ...

    def as_config(self) -> Config:
        """Return the :class:`~physicalai.config.Config` recipe for this instance."""
        ...


# Maximum nesting depth for recursive normalization and instantiation.
# Counts traversal through lists and mappings as well as nested configs.
_MAX_CONFIG_DEPTH = 10

# Private attribute names used by @export_config.
_CAPTURED_INIT_ARGS_ATTR = "_physicalai_captured_init_args"
_EXPORT_DEPTH_ATTR = "_physicalai_export_config_depth"
_EXPORT_MARKER_ATTR = "_physicalai_export_config"
_NORMALIZE_CAPTURED_INIT_ARGS_ATTR = "_physicalai_normalize_captured_init_args"
# Optional public class_path override stored on the decorated __init__ wrapper.
_CONFIG_CLASS_PATH_ATTR = "_physicalai_config_class_path"
# Init-arg names the component consumes as Config *data* rather than as
# a constructed object. instantiate() passes these through undecoded.
_CONFIG_ARGS_ATTR = "_physicalai_config_args"

_REPR_LIMIT = 80
