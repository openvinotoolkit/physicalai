# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Types for JSON-safe component configuration."""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable

JsonScalar = None | bool | int | float | str
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class ComponentConfig(TypedDict):
    """Wire shape shared with jsonargparse: ``class_path`` + ``init_args``."""

    class_path: str
    init_args: dict[str, JsonValue]


@runtime_checkable
class ConfigValue(Protocol):
    """Domain value that encodes to constructor-compatible JSON for export."""

    def to_config_value(self) -> JsonValue:
        """Return ctor-compatible JSON for component-config export."""
        ...


# Maximum nesting depth for recursive normalization and instantiation.
# Counts traversal through lists and mappings as well as nested component configs.
_MAX_CONFIG_DEPTH = 10

# Private attribute names used by @export_config.
_CAPTURED_INIT_ARGS_ATTR = "_physicalai_captured_init_args"
_EXPORT_DEPTH_ATTR = "_physicalai_export_config_depth"
_EXPORT_MARKER_ATTR = "_physicalai_export_config"
_NORMALIZE_CAPTURED_INIT_ARGS_ATTR = "_physicalai_normalize_captured_init_args"
# Optional public class_path override stored on the decorated __init__ wrapper.
_CONFIG_CLASS_PATH_ATTR = "_physicalai_config_class_path"
# Init-arg names the component consumes as ComponentConfig *data* rather than as
# a constructed object. instantiate() passes these through undecoded.
_CONFIG_ARGS_ATTR = "_physicalai_config_args"

_REPR_LIMIT = 80
