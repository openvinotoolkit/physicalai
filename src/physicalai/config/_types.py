# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Types for JSON-safe component configuration."""

from __future__ import annotations

from typing import TypedDict

JsonScalar = None | bool | int | float | str
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class ComponentConfig(TypedDict):
    """Wire shape shared with jsonargparse: ``class_path`` + ``init_args``."""

    class_path: str
    init_args: dict[str, JsonValue]


# Maximum nesting depth for recursive normalization and instantiation.
# Counts traversal through lists and mappings as well as nested component configs.
_MAX_CONFIG_DEPTH = 10

# Private attribute names used by @export_config.
_CAPTURED_INIT_ARGS_ATTR = "_physicalai_captured_init_args"
_EXPORT_DEPTH_ATTR = "_physicalai_export_config_depth"
_EXPORT_MARKER_ATTR = "_physicalai_export_config"
_CONFIG_HOOK_NAME = "__component_config__"
_CONFIG_CLASS_PATH_ATTR = "__config_class_path__"

_REPR_LIMIT = 80
