# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build and save configuration for Physical AI components.

Use :class:`Config` for portable YAML recipes (``class_path`` + ``init_args``)
and :func:`instantiate` for trusted local recipes (robots, cameras, exported
components).

Use jsonargparse directly for known types and
:meth:`Config.from_instance` / :meth:`Config.save` for export.

Opt-in export:

- ``@export_config`` — record constructor arguments for
  :meth:`Config.from_instance` and add ``supports_config_export()`` /
  ``as_config()`` on instances.
- ``@export_config(..., scalar_var_kwargs=True)`` — only JSON-safe ``**kwargs``.
- :meth:`~ConfigValue.to_config_value` — custom JSON fragments for domain types.

Import public names from this package; avoid private ``physicalai.config._*``
modules.
"""

from ._errors import ConfigError, ConfigImportError
from ._export import (
    export_config,
    is_config_exportable,
    resolve_public_class_path,
)
from ._instantiate import instantiate as _strict_instantiate
from ._normalize import validate_config
from ._types import ConfigValue, JsonScalar, JsonValue
from .base import Config

instantiate = _strict_instantiate  # ruff: ignore[RUF067]

__all__ = [
    "Config",
    "ConfigError",
    "ConfigImportError",
    "ConfigValue",
    "JsonScalar",
    "JsonValue",
    "export_config",
    "instantiate",
    "is_config_exportable",
    "resolve_public_class_path",
    "validate_config",
]
