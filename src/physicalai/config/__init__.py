# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Captured component configuration: export live components and instantiate configs.

Trusted local application and parent→child startup configs only. Never pass
network metadata or untrusted peer payloads to :func:`instantiate`.

Opt-in path:

- ``@export_config`` / ``@export_config(class_path=...)`` — remember
  caller-supplied constructor args for :func:`to_config`.
- ``@export_config(..., scalar_var_kwargs=True)`` — seal flattened
  ``**kwargs`` to JSON scalars (non-scalars fail at :func:`to_config`).
- Domain ctor args may implement :meth:`~ConfigValue.to_config_value` to
  return a JSON-compatible fragment (re-normalized).

Transport and other callers import public names from here (no private
``physicalai.config._*`` imports). Studio needs only
:func:`is_config_exportable` and :func:`to_config`, then domain
``from_config`` helpers (for example ``SharedRobot.from_config``).
"""

from ._envelope import (
    normalize_component_config,
    validate_envelope,
)
from ._errors import ComponentConfigError, ComponentImportError
from ._export import (
    export_config,
    is_config_exportable,
    resolve_public_class_path,
    to_config,
)
from ._instantiate import instantiate
from ._normalize import validate_component_config
from ._types import ComponentConfig, ConfigValue, JsonScalar, JsonValue
from ._yaml import load_yaml, save_yaml, to_yaml
from .importing import import_dotted_path

__all__ = [
    "ComponentConfig",
    "ComponentConfigError",
    "ComponentImportError",
    "ConfigValue",
    "JsonScalar",
    "JsonValue",
    "export_config",
    "import_dotted_path",
    "instantiate",
    "is_config_exportable",
    "load_yaml",
    "normalize_component_config",
    "resolve_public_class_path",
    "save_yaml",
    "to_config",
    "to_yaml",
    "validate_component_config",
    "validate_envelope",
]
