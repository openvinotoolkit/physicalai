# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Captured configuration: export live components and instantiate configs.

Trusted local application and parent→child startup configs only. Never pass
network metadata or untrusted peer payloads to :func:`instantiate`.

Two instantiation entry points:

- :func:`instantiate` — strict captured ``class_path`` + ``init_args`` recipes
  (transport, ``@export_config`` components).
- :func:`instantiate_obj` — generic jsonargparse / Studio loaders; see
  :mod:`physicalai.config.loading`.

Opt-in path:

- ``@export_config`` / ``@export_config(class_path=...)`` — remember
  caller-supplied constructor args for :meth:`Config.from_instance`.
- ``@export_config(..., scalar_var_kwargs=True)`` — seal flattened
  ``**kwargs`` to JSON scalars (non-scalars fail during export).
- Domain ctor args may implement :meth:`~ConfigValue.to_config_value` to
  return a JSON-compatible fragment (re-normalized).

Transport and other callers import public names from here (no private
``physicalai.config._*`` imports).
"""

from ._envelope import (
    normalize_config,
    validate_envelope,
)
from ._errors import ConfigError, ConfigImportError
from ._export import (
    export_config,
    is_config_exportable,
    resolve_public_class_path,
    to_config,
)
from ._instantiate import instantiate as _strict_instantiate
from ._normalize import validate_config
from ._types import ConfigValue, JsonScalar, JsonValue
from ._yaml import load_yaml, save_yaml, to_yaml
from .base import Config
from .importing import import_dotted_path
from .loading import import_class, instantiate_obj
from .mixin import FromConfig, from_config

instantiate = _strict_instantiate  # ruff: ignore[RUF067]

__all__ = [
    "Config",
    "ConfigError",
    "ConfigImportError",
    "ConfigValue",
    "FromConfig",
    "JsonScalar",
    "JsonValue",
    "export_config",
    "from_config",
    "import_class",
    "import_dotted_path",
    "instantiate",
    "instantiate_obj",
    "is_config_exportable",
    "load_yaml",
    "normalize_config",
    "resolve_public_class_path",
    "save_yaml",
    "to_config",
    "to_yaml",
    "validate_config",
    "validate_envelope",
]
