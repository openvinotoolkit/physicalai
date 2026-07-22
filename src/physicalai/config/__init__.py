# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Captured component configuration: export live components and instantiate configs.

Trusted local application and parent→child startup configs only. Never pass
network metadata or untrusted peer payloads to :func:`instantiate`.

Opt-in path:

- ``@export_config`` / ``@export_config(class_path=...)`` — remember
  caller-supplied constructor args for :func:`to_config`.
- Domain ctor args may implement :meth:`~ConfigValue.to_config_value` to
  return a JSON-compatible fragment (re-normalized).

Studio needs only :func:`is_config_exportable` and :func:`to_config`, then
domain ``from_config`` helpers (for example ``SharedRobot.from_config``).

Public names are resolved lazily so ``physicalai.config.importing`` can be
imported without loading export/normalize/instantiate.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._errors import ComponentConfigError as ComponentConfigError
    from ._errors import ComponentImportError as ComponentImportError
    from ._export import export_config as export_config
    from ._export import is_config_exportable as is_config_exportable
    from ._export import to_config as to_config
    from ._instantiate import instantiate as instantiate
    from ._types import ComponentConfig as ComponentConfig
    from ._types import ConfigValue as ConfigValue
    from ._types import JsonScalar as JsonScalar
    from ._types import JsonValue as JsonValue
    from .importing import import_dotted_path as import_dotted_path

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
    "to_config",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "ComponentConfig": ("._types", "ComponentConfig"),
    "ConfigValue": ("._types", "ConfigValue"),
    "JsonScalar": ("._types", "JsonScalar"),
    "JsonValue": ("._types", "JsonValue"),
    "ComponentConfigError": ("._errors", "ComponentConfigError"),
    "ComponentImportError": ("._errors", "ComponentImportError"),
    "import_dotted_path": (".importing", "import_dotted_path"),
    "export_config": ("._export", "export_config"),
    "is_config_exportable": ("._export", "is_config_exportable"),
    "to_config": ("._export", "to_config"),
    "instantiate": ("._instantiate", "instantiate"),
}


def __getattr__(name: str) -> object:
    """Lazily resolve public API attributes.

    Returns:
        The requested public attribute.

    Raises:
        AttributeError: If *name* is not part of the public API.
    """
    try:
        module_name, attr = _LAZY_ATTRS[name]
    except KeyError as exc:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg) from exc
    value = getattr(import_module(module_name, __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
