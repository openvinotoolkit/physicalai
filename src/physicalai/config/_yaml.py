# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""YAML round-trip helpers for configs.

``to_yaml`` / ``save_yaml`` serialize a live ``@export_config`` component (or
an existing ``class_path`` + ``init_args`` mapping) to a YAML document that
``load_yaml`` + :func:`~physicalai.config.instantiate` can rebuild. The same
document is accepted by ``physicalai run --config`` when the top-level
component is a ``RobotRuntime``.

Trusted local configs only — never feed network-received YAML to
:func:`~physicalai.config.instantiate`.

.. deprecated::
    Prefer :meth:`~physicalai.config.Config.save` and
    :meth:`~physicalai.config.Config.load`. Emits :class:`DeprecationWarning`
    at runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from ._deprecate import deprecate
from ._errors import ConfigError
from ._export import to_config
from ._normalize import normalize_config
from .base import Config

_REPLACEMENT = "Config.from_instance(...).save(path) and Config.load(path)"


def to_yaml(component: object) -> str:
    """Serialize a component to a YAML ``class_path`` + ``init_args`` document.

    .. deprecated::
        Use :meth:`~physicalai.config.Config.save` or serialize
        :meth:`~physicalai.config.Config.to_dict`.

    Args:
        component: A live ``@export_config`` instance, or an existing
            :class:`Config` mapping (validated, not re-exported).

    Returns:
        YAML text of the validated :class:`Config`.
    """
    deprecate("physicalai.config.to_yaml", _REPLACEMENT)
    if type(component) is Config:
        config = component
    elif isinstance(component, Mapping):
        config = Config.from_dict(component)
    else:
        config = to_config(component)
    return yaml.safe_dump(config.to_dict(), sort_keys=False, default_flow_style=False)


def save_yaml(component: object, path: str | Path) -> None:
    """Write :func:`to_yaml` output to *path* (parent directories must exist).

    .. deprecated::
        Use :meth:`~physicalai.config.Config.save`.

    Args:
        component: A live ``@export_config`` instance or a
            :class:`Config` mapping.
        path: Destination file path.
    """
    deprecate("physicalai.config.save_yaml", _REPLACEMENT)
    Path(path).write_text(to_yaml(component), encoding="utf-8")


def load_yaml(path: str | Path) -> Config:
    """Load a YAML document as a mapping, ready for :func:`~physicalai.config.instantiate`.

    The document is parsed with ``yaml.safe_load`` and only shape-checked to
    be a mapping — full component validation happens in
    :func:`~physicalai.config.instantiate`.

    .. deprecated::
        Use :meth:`~physicalai.config.Config.load`.

    Args:
        path: YAML file previously written by :func:`save_yaml` (or
            hand-authored in the same shape).

    Returns:
        The loaded top-level mapping.

    Raises:
        ConfigError: If the document is not a mapping.
    """
    deprecate("physicalai.config.load_yaml", "Config.load(path)")
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        msg = f"{str(path)!r}: YAML config must be a mapping, got {type(loaded).__name__}"
        raise ConfigError(msg)
    return Config.from_dict(normalize_config(loaded))
