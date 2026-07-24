# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""YAML round-trip helpers for component configs.

``to_yaml`` / ``save_yaml`` serialize a live ``@export_config`` component (or
an existing ``class_path`` + ``init_args`` mapping) to a YAML document that
``load_yaml`` + :func:`~physicalai.config.instantiate` can rebuild. The same
document is accepted by ``physicalai run --config`` when the top-level
component is a ``RobotRuntime``.

Trusted local configs only — never feed network-received YAML to
:func:`~physicalai.config.instantiate`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ._errors import ComponentConfigError
from ._export import to_config
from ._normalize import validate_component_config

if TYPE_CHECKING:
    from ._types import ComponentConfig, JsonValue


def to_yaml(component: object) -> str:
    """Serialize a component to a YAML ``class_path`` + ``init_args`` document.

    Args:
        component: A live ``@export_config`` instance, or an existing
            :class:`ComponentConfig` mapping (validated, not re-exported).

    Returns:
        YAML text of the validated :class:`ComponentConfig`.
    """
    config: ComponentConfig
    config = validate_component_config(dict(component)) if isinstance(component, Mapping) else to_config(component)
    return yaml.safe_dump(dict(config), sort_keys=False, default_flow_style=False)


def save_yaml(component: object, path: str | Path) -> None:
    """Write :func:`to_yaml` output to *path* (parent directories must exist).

    Args:
        component: A live ``@export_config`` instance or a
            :class:`ComponentConfig` mapping.
        path: Destination file path.
    """
    Path(path).write_text(to_yaml(component), encoding="utf-8")


def load_yaml(path: str | Path) -> dict[str, JsonValue]:
    """Load a YAML document as a mapping, ready for :func:`~physicalai.config.instantiate`.

    The document is parsed with ``yaml.safe_load`` and only shape-checked to
    be a mapping — full component validation happens in
    :func:`~physicalai.config.instantiate`.

    Args:
        path: YAML file previously written by :func:`save_yaml` (or
            hand-authored in the same shape).

    Returns:
        The loaded top-level mapping.

    Raises:
        ComponentConfigError: If the document is not a mapping.
    """
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = f"{str(path)!r}: YAML config must be a mapping, got {type(loaded).__name__}"
        raise ComponentConfigError(msg)
    return loaded
