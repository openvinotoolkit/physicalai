# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared validation for transport construction envelopes.

Robot-owner and camera-publisher stdin envelopes carry one nested
:class:`ComponentConfig` plus transport-only keys. These helpers keep the
schema-positive validation in one place; transports supply their key names
and allowlists.

Nothing here imports a ``class_path``. Envelopes are built in the subscriber
process, which must stay free of the driver package — the import happens in
the process that calls :func:`~physicalai.config.instantiate`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ._errors import ComponentConfigError
from ._normalize import validate_component_config

if TYPE_CHECKING:
    from ._types import ComponentConfig


def validate_envelope(
    data: Mapping[str, Any],
    *,
    component_key: str,
    allowed_keys: frozenset[str],
    envelope_name: str,
) -> Mapping[str, object]:
    """Validate a transport stdin envelope schema-positively.

    Requires *component_key* with a valid ComponentConfig shape and allows
    only *allowed_keys*. Unknown keys raise a clear schema error before any
    import or hardware access.

    Args:
        data: Full stdin envelope dict.
        component_key: Envelope key holding the nested ComponentConfig
            (for example ``"robot"`` or ``"camera"``).
        allowed_keys: Complete allowlist of envelope keys.
        envelope_name: Short envelope label for error messages
            (for example ``"owner"`` or ``"publisher"``).

    Returns:
        The validated ComponentConfig mapping (see
        :func:`normalize_component_config` for the JSON-serializability check).

    Raises:
        TypeError: If *data* or the component value is not a mapping.
        ValueError: If the component key is missing or unknown keys are present.
    """
    if not isinstance(data, Mapping):
        msg = f"{envelope_name} config must be a mapping, got {type(data).__name__}"
        raise TypeError(msg)

    unknown = sorted(set(data) - allowed_keys)
    if unknown:
        msg = (
            f"unknown {envelope_name} config keys {unknown}; "
            f"require {component_key!r} with class_path + init_args "
            f"(allowed envelope keys: {sorted(allowed_keys)})"
        )
        raise ValueError(msg)

    if component_key not in data:
        msg = f"{envelope_name} config missing required {component_key!r} ComponentConfig"
        raise ValueError(msg)

    component = data[component_key]
    if not isinstance(component, Mapping):
        msg = f"{envelope_name} {component_key!r} must be a mapping, got {type(component).__name__}"
        raise TypeError(msg)

    return validate_component_config(dict(component), path=component_key)


def normalize_component_config(
    config: Mapping[str, object],
    *,
    component_key: str,
    class_label: str,
    json_hint: str = "",
) -> ComponentConfig:
    """Validate a ComponentConfig without importing its ``class_path``.

    The ``class_path`` is trusted and kept exactly as written: envelopes are
    built in the subscriber process, which must not load the driver package
    (and often cannot — the vendor SDK is only installed where the hardware
    lives). Import errors surface later, in the process that calls
    :func:`~physicalai.config.instantiate`.

    Args:
        config: Candidate ``class_path`` + ``init_args`` mapping.
        component_key: Path prefix for validation errors (``"robot"`` / ``"camera"``).
        class_label: Argument label for ``class_path`` errors.
        json_hint: Optional suffix appended to the JSON-serializability error.

    Returns:
        A validated config whose ``class_path`` is a dotted import path.

    Raises:
        ComponentConfigError: If *config* is not a mapping.
        ValueError: If ``class_path`` is not a dotted path or ``init_args`` is
            not JSON-serializable.
    """
    if not isinstance(config, Mapping):
        msg = f"{component_key} must be a ComponentConfig mapping, got {type(config).__name__}"
        raise ComponentConfigError(msg)
    validated = validate_component_config(dict(config), path=component_key)
    class_path = validated["class_path"]
    if not class_path.strip() or "." not in class_path:
        msg = f"{class_label} must be a nonempty dotted path, got {class_path!r}"
        raise ValueError(msg)
    init_args = validated["init_args"]
    try:
        json.dumps({"class_path": class_path, "init_args": init_args}, allow_nan=False)
    except (TypeError, ValueError) as exc:
        msg = f"{component_key}.init_args must be JSON-serializable{json_hint}: {exc}"
        raise ValueError(msg) from exc
    return {"class_path": class_path, "init_args": dict(init_args)}
