# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared validation for transport construction envelopes.

Robot-owner and camera-publisher stdin envelopes carry one nested
:class:`ComponentConfig` plus transport-only keys. These helpers keep the
schema-positive validation and public-path normalization in one place;
transports supply their key names and allowlists.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ._errors import ComponentConfigError
from ._export import resolve_public_class_path
from ._normalize import validate_component_config
from .importing import import_dotted_path

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
    only *allowed_keys*. Unknown keys (including legacy flat shapes) raise a
    clear schema error before any import or hardware access.

    Args:
        data: Full stdin envelope dict.
        component_key: Envelope key holding the nested ComponentConfig
            (for example ``"robot"`` or ``"camera"``).
        allowed_keys: Complete allowlist of envelope keys.
        envelope_name: Short envelope label for error messages
            (for example ``"owner"`` or ``"publisher"``).

    Returns:
        The validated ComponentConfig mapping (not yet public-path-normalized
        — :func:`normalize_component_config` does that).

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


def normalize_class_reference(ref: type | str, *, label: str) -> str:
    """Normalize a class object or dotted path to its verified public path.

    Matches :func:`to_config` path selection: decorator ``class_path=``
    override when present, otherwise ``__module__.__qualname__``. String
    paths are imported first so a defining-module path becomes the public
    re-export before store, metadata advertise, and conflict compare.

    Args:
        ref: A class object, or its dotted import path as a string.
        label: Argument label for error messages (for example ``"robot_class"``).

    Returns:
        The normalized public dotted path.

    Raises:
        TypeError: If *ref* is neither a string nor a class, or a string path
            does not resolve to a class.
        ValueError: If the path cannot be imported, or the public path cannot
            be resolved (for example a local class).
    """
    if isinstance(ref, str):
        try:
            resolved = import_dotted_path(ref)
        except (ValueError, ImportError, AttributeError) as exc:
            msg = f"could not import {label} {ref!r}: {exc}"
            raise ValueError(msg) from exc
        if not isinstance(resolved, type):
            msg = f"{label} {ref!r} does not resolve to a class (got {type(resolved).__name__})"
            raise TypeError(msg)
        ref = resolved
    if not isinstance(ref, type):
        msg = f"{label} must be a class or a dotted path string, got {type(ref).__name__}"
        raise TypeError(msg)

    try:
        return resolve_public_class_path(ref)
    except ComponentConfigError as exc:
        raise ValueError(str(exc)) from exc


def normalize_component_config(
    config: Mapping[str, object],
    *,
    component_key: str,
    class_label: str,
    json_hint: str = "",
) -> ComponentConfig:
    """Validate a ComponentConfig and normalize its ``class_path`` to the public path.

    Args:
        config: Candidate ``class_path`` + ``init_args`` mapping.
        component_key: Path prefix for validation errors (``"robot"`` / ``"camera"``).
        class_label: Argument label for class-reference errors.
        json_hint: Optional suffix appended to the JSON-serializability error.

    Returns:
        A validated config whose ``class_path`` is the public import path.

    Raises:
        ComponentConfigError: If *config* is not a mapping.
        ValueError: If ``class_path`` cannot be imported or ``init_args`` is
            not JSON-serializable.
    """
    if not isinstance(config, Mapping):
        msg = f"{component_key} must be a ComponentConfig mapping, got {type(config).__name__}"
        raise ComponentConfigError(msg)
    validated = validate_component_config(dict(config), path=component_key)
    class_path = normalize_class_reference(validated["class_path"], label=class_label)
    init_args = validated["init_args"]
    try:
        json.dumps({"class_path": class_path, "init_args": init_args})
    except TypeError as exc:
        msg = f"{component_key}.init_args must be JSON-serializable{json_hint}: {exc}"
        raise ValueError(msg) from exc
    return {"class_path": class_path, "init_args": dict(init_args)}
