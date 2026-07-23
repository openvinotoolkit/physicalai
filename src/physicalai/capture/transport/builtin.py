# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Built-in camera type token ↔ public class_path for SharedCamera.

Single source of truth for service-name derivation and
``create_camera(..., shared=True)``. Only backends with a real driver
package under ``cameras/`` are listed — IP/Genicam stubs are intentionally
absent so shared spawn cannot claim phantom coverage.
"""

from __future__ import annotations

import importlib
from functools import lru_cache

from physicalai.config import ComponentConfigError, resolve_public_class_path

# token → (import module, attribute, decorator-declared public class_path).
# Public paths match ``@export_config(class_path=...)`` on each driver.
# Fallback strings keep service_name derivation working when an optional
# camera extra is not installed (must not import pyrealsense2/pypylon).
_BUILTIN_SHARED: dict[str, tuple[str, str, str]] = {
    "uvc": (
        "physicalai.capture.cameras.uvc",
        "UVCCamera",
        "physicalai.capture.UVCCamera",
    ),
    "realsense": (
        "physicalai.capture.cameras.realsense",
        "RealSenseCamera",
        "physicalai.capture.RealSenseCamera",
    ),
    "basler": (
        "physicalai.capture.cameras.basler",
        "BaslerCamera",
        "physicalai.capture.BaslerCamera",
    ),
}


@lru_cache(maxsize=1)
def _token_to_class_path() -> dict[str, str]:
    """Resolve public class_path per token, preferring live ``@export_config``.

    Returns:
        Mapping from shareable type token to public ``class_path``.
    """
    resolved: dict[str, str] = {}
    for token, (module_name, attr, fallback) in _BUILTIN_SHARED.items():
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, attr)
            resolved[token] = resolve_public_class_path(cls)
        except (ImportError, AttributeError, ComponentConfigError, TypeError, ValueError):
            resolved[token] = fallback
    return resolved


@lru_cache(maxsize=1)
def _class_path_to_token() -> dict[str, str]:
    return {path: token for token, path in _token_to_class_path().items()}


def builtin_shared_type_tokens() -> frozenset[str]:
    """Return CameraType tokens that support shared service-name derivation."""
    return frozenset(_BUILTIN_SHARED)


def builtin_class_path_for_type(token: str) -> str | None:
    r"""Return the public class_path for a built-in shared camera type token.

    Args:
        token: Lowercase ``CameraType`` value (e.g. ``\"uvc\"``).

    Returns:
        Public ``class_path``, or ``None`` if the token is not a shareable built-in.
    """
    return _token_to_class_path().get(token)


def builtin_type_for_class_path(class_path: str) -> str | None:
    """Return the legacy type token for a built-in public class_path.

    Args:
        class_path: Normalized public camera ``class_path``.

    Returns:
        Type token (``uvc`` / ``realsense`` / ``basler``), or ``None`` for
        third-party / stub / unknown paths.
    """
    return _class_path_to_token().get(class_path)
