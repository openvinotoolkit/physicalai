# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Built-in camera type token ↔ public class_path for SharedCamera.

Single source of truth for service-name derivation and
``create_camera(..., shared=True)``. Only backends with a real driver
package under ``cameras/`` are listed — IP/Genicam stubs are intentionally
absent so shared spawn cannot claim phantom coverage.

The table is static on purpose: a subscriber process derives a service name
without the vendor SDK installed, so nothing here may import a driver
module. ``tests/unit/capture/test_factory.py`` asserts the table still
matches each driver's ``@export_config(class_path=...)`` when the optional
camera extras are available.
"""

from __future__ import annotations

# token → accepted class paths. Index 0 is the canonical public path declared
# by ``@export_config(class_path=...)`` on the driver; the rest are the
# sub-package re-export and the defining module, so a hand-written config that
# spells a driver the internal way still derives the same service name instead
# of demanding an explicit one.
_BUILTIN_SHARED: dict[str, tuple[str, ...]] = {
    "uvc": (
        "physicalai.capture.UVCCamera",
        "physicalai.capture.cameras.uvc.UVCCamera",
        "physicalai.capture.cameras.uvc._camera.UVCCamera",
    ),
    "realsense": (
        "physicalai.capture.RealSenseCamera",
        "physicalai.capture.cameras.realsense.RealSenseCamera",
        "physicalai.capture.cameras.realsense._camera.RealSenseCamera",
    ),
    "basler": (
        "physicalai.capture.BaslerCamera",
        "physicalai.capture.cameras.basler.BaslerCamera",
        "physicalai.capture.cameras.basler._camera.BaslerCamera",
    ),
}

_TOKEN_TO_CLASS_PATH: dict[str, str] = {token: paths[0] for token, paths in _BUILTIN_SHARED.items()}
_CLASS_PATH_TO_TOKEN: dict[str, str] = {path: token for token, paths in _BUILTIN_SHARED.items() for path in paths}


def builtin_shared_type_tokens() -> frozenset[str]:
    """Return CameraType tokens that support shared service-name derivation."""
    return frozenset(_BUILTIN_SHARED)


def builtin_class_paths_for_type(token: str) -> tuple[str, ...]:
    r"""Return every class_path spelling accepted for a built-in type token.

    Args:
        token: Lowercase ``CameraType`` value (e.g. ``\"uvc\"``).

    Returns:
        Accepted paths with the canonical public one first, or an empty tuple
        if the token is not a shareable built-in.
    """
    return _BUILTIN_SHARED.get(token, ())


def builtin_class_path_for_type(token: str) -> str | None:
    r"""Return the canonical public class_path for a built-in shared camera type token.

    Args:
        token: Lowercase ``CameraType`` value (e.g. ``\"uvc\"``).

    Returns:
        Public ``class_path``, or ``None`` if the token is not a shareable built-in.
    """
    return _TOKEN_TO_CLASS_PATH.get(token)


def builtin_type_for_class_path(class_path: str) -> str | None:
    """Return the legacy type token for a built-in camera class_path.

    Accepts the canonical public path plus the internal spellings listed in
    :data:`_BUILTIN_SHARED`, so the same physical device derives one service
    name however the config names its driver.

    Args:
        class_path: Camera ``class_path`` as written in the config.

    Returns:
        Type token (``uvc`` / ``realsense`` / ``basler``), or ``None`` for
        third-party / stub / unknown paths.
    """
    return _CLASS_PATH_TO_TOKEN.get(class_path)
