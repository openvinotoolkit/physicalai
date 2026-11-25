# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Capture error hierarchy.

All capture-related exceptions inherit from :class:`CaptureError`, which
itself extends :class:`RuntimeError`. This allows callers to catch broad
(``except CaptureError``) or narrow (``except CaptureTimeoutError``).
"""


class CaptureError(RuntimeError):
    """Base error for capture failures."""


class NotConnectedError(CaptureError):
    """Raised when read methods are called before connect()."""


class CaptureTimeoutError(CaptureError):
    """Raised when a read or connect operation exceeds its timeout."""


class CaptureSkewError(CaptureError):
    """Raised when camera timestamp skew exceeds the maximum allowed threshold."""

    def __init__(self, skew_ms: float, max_skew_ms: float) -> None:
        """Create a capture skew error.

        Args:
            skew_ms: The actual observed skew across camera frames in milliseconds.
            max_skew_ms: The maximum allowed skew threshold in milliseconds.
        """
        self.skew_ms = skew_ms
        self.max_skew_ms = max_skew_ms
        super().__init__(
            f"Camera timestamp skew of {skew_ms:.1f}ms exceeds maximum allowed threshold of {max_skew_ms:.1f}ms"
        )


class MissingDependencyError(CaptureError):
    """Raised when a camera SDK extra is not installed."""

    def __init__(self, package: str, extra: str) -> None:
        """Create a missing-dependency error.

        Args:
            package: The Python package that is missing (e.g. ``"pyrealsense2"``).
            extra: The pip extra to install (e.g. ``"realsense"``).
        """
        self.package = package
        self.extra = extra
        super().__init__(f"{package} is required but not installed. Install it with: pip install physicalai[{extra}]")
