# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared deprecation warnings for compatibility facades."""

from __future__ import annotations

import warnings


def deprecate(name: str, replacement: str) -> None:
    """Emit a :class:`DeprecationWarning` for a legacy public API."""
    warnings.warn(
        f"{name} is deprecated and will be removed in a future release; use {replacement} instead.",
        DeprecationWarning,
        stacklevel=3,
    )
