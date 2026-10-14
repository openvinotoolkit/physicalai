# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dotted-path imports for robot owner configuration.

Canonical implementation lives in :mod:`physicalai.config._importing`. This
module re-exports the shared resolver for robot transport callers.
"""

from __future__ import annotations

# Keep transport's import path backed by the shared resolver.
from physicalai.config._importing import import_dotted_path  # ruff: ignore[PLC2701]

__all__ = ["import_dotted_path"]
