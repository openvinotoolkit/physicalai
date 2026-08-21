# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dotted-path imports for robot owner configuration.

Canonical implementation lives in :mod:`physicalai.config._importing`. This
module re-exports for backward-compatible imports without loading export or
instantiate.
"""

from __future__ import annotations

# This private re-export keeps transport imports independent of config construction.
from physicalai.config._importing import import_dotted_path  # ruff: ignore[PLC2701]

__all__ = ["import_dotted_path"]
