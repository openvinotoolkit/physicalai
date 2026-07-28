# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dotted-path imports for robot owner configuration.

Canonical implementation lives in :mod:`physicalai.config.importing`. This
module re-exports for backward-compatible imports without loading export or
instantiate.
"""

from __future__ import annotations

from physicalai.config.importing import import_dotted_path

__all__ = ["import_dotted_path"]
