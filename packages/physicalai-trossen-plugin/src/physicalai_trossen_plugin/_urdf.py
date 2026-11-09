# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""URDF path utility for bundled Trossen robot descriptions."""

from __future__ import annotations

import importlib.resources as ir  # nosemgrep: python.lang.compatibility.python37.python37-compatibility-importlib2
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def get_urdf_path() -> Path:
    """Return the directory containing bundled Trossen description assets."""
    traversal = ir.files("physicalai_trossen_plugin")
    with ir.as_file(traversal) as package_path:
        candidates = (package_path.parent / "urdf", package_path.parent.parent / "urdf")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[1]
