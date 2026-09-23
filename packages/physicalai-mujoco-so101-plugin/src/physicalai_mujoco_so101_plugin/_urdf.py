# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""URDF path utility for bundled MuJoCo scene assets."""

from __future__ import annotations

# importlib.resources is safe here because this package requires Python >=3.12.
import importlib.resources as ir  # nosemgrep: python.lang.compatibility.python37.python37-compatibility-importlib2
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def get_urdf_path() -> Path:
    """Return the path to bundled MuJoCo scene assets."""
    traversal = ir.files("physicalai_mujoco_so101_plugin")
    with ir.as_file(traversal) as p:
        candidates = (p / "urdf", p.parent.parent / "urdf")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[1]
