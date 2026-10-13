# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""URDF path utility for the bundled Franka description package."""

from __future__ import annotations

import importlib.resources as ir  # nosemgrep: python.lang.compatibility.python37.python37-compatibility-importlib2
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def get_urdf_path() -> Path:
    """Return the path to the bundled URDF directory.

    The URDF files are installed alongside the Python package. This function
    locates them regardless of whether the package is installed in development
    mode (editable) or from a wheel.

    Returns:
        Path to the ``urdf/`` directory containing robot description packages.

    Example:
        >>> from physicalai_franka_plugin import get_urdf_path
        >>> fr3_urdf = get_urdf_path() / "franka_description" / "urdf" / "fr3.urdf"

    """
    traversal = ir.files("physicalai_franka_plugin")
    with ir.as_file(traversal) as p:
        return p.parent.parent.joinpath("urdf")
