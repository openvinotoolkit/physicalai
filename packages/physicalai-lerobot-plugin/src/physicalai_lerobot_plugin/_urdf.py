# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""URDF path utility for bundled robot description packages."""

from __future__ import annotations

# importlib.resources is safe here because this package requires Python >=3.12.
import importlib.resources as ir  # nosemgrep: python.lang.compatibility.python37.python37-compatibility-importlib2
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def get_urdf_path() -> Path:
    traversal = ir.files("physicalai_lerobot_plugin")
    with ir.as_file(traversal) as p:
        return p.parent.joinpath("urdf") if p.parent.joinpath("urdf").exists() else p.parent.parent.joinpath("urdf")
