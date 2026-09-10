"""URDF path utility for bundled robot description packages."""

from __future__ import annotations

# importlib.resources is safe here because this package requires Python >=3.12.
import importlib.resources as ir  # nosemgrep: python.lang.compatibility.python37.python37-compatibility-importlib2
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def get_urdf_path() -> Path:
    """Return the path to the bundled URDF directory.

    The package supports both editable installs (repo layout) and wheel installs
    (site-packages layout). We probe both expected locations and return the first
    existing directory.

    Returns:
        Path to the ``urdf/`` directory containing robot description assets.
    """
    traversal = ir.files("physicalai_bimanual_so101_plugin")
    with ir.as_file(traversal) as p:
        candidates = (p.parent / "urdf", p.parent.parent / "urdf")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[1]
