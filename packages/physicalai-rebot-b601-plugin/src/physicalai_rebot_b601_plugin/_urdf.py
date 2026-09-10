"""URDF path utility for bundled robot description packages."""

from __future__ import annotations

# importlib.resources is safe here because this package requires Python >=3.11.
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
        >>> from physicalai_rebot_b601_plugin import get_urdf_path
        >>> urdf_dir = get_urdf_path()
        >>> dm_urdf = urdf_dir / "rebot-b601-dm" / "urdf" / "reBot-DevArm_fixend.urdf"
        >>> rs_urdf = urdf_dir / "rebot-b601-rs" / "urdf" / "00-arm-rs_asm-v3.urdf"
    """
    traversal = ir.files("physicalai_rebot_b601_plugin")
    with ir.as_file(traversal) as p:
        return p.parent.parent.joinpath("urdf")
