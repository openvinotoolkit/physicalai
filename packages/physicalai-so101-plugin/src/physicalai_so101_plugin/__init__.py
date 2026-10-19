# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""SO-101 robot driver, Studio catalog integration, and bundled URDF assets."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from physicalai_so101_plugin._urdf import get_urdf_path as get_urdf_path

if TYPE_CHECKING:
    from physicalai_so101_plugin.bimanual_so101 import BimanualSO101 as BimanualSO101
    from physicalai_so101_plugin.bimanual_so101 import BimanualSO101Observation as BimanualSO101Observation
    from physicalai_so101_plugin.calibration import SO101Calibration as SO101Calibration
    from physicalai_so101_plugin.calibration import SO101JointCalibration as SO101JointCalibration
    from physicalai_so101_plugin.so101 import SO101 as SO101
    from physicalai_so101_plugin.so101 import SO101Observation as SO101Observation

__all__ = [
    "SO101",
    "BimanualSO101",
    "BimanualSO101Observation",
    "SO101Calibration",
    "SO101JointCalibration",
    "SO101Observation",
    "get_urdf_path",
]


def __getattr__(name: str) -> object:
    """Lazily import the hardware driver without loading its SDK on package import.

    Returns:
        The requested SO-101 driver or calibration type.

    Raises:
        AttributeError: If ``name`` is not exported by this package.
    """
    if name in {"SO101Calibration", "SO101JointCalibration"}:
        return getattr(import_module("physicalai_so101_plugin.calibration"), name)
    if name in {"SO101", "SO101Observation"}:
        return getattr(import_module("physicalai_so101_plugin.so101"), name)
    if name in {"BimanualSO101", "BimanualSO101Observation"}:
        return getattr(import_module("physicalai_so101_plugin.bimanual_so101"), name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
