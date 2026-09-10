# ruff: noqa: PLC0415

"""Bimanual SO-101 robot arm plugin for PhysicalAI.

Composes two :class:`physicalai.robot.so101.SO101` arms (left + right) behind the
``physicalai.robot.Robot`` protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai_bimanual_so101_plugin._urdf import get_urdf_path as get_urdf_path

if TYPE_CHECKING:
    from physicalai_bimanual_so101_plugin.bimanual import BimanualSO101 as BimanualSO101
    from physicalai_bimanual_so101_plugin.bimanual import (
        BimanualSO101Observation as BimanualSO101Observation,
    )

__all__ = [
    "BimanualSO101",
    "BimanualSO101Observation",
    "get_urdf_path",
]


def __getattr__(name: str) -> object:
    if name == "BimanualSO101":
        from physicalai_bimanual_so101_plugin.bimanual import BimanualSO101

        return BimanualSO101
    if name == "BimanualSO101Observation":
        from physicalai_bimanual_so101_plugin.bimanual import BimanualSO101Observation

        return BimanualSO101Observation
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
