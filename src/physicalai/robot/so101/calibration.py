# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible SO-101 calibration module forwarded to the plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.robot.plugins import load_robot_plugin

if TYPE_CHECKING:
    from physicalai_so101_plugin.calibration import SO101Calibration, SO101JointCalibration

__all__ = ["SO101Calibration", "SO101JointCalibration"]


def __getattr__(name: str) -> object:
    if name in __all__:
        return getattr(load_robot_plugin("physicalai_so101_plugin.calibration", "so101"), name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
