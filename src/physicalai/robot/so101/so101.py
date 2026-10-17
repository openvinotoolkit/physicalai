# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible SO-101 driver module forwarded to the plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.robot.plugins import load_robot_plugin

if TYPE_CHECKING:
    from physicalai_so101_plugin.so101 import SO101, SO101Observation

__all__ = ["SO101", "SO101Observation"]


def __getattr__(name: str) -> object:
    if name in __all__:
        return getattr(load_robot_plugin("physicalai_so101_plugin.so101", "so101"), name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
