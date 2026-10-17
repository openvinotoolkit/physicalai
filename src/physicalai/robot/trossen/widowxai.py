# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible WidowX AI driver module forwarded to the plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.robot.plugins import load_robot_plugin

if TYPE_CHECKING:
    from physicalai_trossen_plugin.widowxai import WidowXAI, WidowXAIObservation

__all__ = ["WidowXAI", "WidowXAIObservation"]


def __getattr__(name: str) -> object:
    if name in __all__:
        return getattr(load_robot_plugin("physicalai_trossen_plugin.widowxai", "trossen"), name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
