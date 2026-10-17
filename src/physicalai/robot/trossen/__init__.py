# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible Trossen imports forwarded to the plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.robot.plugins import load_robot_plugin

if TYPE_CHECKING:
    from physicalai_trossen_plugin import BimanualWidowXAI, BimanualWidowXAIObservation, WidowXAI, WidowXAIObservation

__all__ = ["BimanualWidowXAI", "BimanualWidowXAIObservation", "WidowXAI", "WidowXAIObservation"]


def __getattr__(name: str) -> object:
    if name in __all__:
        return getattr(load_robot_plugin("physicalai_trossen_plugin", "trossen"), name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
