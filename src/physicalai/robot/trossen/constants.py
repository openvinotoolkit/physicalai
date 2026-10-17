# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible Trossen constants module forwarded to the plugin."""

from __future__ import annotations

from physicalai.robot.plugins import load_robot_plugin


def __getattr__(name: str) -> object:
    return getattr(load_robot_plugin("physicalai_trossen_plugin.constants", "trossen"), name)
