# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Helpers for loading optional first-party robot plugins."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def load_robot_plugin(module_name: str, extra_name: str) -> ModuleType:
    """Import an optional robot plugin with actionable installation guidance.

    Args:
        module_name: Import path for the robot plugin or one of its modules.
        extra_name: Root package extra that installs the plugin.

    Returns:
        The imported plugin module.

    Raises:
        ModuleNotFoundError: If the requested plugin is not installed.
    """
    try:
        return import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name is None or not module_name.startswith(error.name):
            raise
        msg = f"{module_name} is required for this robot. Install it with: pip install 'physicalai[{extra_name}]'"
        raise ModuleNotFoundError(msg, name=module_name) from error
