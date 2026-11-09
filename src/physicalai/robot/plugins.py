# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Helpers for loading optional first-party robot plugins."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


_PLUGIN_PACKAGES = frozenset({"physicalai_so101_plugin", "physicalai_trossen_plugin"})


def load_robot_plugin(module_name: str, extra_name: str) -> ModuleType:
    """Import an approved optional robot plugin with installation guidance.

    Args:
        module_name: Import path for an approved robot plugin or one of its modules.
        extra_name: Root package extra that installs the plugin.

    Returns:
        The imported plugin module.

    Raises:
        ValueError: If the plugin module is not approved.
        ModuleNotFoundError: If the requested plugin is not installed.
    """
    try:
        # Only modules from first-party robot plugin packages are allowed.
        package_name, _, _ = module_name.partition(".")
        if package_name not in _PLUGIN_PACKAGES:
            msg = f"Unsupported robot plugin module: {module_name!r}"
            raise ValueError(msg)
        # module_name is constrained to an approved first-party package above.
        # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
        return import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name is None or not module_name.startswith(error.name):
            raise
        msg = f"{module_name} is required for this robot. Install it with: pip install 'physicalai[{extra_name}]'"
        raise ModuleNotFoundError(msg, name=module_name) from error
