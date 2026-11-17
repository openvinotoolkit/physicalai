# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Passive USB diagnostics shared by terminal and application callers.

``diagnose()`` is blocking; async applications should call it in a worker thread.
Importing this package neither enumerates hardware nor imports platform SDKs.
"""

from physicalai.diagnostics.api import collect_inventory, diagnose
from physicalai.diagnostics.checks import assess
from physicalai.diagnostics.models import Alias, Device, Endpoint, Finding, Inventory, Report

__all__ = ["Alias", "Device", "Endpoint", "Finding", "Inventory", "Report", "assess", "collect_inventory", "diagnose"]
