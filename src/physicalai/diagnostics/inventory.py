# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Public inventory entry point with an injectable platform strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.diagnostics.platforms import create_collector

if TYPE_CHECKING:
    from physicalai.diagnostics.models import Inventory
    from physicalai.diagnostics.platforms.base import Collector


def collect_inventory(collector: Collector | None = None) -> Inventory:
    """Collect hardware without importing robot plugins or constructing drivers.

    Returns:
        A snapshot including explicit collection limitations.
    """
    return (collector or create_collector()).collect()
