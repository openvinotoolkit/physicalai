# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Small injectable OS strategy contract."""

from typing import Protocol

from physicalai.diagnostics.models import Inventory


class Collector(Protocol):
    """Collect passive evidence; implementations never connect robot drivers."""

    def collect(self) -> Inventory:
        """Collect current devices and bounded system evidence."""
