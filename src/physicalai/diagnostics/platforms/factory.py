# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lazy platform selection; importing diagnostics never imports OS-only APIs."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physicalai.diagnostics.platforms.base import Collector


def create_collector(platform: str | None = None) -> Collector:
    """Select a collector without loading other platforms' dependencies.

    Returns:
        The platform-specific passive collector.

    Raises:
        ValueError: If the platform is not supported.
    """
    selected = platform or sys.platform
    if selected == "linux":
        from physicalai.diagnostics.platforms.linux import LinuxCollector  # noqa: PLC0415

        return LinuxCollector()
    if selected in {"darwin", "win32"}:
        from physicalai.diagnostics.platforms.portable import PortableCollector  # noqa: PLC0415

        return PortableCollector(selected)
    msg = f"Hardware diagnostics do not support platform {selected!r}."
    raise ValueError(msg)
