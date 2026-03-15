# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Robot control interfaces.

Public API::

    from physicalai.robot import Robot, connect, check_robot_conformance
    from physicalai.robot import SO101  # requires: pip install physicalai[so101]
"""

from __future__ import annotations

from physicalai.robot.protocol import Robot
from physicalai.robot.testing import check_robot_conformance
from physicalai.robot.utils import connect

__all__ = [  # noqa: F822, RUF022
    "Robot",
    "SO101",
    "check_robot_conformance",
    "connect",
]


def __getattr__(name: str) -> object:
    """Lazy-load concrete robot implementations.

    This avoids pulling in hardware SDKs (e.g. ``feetech-servo-sdk``)
    at package import time.

    Args:
        name: The attribute name being looked up.

    Returns:
        The requested class (e.g. ``SO101``).

    Raises:
        AttributeError: If ``name`` does not match a known lazy-loaded symbol.
    """
    if name == "SO101":
        from physicalai.robot.so101.so101 import SO101  # noqa: PLC0415

        return SO101

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
