# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Robot control interfaces.

Public API::

    from physicalai.robot import Robot, connect, verify_robot
    from physicalai.robot import SO101            # requires: pip install physicalai[so101]
    from physicalai.robot import WidowXAI         # requires: pip install physicalai[trossen]
    from physicalai.robot import BimanualWidowXAI # requires: pip install physicalai[trossen]
    from physicalai.robot import SharedRobot      # requires: pip install physicalai[transport]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.robot.connect import connect
from physicalai.robot.errors import RobotError, RobotIdConflict, RobotNotConnectedError, RobotTransportError
from physicalai.robot.interface import Robot, RobotObservation
from physicalai.robot.verify import verify_robot

if TYPE_CHECKING:
    from physicalai.robot.so101 import SO101 as SO101
    from physicalai.robot.transport import SharedRobot as SharedRobot
    from physicalai.robot.trossen import BimanualWidowXAI as BimanualWidowXAI
    from physicalai.robot.trossen import WidowXAI as WidowXAI

__all__ = [
    "Robot",
    "RobotError",
    "RobotIdConflict",
    "RobotNotConnectedError",
    "RobotObservation",
    "RobotTransportError",
    "connect",
    "verify_robot",
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
        from physicalai.robot.so101 import SO101  # noqa: PLC0415

        return SO101

    if name == "WidowXAI":
        from physicalai.robot.trossen import WidowXAI  # noqa: PLC0415

        return WidowXAI

    if name == "BimanualWidowXAI":
        from physicalai.robot.trossen import BimanualWidowXAI  # noqa: PLC0415

        return BimanualWidowXAI

    if name == "SharedRobot":
        from physicalai.robot.transport import SharedRobot  # noqa: PLC0415

        return SharedRobot

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
