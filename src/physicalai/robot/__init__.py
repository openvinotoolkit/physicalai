# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: PLC0415

"""Robot control interfaces.

Public API::

    from physicalai.robot import Robot, connect, verify_robot
    from physicalai.robot import SharedRobot      # requires: pip install physicalai[transport]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.robot.composite import BimanualRobot, CompositeRobot, CompositeRobotObservation
from physicalai.robot.connect import connect
from physicalai.robot.device_ids import device_id_from_serial_port
from physicalai.robot.errors import (
    RobotDeviceAlreadyOwned,
    RobotError,
    RobotNameConflict,
    RobotNotConnectedError,
    RobotProtocolMismatch,
    RobotTransportError,
)
from physicalai.robot.interface import Robot, RobotObservation
from physicalai.robot.verify import verify_robot

if TYPE_CHECKING:
    from physicalai_so101_plugin import SO101 as SO101
    from physicalai_trossen_plugin import BimanualWidowXAI as BimanualWidowXAI
    from physicalai_trossen_plugin import WidowXAI as WidowXAI

    from physicalai.robot.transport import SharedRobot as SharedRobot

__all__ = [
    "BimanualRobot",
    "BimanualWidowXAI",
    "CompositeRobot",
    "CompositeRobotObservation",
    "Robot",
    "RobotDeviceAlreadyOwned",
    "RobotError",
    "RobotNameConflict",
    "RobotNotConnectedError",
    "RobotObservation",
    "RobotProtocolMismatch",
    "RobotTransportError",
    "WidowXAI",
    "connect",
    "device_id_from_serial_port",
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
        from physicalai_so101_plugin import SO101

        return SO101

    if name == "SharedRobot":
        from physicalai.robot.transport import SharedRobot

        return SharedRobot

    if name in {"BimanualWidowXAI", "WidowXAI"}:
        from physicalai_trossen_plugin import BimanualWidowXAI, WidowXAI

        return {"BimanualWidowXAI": BimanualWidowXAI, "WidowXAI": WidowXAI}[name]

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
