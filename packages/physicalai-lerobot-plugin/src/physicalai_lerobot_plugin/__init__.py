# ruff: noqa: PLC0415

"""LeRobot adapter plugin for PhysicalAI.

Wraps any :class:`lerobot.robots.robot.Robot` into the ``physicalai.robot.Robot``
protocol so that LeRobot-compatible robots can be driven from Physical AI Studio.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai_lerobot_plugin._urdf import get_urdf_path as get_urdf_path

if TYPE_CHECKING:
    from physicalai_lerobot_plugin.lerobot_adapter import LeRobotAdapter as LeRobotAdapter
    from physicalai_lerobot_plugin.lerobot_adapter import (
        LeRobotAdapterObservation as LeRobotAdapterObservation,
    )

__all__ = [
    "LeRobotAdapter",
    "LeRobotAdapterObservation",
    "get_urdf_path",
]


def __getattr__(name: str) -> object:
    if name == "LeRobotAdapter":
        from physicalai_lerobot_plugin.lerobot_adapter import LeRobotAdapter

        return LeRobotAdapter
    if name == "LeRobotAdapterObservation":
        from physicalai_lerobot_plugin.lerobot_adapter import LeRobotAdapterObservation

        return LeRobotAdapterObservation
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
