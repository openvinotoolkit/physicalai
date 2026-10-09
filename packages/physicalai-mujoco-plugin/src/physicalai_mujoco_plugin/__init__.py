# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo simulation plugin for Physical AI Runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physicalai_mujoco_plugin.robot import MuJoCoObservation as MuJoCoObservation
    from physicalai_mujoco_plugin.robot import MuJoCoRobot as MuJoCoRobot

__all__ = ["MuJoCoObservation", "MuJoCoRobot"]


def __getattr__(name: str) -> object:
    if name in __all__:
        from physicalai_mujoco_plugin import robot  # noqa: PLC0415

        return getattr(robot, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
