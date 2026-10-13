# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ruff: file-ignore[import-outside-top-level]

"""Third-party Franka Research 3 robot arm plugin for PhysicalAI.

Provides robot drivers compatible with the ``physicalai.robot.Robot`` protocol,
controlled over the Franka Control Interface (FCI) through the official
``pylibfranka`` Python bindings:

* :class:`FrankaFR3` -- 7-DOF FR3 follower with an optional Franka Hand gripper.
* :class:`BimanualFrankaFR3` -- left/right composition of compatible FR3 arms.
* :class:`GelloLeader` / :class:`BimanualGelloLeader` -- passive USB leaders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai_franka_plugin._urdf import get_urdf_path as get_urdf_path

if TYPE_CHECKING:
    from physicalai_franka_plugin.bimanual import BimanualFrankaFR3 as BimanualFrankaFR3
    from physicalai_franka_plugin.bimanual import BimanualFrankaFR3Observation as BimanualFrankaFR3Observation
    from physicalai_franka_plugin.gello import BimanualGelloLeader as BimanualGelloLeader
    from physicalai_franka_plugin.gello import GelloCalibration as GelloCalibration
    from physicalai_franka_plugin.gello import GelloLeader as GelloLeader
    from physicalai_franka_plugin.gello import GelloObservation as GelloObservation
    from physicalai_franka_plugin.robot import FrankaFR3 as FrankaFR3
    from physicalai_franka_plugin.robot import FrankaFRObservation as FrankaFRObservation

__all__ = [
    "BimanualFrankaFR3",
    "BimanualFrankaFR3Observation",
    "BimanualGelloLeader",
    "FrankaFR3",
    "FrankaFRObservation",
    "GelloCalibration",
    "GelloLeader",
    "GelloObservation",
    "get_urdf_path",
]


def __getattr__(name: str) -> object:
    if name in {"BimanualGelloLeader", "GelloCalibration", "GelloLeader", "GelloObservation"}:
        from physicalai_franka_plugin.gello import BimanualGelloLeader, GelloCalibration, GelloLeader, GelloObservation

        return {
            "BimanualGelloLeader": BimanualGelloLeader,
            "GelloCalibration": GelloCalibration,
            "GelloLeader": GelloLeader,
            "GelloObservation": GelloObservation,
        }[name]
    if name in {"BimanualFrankaFR3", "BimanualFrankaFR3Observation"}:
        from physicalai_franka_plugin.bimanual import BimanualFrankaFR3, BimanualFrankaFR3Observation

        return {
            "BimanualFrankaFR3": BimanualFrankaFR3,
            "BimanualFrankaFR3Observation": BimanualFrankaFR3Observation,
        }[name]
    if name == "FrankaFR3":
        from physicalai_franka_plugin.robot import FrankaFR3

        return FrankaFR3
    if name == "FrankaFRObservation":
        from physicalai_franka_plugin.robot import FrankaFRObservation

        return FrankaFRObservation
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
