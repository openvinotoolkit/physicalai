# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo SO-101 simulation plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physicalai_mujoco_so101_plugin.mujoco_robot import MuJoCoSO101 as MuJoCoSO101
    from physicalai_mujoco_so101_plugin.mujoco_robot import (
        MuJoCoSO101Observation as MuJoCoSO101Observation,
    )

__all__ = [
    "MuJoCoSO101",
    "MuJoCoSO101Observation",
]


def __getattr__(name: str) -> object:
    if name == "MuJoCoSO101":
        from physicalai_mujoco_so101_plugin.mujoco_robot import MuJoCoSO101  # noqa: PLC0415

        return MuJoCoSO101
    if name == "MuJoCoSO101Observation":
        from physicalai_mujoco_so101_plugin.mujoco_robot import MuJoCoSO101Observation  # noqa: PLC0415

        return MuJoCoSO101Observation
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
