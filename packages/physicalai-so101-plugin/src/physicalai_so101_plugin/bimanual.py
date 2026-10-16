# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Bimanual SO-101 robot arm driver."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from physicalai.config import export_config
from physicalai.robot import BimanualRobot, CompositeRobotObservation
from physicalai_so101_plugin.bimanual_constants import BIMANUAL_SO101_JOINT_ORDER, NUM_BIMANUAL_JOINTS
from physicalai_so101_plugin.so101 import SO101

if TYPE_CHECKING:
    import numpy as np


# Kept as a public compatibility name for existing callers.
BimanualSO101Observation = CompositeRobotObservation


@export_config(class_path="physicalai_so101_plugin.BimanualSO101")
class BimanualSO101(BimanualRobot[SO101, SO101]):
    """Two-arm SO-101 driver composing a left and right :class:`SO101`.

    Both arms must have the same role and concrete driver type. Joint vectors
    are ordered left arm then right arm.
    """

    JOINT_ORDER: ClassVar[tuple[str, ...]] = BIMANUAL_SO101_JOINT_ORDER
    NUM_JOINTS: int = NUM_BIMANUAL_JOINTS

    def __init__(self, left: SO101, right: SO101) -> None:
        """Initialize the bimanual driver with left and right SO101 arms.

        Raises:
            ValueError: If arm roles or concrete driver types differ.
        """
        if left.role != right.role:
            msg = f"Both arms must have the same role; got left={left.role!r}, right={right.role!r}."
            raise ValueError(msg)
        if type(left) is not type(right):
            msg = (
                f"Both arms must be the same driver type; got left={type(left).__name__}, right={type(right).__name__}."
            )
            raise ValueError(msg)
        super().__init__(left=left, right=right)

    @property
    def role(self) -> str:
        """Shared role of both arms."""
        return self._left.role

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Send an exact 12-dimensional action to the two follower arms.

        Raises:
            RuntimeError: If the bimanual robot is a leader.
        """
        if self.role == "leader":
            msg = "Cannot send actions to a leader robot."
            raise RuntimeError(msg)
        super().send_action(action, goal_time=goal_time)
