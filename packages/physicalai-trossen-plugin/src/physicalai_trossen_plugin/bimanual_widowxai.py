# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Bimanual Trossen WidowX AI robot driver."""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.config import export_config
from physicalai.robot import BimanualRobot, CompositeRobotObservation
from physicalai_trossen_plugin.widowxai import WidowXAI

if TYPE_CHECKING:
    import numpy as np


# Kept as a public compatibility name for existing callers.
BimanualWidowXAIObservation = CompositeRobotObservation


@export_config(class_path="physicalai_trossen_plugin.BimanualWidowXAI")
class BimanualWidowXAI(BimanualRobot[WidowXAI, WidowXAI]):
    """Two-arm WidowX AI driver composing a left and right :class:`WidowXAI`."""

    def __init__(self, left: WidowXAI, right: WidowXAI) -> None:
        """Initialize the bimanual driver with two matching single-arm drivers.

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
        """Shared robot role."""
        return self._left.role

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Send the first 14 action values to follower arms.

        Trossen policy outputs can include trailing values such as predicted
        velocities, which this hardware driver intentionally ignores.

        Raises:
            RuntimeError: If the bimanual robot is a leader.
            ValueError: If action does not contain both arm joint vectors.
        """
        if self.role == "leader":
            msg = "Cannot send actions to a leader robot."
            raise RuntimeError(msg)

        joint_count = len(self.joint_names)
        if action.ndim != 1 or action.shape[0] < joint_count:
            msg = f"Expected at least {joint_count} action dims, got {action.shape}"
            raise ValueError(msg)
        super().send_action(action[:joint_count], goal_time=goal_time)

    def set_external_efforts(self, efforts: np.ndarray, gain: float = 1.0) -> None:
        """Apply force feedback to each leader arm.

        Args:
            efforts: Left then right effort vectors, each with seven elements.
            gain: Scaling factor for the effort command.

        Raises:
            RuntimeError: If the bimanual robot is not a leader.
            ValueError: If efforts does not match both arm joint vectors.
        """
        if self.role != "leader":
            msg = "set_external_efforts is only available for leader robots."
            raise RuntimeError(msg)

        joint_count = len(self._left.joint_names)
        expected_shape = (2 * joint_count,)
        if efforts.shape != expected_shape:
            msg = f"Expected efforts shape {expected_shape}, got {efforts.shape}"
            raise ValueError(msg)
        self._left.set_external_efforts(efforts[:joint_count], gain)
        self._right.set_external_efforts(efforts[joint_count:], gain)
