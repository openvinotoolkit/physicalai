# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime policy interfaces and implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from physicalai.robot.interface import Robot


@runtime_checkable
class RuntimePolicy(Protocol):
    """Policy-like object that can produce action chunks for ``PolicyRuntime``."""

    def predict_action_chunk(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Return an action chunk with shape ``(chunk_size, action_dim)``."""
        ...

    def reset(self) -> None:
        """Reset policy state for a new episode."""
        ...


class TeleoperatorPolicy:
    """Policy that mirrors a leader robot's joint state as follower actions."""

    def __init__(self, leader_robot: Robot) -> None:
        """Initialize with the leader robot to read each control tick."""
        self.leader_robot = leader_robot

    def predict_action_chunk(self, observation: dict[str, np.ndarray]) -> np.ndarray:  # noqa: ARG002
        """Read the leader and return one action for the follower."""
        leader_obs = self.leader_robot.get_observation()
        action = np.asarray(leader_obs.joint_positions, dtype=np.float32)
        return action[None, :]

    def reset(self) -> None:
        """No-op; teleoperation has no model state."""
