# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Teleop action source: forwards a leader arm's observation as the follower action."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from physicalai.config import export_config
from physicalai.runtime.action_sources.base import ActionSource

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import numpy as np

    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import Robot, RobotObservation
    from physicalai.runtime._callback_bus import _CallbackBus


@export_config(class_path="physicalai.runtime.TeleopSource")
class TeleopSource(ActionSource):
    """Action source that reads a leader arm and writes to the follower.

    The action source is the leader device, not the follower's observation or
    any inference model. Both ``robot_state``/``camera_frames`` are ignored —
    a teleop tick with no recording attached performs zero extra reads beyond
    what the runtime already does for telemetry.

    Args:
        leader: The leader robot (same ``Robot`` protocol; must support
            ``get_observation()``).
        to_action: Optional callable mapping a ``RobotObservation`` from the
            leader to an action array for the follower. Defaults to
            ``obs.joint_positions`` (identity for same-morphology leader/follower).
    """

    def __init__(  # noqa: D107
        self,
        leader: Robot,
        *,
        to_action: Callable[[RobotObservation], np.ndarray] | None = None,
    ) -> None:
        self._leader = leader
        self._to_action = to_action or (lambda obs: obs.joint_positions)
        self._leader_owned = False

    def connect(self, *, bus: _CallbackBus, session_id: str) -> None:  # noqa: ARG002
        """Connect leader if not already connected."""
        if not self._leader.is_connected():
            self._leader.connect()
            self._leader_owned = True

    def update(self, robot_state: RobotObservation, camera_frames: Mapping[str, Frame], step: int) -> np.ndarray:  # noqa: ARG002
        """Read the leader arm and return the action for the follower.

        Returns:
            Action array for the follower robot.
        """
        return self._to_action(self._leader.get_observation())

    def disconnect(self) -> None:
        """Disconnect leader if we connected it."""
        if self._leader_owned:
            with contextlib.suppress(Exception):
                self._leader.disconnect()
