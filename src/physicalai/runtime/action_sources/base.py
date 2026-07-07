# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action source protocol — the seam RobotRuntime plugs into."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np

    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import RobotObservation
    from physicalai.runtime._callback_bus import _CallbackBus


class ActionSource(Protocol):
    """The minimum a developer must implement to plug an action source into RobotRuntime.

    Three required methods, nothing optional — no capability protocols, no
    ``isinstance`` anywhere in the runtime.
    """

    def connect(self, *, bus: _CallbackBus, session_id: str) -> None:
        """Set up resources (spawn threads, connect a leader device, etc.).

        Called fresh every ``run()``, which is exactly when the runtime
        generates a new ``session_id`` — construction-time injection would
        miss that.
        """
        ...

    def update(self, robot_state: RobotObservation, camera_frames: Mapping[str, Frame], step: int) -> np.ndarray:
        """Return the action to send this tick.

        Always returns a sendable action — no ``None`` sentinel. What to do
        when there is nothing new to decide (repeat the last action, go to a
        safe pose, whatever) is entirely this action source's own call, made
        internally. If it truly cannot produce anything, it raises.

        Returns:
            Action vector to send to the robot this tick.
        """
        ...

    def disconnect(self) -> None:
        """Tear down only (stop threads, release a leader device).

        Returns nothing — any queued-but-unsent actions are discarded, not
        flushed. The action source never receives a robot reference.
        """
        ...
