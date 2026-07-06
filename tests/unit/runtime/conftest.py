# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from physicalai.capture.frame import Frame
    from physicalai.runtime._callback_bus import _CallbackBus


@dataclass
class FakeRobotObservation:
    """Test double satisfying the RobotObservation protocol."""

    joint_positions: np.ndarray
    timestamp: float = 0.0
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict | None = None

    @property
    def state(self) -> np.ndarray:
        return self.joint_positions


@dataclass
class FakeActionSource:
    """Minimal test double satisfying the ActionSource protocol (3 methods).

    Records every ``update()`` call's ``(robot_state, camera_frames, step)``
    args for assertions, and always returns ``next_action`` (or an echo of the
    last-seen robot state's joint positions if unset).
    """

    next_action: np.ndarray | None = None
    connected: bool = False
    disconnected: bool = False
    bus: _CallbackBus | None = field(default=None, repr=False)
    session_id: str = ""
    calls: list[tuple[Any, Mapping[str, Frame], int]] = field(default_factory=list)

    def connect(self, *, bus: _CallbackBus, session_id: str) -> None:
        self.connected = True
        self.bus = bus
        self.session_id = session_id

    def update(self, robot_state: Any, camera_frames: Mapping[str, Frame], step: int) -> np.ndarray:
        self.calls.append((robot_state, camera_frames, step))
        if self.next_action is not None:
            return self.next_action
        return np.asarray(robot_state.joint_positions)

    def disconnect(self) -> None:
        self.disconnected = True

