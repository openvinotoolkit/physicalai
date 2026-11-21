# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Kinematic replay of recorded joint positions (``POST /replay``).

A replay runs on the owner's control tick: each tick shows the frame due at the replay's own
clock, which advances by one control period per tick, like the simulation clock it replaces.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

_FRAME_EPSILON = 1e-9
"""Absorbs rounding in ``ticks · control_dt · fps``, so a tick that lands on a frame time shows that frame."""


class Replay:
    """A trajectory being played back, and the physics state to restore when it ends."""

    def __init__(self, joint_positions: np.ndarray, fps: float, saved_state: np.ndarray) -> None:
        """Start a replay at its first frame.

        Args:
            joint_positions: ``(frames, joints)`` positions in the robot's public units.
            fps: Recorded frame rate.
            saved_state: The ``mjSTATE_INTEGRATION`` state from before the replay.
        """
        self.joint_positions = joint_positions
        self.fps = float(fps)
        self.saved_state = saved_state
        self._shown = (0, False)
        """``(frame, finished)``, replaced as one value so other threads never read a torn pair."""
        self._ticks = 0

    @property
    def frame(self) -> int:
        """Index of the frame shown last."""
        return self._shown[0]

    @property
    def finished(self) -> bool:
        """Whether the replay clock passed the last frame's time, so the last frame is held."""
        return self._shown[1]

    def next_frame(self, control_dt: float) -> np.ndarray:
        """Return the frame to show this tick, then advance the replay clock by one tick.

        After the last frame's time the replay holds the last frame and reports ``finished``.

        Args:
            control_dt: Simulated seconds per control tick.

        Returns:
            One row of ``joint_positions``.
        """
        due = math.floor(self._ticks * control_dt * self.fps + _FRAME_EPSILON)
        self._ticks += 1
        count = len(self.joint_positions)
        frame = min(due, count - 1)
        self._shown = (frame, due >= count)
        return self.joint_positions[frame]

    def status(self) -> dict[str, object]:
        """Return the replay status served at ``GET /health``.

        Returns:
            ``active``, ``finished``, the shown ``frame``, the frame count and ``fps``.
        """
        frame, finished = self._shown
        return {
            "active": True,
            "finished": finished,
            "frame": frame,
            "frames": len(self.joint_positions),
            "fps": self.fps,
        }


__all__ = ["Replay"]
