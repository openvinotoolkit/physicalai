"""Motion and observation helpers for running examples via ``physicalai run``.

Implements the :class:`physicalai.runtime.ActionSource` protocol (plus a
console callback) so the ``move_joints`` / ``read_joints`` examples run through
``physicalai run --config`` instead of hand-written control loops:

- :class:`SineWaveSource` — sinusoidal joint targets (``move_joints``).
- :class:`HoldPoseSource` — echo the current observation (``read_joints``).
- :class:`JointLogger` — print observed joint positions each tick.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np
from physicalai.config import export_config

if TYPE_CHECKING:
    from collections.abc import Mapping

    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import RobotObservation
    from physicalai.runtime._callback_bus import _CallbackBus
    from physicalai.runtime.events import TickEvent


@export_config(class_path="physicalai_rebot_b601_plugin.motion.SineWaveSource")
class SineWaveSource:
    """Generate sinusoidal joint targets, one sine per joint.

    Produces ``amplitude * sin(2*pi*frequency*t + phase)`` for each joint,
    with ``t`` measured from :meth:`connect`. The joint count is inferred from
    the first observation, so the same source works for any robot.

    Args:
        amplitude: Sine amplitude applied to every joint (unless
            ``joint_amplitudes`` is given).
        frequency: Sine frequency in Hz.
        joint_amplitudes: Optional per-joint amplitude list, overriding
            ``amplitude``. Lets some joints stay fixed (amplitude 0).
        phase_offsets: Optional per-joint phase offsets in radians. Defaults
            to ``i * 2*pi / num_joints``.

    Raises:
        ValueError: If ``frequency`` is not positive.
    """

    def __init__(
        self,
        *,
        amplitude: float = 10.0,
        frequency: float = 0.25,
        joint_amplitudes: list[float] | None = None,
        phase_offsets: list[float] | None = None,
    ) -> None:
        """Initialize the sine source with the given motion parameters.

        Raises:
            ValueError: If ``frequency`` is not positive.
        """
        if frequency <= 0:
            msg = "frequency must be positive"
            raise ValueError(msg)
        self._amplitude = amplitude
        self._frequency = frequency
        self._joint_amplitudes = list(joint_amplitudes) if joint_amplitudes else None
        self._phase_offsets = list(phase_offsets) if phase_offsets else None
        self._start_time: float | None = None

    def connect(self, *, bus: _CallbackBus, session_id: str) -> None:  # noqa: ARG002
        """Reset the sine timer; part of the ``ActionSource`` protocol."""
        self._start_time = time.monotonic()

    def update(
        self,
        robot_state: RobotObservation,
        camera_frames: Mapping[str, Frame],  # noqa: ARG002
        step: int,  # noqa: ARG002
    ) -> np.ndarray:
        """Return the sinusoidal action for this tick.

        Args:
            robot_state: Used to infer the number of joints.
            camera_frames: Unused; part of the ``ActionSource`` protocol.
            step: Unused; part of the ``ActionSource`` protocol.

        Returns:
            Action vector of sine targets, one per joint.

        Raises:
            ValueError: If the amplitude or phase lists do not match the joint
                count.
        """
        num_joints = len(robot_state.joint_positions)
        amplitudes = self._joint_amplitudes or [self._amplitude] * num_joints
        phases = self._phase_offsets or [i * math.tau / num_joints for i in range(num_joints)]
        if len(amplitudes) != num_joints or len(phases) != num_joints:
            msg = f"joint_amplitudes/phase_offsets must match the robot's joint count ({num_joints})"
            raise ValueError(msg)
        if self._start_time is None:
            self._start_time = time.monotonic()
        t = time.monotonic() - self._start_time
        return np.array(
            [amplitudes[i] * math.sin(math.tau * self._frequency * t + phases[i]) for i in range(num_joints)],
            dtype=np.float32,
        )

    def disconnect(self) -> None:
        """Release the timer; part of the ``ActionSource`` protocol."""
        self._start_time = None


@export_config(class_path="physicalai_rebot_b601_plugin.motion.HoldPoseSource")
class HoldPoseSource:
    """Echo the current observation back as the action (hold).

    The runtime always sends an action; this source holds the robot at its
    current observed pose.
    """

    def __init__(self) -> None:
        """Initialize the hold source (no configuration)."""

    def connect(self, *, bus: _CallbackBus, session_id: str) -> None:
        """No resources to set up; part of the ``ActionSource`` protocol."""

    def update(  # noqa: PLR6301
        self,
        robot_state: RobotObservation,
        camera_frames: Mapping[str, Frame],  # noqa: ARG002
        step: int,  # noqa: ARG002
    ) -> np.ndarray:
        """Return the current joint positions unchanged.

        Args:
            robot_state: The observation whose positions are echoed.
            camera_frames: Unused; part of the ``ActionSource`` protocol.
            step: Unused; part of the ``ActionSource`` protocol.

        Returns:
            The observation's ``joint_positions`` as the hold action.
        """
        return np.asarray(robot_state.joint_positions, dtype=np.float32)

    def disconnect(self) -> None:
        """Nothing to tear down; part of the ``ActionSource`` protocol."""


@export_config(class_path="physicalai_rebot_b601_plugin.motion.JointLogger")
class JointLogger:
    """Print observed joint positions periodically, like ``read_joints``.

    Args:
        throttle_steps: Print every Nth tick (default ``1`` = every tick).
    """

    def __init__(self, throttle_steps: int = 1) -> None:  # noqa: D107
        if throttle_steps <= 0:
            msg = "throttle_steps must be positive"
            raise ValueError(msg)
        self._throttle_steps = throttle_steps

    def on_tick(self, event: TickEvent) -> None:
        """Print the joint positions observed on this tick."""
        if event.step % self._throttle_steps != 0:
            return
        values = "  ".join(f"{v:8.2f}" for v in event.robot_state.joint_positions)
        print(f"[{event.robot_state.timestamp:13.3f}]  {values}")  # noqa: T201
