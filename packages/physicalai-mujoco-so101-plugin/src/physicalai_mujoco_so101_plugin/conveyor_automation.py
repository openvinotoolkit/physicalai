# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Conveyor-scene automation that the simulation loop runs once per control tick.

One object owns everything the conveyor scene adds on top of plain
simulation, so ``MuJoCoSO101`` only forwards to it:

- the conveyor controller for the current scene, and the belt speed remembered
  across scene switches;
- the autopilot (the scripted demonstrator, see ``autopilot.py``);
- the virtual leader pose served on ``GET /leader``;
- automatic Studio episodes (``studio_recorder.py``), including holding the
  belt feed between episodes.

All methods except `leader_snapshot` and `status` run on the simulation thread.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from physicalai_mujoco_so101_plugin.autopilot import Autopilot
from physicalai_mujoco_so101_plugin.conveyor import DEFAULT_BELT_SPEED, ConveyorSort
from physicalai_mujoco_so101_plugin.studio_recorder import (
    DEFAULT_STUDIO_URL,
    AutoRecorder,
    RecordingOptions,
    StudioLink,
    validate_studio_url,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from physicalai_mujoco_so101_plugin.autopilot import AutopilotMode


class ConveyorAutomation:
    """Belt, autopilot, virtual leader and Studio recording for the current scene."""

    def __init__(
        self,
        studio_url: str = DEFAULT_STUDIO_URL,
        *,
        joint_names: tuple[str, ...],
        unit: str,
        to_units: Callable[[np.ndarray], np.ndarray],
        owner_name: Callable[[], str],
    ) -> None:
        """Start with no scene attached.

        Args:
            studio_url: Studio backend that automatic recording attaches to.
            joint_names: The robot's joint order; leader poses use it.
            unit: The robot's joint unit, reported with each leader pose.
            to_units: Converts joint radians to the robot's unit.
            owner_name: Returns the simulation's transport name, used to find its Studio session.
        """
        self.studio_url = validate_studio_url(studio_url)
        self.autopilot = Autopilot()
        self.recorder = AutoRecorder(self._open_link)
        self.belt_speed = DEFAULT_BELT_SPEED
        """Belt speed in m/s, kept across scene switches like the auto-reset settings."""
        self._joint_names = tuple(joint_names)
        self._unit = unit
        self._to_units = to_units
        self._owner_name = owner_name
        self._conveyor: ConveyorSort | None = None
        self._holds_feed = False
        self._lock = threading.Lock()
        self._leader_seq = 0
        self._leader: dict[str, object] = {"seq": 0, "joint_names": list(self._joint_names)}

    @property
    def conveyor(self) -> ConveyorSort | None:
        """The current scene's conveyor controller, if it has a belt."""
        return self._conveyor

    @property
    def drives_arm(self) -> bool:
        """Whether the autopilot writes the actuators, so client actions must be ignored."""
        return self.autopilot.drives_arm

    # ------------------------------------------------------------------
    # Scene lifecycle
    # ------------------------------------------------------------------

    def attach_scene(self, model: object | None, *, rng: np.random.Generator, active: bool) -> ConveyorSort | None:
        """Build the conveyor controller for a freshly loaded scene.

        Returns:
            The controller, or ``None`` for scenes without a belt.
        """
        conveyor = None
        if model is not None:
            conveyor = ConveyorSort.maybe_create(model, rng=rng, belt_speed=self.belt_speed, active=active)
        self._conveyor = conveyor
        self._holds_feed = False  # a new conveyor starts unheld
        self.autopilot.bind(model, conveyor)
        if conveyor is None and self.recorder.phase != "off":
            self.recorder.disable("Automatic recording needs the conveyor scene.")
        elif conveyor is not None:
            self.recorder.restart_episode("The scene was reloaded.")
        return conveyor

    def reset(self) -> None:
        """Forget the autopilot's current pick and restart a Studio episode; call after a scene reset."""
        self.autopilot.reset()
        self.recorder.restart_episode("The scene was reset.")

    def close(self) -> None:
        """Detach from Studio; call when the simulation disconnects."""
        self.recorder.disable()

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def set_belt_speed(self, speed: float) -> bool:
        """Remember `speed` (m/s) and apply it to the current belt.

        Returns:
            Whether the current scene has a belt.
        """
        self.belt_speed = float(speed)
        if self._conveyor is None:
            return False
        self._conveyor.set_belt_speed(self.belt_speed)
        return True

    def set_autopilot(self, mode: AutopilotMode) -> bool:
        """Switch the autopilot mode.

        Returns:
            Whether the current scene has an autopilot.
        """
        if not self.autopilot.available:
            logger.warning("The current scene has no autopilot")
            return False
        self.autopilot.set_mode(mode)
        logger.info("Autopilot {}", mode)
        return True

    def set_studio_recording(self, options: RecordingOptions | None) -> bool:
        """Switch automatic Studio recording on with `options`, or off with ``None``.

        Returns:
            ``False`` if recording was requested outside the conveyor scene.
        """
        if options is None:
            self.recorder.disable("Switched off.")
            return True
        if self._conveyor is None:
            logger.warning("Automatic Studio recording needs the conveyor scene")
            return False
        self.recorder.enable(options)
        return True

    # ------------------------------------------------------------------
    # Per control tick
    # ------------------------------------------------------------------

    def tick(self, model: object, data: object, dt: float, arm_targets: np.ndarray) -> None:
        """Run the Studio episode cycle, step the autopilot and publish the leader pose.

        Args:
            model: The loaded MuJoCo model.
            data: Its data, before this tick's physics steps.
            dt: Control period in seconds.
            arm_targets: The arm's current actuator targets in radians, published as the
                leader pose while the autopilot is off.
        """
        conveyor = self._conveyor
        if conveyor is not None:
            status = conveyor.status()
            directive = self.recorder.update(int(status["episode_count"]), status.get("last_episode"))  # pyrefly: ignore [bad-argument-type]
            if directive.hold_feed != self._holds_feed:  # only on change: leave other holds alone
                conveyor.set_feed_hold(directive.hold_feed)
                self._holds_feed = directive.hold_feed
            if directive.clear_belt:
                conveyor.reset_items(model, data)
                self.autopilot.reset()
        targets = self.autopilot.step(data, dt)
        self._publish_leader(arm_targets if targets is None else targets)

    def _publish_leader(self, targets: np.ndarray) -> None:
        """Publish the virtual leader pose in the robot's unit.

        While the autopilot is off this is the arm's actuator targets, not its
        measured joints: echoing measured joints would let a teleoperated
        follower sag under gravity.
        """
        positions = self._to_units(np.asarray(targets, dtype=np.float64))
        self._leader_seq += 1
        state: dict[str, object] = {
            "seq": self._leader_seq,
            "mode": self.autopilot.mode,
            "unit": self._unit,
            "joint_names": list(self._joint_names),
            "joint_positions": [float(value) for value in positions],
        }
        with self._lock:
            self._leader = state

    # ------------------------------------------------------------------
    # Readers (any thread)
    # ------------------------------------------------------------------

    def leader_snapshot(self) -> dict[str, object]:
        """Latest virtual leader pose for ``GET /leader``.

        Returns:
            ``seq``, ``mode``, ``unit``, ``joint_names`` and ``joint_positions``.
        """
        with self._lock:
            return dict(self._leader)

    def status(self) -> dict[str, dict[str, object]]:
        """Return the ``autopilot`` and ``studio`` sections of ``/health`` and the viewer."""
        return {"autopilot": self.autopilot.status(), "studio": self.recorder.status()}

    def _open_link(self) -> StudioLink:
        return StudioLink(self.studio_url, self._owner_name())
