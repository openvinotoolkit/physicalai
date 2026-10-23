# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run a scene's scripted demonstrator inside the simulation loop.

The conveyor scene has ``conveyor_demo.ConveyorDemonstrator`` (one SO-101), the bucket_sort scene
``bucket_sort_demo.BucketSortDemonstrator`` (one Franka FR3).

Two ways to use it:

- ``drive``: the demonstrator writes the arm's actuator targets itself, and
  actions from clients are ignored. For demos and quick checks.
- ``leader``: the demonstrator only publishes its targets, as a virtual leader
  arm (``GET /leader``). A teleoperation client, such as Physical AI Studio
  with the *MuJoCo SO-101 Virtual Leader*, reads them and sends them back as
  follower actions, so the recorded actions are the demonstrator's commands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

import numpy as np

if TYPE_CHECKING:
    from physicalai_mujoco_plugin.bucket_sort import BucketSortTask
    from physicalai_mujoco_plugin.bucket_sort_demo import BucketSortDemonstrator
    from physicalai_mujoco_plugin.conveyor import ConveyorSort
    from physicalai_mujoco_plugin.conveyor_demo import ConveyorDemonstrator

AutopilotMode = Literal["off", "drive", "leader"]
AUTOPILOT_MODES: tuple[AutopilotMode, ...] = get_args(AutopilotMode)


class Autopilot:
    """Hold the demonstrator for the current scene and step it once per control tick."""

    def __init__(self) -> None:
        """Start switched off, with no scene bound."""
        self._mode: AutopilotMode = "off"
        self._demo: ConveyorDemonstrator | BucketSortDemonstrator | None = None
        self._unavailable: str | None = None
        """Why a conveyor scene has no autopilot, for the viewer and ``POST /autopilot``."""

    @property
    def available(self) -> bool:
        """Whether the current scene has a demonstrator (the conveyor and bucket_sort scenes)."""
        return self._demo is not None

    @property
    def demonstrator(self) -> ConveyorDemonstrator | BucketSortDemonstrator | None:
        """The current scene's demonstrator, if it has one."""
        return self._demo

    @property
    def mode(self) -> AutopilotMode:
        """Current mode; ``off`` whenever the scene has no demonstrator."""
        return self._mode if self._demo is not None else "off"

    @property
    def drives_arm(self) -> bool:
        """Whether the demonstrator writes the actuators, so client actions must be ignored."""
        return self.mode == "drive"

    def bind(
        self,
        model: object | None,
        conveyor: ConveyorSort | None,
        *,
        robots: int = 1,
        bucket_task: BucketSortTask | None = None,
        seed: int | None = None,
    ) -> None:
        """Attach to a freshly loaded scene, switched off.

        Only the conveyor and bucket_sort scenes have an autopilot, and a conveyor with several arms
        has none: its demonstrator drives one SO-101.

        Args:
            model: The loaded model, or ``None``.
            conveyor: The scene's conveyor controller, if it has a belt.
            robots: Number of arms.
            bucket_task: The bucket_sort scene's task (its reset callback), if that is the scene.
            seed: Seeds the bucket_sort demonstrator's draws; ``None`` draws a fresh seed.
        """
        self._mode = "off"  # a mode chosen for an earlier scene must not come back with this one
        self._unavailable = None
        if model is not None and conveyor is None and bucket_task is not None:
            from physicalai_mujoco_plugin.bucket_sort_demo import BucketSortDemonstrator  # noqa: PLC0415

            self._demo = BucketSortDemonstrator(model, bucket_task, rng=np.random.default_rng(seed))
            return
        if model is None or conveyor is None:
            self._demo = None
            return
        if robots != 1:
            self._demo = None
            self._unavailable = f"The conveyor autopilot drives one SO-101 arm; this simulation has {robots} arms."
            return
        from physicalai_mujoco_plugin.conveyor_demo import ConveyorDemonstrator  # noqa: PLC0415

        self._demo = ConveyorDemonstrator(model, conveyor)

    def set_mode(self, mode: AutopilotMode) -> None:
        """Switch mode; the next cycle starts from the arm's current pose.

        Raises:
            ValueError: If `mode` is not one of ``AUTOPILOT_MODES``.
        """
        if mode not in AUTOPILOT_MODES:
            msg = f"Unknown autopilot mode {mode!r}; expected one of {AUTOPILOT_MODES}"
            raise ValueError(msg)
        self._mode = mode
        self.reset()

    def reset(self) -> None:
        """Forget the current pick; call after anything teleports the arm or the items."""
        if self._demo is not None:
            self._demo.reset()

    def take_reset_request(self) -> bool:
        """Return whether the demonstrator gave up on an episode and wants the scene reset, once.

        Returns:
            ``True`` once per failed bucket_sort episode while the autopilot runs.
        """
        demo = self._demo
        take = getattr(demo, "take_reset_request", None)
        return bool(take()) if take is not None and self.mode != "off" else False

    def step(self, data: object, dt: float) -> np.ndarray | None:
        """Advance the demonstrator by one tick.

        Returns:
            The demonstrator's targets in model units, one per channel (joints in radians; the
            Franka Hand's opening in metres), or ``None`` when the autopilot is off.
        """
        if self._demo is None or self._mode == "off":
            return None
        return self._demo.step(data, dt, apply=self._mode == "drive")

    def status(self) -> dict[str, object]:
        """Return a JSON-friendly snapshot for ``/health`` and the viewer."""
        demo = self._demo
        target = demo.target if demo is not None and self.mode != "off" else None
        status: dict[str, object] = {
            "available": demo is not None,
            "unavailable_reason": self._unavailable,
            "mode": self.mode,
            "phase": demo.phase if demo is not None and self.mode != "off" else None,
            "target": getattr(target, "name", target),
        }
        episode = getattr(demo, "status", None)
        if episode is not None:
            # bucket_sort: the instruction, the attempt and how the episode ended.
            status.update(episode() if self.mode != "off" else dict.fromkeys(episode(), None))
        return status
