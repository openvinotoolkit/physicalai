# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run a scene's scripted demonstrator inside the simulation loop.

The autopilot steps any :class:`Demonstrator`; the scene picks which one (see
``ConveyorAutomation.attach_scene``). The conveyor scene has
``conveyor_demo.ConveyorDemonstrator`` (one SO-101), the bucket_sort scene
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

from typing import TYPE_CHECKING, Literal, Protocol, get_args

if TYPE_CHECKING:
    import numpy as np

AutopilotMode = Literal["off", "drive", "leader"]
AUTOPILOT_MODES: tuple[AutopilotMode, ...] = get_args(AutopilotMode)


class Demonstrator(Protocol):
    """A scripted controller for one scene that the autopilot steps once per control tick."""

    @property
    def phase(self) -> str:
        """Current phase, for the viewer and ``/health``."""
        ...

    @property
    def target(self) -> object | None:
        """What the demonstrator works on now: an object with a ``name``, a name, or ``None``."""
        ...

    def reset(self) -> None:
        """Forget the work in progress; the next :meth:`step` starts from the arm's current pose."""
        ...

    def step(self, data: object, dt: float, *, apply: bool = True) -> np.ndarray:
        """Advance by one control tick of *dt* seconds and return the targets, writing them if *apply*."""
        ...

    def status(self) -> dict[str, object]:
        """Return scene-specific status fields for ``/health``; the same keys every call."""
        ...

    def take_reset_request(self) -> bool:
        """Return whether the demonstrator gave up and wants the scene reset, once."""
        ...


class Autopilot:
    """Hold the demonstrator for the current scene and step it once per control tick."""

    def __init__(self) -> None:
        """Start switched off, with no scene bound."""
        self._mode: AutopilotMode = "off"
        self._demo: Demonstrator | None = None
        self._unavailable: str | None = None
        """Why the scene has no autopilot, for the viewer and ``POST /autopilot``."""

    @property
    def available(self) -> bool:
        """Whether the current scene has a demonstrator (the conveyor and bucket_sort scenes)."""
        return self._demo is not None

    @property
    def demonstrator(self) -> Demonstrator | None:
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

    def bind(self, demonstrator: Demonstrator | None, *, unavailable: str | None = None) -> None:
        """Attach to a freshly loaded scene, switched off.

        Args:
            demonstrator: The scene's demonstrator, or ``None`` when the scene has none.
            unavailable: Why the scene has no demonstrator, when it is one that usually has one.
        """
        self._mode = "off"  # a mode chosen for an earlier scene must not come back with this one
        self._demo = demonstrator
        self._unavailable = unavailable if demonstrator is None else None

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
            ``True`` once per episode the demonstrator gave up on while the autopilot runs.
        """
        return self._demo.take_reset_request() if self._demo is not None and self.mode != "off" else False

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
        if demo is not None:
            # Scene fields, e.g. bucket_sort's instruction, attempt and how the episode ended.
            extra = demo.status()
            status.update(extra if self.mode != "off" else dict.fromkeys(extra, None))
        return status
