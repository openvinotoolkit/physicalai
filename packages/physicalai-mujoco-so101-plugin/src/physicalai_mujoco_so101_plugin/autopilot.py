# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run the scripted conveyor demonstrator inside the simulation loop.

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

if TYPE_CHECKING:
    import numpy as np

    from physicalai_mujoco_so101_plugin.conveyor import ConveyorSort
    from physicalai_mujoco_so101_plugin.conveyor_demo import ConveyorDemonstrator

AutopilotMode = Literal["off", "drive", "leader"]
AUTOPILOT_MODES: tuple[AutopilotMode, ...] = get_args(AutopilotMode)


class Autopilot:
    """Hold the demonstrator for the current scene and step it once per control tick."""

    def __init__(self) -> None:
        """Start switched off, with no scene bound."""
        self._mode: AutopilotMode = "off"
        self._demo: ConveyorDemonstrator | None = None

    @property
    def available(self) -> bool:
        """Whether the current scene has a demonstrator (conveyor scenes only)."""
        return self._demo is not None

    @property
    def mode(self) -> AutopilotMode:
        """Current mode; ``off`` whenever the scene has no demonstrator."""
        return self._mode if self._demo is not None else "off"

    @property
    def drives_arm(self) -> bool:
        """Whether the demonstrator writes the actuators, so client actions must be ignored."""
        return self.mode == "drive"

    def bind(self, model: object | None, conveyor: ConveyorSort | None) -> None:
        """Attach to a freshly loaded scene, switched off; scenes without a conveyor have no autopilot."""
        self._mode = "off"  # a mode chosen for an earlier scene must not come back with this one
        if model is None or conveyor is None:
            self._demo = None
            return
        from physicalai_mujoco_so101_plugin.conveyor_demo import ConveyorDemonstrator  # noqa: PLC0415

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

    def step(self, data: object, dt: float) -> np.ndarray | None:
        """Advance the demonstrator by one tick.

        Returns:
            The demonstrator's joint targets in radians (``SO101_JOINT_ORDER``),
            or ``None`` when the autopilot is off.
        """
        if self._demo is None or self._mode == "off":
            return None
        return self._demo.step(data, dt, apply=self._mode == "drive")

    def status(self) -> dict[str, object]:
        """Return a JSON-friendly snapshot for ``/health`` and the viewer."""
        demo = self._demo
        target = demo.target if demo is not None and self.mode != "off" else None
        return {
            "available": demo is not None,
            "mode": self.mode,
            "phase": demo.phase if demo is not None and self.mode != "off" else None,
            "target": target.name if target is not None else None,
        }
