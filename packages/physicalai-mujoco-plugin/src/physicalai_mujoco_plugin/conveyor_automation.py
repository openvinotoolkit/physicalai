# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Conveyor-scene automation that the simulation loop runs once per control tick.

One object owns everything the conveyor scene adds on top of plain
simulation, so ``MuJoCoRobot`` only forwards to it:

- the conveyor controller for the current scene, and the belt speed remembered
  across scene switches;
- the autopilot (the scripted demonstrator, see ``autopilot.py``);
- the virtual leader pose served on ``GET /leader``;
- automatic Studio episodes (``studio_recorder.py``), including holding the
  belt feed between episodes; in the bucket_sort scene, holding the autopilot
  between episodes and resetting the scene before each one.

All methods except `leader_snapshot` and `status` run on the simulation thread.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from physicalai_mujoco_plugin.autopilot import Autopilot
from physicalai_mujoco_plugin.conveyor import DEFAULT_BELT_SPEED, ConveyorSort
from physicalai_mujoco_plugin.studio_recorder import (
    DEFAULT_STUDIO_URL,
    AutoRecorder,
    BucketSortProgress,
    BucketSortRecorder,
    RecordingOptions,
    StudioLink,
    validate_studio_url,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from physicalai_mujoco_plugin.autopilot import AutopilotMode, Demonstrator
    from physicalai_mujoco_plugin.bucket_sort import BucketSortTask


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
        """Records conveyor episodes."""
        self.bucket_recorder = BucketSortRecorder(self._open_link)
        """Records bucket_sort episodes that the autopilot drives."""
        self.belt_speed = DEFAULT_BELT_SPEED
        """Belt speed in m/s, kept across scene switches like the auto-reset settings."""
        self._joint_names = tuple(joint_names)
        self._unit = unit
        self._to_units = to_units
        self._owner_name = owner_name
        self._conveyor: ConveyorSort | None = None
        self._bucket_task: BucketSortTask | None = None
        self._recorder_reset = False
        self._holds_feed = False
        self._lock = threading.Lock()
        self._leader_seq = 0
        self._leader: dict[str, object] = {"seq": 0, "joint_names": list(self._joint_names)}

    @property
    def conveyor(self) -> ConveyorSort | None:
        """The current scene's conveyor controller, if it has a belt."""
        return self._conveyor

    @property
    def studio_recorder(self) -> AutoRecorder | BucketSortRecorder:
        """The recorder for the current scene: bucket_sort's, else the conveyor's."""
        return self.bucket_recorder if self._bucket_task is not None else self.recorder

    @property
    def recording_active(self) -> bool:
        """Whether automatic Studio recording runs (so the overview camera must stay put)."""
        return self.recorder.active or self.bucket_recorder.active

    @property
    def drives_arm(self) -> bool:
        """Whether the autopilot writes the actuators, so client actions must be ignored."""
        return self.autopilot.drives_arm

    # ------------------------------------------------------------------
    # Scene lifecycle
    # ------------------------------------------------------------------

    def attach_scene(
        self,
        model: object | None,
        *,
        rng: np.random.Generator,
        active: bool,
        robots: int = 1,
        robot_roots: tuple[int, ...] | None = None,
        bucket_task: BucketSortTask | None = None,
        seed: int | None = None,
    ) -> ConveyorSort | None:
        """Build the conveyor controller for a freshly loaded scene.

        Args:
            model: The loaded model, or ``None`` when nothing is loaded.
            rng: Generator for the belt feed.
            active: Whether the belt and its episodes start running.
            robots: Number of arms in the scene; the autopilot drives a single one.
            robot_roots: Root body of every arm, for the held-item test; ``None`` takes a single
                unprefixed SO-101.
            bucket_task: The bucket_sort scene's task, which gets that scene's demonstrator.
            seed: Seeds the bucket_sort demonstrator; ``None`` draws a fresh seed.

        Returns:
            The controller, or ``None`` for scenes without a belt.
        """
        conveyor = None
        if model is not None:
            conveyor = ConveyorSort.maybe_create(
                model, rng=rng, belt_speed=self.belt_speed, active=active, robot_roots=robot_roots
            )
        self._conveyor = conveyor
        self._holds_feed = False  # a new conveyor starts unheld
        self._recorder_reset = False
        demonstrator, unavailable = _scene_demonstrator(
            model, conveyor, robots=robots, bucket_task=bucket_task, seed=seed
        )
        self.autopilot.bind(demonstrator, unavailable=unavailable)
        self._bucket_task = bucket_task if model is not None and conveyor is None else None
        if conveyor is None and self.recorder.phase != "off":
            self.recorder.disable("Automatic recording needs the conveyor scene.")
        elif conveyor is not None:
            self.recorder.restart_episode("The scene was reloaded.")
        if self._bucket_task is None and self.bucket_recorder.phase != "off":
            self.bucket_recorder.disable("Automatic recording of bucket_sort episodes needs the bucket_sort scene.")
        # A reloaded bucket_sort scene is a new task object: the recorder discards the episode itself.
        return conveyor

    def take_reset_request(self) -> bool:
        """Return whether the scene should be reset now, once.

        Returns:
            ``True`` once per failed bucket_sort episode the autopilot gave up on, and once before
            each bucket_sort episode Studio records.
        """
        requested = self.autopilot.take_reset_request()
        recorder, self._recorder_reset = self._recorder_reset, False
        return requested or recorder

    def reset(self) -> None:
        """Forget the autopilot's current pick and restart a Studio episode; call after a scene reset."""
        self.autopilot.reset()
        self.recorder.restart_episode("The scene was reset.")

    def close(self) -> None:
        """Detach from Studio; call when the simulation disconnects."""
        self.recorder.disable()
        self.bucket_recorder.disable()

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

        In the bucket_sort scene the task of `options` is ignored: each episode's instruction is its task.

        Returns:
            ``False`` if recording was requested outside the conveyor and bucket_sort scenes.
        """
        if options is None:
            self.studio_recorder.disable("Switched off.")
            return True
        if self._conveyor is not None:
            self.recorder.enable(options)
            return True
        if self._bucket_task is not None:
            self.bucket_recorder.enable(options)
            return True
        logger.warning("Automatic Studio recording needs the conveyor or bucket_sort scene")
        return False

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
        hold_arm = False
        if self._bucket_task is not None:
            bucket = self.bucket_recorder.update(self._bucket_progress())
            hold_arm = bucket.hold_arm
            self._recorder_reset = self._recorder_reset or bucket.reset_scene
        targets = None if hold_arm else self.autopilot.step(data, dt)
        self._publish_leader(arm_targets if targets is None else targets)

    def _bucket_progress(self) -> BucketSortProgress:
        """Read what the bucket_sort recorder needs this tick.

        Returns:
            The layout generation, the instruction and the autopilot's mode and episode outcome.
        """
        task = self._bucket_task
        instruction = task.instruction if task is not None else None
        episode = self.autopilot.status()
        outcome, clean = episode.get("outcome"), episode.get("clean")
        return BucketSortProgress(
            generation=(id(task), task.reset_count if task is not None else 0),
            instruction=instruction.text if instruction is not None else None,
            leading=self.autopilot.mode == "leader",
            outcome=outcome if isinstance(outcome, str) else None,
            clean=clean if isinstance(clean, bool) else None,
        )

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
        return {"autopilot": self.autopilot.status(), "studio": self.studio_recorder.status()}

    def _open_link(self) -> StudioLink:
        return StudioLink(self.studio_url, self._owner_name())


def _scene_demonstrator(
    model: object | None,
    conveyor: ConveyorSort | None,
    *,
    robots: int,
    bucket_task: BucketSortTask | None,
    seed: int | None,
) -> tuple[Demonstrator | None, str | None]:
    """Pick the demonstrator for a freshly loaded scene.

    Only the conveyor and bucket_sort scenes have one, and a conveyor with several arms has none:
    its demonstrator drives one SO-101.

    Returns:
        The demonstrator, or ``None`` with the reason the viewer shows (``None`` when the scene
        never has one).
    """
    if model is None:
        return None, None
    if conveyor is None:
        if bucket_task is None:
            return None, None
        from physicalai_mujoco_plugin.bucket_sort_demo import BucketSortDemonstrator  # noqa: PLC0415

        return BucketSortDemonstrator(model, bucket_task, rng=np.random.default_rng(seed)), None
    if robots != 1:
        return None, f"The conveyor autopilot drives one SO-101 arm; this simulation has {robots} arms."
    from physicalai_mujoco_plugin.conveyor_demo import ConveyorDemonstrator  # noqa: PLC0415

    return ConveyorDemonstrator(model, conveyor), None
