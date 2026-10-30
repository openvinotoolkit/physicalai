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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from physicalai_mujoco_plugin.autopilot import Autopilot
from physicalai_mujoco_plugin.conveyor import DEFAULT_BELT_SPEED, ConveyorSort
from physicalai_mujoco_plugin.studio_recorder import (
    DEFAULT_MAX_ATTEMPTS,
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
    from physicalai_mujoco_plugin.bucket_sort import BucketSortEpisode, BucketSortTask
    from physicalai_mujoco_plugin.bucket_sort_demo import BucketSortDemonstrator
    from physicalai_mujoco_plugin.studio_recorder import StuckStartState

STUCK_SETTLE_S = 0.3
"""After the move to a stuck pose, how long the arm holds there before the layout is checked."""
STUCK_DISTURBANCE = 0.005
"""A move to a stuck pose that moved an object or a bucket further than this (m) is not a clean start."""
STUCK_REST_SPEED = 0.005
"""The move to a stuck pose waits until every object and bucket is slower than this (m/s), dropped at the reset."""
STUCK_REST_TIMEOUT_S = 3.0


@dataclass
class _StuckMove:
    """The move to a stuck start in progress (bucket_sort recording)."""

    state: StuckStartState
    dt: float = 0.0
    """Control period, for the move once it is planned."""
    targets: np.ndarray | None = None
    """Model-unit targets per tick, every channel; ``None`` when unavailable."""
    tick: int = 0
    settle: int = 0
    """Ticks to hold the last target before checking the layout."""
    bodies: tuple[int, ...] = ()
    """Every object and bucket."""
    wait: int = 0
    """Ticks left to wait for them to rest before the move; the move starts at the latest then."""
    wait_total: int = 0
    spawn: dict[int, np.ndarray] | None = None
    """Every object's and bucket's position when the move starts, by body id."""


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
        self._bucket_episode: BucketSortEpisode | None = None
        """The bucket_sort scene's episode helper, whose success reset waits while the recorder owns episodes."""
        self._bucket_demo: BucketSortDemonstrator | None = None
        """The bucket_sort scene's demonstrator, which recording options and stuck starts reach."""
        self._max_attempts = DEFAULT_MAX_ATTEMPTS
        self._stuck: _StuckMove | None = None
        self._stuck_rng = np.random.default_rng()
        self._reset_pose: np.ndarray | None = None
        """The arm's targets right after a scene reset while bucket_sort recording holds the arm: the
        leader publishes them and client actions are ignored, so Studio's last action of the previous
        episode cannot pull the arm back before the next episode starts."""
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
        """Whether to ignore client actions: the autopilot drives the arm, or it waits at its reset pose."""
        return self.autopilot.drives_arm or self._reset_pose is not None

    @property
    def planner_running(self) -> bool:
        """Whether the bucket_sort demonstrator's planning thread runs, which reads the model."""
        return self._bucket_demo is not None and self._bucket_demo.planner_running

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
        profile: str = "franka_fr3",
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
            profile: The robot's profile name, which picks the bucket_sort demonstrator's arms.

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
        self._reset_pose = None
        self._bucket_episode = None
        self._stuck = None
        self._stuck_rng = np.random.default_rng(None if seed is None else seed + 1)
        demonstrator, unavailable = _scene_demonstrator(
            model,
            conveyor,
            robots=robots,
            bucket_task=bucket_task,
            seed=seed,
            profile=profile,
            max_attempts=self._max_attempts,
        )
        from physicalai_mujoco_plugin.bucket_sort_demo import BucketSortDemonstrator  # noqa: PLC0415

        if self._bucket_demo is not None:
            self._bucket_demo.close()  # its scene is gone: stop a plan in flight
        self._bucket_demo = demonstrator if isinstance(demonstrator, BucketSortDemonstrator) else None
        self.autopilot.bind(demonstrator, unavailable=unavailable)
        self._bucket_task = bucket_task if model is not None and conveyor is None else None
        if conveyor is None and self.recorder.phase != "off":
            self.recorder.disable("Automatic recording needs the conveyor scene.")
        elif conveyor is not None:
            self.recorder.restart_episode("The scene was reloaded.")
        if self._bucket_task is None and self.bucket_recorder.phase != "off":
            # The episode Studio records showed the scene that is gone: discard it.
            self.bucket_recorder.disable(
                "Automatic recording of bucket_sort episodes needs the bucket_sort scene.", discard=True
            )
        # A reloaded bucket_sort scene is a new task object: the recorder discards the episode itself.
        return conveyor

    def bind_bucket_episode(self, episode: BucketSortEpisode | None) -> None:
        """Attach the bucket_sort scene's episode helper, after :meth:`attach_scene`.

        While bucket_sort recording runs, the recorder ends and resets every episode, so the helper's
        own reset after a success waits: a short success dwell would otherwise reset the scene before
        the recorder saves the episode.
        """
        self._bucket_episode = episode

    def take_reset_request(self) -> bool:
        """Return whether the scene should be reset now, once.

        Returns:
            ``True`` once per failed bucket_sort episode the autopilot gave up on, and once before
            each bucket_sort episode Studio records.
        """
        requested = self.autopilot.take_reset_request()
        recorder, self._recorder_reset = self._recorder_reset, False
        return requested or recorder

    def reset(self, arm_targets: np.ndarray | None = None) -> None:
        """Forget the autopilot's current pick and restart a Studio episode; call after a scene reset.

        Args:
            arm_targets: The arm's targets after the reset, in model units; held until the next
                bucket_sort episode Studio records starts.
        """
        self.autopilot.reset()
        self.recorder.restart_episode("The scene was reset.")
        self._stuck = None
        if arm_targets is not None and self._bucket_task is not None and self.bucket_recorder.active:
            self._reset_pose = np.array(arm_targets, dtype=np.float64)
            # Before the next tick, which may start the Studio episode: Studio must already read this pose.
            self._publish_leader(self._reset_pose)

    def close(self) -> None:
        """Detach from Studio and stop a plan in flight; call when the simulation disconnects."""
        self.recorder.disable()
        self.bucket_recorder.disable()
        if self._bucket_demo is not None:
            self._bucket_demo.close()

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
            self._set_max_attempts(DEFAULT_MAX_ATTEMPTS)
            return True
        if self._conveyor is not None:
            self.recorder.enable(options)
            return True
        if self._bucket_task is not None:
            self._set_max_attempts(options.max_attempts)
            self.bucket_recorder.enable(options)
            return True
        logger.warning("Automatic Studio recording needs the conveyor or bucket_sort scene")
        return False

    def _set_max_attempts(self, attempts: int) -> None:
        """Give the bucket_sort demonstrator *attempts* per episode, now and for a reloaded scene."""
        self._max_attempts = attempts
        if self._bucket_demo is not None:
            self._bucket_demo.max_attempts = attempts

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
        if self._bucket_demo is not None:
            self._bucket_demo.reap()  # a plan cancelled while the autopilot is off is never stepped
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
            bucket = self.bucket_recorder.update(self._bucket_progress(model, data))
            hold_arm = bucket.hold_arm
            self._recorder_reset = self._recorder_reset or bucket.reset_scene
            if bucket.abort_stuck_start:
                # Nothing more is written to the arm, and client actions count again.
                self._drop_stuck_start()
                self._reset_pose = None
            if bucket.stuck_start:
                self._begin_stuck_start(model, data, dt)
        if not hold_arm or not self.bucket_recorder.active:
            self._reset_pose = None
            self._drop_stuck_start()
        elif self._stuck is not None:
            self._advance_stuck_start(model, data)
        if self._bucket_episode is not None:
            self._bucket_episode.suspend_resets(self.bucket_recorder.active)
        targets = None if hold_arm else self.autopilot.step(data, dt)
        if targets is None and self._reset_pose is not None:
            targets = self._reset_pose
        self._publish_leader(arm_targets if targets is None else targets)

    def _drop_stuck_start(self) -> None:
        """Forget the stuck start; one still being planned is cancelled."""
        if self._stuck is not None and self._stuck.state == "planning" and self._bucket_demo is not None:
            self._bucket_demo.reset()
        self._stuck = None

    def _begin_stuck_start(self, model: object, data: object, dt: float) -> None:
        """Plan the move to a stuck start (:meth:`~.bucket_sort_demo.BucketSortDemonstrator.stuck_start`).

        The demonstrator plans on its own thread; the state is ``planning`` until it is done.
        """
        demo = self._bucket_demo
        if demo is None:
            self._stuck = _StuckMove(state="unavailable")
            return
        demo.begin_stuck_start(data, dt, self._stuck_rng)
        self._stuck = _StuckMove(state="planning", dt=dt)
        self._take_stuck_plan(model)

    def _take_stuck_plan(self, model: object) -> None:
        """Once the move to the stuck pose is planned, start it (``moving``), or report none (``unavailable``)."""
        from physicalai_mujoco_plugin.bucket_sort import BUCKET_COLORS, OBJECTS  # noqa: PLC0415

        stuck, demo = self._stuck, self._bucket_demo
        if stuck is None or stuck.state != "planning" or demo is None:
            return
        planning, move = demo.poll_stuck_start(stuck.dt)
        if planning:
            return
        if move is None:
            self._stuck = _StuckMove(state="unavailable")
            return
        dt = stuck.dt
        names = (*OBJECTS, *(f"bucket_{color}" for color in BUCKET_COLORS))
        self._stuck = _StuckMove(
            state="moving",
            dt=dt,
            targets=move.targets,
            settle=max(1, round(STUCK_SETTLE_S / dt)),
            bodies=tuple(int(model.body(name).id) for name in names),  # type: ignore[attr-defined]
            wait=max(1, round(STUCK_REST_TIMEOUT_S / dt)),
            wait_total=max(1, round(STUCK_REST_TIMEOUT_S / dt)),
        )

    def _advance_stuck_start(self, model: object, data: object) -> None:
        """Drive the arm one tick along the move to the stuck pose, then check that nothing moved.

        The move starts once it is planned and the objects and buckets the reset dropped rest (or
        after :data:`STUCK_REST_TIMEOUT_S`); the arm holds its reset targets meanwhile.

        Client actions are ignored meanwhile (:attr:`drives_arm`), and the leader publishes the arm's
        targets: when the episode starts, it already publishes the stuck pose.
        """
        self._take_stuck_plan(model)
        stuck, demo = self._stuck, self._bucket_demo
        if stuck is None or stuck.state != "moving" or stuck.targets is None or demo is None:
            return
        if stuck.spawn is None:
            # The reset dropped the objects: measure the layout once it rests, then move.
            # They start at rest just above the table, so give them STUCK_SETTLE_S to land first.
            speed = max(float(np.linalg.norm(data.cvel[body][3:])) for body in stuck.bodies)  # type: ignore[attr-defined]
            stuck.wait -= 1
            landed = stuck.wait <= stuck.wait_total - stuck.settle
            if (not landed or speed > STUCK_REST_SPEED) and stuck.wait > 0:
                return
            stuck.spawn = {body: np.array(data.xpos[body]) for body in stuck.bodies}  # type: ignore[attr-defined]
        targets = stuck.targets[min(stuck.tick, len(stuck.targets) - 1)]
        demo.write_targets(data, targets)
        self._reset_pose = np.array(targets)
        if stuck.tick < len(stuck.targets):
            stuck.tick += 1
            return
        stuck.settle -= 1
        if stuck.settle > 0:
            return
        moved = max(float(np.linalg.norm(data.xpos[body] - start)) for body, start in (stuck.spawn or {}).items())  # type: ignore[attr-defined]
        stuck.state = "disturbed" if moved > STUCK_DISTURBANCE else "ready"
        logger.info("Stuck start {}: the layout moved {:.1f} mm", stuck.state, moved * 1000)

    def _bucket_progress(self, model: object, data: object) -> BucketSortProgress:
        """Read what the bucket_sort recorder needs this tick.

        Returns:
            The layout generation, the instruction, the autopilot's mode and episode outcome, and
            whether the task's success holds (evaluated only after the demonstrator's success).
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
            placed=outcome == "success" and task is not None and task.evaluate(model, data).success,
            sim_time=float(data.time),  # type: ignore[attr-defined]
            stuck_start=self._stuck.state if self._stuck is not None else None,
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
    profile: str = "franka_fr3",
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> tuple[Demonstrator | None, str | None]:
    """Pick the demonstrator for a freshly loaded scene.

    Only the conveyor and bucket_sort scenes have one. A conveyor with several arms has none: its
    demonstrator drives one SO-101. The bucket_sort one drives the arms of the profiles in
    :data:`~.bucket_sort_arms.BUCKET_SORT_ARMS` (one robot: the Duo's two arms are one).

    Returns:
        The demonstrator, or ``None`` with the reason the viewer shows (``None`` when the scene
        never has one).
    """
    if model is None:
        return None, None
    if conveyor is None:
        if bucket_task is None:
            return None, None
        from physicalai_mujoco_plugin.bucket_sort_arms import BUCKET_SORT_ARMS, bucket_sort_arms  # noqa: PLC0415
        from physicalai_mujoco_plugin.bucket_sort_demo import BucketSortDemonstrator  # noqa: PLC0415

        arms = bucket_sort_arms(profile, robots)
        if arms is None:
            names = ", ".join(BUCKET_SORT_ARMS)
            return None, f"The bucket_sort autopilot drives one robot of these profiles ({names}) or two WidowX AI."
        demonstrator = BucketSortDemonstrator(
            model,
            bucket_task,
            arms=arms,
            rng=np.random.default_rng(seed),
            max_attempts=max_attempts,
            background_planning=True,  # plans take up to seconds: the control tick must not wait
        )
        return demonstrator, None
    if robots != 1:
        return None, f"The conveyor autopilot drives one SO-101 arm; this simulation has {robots} arms."
    from physicalai_mujoco_plugin.conveyor_demo import ConveyorDemonstrator  # noqa: PLC0415

    return ConveyorDemonstrator(model, conveyor), None
