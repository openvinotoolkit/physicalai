# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Conveyor-scene automation that the simulation loop runs once per control tick.

One object owns everything the conveyor and bucket_sort scenes add on top of plain
simulation, so ``MuJoCoRobot`` only forwards to it:

- the conveyor controller for the current scene, and the belt speed remembered
  across scene switches;
- the autopilot (the scripted demonstrator, see ``autopilot.py``);
- the virtual leader pose served on ``GET /leader``, with the current episode's id and phase;
- Physical AI Studio's autopilot episodes (the :class:`~.autopilot.EpisodeBook` behind
  ``/autopilot/v1``), applied here on the simulation thread: each episode lays the scene out
  (a cleared belt; or a reset bucket_sort layout, optionally with a stuck start) with the feed
  held and the scene's own resets suspended, waits ``ready``, lets the demonstrator lead from
  ``begin``, latches the outcome, and then gives the scene settings back.

All methods except `leader_snapshot`, `read_leader` and `status` run on the simulation thread.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from loguru import logger

from physicalai_mujoco_plugin.autopilot import Autopilot, EpisodeBook
from physicalai_mujoco_plugin.conveyor import DEFAULT_BELT_SPEED, ConveyorSort

if TYPE_CHECKING:
    from collections.abc import Callable

    from physicalai_studio_plugin.autopilot import Outcome

    from physicalai_mujoco_plugin.autopilot import AutopilotMode, Demonstrator, EpisodeView
    from physicalai_mujoco_plugin.bucket_sort import BucketSortEpisode, BucketSortTask
    from physicalai_mujoco_plugin.bucket_sort_demo import BucketSortDemonstrator

DEFAULT_MAX_ATTEMPTS = 2
"""The demonstrator's attempts per episode by default (``bucket_sort_demo.DEFAULT_MAX_ATTEMPTS``; kept
here so that this module does not import the demonstrator)."""
CONVEYOR_TASK = "Sort the items: cracked or purple ones into reject, the others into the bin of their color."
"""The instruction of every conveyor episode."""
BUCKET_SORT_OPTIONS = ("max_attempts", "recovery_fraction")
"""The episode options the bucket_sort scene takes; the conveyor takes none."""
STUCK_SETTLE_S = 0.3
"""After the move to a stuck pose, how long the arm holds there before the layout is checked."""
STUCK_DISTURBANCE = 0.005
"""A move to a stuck pose that moved an object or a bucket further than this (m) is not a clean start."""
STUCK_REST_SPEED = 0.005
"""The move to a stuck pose waits until every object and bucket is slower than this (m/s), dropped at the reset."""
STUCK_REST_TIMEOUT_S = 3.0
PREPARE_TIMEOUT_S = 15.0
"""Wall seconds a scene reset, or the move to a stuck pose once planned, may take while preparing an episode."""
SUCCESS_HOLD_S = 1.0
"""Simulated seconds a bucket_sort success must hold before the episode ends as a success: the object stays
in its bucket, and the episode ends with the arm settling instead of on the tick the object landed."""
SUCCESS_HOLD_TIMEOUT_S = 4.0
"""Simulated seconds after the demonstrator's success the object may take to hold; then it is a failure."""
EPISODE_LIMIT_S = {"conveyor": 300.0, "bucket_sort": 120.0}
"""Simulated seconds an episode may run; then it ends as ``stalled``."""

StuckStartState = Literal["planning", "moving", "ready", "disturbed", "unavailable"]
"""``planning`` the move (on its own thread; not counted against the timeout); ``moving`` to the stuck pose;
``ready`` there; ``disturbed``: the move moved an object or a bucket more than 5 mm; ``unavailable``: no
stuck pose for this layout (the arm did not move)."""
EpisodeStage = Literal["resetting", "positioning", "ready", "running"]


@dataclass
class _StuckMove:
    """The move to a stuck start in progress (bucket_sort episodes)."""

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


@dataclass
class _Applied:
    """The episode the simulation thread applies, and the scene settings it gives back afterwards."""

    episode_id: str
    stage: EpisodeStage
    belt_active: bool | None
    """The conveyor's own running state before the episode; ``None`` without a conveyor."""
    since: float = 0.0
    """Wall time the current preparation step started, for :data:`PREPARE_TIMEOUT_S`."""
    generation: tuple[int, int] | None = None
    """The bucket_sort layout before the reset this episode asked for."""
    normal_start: bool = False
    """The stuck start of this episode disturbed the layout: the next start is normal."""
    started_at: float = 0.0
    """Simulated time at ``begin``."""
    base_count: int = 0
    """The conveyor's episode count at ``begin``."""
    success_at: float | None = None
    placed_since: float | None = None


class ConveyorAutomation:
    """Belt, autopilot, virtual leader and Physical AI Studio's autopilot episodes for the current scene."""

    def __init__(
        self,
        *,
        joint_names: tuple[str, ...],
        unit: str,
        to_units: Callable[[np.ndarray], np.ndarray],
        chance: Callable[[], float] | None = None,
    ) -> None:
        """Start with no scene attached.

        Args:
            joint_names: The robot's joint order; leader poses use it.
            unit: The robot's joint unit, reported with each leader pose.
            to_units: Converts joint radians to the robot's unit.
            chance: Draws a number in [0, 1) per bucket_sort episode; below the episode's
                ``recovery_fraction`` it starts stuck. ``None`` uses a fresh generator.
        """
        self.autopilot = Autopilot()
        self.book = EpisodeBook()
        """Physical AI Studio's episode, shared with the HTTP server's ``/autopilot/v1`` routes."""
        self.belt_speed = DEFAULT_BELT_SPEED
        """Belt speed in m/s, kept across scene switches like the auto-reset settings."""
        self._joint_names = tuple(joint_names)
        self._unit = unit
        self._to_units = to_units
        self._chance = chance if chance is not None else np.random.default_rng().random  # not a secret
        self._conveyor: ConveyorSort | None = None
        self._bucket_task: BucketSortTask | None = None
        self._bucket_episode: BucketSortEpisode | None = None
        """The bucket_sort scene's episode helper, whose own resets wait while an episode is applied."""
        self._bucket_demo: BucketSortDemonstrator | None = None
        """The bucket_sort scene's demonstrator, which episode options and stuck starts reach."""
        self._max_attempts = DEFAULT_MAX_ATTEMPTS
        self._stuck: _StuckMove | None = None
        self._stuck_rng = np.random.default_rng()
        self.stuck_starts = 0
        """Episodes that started from a stuck pose, for ``/health``."""
        self._reset_pose: np.ndarray | None = None
        """The arm's targets while an episode is prepared and waits to begin: the leader publishes them and
        client actions are ignored, so a client's last action cannot move the arm off the start pose."""
        self._kept_target: np.ndarray | None = None
        """The demonstrator's last target after an episode ended: the leader keeps publishing it, so the
        follower settles there, until the next episode or a manual change."""
        self._last_target: np.ndarray | None = None
        self._applied: _Applied | None = None
        """The episode this thread applies; ``None`` while none is in progress."""
        self._episode_reset = False
        """The episode asked for a scene reset at the end of this tick."""
        self._expect_reset = False
        """The reset running now is the episode's own (any other one ends the episode)."""
        self._lock = threading.Lock()
        self._leader_seq = 0
        self._leader: dict[str, object] = {"seq": 0, "joint_names": list(self._joint_names)}

    @property
    def conveyor(self) -> ConveyorSort | None:
        """The current scene's conveyor controller, if it has a belt."""
        return self._conveyor

    @property
    def drives_arm(self) -> bool:
        """Whether to ignore client actions: the autopilot drives the arm, or an episode holds its start pose."""
        return self.autopilot.drives_arm or self._reset_pose is not None

    @property
    def planner_running(self) -> bool:
        """Whether the bucket_sort demonstrator's planning thread runs, which reads the model."""
        return self._bucket_demo is not None and self._bucket_demo.planner_running

    @property
    def episode_options(self) -> tuple[str, ...]:
        """The episode options the current scene takes (``GET /autopilot/v1`` ``options``)."""
        return BUCKET_SORT_OPTIONS if self._bucket_task is not None and self._conveyor is None else ()

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
        """Build the conveyor controller for a freshly loaded scene; an episode in progress ends with the old scene.

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
        self.book.end("The scene was loaded again.")
        conveyor = None
        if model is not None:
            conveyor = ConveyorSort.maybe_create(
                model, rng=rng, belt_speed=self.belt_speed, active=active, robot_roots=robot_roots
            )
        self._conveyor = conveyor
        # The old scene's objects are gone: nothing the episode changed on them needs giving back.
        self._applied = None
        self._episode_reset = self._expect_reset = False
        self._reset_pose = self._kept_target = None
        self._bucket_episode = None
        self._stuck = None
        self._stuck_rng = np.random.default_rng(None if seed is None else seed + 1)
        self._max_attempts = DEFAULT_MAX_ATTEMPTS
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
        return conveyor

    def bind_bucket_episode(self, episode: BucketSortEpisode | None) -> None:
        """Attach the bucket_sort scene's episode helper, after :meth:`attach_scene`.

        While an episode is applied, this object resets and ends it, so the helper's own reset after a
        success waits: a short success dwell would otherwise reset the scene before Studio saves.
        """
        self._bucket_episode = episode

    def take_reset_request(self) -> bool:
        """Return whether the scene should be reset now, once.

        Returns:
            While an episode is applied, ``True`` once before each bucket_sort layout it asks for (the
            autopilot's own requests are dropped: the episode decides). Otherwise ``True`` once per
            failed bucket_sort episode the autopilot gave up on.
        """
        requested = self.autopilot.take_reset_request()
        if self._applied is None:
            return requested
        asked, self._episode_reset = self._episode_reset, False
        self._expect_reset = asked
        return asked

    def reset(self, arm_targets: np.ndarray | None = None) -> None:
        """Forget the autopilot's current pick; call after a scene reset.

        A reset the episode asked for holds the arm at *arm_targets* (the targets right after the reset,
        in model units) until the episode begins; any other reset ends the episode in progress.
        """
        self.autopilot.reset()
        self._stuck = None
        expected, self._expect_reset = self._expect_reset, False
        if self._applied is None:
            return
        if not expected:
            self.end_episode_now("The scene was reset.")
            return
        if arm_targets is not None:
            self._reset_pose = np.array(arm_targets, dtype=np.float64)
            # Before the next tick, which may report ready: the leader must already publish this pose.
            self._publish_leader(self._reset_pose)

    def end_episode_now(self, reason: str) -> None:
        """End the episode in progress as ``aborted`` for *reason*, and give the scene settings back now.

        Runs on the simulation thread before a change to the scene applies (a command from HTTP or the
        viewer, the native viewer's reset), so the change finds the scene as it was before the episode.
        """
        self.book.end(reason)
        if self._applied is not None:
            self._finish()
        self._kept_target = None  # a manual change: the leader follows the arm again

    def close(self) -> None:
        """End an episode in progress and stop a plan in flight; call when the simulation disconnects."""
        self.book.end("The simulation stopped.")
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
        self._kept_target = None
        logger.info("Autopilot {}", mode)
        return True

    def _set_max_attempts(self, attempts: int) -> None:
        """Give the bucket_sort demonstrator *attempts* per episode."""
        self._max_attempts = attempts
        if self._bucket_demo is not None:
            self._bucket_demo.max_attempts = attempts

    # ------------------------------------------------------------------
    # Per control tick
    # ------------------------------------------------------------------

    def tick(self, model: object, data: object, dt: float, arm_targets: np.ndarray) -> None:
        """Apply Studio's episode, step the autopilot and publish the leader pose.

        Args:
            model: The loaded MuJoCo model.
            data: The simulation state.
            dt: Control period in seconds.
            arm_targets: The arm's current actuator targets in model units, published as the leader
                pose while the autopilot is off.
        """
        if self._bucket_demo is not None:
            self._bucket_demo.reap()  # a plan cancelled while the autopilot is off is never stepped
        step = self._episode_tick(model, data, dt, arm_targets)
        applied = self._applied
        if applied is not None and applied.stage == "positioning":
            self._advance_stuck_start(model, data)
        if self._bucket_episode is not None:
            self._bucket_episode.suspend_resets(applied is not None)
        targets = self.autopilot.step(data, dt) if step else None
        if targets is None:
            targets = self._reset_pose if self._reset_pose is not None else self._kept_target
        self._publish_leader(arm_targets if targets is None else targets)

    # -- the episode ----------------------------------------------------

    def _episode_tick(self, model: object, data: object, dt: float, arm_targets: np.ndarray) -> bool:
        """Apply the episode's phase for this tick.

        Returns:
            Whether the autopilot may step: always outside an episode (as its mode says); in one, only
            while it runs (checked against the book right before the step).
        """
        view = self.book.view()
        applied = self._applied
        if applied is not None and (view is None or view.id != applied.episode_id or view.phase == "ended"):
            self._finish()  # ended over HTTP or by liveness, or replaced by a new episode
            applied = None
        if view is None or view.phase == "ended":
            return True
        if applied is None:
            if view.phase != "preparing":
                return True  # not applied here (the scene was loaded again under it): nothing to do
            applied = self._start(model, data, view, arm_targets)
        if view.phase == "preparing":
            self._prepare(model, data, dt, view, applied)
            return False
        if view.phase == "ready":
            return False
        if applied.stage != "running":
            self._begin(data, applied)
        ended = self._evaluate(model, data, applied)
        if ended is None:
            return self.book.may_step(applied.episode_id)
        outcome, clean, reason = ended
        self.book.latch(applied.episode_id, outcome, clean=clean, reason=reason)
        self._finish()
        return False

    def _start(self, model: object, data: object, view: EpisodeView, arm_targets: np.ndarray) -> _Applied:
        """Take the scene for a new episode: the feed and the arm hold, the autopilot leads (not stepped yet).

        Then the belt is cleared, or a new bucket_sort layout is asked for.

        Returns:
            The episode's progress on this thread.
        """
        conveyor = self._conveyor
        applied = _Applied(
            view.id,
            stage="resetting",
            belt_active=conveyor.active if conveyor is not None else None,
            since=time.monotonic(),
        )
        self._applied = applied
        self._kept_target = None
        self._drop_stuck_start()
        if self.autopilot.available:
            self.autopilot.set_mode("leader")  # not stepped until the episode begins
        self._reset_pose = np.array(arm_targets, dtype=np.float64)
        if self._bucket_episode is not None:
            self._bucket_episode.suspend_resets(True)  # noqa: FBT003
        if conveyor is not None:
            conveyor.set_feed_hold(True)
            conveyor.reset_items(model, data)
            self.autopilot.reset()
        else:
            self._set_max_attempts(view.options.max_attempts or DEFAULT_MAX_ATTEMPTS)
            task = self._bucket_task
            applied.generation = (id(task), task.reset_count) if task is not None else None
            self._episode_reset = True  # the robot resets the scene at the end of this tick
        return applied

    def _finish(self) -> None:
        """Give back the scene settings from before the episode: autopilot off, feed and resets as they were.

        The leader keeps publishing the demonstrator's last target, so the follower settles there.
        """
        applied, self._applied = self._applied, None
        self._kept_target = self._last_target
        self._episode_reset = self._expect_reset = False
        self._drop_stuck_start()
        self._reset_pose = None  # client actions count again; the arm keeps its targets
        if self.autopilot.available:
            self.autopilot.set_mode("off")
        conveyor = self._conveyor
        if conveyor is not None:
            conveyor.set_feed_hold(False)
            if applied is not None and applied.belt_active is not None:
                conveyor.set_active(applied.belt_active)
        if self._bucket_episode is not None:
            self._bucket_episode.suspend_resets(False)  # noqa: FBT003
        self._set_max_attempts(DEFAULT_MAX_ATTEMPTS)

    def _prepare(  # noqa: PLR0911 - one return per preparation step
        self, model: object, data: object, dt: float, view: EpisodeView, applied: _Applied
    ) -> None:
        """Advance the episode's preparation, and report it ``ready`` once the scene is laid out and holds."""
        episode_id = applied.episode_id
        now = time.monotonic()
        if not self.autopilot.available:
            self.book.latch(episode_id, "aborted", reason="The scene has no autopilot.")
            return
        if self._conveyor is not None:
            self._ready(applied, CONVEYOR_TASK)
            return
        task = self._bucket_task
        if task is None:
            self.book.latch(episode_id, "aborted", reason="The scene has no bucket_sort task.")
            return
        generation = (id(task), task.reset_count)
        instruction = task.instruction.text if task.instruction is not None else None
        if applied.stage == "resetting":
            if generation == applied.generation or not instruction:
                if now - applied.since > PREPARE_TIMEOUT_S:
                    self.book.latch(episode_id, "aborted", reason="The scene did not reset for a new episode.")
                return
            applied.generation = generation
            fraction = view.options.recovery_fraction or 0.0
            if not applied.normal_start and fraction > 0.0 and self._chance() < fraction:
                applied.stage, applied.since = "positioning", now
                self._begin_stuck_start(model, data, dt)
                return
            applied.normal_start = False
            self._ready(applied, instruction)
            return
        if applied.stage != "positioning":
            return
        if generation != applied.generation:
            self._reset_again(applied, generation, now)
            return
        state = self._stuck.state if self._stuck is not None else "unavailable"
        if state in {"ready", "unavailable"}:
            # Unavailable: the arm did not move, so the episode starts normally as the scene stands.
            self.stuck_starts += state == "ready"
            self._ready(applied, instruction or "")
        elif state == "planning":
            applied.since = now  # planning takes up to seconds on its own thread: the timeout counts after it
        elif state == "disturbed" or now - applied.since > PREPARE_TIMEOUT_S:
            # A start with something knocked out of place is not a clean stuck state: lay it out anew.
            logger.info("The stuck start was {}; resetting for a normal start", state)
            applied.normal_start = True
            self._reset_again(applied, generation, now)

    def _reset_again(self, applied: _Applied, generation: tuple[int, int], now: float) -> None:
        self._drop_stuck_start()
        applied.stage, applied.since, applied.generation = "resetting", now, generation
        self._episode_reset = True

    def _ready(self, applied: _Applied, task: str) -> None:
        if self.book.mark_ready(applied.episode_id, task):
            applied.stage = "ready"
            logger.info("Autopilot episode {} ready: {}", applied.episode_id, task)

    def _begin(self, data: object, applied: _Applied) -> None:
        """The episode begins: the demonstrator leads from here, the feed runs and client actions count."""
        self._reset_pose = None
        self._drop_stuck_start()
        conveyor = self._conveyor
        if conveyor is not None:
            conveyor.set_active(True)
            conveyor.set_feed_hold(False)
            applied.base_count = int(conveyor.status()["episode_count"])  # pyrefly: ignore [bad-argument-type]
        applied.stage = "running"
        applied.started_at = float(data.time)  # type: ignore[attr-defined]
        applied.success_at = applied.placed_since = None

    def _evaluate(
        self, model: object, data: object, episode: _Applied
    ) -> tuple[Outcome, bool | None, str | None] | None:
        """Return how the running episode ended, or ``None`` while it runs.

        Returns:
            The outcome, whether it was clean, and why; ``None`` while the episode goes on.
        """
        now = float(data.time)  # type: ignore[attr-defined]
        conveyor = self._conveyor
        kind = "conveyor" if conveyor is not None else "bucket_sort"
        if conveyor is not None:
            status = conveyor.status()
            if int(status["episode_count"]) > episode.base_count:  # pyrefly: ignore [bad-argument-type]
                score = dict(status.get("last_episode") or {})  # pyrefly: ignore [no-matching-overload]
                wrong, missed = int(score.get("wrong", 0)), int(score.get("missed", 0))
                if wrong == 0 and missed == 0:
                    return "success", True, None
                return "failure", False, f"{wrong} wrong, {missed} missed"
        else:
            ended = self._bucket_outcome(model, data, episode, now)
            if ended is not None:
                return ended
        limit = EPISODE_LIMIT_S[kind]
        if now - episode.started_at > limit:
            return "stalled", None, f"The episode ran out of time ({limit:.0f} s simulated)."
        return None

    def _bucket_outcome(
        self, model: object, data: object, episode: _Applied, now: float
    ) -> tuple[Outcome, bool | None, str | None] | None:
        """Read the demonstrator's outcome; a success counts once the object has stayed in its bucket.

        Returns:
            The outcome, whether it was clean, and why; ``None`` while the episode goes on.
        """
        status = self.autopilot.status()
        outcome, clean = status.get("outcome"), status.get("clean")
        clean = clean if isinstance(clean, bool) else None
        failure = status.get("failure")
        if outcome == "failed":
            return "failure", False, str(failure or "The autopilot gave the episode up.")
        if outcome == "stalled":
            return "stalled", None, str(failure or "The arm did not follow the autopilot.")
        if outcome != "success":
            return None
        task = self._bucket_task
        placed = task is not None and task.evaluate(model, data).success
        if episode.success_at is None:
            episode.success_at = now
        if not placed:
            episode.placed_since = None
        elif episode.placed_since is None:
            episode.placed_since = now
        if episode.placed_since is not None and now - episode.placed_since >= SUCCESS_HOLD_S:
            return "success", clean, None if clean is not False else "Succeeded after a retry."
        if now - episode.success_at > SUCCESS_HOLD_TIMEOUT_S:
            return "failure", clean, "The object did not stay in its bucket."
        return None

    # -- stuck starts ---------------------------------------------------

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

    def _publish_leader(self, targets: np.ndarray) -> None:
        """Publish the virtual leader pose in the robot's unit.

        While the autopilot is off this is the arm's actuator targets, not its
        measured joints: echoing measured joints would let a teleoperated
        follower sag under gravity.
        """
        self._last_target = np.array(targets, dtype=np.float64)
        positions = self._to_units(self._last_target)
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
        """Latest virtual leader pose, with the current or last episode's id and phase; no sign of life.

        Returns:
            ``seq``, ``mode``, ``unit``, ``joint_names``, ``joint_positions``, ``episode_id`` and
            ``phase`` (both ``None`` before the first episode).
        """
        with self._lock:
            sample = dict(self._leader)
        episode = self.book.summary()["episode"]
        sample["episode_id"] = episode.get("id") if isinstance(episode, dict) else None
        sample["phase"] = episode.get("phase") if isinstance(episode, dict) else None
        return sample

    def read_leader(self) -> dict[str, object]:
        """Answer ``GET /leader``: as :meth:`leader_snapshot`, and a sign of life for the episode in progress.

        Returns:
            The leader sample.
        """
        fields = self.book.leader_fields()
        with self._lock:
            sample = dict(self._leader)
        sample.update(fields)
        return sample

    def status(self) -> dict[str, dict[str, object]]:
        """Return the ``autopilot`` and ``autopilot_episode`` sections of ``/health`` and the viewer."""
        episode = self.book.summary()
        episode["stuck_starts"] = self.stuck_starts
        return {"autopilot": self.autopilot.status(), "autopilot_episode": episode}


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
