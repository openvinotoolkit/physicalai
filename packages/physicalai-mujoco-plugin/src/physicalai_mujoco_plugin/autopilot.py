# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run a scene's scripted demonstrator inside the simulation loop.

The autopilot steps any :class:`Demonstrator`; the scene picks which one (see
``ConveyorAutomation.attach_scene``). The conveyor scene has
``conveyor_demo.ConveyorDemonstrator`` (one SO-101), the bucket_sort scene
``bucket_sort_demo.BucketSortDemonstrator`` (a Franka FR3, Panda or Duo).

Two ways to use it:

- ``drive``: the demonstrator writes the arm's actuator targets itself, and
  actions from clients are ignored. For demos and quick checks.
- ``leader``: the demonstrator only publishes its targets, as a virtual leader
  arm (``GET /leader``). A teleoperation client, such as Physical AI Studio
  with the *MuJoCo Virtual Leader*, reads them and sends them back as
  follower actions, so the recorded actions are the demonstrator's commands.

Physical AI Studio records the demonstrator through the autopilot protocol
(``physicalai_studio_plugin.autopilot``, served under ``/autopilot/v1``): one
episode at a time, kept in an :class:`EpisodeBook` that the HTTP thread and the
simulation thread share. ``ConveyorAutomation`` applies the episode on the
simulation thread.
"""

from __future__ import annotations

import itertools
import secrets
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, get_args

from loguru import logger
from physicalai_studio_plugin.autopilot import (
    AUTOPILOT_PROTOCOL,
    LEADER_TIMEOUT_S,
    AutopilotError,
    AutopilotInfo,
    Episode,
    EpisodeOptions,
    EpisodeRequest,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from physicalai_studio_plugin.autopilot import Outcome, Phase

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


# ----------------------------------------------------------------------
# Episodes for Physical AI Studio (the autopilot protocol)
# ----------------------------------------------------------------------

LIVE_PHASES: frozenset[str] = frozenset({"preparing", "ready", "running"})
"""Phases of an episode in progress, which liveness and any change to the scene end."""
NATIVE_VIEWER_REASON = (
    "The native MuJoCo viewer is open: its mouse forces move objects past the simulation's controls. "
    "Start the simulation with the web viewer (viser) to record it automatically."
)
REPLACED_REASON = "Replaced by a new episode."
STOPPED_READING_REASON = "Physical AI Studio stopped reading the leader."
STUDIO_ABORT_REASON = "Physical AI Studio aborted the episode."

_STATUS = {"unsupported": 409, "bad_option": 422, "not_found": 404, "wrong_phase": 409}
_PROCESS_TOKEN = secrets.token_hex(4)
"""Starts every episode id of this process, so a restarted simulation does not reuse its ids."""
_EPISODE_NUMBERS = itertools.count(1)


def _new_episode_id() -> str:
    """Return an episode id that is unique for this process's lifetime (and very likely across restarts)."""
    return f"{_PROCESS_TOKEN}-{next(_EPISODE_NUMBERS)}"


class AutopilotRefusal(Exception):  # noqa: N818 - a refusal, not an error of this process
    """A request the protocol refuses; the HTTP layer answers it with :attr:`status_code`."""

    def __init__(self, message: str, *, code: Literal["unsupported", "bad_option", "not_found", "wrong_phase"]) -> None:
        """Refuse with a message for people and a machine-readable *code*."""
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def status_code(self) -> int:
        """The HTTP status of the refusal."""
        return _STATUS[self.code]

    def detail(self) -> dict[str, object]:
        """Return the ``detail`` of the error answer: an ``AutopilotError``."""
        return AutopilotError(code=self.code, message=self.message[:1000]).model_dump()


@dataclass(frozen=True)
class SceneSupport:
    """What the simulation thread reports about the running scene, for ``GET /autopilot/v1`` and new episodes."""

    scene: str = ""
    supported: bool = False
    reason: str | None = "No scene is loaded."
    options: tuple[str, ...] = ()
    joint_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class EpisodeView:
    """The current or last episode, as the simulation thread reads it at the start of a tick."""

    id: str
    phase: Phase
    options: EpisodeOptions


@dataclass
class _Record:
    id: str
    phase: Phase
    options: EpisodeOptions
    task: str | None = None
    outcome: Outcome | None = None
    reason: str | None = None
    clean: bool | None = None

    def model(self) -> Episode:
        return Episode(
            id=self.id,
            phase=self.phase,
            task=self.task,
            outcome=self.outcome,
            reason=self.reason[:500] if self.reason else None,
            clean=self.clean,
        )


class EpisodeBook:
    """The current or last episode, shared by the HTTP thread and the simulation thread under one lock.

    The HTTP thread answers the protocol from it (``info`` to ``abort``, which raise
    :class:`AutopilotRefusal`); nothing in it touches MuJoCo. The simulation thread reads it every
    tick (:meth:`view`), reports ``ready`` and the outcome, and ends the episode for any other change
    to the scene (:meth:`end`).

    Liveness: while an episode is in progress, every protocol request and every leader read
    (:meth:`leader_fields`) is a sign of life. After ``timeout_s`` of wall time without one, the
    episode ends as ``aborted`` (:data:`STOPPED_READING_REASON`), checked by the next tick or
    request, whichever comes first.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic, timeout_s: float = LEADER_TIMEOUT_S) -> None:
        """Start with no scene and no episode.

        Args:
            clock: Monotonic seconds, for liveness.
            timeout_s: How long an episode in progress may go without a sign of life.
        """
        self._lock = threading.Lock()
        self._clock = clock
        self._timeout_s = float(timeout_s)
        self._episode: _Record | None = None
        self._last_sign = clock()
        self._scene = SceneSupport()
        self._native_viewer: str | None = None
        self._busy: str | None = None

    # -- the simulation thread -------------------------------------------

    def set_scene(self, support: SceneSupport) -> None:
        """Report the scene now running and what it supports."""
        with self._lock:
            self._scene = support

    def set_native_viewer(self, open_: bool) -> None:  # noqa: FBT001
        """Report whether the native viewer is open: its mouse forces bypass the simulation's controls.

        While it is open the scene supports no episode, and one in progress ends.
        """
        with self._lock:
            self._native_viewer = NATIVE_VIEWER_REASON if open_ else None
            if open_:
                self._end_locked(NATIVE_VIEWER_REASON)

    def set_busy(self, reason: str | None) -> None:
        """Report that something else drives the simulation (a replay), so no episode may start, or that it ended."""
        with self._lock:
            self._busy = reason

    def view(self) -> EpisodeView | None:
        """Return the current or last episode for the simulation thread, after the liveness check."""
        with self._lock:
            self._expire_locked()
            record = self._episode
            return None if record is None else EpisodeView(record.id, record.phase, record.options)

    def in_progress(self) -> bool:
        """Return whether an episode is preparing, ready or running."""
        with self._lock:
            self._expire_locked()
            return self._episode is not None and self._episode.phase in LIVE_PHASES

    def may_step(self, episode_id: str) -> bool:
        """Return whether *episode_id* still runs, checked right before each demonstrator step."""
        with self._lock:
            self._expire_locked()
            record = self._episode
            return record is not None and record.id == episode_id and record.phase == "running"

    def mark_ready(self, episode_id: str, task: str) -> bool:
        """The scene is laid out for *episode_id*: ``preparing`` -> ``ready``, with its *task*.

        Returns:
            Whether the episode was still preparing.
        """
        with self._lock:
            record = self._episode
            if record is None or record.id != episode_id or record.phase != "preparing":
                return False
            record.phase, record.task = "ready", task
            return True

    def latch(self, episode_id: str, outcome: Outcome, *, clean: bool | None = None, reason: str | None = None) -> bool:
        """End *episode_id* with *outcome*, once.

        Returns:
            Whether it was still in progress (an abort may have ended it first).
        """
        with self._lock:
            record = self._episode
            if record is None or record.id != episode_id or record.phase not in LIVE_PHASES:
                return False
            record.phase, record.outcome, record.clean, record.reason = "ended", outcome, clean, reason
            logger.info("Autopilot episode {} ended: {}{}", episode_id, outcome, f" ({reason})" if reason else "")
            return True

    def end(self, reason: str) -> bool:
        """End the episode in progress as ``aborted`` for *reason*: a change to the scene, or a shutdown.

        Returns:
            Whether an episode was in progress.
        """
        with self._lock:
            return self._end_locked(reason)

    def summary(self) -> dict[str, object]:
        """Return the ``autopilot_episode`` section of ``/health`` and the viewer panel."""
        with self._lock:
            self._expire_locked()
            supported, reason = self._supported_locked()
            record = self._episode
            return {
                "supported": supported,
                "reason": reason,
                "episode": record.model().model_dump(mode="json") if record is not None else None,
            }

    # -- the HTTP thread ---------------------------------------------------

    def leader_fields(self) -> dict[str, object]:
        """A read of ``GET /leader``: a sign of life. Return the episode fields the sample carries.

        Returns:
            ``episode_id`` and ``phase`` of the current or last episode, both ``None`` before the first.
        """
        with self._lock:
            self._alive_locked()
            record = self._episode
            return {"episode_id": record.id if record else None, "phase": record.phase if record else None}

    def info(self) -> AutopilotInfo:
        """Answer ``GET /autopilot/v1``.

        Returns:
            The scene, whether it has an autopilot, and the current or last episode.
        """
        with self._lock:
            self._alive_locked()
            supported, reason = self._supported_locked()
            scene, record = self._scene, self._episode
            return AutopilotInfo(
                protocol=AUTOPILOT_PROTOCOL,
                scene=scene.scene,
                supported=supported,
                reason=reason,
                options=list(scene.options),
                joint_names=list(scene.joint_names),
                episode=record.model() if record is not None else None,
            )

    def create(self, request: EpisodeRequest) -> Episode:
        """Answer ``POST /autopilot/v1/episodes``: start an episode; one in progress ends first, replaced.

        Returns:
            The new episode, ``preparing``.

        Raises:
            AutopilotRefusal: ``unsupported`` (no autopilot, the native viewer, a replay) or
                ``bad_option`` (an option the scene does not take).
        """
        with self._lock:
            self._alive_locked()
            supported, reason = self._supported_locked()
            if not supported:
                raise AutopilotRefusal(reason or "The scene has no autopilot.", code="unsupported")
            extra = request.options.set_fields() - set(self._scene.options)
            if extra:
                message = f"Scene {self._scene.scene!r} does not take {', '.join(sorted(extra))}."
                raise AutopilotRefusal(message, code="bad_option")
            self._end_locked(REPLACED_REASON)
            self._episode = _Record(id=_new_episode_id(), phase="preparing", options=request.options)
            logger.info("Autopilot episode {} starts preparing", self._episode.id)
            return self._episode.model()

    def get(self, episode_id: str) -> Episode:
        """Answer ``GET /autopilot/v1/episodes/{id}``.

        Returns:
            The episode.
        """
        with self._lock:
            self._alive_locked()
            return self._find_locked(episode_id).model()

    def begin(self, episode_id: str) -> Episode:
        """Answer ``POST .../begin``: ``ready`` -> ``running``; a running episode is returned as is.

        Returns:
            The episode, ``running``.

        Raises:
            AutopilotRefusal: ``wrong_phase`` when it is not ready (or running).
        """
        with self._lock:
            self._alive_locked()
            record = self._find_locked(episode_id)
            if record.phase == "ready":
                record.phase = "running"
                logger.info("Autopilot episode {} begins", episode_id)
            elif record.phase != "running":
                message = f"Episode {episode_id!r} is {record.phase}, not ready."
                raise AutopilotRefusal(message, code="wrong_phase")
            return record.model()

    def abort(self, episode_id: str) -> Episode:
        """Answer ``POST .../abort``: end it as ``aborted``; an ended episode is returned as is.

        Returns:
            The episode, ``ended``.
        """
        with self._lock:
            self._alive_locked()
            record = self._find_locked(episode_id)
            self._end_locked(STUDIO_ABORT_REASON)
            return record.model()

    # -- under the lock ----------------------------------------------------

    def _supported_locked(self) -> tuple[bool, str | None]:
        scene = self._scene
        reason = self._native_viewer or self._busy
        if not scene.supported:
            return False, scene.reason or "The scene has no autopilot."
        return reason is None, reason

    def _find_locked(self, episode_id: str) -> _Record:
        record = self._episode
        if record is None or record.id != episode_id:
            message = f"Episode {episode_id!r} is neither the current nor the last episode."
            raise AutopilotRefusal(message, code="not_found")
        return record

    def _end_locked(self, reason: str) -> bool:
        record = self._episode
        if record is None or record.phase not in LIVE_PHASES:
            return False
        record.phase, record.outcome, record.clean, record.reason = "ended", "aborted", None, reason
        logger.info("Autopilot episode {} aborted: {}", record.id, reason)
        return True

    def _expire_locked(self) -> None:
        record = self._episode
        if record is not None and record.phase in LIVE_PHASES and self._clock() - self._last_sign > self._timeout_s:
            self._end_locked(STOPPED_READING_REASON)

    def _alive_locked(self) -> None:
        """A sign of life: an episode that already went too long without one ends first."""
        self._expire_locked()
        self._last_sign = self._clock()


__all__ = [
    "AUTOPILOT_MODES",
    "LIVE_PHASES",
    "NATIVE_VIEWER_REASON",
    "REPLACED_REASON",
    "STOPPED_READING_REASON",
    "STUDIO_ABORT_REASON",
    "Autopilot",
    "AutopilotMode",
    "AutopilotRefusal",
    "Demonstrator",
    "EpisodeBook",
    "EpisodeView",
    "SceneSupport",
]
