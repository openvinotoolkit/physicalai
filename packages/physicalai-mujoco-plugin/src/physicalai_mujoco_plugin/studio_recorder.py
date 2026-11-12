# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Save conveyor and bucket_sort episodes into a Physical AI Studio dataset automatically.

The simulation does not reach Studio itself. Studio's recording page, open on this
simulation's robot, relays: while automatic recording runs (``/health`` ``studio.phase``
is neither ``off``, ``done``, ``stopped`` nor ``error``), it posts its session's state to
``POST /studio/relay`` a few times a second, runs the commands it gets back through its own
session, exactly as its buttons do, and posts the results with the next exchange.
:class:`StudioRelay` is that endpoint; each run of the recorder gets a :class:`RelayLink`
from it. The episode logic around the link:

1. Wait until Studio is ready: a dataset loaded, teleoperation running, and no
   recording in progress.
2. Clear the belt, start a Studio episode with the task, and let the belt run.
3. When the conveyor episode ends, hold the feed, then save the episode (or
   discard it when it had mistakes and only perfect episodes are kept).
4. Repeat from 1 until switched off or the episode budget is used up.

The bucket_sort scene (`BucketSortRecorder`) runs the same cycle, also through
Studio's teleoperation with the autopilot as leader: the autopilot runs in ``leader``
mode, so Studio records the demonstrator's commands as the actions, including a
gripper closed past the object, not just where the fingers stopped. It differs
in two ways. It resets the scene before each episode, and starts the Studio
episode with that episode's worded instruction as the task. And it ends the
episode on the demonstrator's outcome: a success is saved (``keep: "perfect"``
only when no attempt went wrong), a failure or a stall discarded.

The relay exchange, request then response::

    {"client": "<the page's own id>",
     "session": {"session_id": "...", "dataset_id": "...", "dataset": true, "recording": false,
                 "teleop": true, "follower": "Sim arm", "leader": "Sim arm autopilot"},
     "results": [{"request_id": "...", "ok": true, "error": null}]}

    {"active": true, "owner": true,
     "commands": [{"command": "start_episode", "request_id": "...", "task": "..."}]}

``commands`` are Studio's own session commands: ``start_episode`` (with ``task``),
``save_episode`` and ``discard_episode``. The first page to exchange owns a run (``owner``);
another page gets no commands, and the run fails if its page turns up on another session or
dataset. A command comes with every exchange until its result does, so the page runs each
``request_id`` once and sends its result until the command no longer comes. ``active`` turns
false once the run ends; its discards still come until answered.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

MAX_TASK_CHARS = 200
MAX_EPISODES = 10_000
MAX_RECOVERY_FRACTION = 0.5
"""Largest share of bucket_sort episodes that start from a stuck pose."""
DEFAULT_MAX_ATTEMPTS = 2
"""The demonstrator's attempts per episode by default (``bucket_sort_demo.DEFAULT_MAX_ATTEMPTS``)."""
MAX_ATTEMPTS = 5
DEFAULT_TASK = "Sort the items: cracked or purple ones into reject, the others into the bin of their color."
CONNECT_TIMEOUT_S = 30.0
"""How long Studio's recording page may take to pick up a run (it checks ``/health`` about every second)."""
RELAY_SILENCE_S = 10.0
"""How long the recording page may stay silent before the run stops: it was closed, or lost its session."""
START_TIMEOUT_S = 15.0
SAVE_TIMEOUT_S = 120.0
SUCCESS_HOLD_S = 1.0
"""Simulated seconds a bucket_sort success must hold before the episode is saved: the object stays in
its bucket, and the episode ends with the arm settling instead of on the tick the object landed."""
SUCCESS_HOLD_TIMEOUT_S = 4.0
"""Simulated seconds after the demonstrator's success the object may take to hold, then the episode is
discarded; below the scene's 5 s success dwell, after which it starts the next layout."""
NO_STUDIO = "Studio did not answer: open its recording page for this robot, with a dataset and teleoperation."
"""Why a run stops when the recording page never picked it up, or stopped relaying."""

KeepPolicy = Literal["all", "perfect"]
RecorderPhase = Literal[
    "off",
    "connecting",
    "waiting",
    "resetting",
    "positioning",
    "starting",
    "recording",
    "saving",
    "done",
    "stopped",
    "error",
]


@dataclass(frozen=True)
class StudioTarget:
    """The Studio session the relaying page drives: its follower and leader, by name."""

    follower_name: str
    leader_name: str | None


# ----------------------------------------------------------------------
# The relay, between the simulation thread and Studio's recording page
# ----------------------------------------------------------------------


class StudioLinkLike(Protocol):
    """What the recorders need from a link; a `RelayLink` or a test double."""

    @property
    def phase(self) -> str:
        """``connecting``, ``connected``, ``closed`` or ``error``."""
        ...

    @property
    def error(self) -> str | None:
        """Why the link failed, if it did."""
        ...

    @property
    def target(self) -> StudioTarget | None:
        """The session the link relays for."""
        ...

    @property
    def state(self) -> dict[str, Any] | None:
        """Studio's session as the recorders read it: ``dataset_loaded``, ``is_recording``, ``follower_source``."""
        ...

    def start(self) -> None:
        """Start waiting for Studio."""
        ...

    def send(self, event: str, data: Mapping[str, Any] | None = None, request_id: str | None = None) -> None:
        """Queue one command for Studio."""
        ...

    def take_ack(self, request_id: str) -> dict[str, Any] | None:
        """Pop Studio's acknowledgement for `request_id`, once it arrived."""
        ...

    def close(self) -> None:
        """Stop relaying; Studio's session keeps running."""
        ...


_COMMANDS = {"start_recording": "start_episode", "save_episode": "save_episode", "discard_episode": "discard_episode"}
"""The recorders' commands, as Studio's session names them."""


@dataclass(frozen=True)
class RelayContext:
    """Who relays a run: one recording page (``client``), on one session and dataset."""

    client: str
    session_id: str | None
    dataset_id: str | None


class RelayLink:
    """One recording run's commands for Studio, and what the recording page relaying it answered.

    The simulation thread sends and reads; :meth:`StudioRelay.exchange` (the HTTP server's
    thread) hands the commands over and brings the session state and the results back.

    The first page to exchange owns the run, on its session and dataset; another page gets
    nothing, and the run fails if the owner turns up on another session or dataset. A command is
    handed over on every exchange until its result comes back, so one lost on the way is sent
    again; the page runs each request id once. Ending the run cancels what Studio has not
    answered yet, except discards: those clean up the episode the run started.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic, silence_s: float = RELAY_SILENCE_S) -> None:
        """Start idle; :meth:`start` waits for the first exchange."""
        self._clock = clock
        self._silence_s = silence_s
        self._lock = threading.Lock()
        self._started = False
        self._closed = False
        self._failure: str | None = None
        self._owner: RelayContext | None = None
        self._last_exchange: float | None = None
        self._session: dict[str, Any] | None = None
        self._outgoing: dict[str, dict[str, Any]] = {}
        """Commands handed over until their result comes back, by request id, oldest first."""
        self._acks: dict[str, dict[str, Any]] = {}

    @property
    def phase(self) -> str:
        """``idle``, ``connecting`` until a page answers, ``connected``, ``closed``, or ``error``."""
        with self._lock:
            return self._phase()

    def _phase(self) -> str:
        if self._closed:
            return "closed"
        if not self._started:
            return "idle"
        if self._failure is not None:
            return "error"
        if self._last_exchange is None:
            return "connecting"
        if self._clock() - self._last_exchange > self._silence_s:
            return "error"
        return "connected"

    @property
    def error(self) -> str | None:
        """Why the link failed: the page stopped relaying, or moved to another session or dataset."""
        with self._lock:
            if self._phase() != "error":
                return None
            return self._failure or NO_STUDIO

    @property
    def target(self) -> StudioTarget | None:
        """The follower and leader of the session the relaying page drives."""
        with self._lock:
            session = self._session
        if session is None or not session.get("follower"):
            return None
        leader = session.get("leader")
        return StudioTarget(str(session["follower"]), str(leader) if leader else None)

    @property
    def state(self) -> dict[str, Any] | None:
        """The relaying page's session, as the recorders read it; ``None`` before the first exchange."""
        with self._lock:
            session = self._session
        if session is None:
            return None
        return {
            "dataset_loaded": bool(session.get("dataset")),
            "is_recording": bool(session.get("recording")),
            "follower_source": "teleop" if session.get("teleop") else None,
        }

    @property
    def owned(self) -> bool:
        """Whether a page relays this run."""
        with self._lock:
            return self._owner is not None

    @property
    def pending(self) -> bool:
        """Whether commands still wait for Studio's answer."""
        with self._lock:
            return bool(self._outgoing)

    def start(self) -> None:
        """Wait for Studio's recording page."""
        with self._lock:
            self._started = True

    def send(self, event: str, data: Mapping[str, Any] | None = None, request_id: str | None = None) -> None:
        """Queue one command for the page, handed over until it answers.

        Raises:
            ValueError: For a command Studio's session does not take from the relay.
        """
        command = _COMMANDS.get(event)
        if command is None:
            msg = f"Studio's recording page relays no {event!r}"
            raise ValueError(msg)
        message: dict[str, Any] = {"command": command, "request_id": request_id or uuid.uuid4().hex}
        if command == "start_episode":
            message["task"] = str((data or {}).get("task", ""))
        with self._lock:
            if not self._closed:
                self._outgoing[message["request_id"]] = message

    def take_ack(self, request_id: str) -> dict[str, Any] | None:
        """Pop Studio's result for `request_id`, once a relay brought it.

        Returns:
            ``{"request_id", "ok", "error"}``, or ``None`` while still pending.
        """
        with self._lock:
            return self._acks.pop(request_id, None)

    def close(self) -> None:
        """Stop relaying: cancel what Studio has not answered yet, except discards, which still go out."""
        with self._lock:
            self._closed = True
            self._acks.clear()
            self._outgoing = {
                request_id: command
                for request_id, command in self._outgoing.items()
                if command["command"] == "discard_episode"
            }

    def exchange(
        self, context: RelayContext, session: Mapping[str, Any], results: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]] | None:
        """Take a page's session state and results, and hand it the commands Studio has not answered.

        Returns:
            The outstanding commands, oldest first; ``None`` for a page that does not own the run.
        """
        with self._lock:
            if self._owner is None and not self._closed and self._started:
                self._owner = context
            if self._owner is None or context.client != self._owner.client:
                return None
            if context != self._owner:
                # The page loaded another session or dataset: what this run asked for no longer applies.
                self._failure = "Studio's recording page moved to another session or dataset."
                self._closed = True
                self._outgoing.clear()
                return None
            for result in results:
                request_id = str(result.get("request_id", ""))
                if self._outgoing.pop(request_id, None) is not None and not self._closed:
                    self._acks[request_id] = {
                        "request_id": request_id,
                        "ok": bool(result.get("ok")),
                        "error": result.get("error"),
                    }
            if not self._closed:
                self._last_exchange = self._clock()
                self._session = dict(session)
            return list(self._outgoing.values())


MAX_ENDED_RUNS = 4
"""Ended runs whose discards are kept for their page; older ones are dropped."""


class StudioRelay:
    """``POST /studio/relay``: where Studio's recording page picks up the current run's commands."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, silence_s: float = RELAY_SILENCE_S) -> None:
        """Start with no run."""
        self._clock = clock
        self._silence_s = silence_s
        self._lock = threading.Lock()
        self._link: RelayLink | None = None
        self._ended: list[RelayLink] = []
        """Ended runs whose discards Studio has not answered yet."""

    def open(self) -> RelayLink:
        """Return a link for a new run; an ended run's discards still go out with the next exchanges."""
        link = RelayLink(self._clock, self._silence_s)
        with self._lock:
            if self._link is not None:
                self._link.close()
                self._ended.append(self._link)
                del self._ended[:-MAX_ENDED_RUNS]
            self._link = link
        return link

    def exchange(
        self, context: RelayContext, session: Mapping[str, Any], results: Iterable[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Relay one exchange with a recording page (the HTTP server's thread).

        Returns:
            ``{"active", "owner", "commands"}``: whether a run is under way, whether this page
            relays it, and the commands Studio has not answered yet, an ended run's discards first.
        """
        results = list(results)
        with self._lock:  # one exchange at a time, and never with a run being replaced
            commands: list[dict[str, Any]] = []
            owner = False
            for ended in list(self._ended):
                handed = ended.exchange(context, session, results)
                if handed is not None:
                    owner = True
                    commands.extend(handed)
                if not ended.pending or not ended.owned:  # answered, or no page ever relayed it
                    self._ended.remove(ended)
            link = self._link
            active = link is not None and link.phase not in {"idle", "closed"}
            if link is not None:
                handed = link.exchange(context, session, results)
                if handed is not None:
                    owner = True
                    commands.extend(handed)
                if link.phase == "closed" and link.pending:
                    self._ended.append(link)
                    self._link = None
            del self._ended[:-MAX_ENDED_RUNS]
            return {"active": active, "owner": owner, "commands": commands}


# ----------------------------------------------------------------------
# Episode logic, called from the simulation thread
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class RecordingOptions:
    """What to record and which episodes to keep."""

    task: str = DEFAULT_TASK
    keep: KeepPolicy = "perfect"
    max_episodes: int = 0
    """Stop after this many saved episodes; ``0`` records until switched off."""
    recovery_fraction: float = 0.0
    """bucket_sort: the share of episodes that start from a stuck pose (0 to :data:`MAX_RECOVERY_FRACTION`)."""
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    """bucket_sort: the demonstrator's attempts per episode, the retries included (1 to :data:`MAX_ATTEMPTS`)."""

    def __post_init__(self) -> None:
        """Reject values the HTTP layer would also reject.

        Raises:
            ValueError: For an empty or overlong task, an unknown keep policy, a bad budget, recovery
                share or attempt count.
        """
        if not 0.0 <= self.recovery_fraction <= MAX_RECOVERY_FRACTION:  # also rejects NaN
            msg = f"recovery_fraction must be 0..{MAX_RECOVERY_FRACTION}"
            raise ValueError(msg)
        if not 1 <= self.max_attempts <= MAX_ATTEMPTS:
            msg = f"max_attempts must be 1..{MAX_ATTEMPTS}"
            raise ValueError(msg)
        if not self.task.strip() or len(self.task) > MAX_TASK_CHARS:
            msg = f"task must be 1..{MAX_TASK_CHARS} characters"
            raise ValueError(msg)
        if self.keep not in {"all", "perfect"}:
            msg = f"keep must be 'all' or 'perfect', got {self.keep!r}"
            raise ValueError(msg)
        if not 0 <= self.max_episodes <= MAX_EPISODES:
            msg = f"max_episodes must be 0..{MAX_EPISODES}"
            raise ValueError(msg)


@dataclass(frozen=True)
class RecorderDirective:
    """What the simulation must do this tick."""

    hold_feed: bool = False
    """Keep the belt and the item feed stopped."""
    clear_belt: bool = False
    """Park every item and restart the conveyor episode now (one tick only)."""


@dataclass
class _Counters:
    saved: int = 0
    discarded: int = 0
    scores: list[dict[str, int]] = field(default_factory=list)


class _RecorderBase:
    """The link, the options, the phase and the counters every scene's recorder shares."""

    def __init__(self, link_factory: Callable[[], StudioLinkLike], clock: Callable[[], float] = time.monotonic) -> None:
        """Use `link_factory` to open a link each time recording is switched on."""
        self._link_factory = link_factory
        self._clock = clock
        self._link: StudioLinkLike | None = None
        self._options = RecordingOptions()
        self._phase: RecorderPhase = "off"
        self._message = ""
        self._since = 0.0
        self._episode_base = 0
        self._request_id: str | None = None
        self._pending_keep = True
        self._counters = _Counters()

    @property
    def phase(self) -> RecorderPhase:
        """Where the recorder is in its episode cycle."""
        return self._phase

    @property
    def options(self) -> RecordingOptions:
        """The options of the current (or last) run."""
        return self._options

    @property
    def active(self) -> bool:
        """Whether a run is under way: switched on and not yet done, stopped or failed."""
        return self._phase not in {"off", "done", "stopped", "error"}

    def enable(self, options: RecordingOptions) -> None:
        """Start recording episodes with `options`, replacing any previous run."""
        self.disable()
        self._options = options
        self._counters = _Counters()
        self._link = self._link_factory()
        self._link.start()
        self._set_phase("connecting", "Waiting for Studio's recording page...")

    def disable(self, message: str = "", *, discard: bool = False) -> None:
        """Stop recording; an episode Studio is still recording stays for the user to save or discard.

        Args:
            message: Shown in the viewer and ``/health``.
            discard: Discard the episode Studio records instead, when the scene it shows is gone.
        """
        link = self._link
        if discard and link is not None and link.phase == "connected" and self._phase in {"starting", "recording"}:
            link.send("discard_episode", request_id=uuid.uuid4().hex)  # sent before the link closes
        self._close_link()
        if self._phase not in {"done", "stopped", "error"} or message:
            self._set_phase("off", message)

    def status(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot for ``/health`` and the viewer."""
        link = self._link
        target = link.target if link is not None else None
        return {
            "phase": self._phase,
            "message": self._message,
            "task": self._options.task,
            "keep": self._options.keep,
            "max_episodes": self._options.max_episodes,
            "recovery_fraction": self._options.recovery_fraction,
            "max_attempts": self._options.max_attempts,
            "saved": self._counters.saved,
            "discarded": self._counters.discarded,
            "follower": target.follower_name if target is not None else None,
            "leader": target.leader_name if target is not None else None,
        }

    # -- helpers --------------------------------------------------------

    def _set_phase(self, phase: RecorderPhase, message: str) -> None:
        self._phase, self._message, self._since = phase, message, self._clock()

    def _finish(self, phase: RecorderPhase, message: str) -> None:
        logger.info("Studio recording {}: {}", phase, message)
        self._close_link()
        self._set_phase(phase, message)

    def _close_link(self) -> None:
        if self._link is not None:
            self._link.close()
        self._link = None

    def _check_save(self, link: StudioLinkLike) -> Literal["pending", "ended", "next"]:
        """Count Studio's answer to the save or discard in flight.

        Returns:
            ``pending`` while Studio has not answered, ``ended`` once the run stopped (an error or
            the episode budget used up), ``next`` to start the next episode.
        """
        ack = link.take_ack(self._request_id or "")
        if ack is None:
            if self._clock() - self._since > SAVE_TIMEOUT_S:
                self._finish("error", "Studio did not confirm saving the episode.")
                return "ended"
            return "pending"
        if not ack.get("ok"):
            self._finish("error", f"Studio could not save the episode: {ack.get('error') or 'unknown error'}")
            return "ended"
        if self._pending_keep:
            self._counters.saved += 1
        else:
            self._counters.discarded += 1
        budget = self._options.max_episodes
        if budget and self._counters.saved >= budget:
            self._finish("done", f"Recorded {self._counters.saved} episodes.")
            return "ended"
        return "next"

    def _start_went_through(self, link: StudioLinkLike) -> bool:
        """Read Studio's answer to the start in flight; stop with its reason if it refused.

        Returns:
            ``False`` once the run stopped.
        """
        ack = link.take_ack(self._request_id or "")
        if ack is not None and not ack.get("ok"):
            self._finish("error", f"Studio could not start the episode: {ack.get('error') or 'unknown error'}")
            return False
        return True

    def _send_end(self, link: StudioLinkLike, *, keep: bool, message: str) -> None:
        """Ask Studio to save (`keep`) or discard the episode it records, and wait for its answer."""
        self._request_id = uuid.uuid4().hex
        self._pending_keep = keep
        link.send("save_episode" if keep else "discard_episode", request_id=self._request_id)
        self._set_phase("saving", message)


class AutoRecorder(_RecorderBase):
    """Turn conveyor episodes into Studio episodes, one tick at a time."""

    def restart_episode(self, reason: str) -> None:
        """Discard the Studio episode being recorded, because the scene jumped (a reset or a reload).

        The episode then holds a teleport, and a reloaded conveyor counts its
        episodes from zero again. After the discard the cycle starts a fresh
        episode as usual. Other phases need nothing: an episode's start is
        taken when Studio begins recording it.
        """
        if self._phase != "recording" or self._link is None:
            return
        self._request_id = uuid.uuid4().hex
        self._pending_keep = False
        self._link.send("discard_episode", request_id=self._request_id)
        self._set_phase("saving", f"{reason} Discarding the episode in progress...")

    def update(self, episode_count: int, last_episode: Mapping[str, int] | None) -> RecorderDirective:
        """Advance the cycle for one simulation tick.

        Returns:
            Whether to hold the feed and whether to clear the belt now.
        """
        link = self._link
        if link is None or self._phase in {"off", "done", "stopped", "error"}:
            return RecorderDirective()
        if link.phase in {"error", "closed"}:
            self._finish("error", link.error or "Studio stopped relaying.")
            return RecorderDirective()
        state = link.state
        if self._phase == "connecting":
            if link.phase == "connected" and state is not None:
                self._set_phase("waiting", "")
            elif self._clock() - self._since > CONNECT_TIMEOUT_S:
                self._finish("error", NO_STUDIO)
                return RecorderDirective()
            return RecorderDirective(hold_feed=True)
        if state is None:
            return RecorderDirective(hold_feed=True)
        handler = {
            "waiting": self._update_waiting,
            "starting": self._update_starting,
            "recording": self._update_recording,
            "saving": self._update_saving,
        }[self._phase]
        return handler(link, state, episode_count, last_episode)

    # -- phases ---------------------------------------------------------

    def _update_waiting(
        self, link: StudioLinkLike, state: Mapping[str, Any], episode_count: int, _last: Mapping[str, int] | None
    ) -> RecorderDirective:
        if not state.get("dataset_loaded"):
            self._message = "Waiting for Studio: open a dataset in the recording view."
        elif state.get("follower_source") != "teleop":
            self._message = "Waiting for Studio: turn teleoperation on."
        elif state.get("is_recording"):
            self._message = "Waiting for Studio: save or discard the episode it is recording."
        else:
            self._request_id = uuid.uuid4().hex
            link.send("start_recording", {"task": self._options.task}, request_id=self._request_id)
            self._episode_base = episode_count
            self._set_phase("starting", "Starting a Studio episode...")
            return RecorderDirective(hold_feed=True, clear_belt=True)
        return RecorderDirective(hold_feed=True)

    def _update_starting(
        self, link: StudioLinkLike, state: Mapping[str, Any], episode_count: int, _last: Mapping[str, int] | None
    ) -> RecorderDirective:
        if not self._start_went_through(link):
            return RecorderDirective()
        if state.get("is_recording"):
            self._episode_base = episode_count
            self._set_phase("recording", f"Recording episode {self._counters.saved + 1}.")
            return RecorderDirective()
        if self._clock() - self._since > START_TIMEOUT_S:
            self._finish("error", "Studio did not start recording; is a dataset open?")
            return RecorderDirective()
        return RecorderDirective(hold_feed=True)

    def _update_recording(
        self, link: StudioLinkLike, state: Mapping[str, Any], episode_count: int, last: Mapping[str, int] | None
    ) -> RecorderDirective:
        if not state.get("is_recording"):
            self._finish("stopped", "Studio stopped the recording, so automatic recording stopped too.")
            return RecorderDirective()
        if episode_count <= self._episode_base:
            return RecorderDirective()
        score = dict(last or {})
        keep = self._options.keep == "all" or (score.get("wrong", 0) == 0 and score.get("missed", 0) == 0)
        self._request_id = uuid.uuid4().hex
        self._pending_keep = keep
        self._counters.scores.append(score)
        link.send("save_episode" if keep else "discard_episode", request_id=self._request_id)
        self._set_phase("saving", "Saving the episode..." if keep else "Discarding an episode with mistakes...")
        return RecorderDirective(hold_feed=True)

    def _update_saving(
        self, link: StudioLinkLike, _state: Mapping[str, Any], _count: int, _last: Mapping[str, int] | None
    ) -> RecorderDirective:
        result = self._check_save(link)
        if result == "ended":
            return RecorderDirective()
        if result == "next":
            self._set_phase("waiting", "")
        return RecorderDirective(hold_feed=True)


# ----------------------------------------------------------------------
# bucket_sort: the autopilot leads Studio's teleoperation
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BucketSortProgress:
    """What the bucket_sort recorder reads from the scene each tick."""

    generation: tuple[int, int]
    """Identifies the scene layout: the task object and its reset count. A change means a reset."""
    instruction: str | None
    """The current episode's worded instruction, the Studio episode's task."""
    leading: bool
    """Whether the autopilot leads Studio's teleoperation (mode ``leader``)."""
    outcome: str | None = None
    """The demonstrator's episode outcome: ``running``, ``success``, ``failed``, ``stalled`` or ``None``."""
    clean: bool | None = None
    """Whether no attempt of the episode went wrong."""
    placed: bool = False
    """Whether the task's success holds this tick (the named object resting in the named bucket)."""
    sim_time: float = 0.0
    """Simulated time in seconds, which the success hold counts in (the recorded episode's clock)."""
    stuck_start: StuckStartState | None = None
    """How the move to a stuck start goes, after :attr:`BucketSortDirective.stuck_start`."""


StuckStartState = Literal["planning", "moving", "ready", "disturbed", "unavailable"]
"""``planning`` the move (on its own thread; not counted against the timeout); ``moving`` to the stuck pose;
``ready`` there; ``disturbed``: the move moved an object or a bucket
more than 5 mm; ``unavailable``: no stuck pose for this layout (the arm did not move)."""


@dataclass(frozen=True)
class BucketSortDirective:
    """What the bucket_sort scene must do this tick."""

    hold_arm: bool = False
    """Keep the autopilot paused (not stepped), so the arm waits for Studio."""
    reset_scene: bool = False
    """Reset the scene now, for a fresh episode (one tick only)."""
    stuck_start: bool = False
    """Move the arm to a stuck pose now, before the episode starts (one tick only; :class:`BucketSortProgress`)."""
    abort_stuck_start: bool = False
    """Stop the move to a stuck pose now and give the arm back to client actions (one tick only)."""


def _studio_ready(state: Mapping[str, Any]) -> bool:
    """Return whether Studio can start a bucket_sort episode: a dataset open, teleoperating, not recording."""
    return (
        bool(state.get("dataset_loaded")) and not state.get("is_recording") and state.get("follower_source") == "teleop"
    )


class BucketSortRecorder(_RecorderBase):
    """Turn bucket_sort episodes that the autopilot leads into Studio teleoperation episodes.

    Each Studio episode starts from a scene reset, with the episode's instruction as its task,
    and ends on the demonstrator's outcome. The arm waits (the autopilot is not stepped) whenever
    no Studio episode is recording, so every episode is recorded from its first move.

    With ``recovery_fraction``, that share of episodes starts from a stuck pose: after the reset,
    the scene moves the arm there (not recorded), and the episode starts once it is there. A move
    that disturbed the layout resets the scene again for a normal start.
    """

    def __init__(
        self,
        link_factory: Callable[[], StudioLinkLike],
        clock: Callable[[], float] = time.monotonic,
        chance: Callable[[], float] | None = None,
    ) -> None:
        """Use `link_factory` to open a link each time recording is switched on.

        Args:
            link_factory: Opens a link to Studio.
            clock: Wall-clock seconds, for the timeouts.
            chance: Draws a number in [0, 1) per episode; below ``recovery_fraction`` the episode
                starts stuck. ``None`` uses a fresh generator.
        """
        super().__init__(link_factory, clock)
        if chance is None:
            import numpy as np  # noqa: PLC0415

            chance = np.random.default_rng().random  # which episodes start stuck, not a secret
        self._chance = chance
        self._normal_start = False
        """The next start is normal: the stuck start of this episode disturbed the layout."""
        self._stuck_starts = 0
        self._task: str | None = None
        self._generation: tuple[int, int] | None = None
        self._success_at: float | None = None
        """When the demonstrator reported this episode's success."""
        self._placed_since: float | None = None
        """Since when the success holds without a break."""

    def enable(self, options: RecordingOptions) -> None:
        """Start recording episodes with `options` (its task is ignored: each instruction is the task)."""
        super().enable(options)
        self._task = None
        self._generation = None
        self._normal_start = False
        self._stuck_starts = 0

    def status(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot for ``/health`` and the viewer; ``task`` is the episode's instruction."""
        return {
            **super().status(),
            "task": self._task,
            "task_source": "instruction",
            "stuck_starts": self._stuck_starts,
        }

    def update(self, progress: BucketSortProgress) -> BucketSortDirective:
        """Advance the cycle for one simulation tick.

        Returns:
            Whether to keep the arm waiting and whether to reset the scene now.
        """
        link = self._link
        if link is None or not self.active:
            return BucketSortDirective()
        if link.phase in {"error", "closed"}:
            self._finish("error", link.error or "Studio stopped relaying.")
            return BucketSortDirective()
        state = link.state
        if self._phase == "connecting":
            return self._update_connecting(link)
        if state is None:
            return BucketSortDirective(hold_arm=True)
        handler = {
            "waiting": self._update_waiting,
            "resetting": self._update_resetting,
            "positioning": self._update_positioning,
            "starting": self._update_starting,
            "recording": self._update_recording,
            "saving": self._update_saving,
        }[self._phase]
        return handler(link, state, progress)

    # -- phases ---------------------------------------------------------

    def _update_waiting(
        self, _link: StudioLinkLike, state: Mapping[str, Any], progress: BucketSortProgress
    ) -> BucketSortDirective:
        if not state.get("dataset_loaded"):
            self._message = "Waiting for Studio: open a dataset in the recording view."
        elif state.get("is_recording"):
            self._message = "Waiting for Studio: save or discard the episode it is recording."
        elif state.get("follower_source") != "teleop":
            self._message = "Waiting for Studio: turn teleoperation on, with the autopilot as leader."
        elif not progress.leading:
            self._message = "Waiting for the autopilot to lead: teleoperate with the autopilot as leader in Studio."
        else:
            self._generation = progress.generation
            self._set_phase("resetting", "Resetting the scene for a new episode...")
            return BucketSortDirective(hold_arm=True, reset_scene=True)
        return BucketSortDirective(hold_arm=True)

    def _update_connecting(self, link: StudioLinkLike) -> BucketSortDirective:
        if link.phase == "connected" and link.state is not None:
            self._set_phase("waiting", "")
        elif self._clock() - self._since > CONNECT_TIMEOUT_S:
            self._finish("error", NO_STUDIO)
            return BucketSortDirective()
        return BucketSortDirective(hold_arm=True)

    def _update_resetting(
        self, link: StudioLinkLike, _state: Mapping[str, Any], progress: BucketSortProgress
    ) -> BucketSortDirective:
        if progress.generation == self._generation or not progress.instruction:
            if self._clock() - self._since > START_TIMEOUT_S:
                self._finish("error", "The scene did not reset for a new episode.")
                return BucketSortDirective()
            return BucketSortDirective(hold_arm=True)
        self._generation = progress.generation
        fraction = self._options.recovery_fraction
        if not self._normal_start and fraction > 0.0 and self._chance() < fraction:
            self._set_phase("positioning", "Moving the arm to a stuck pose for a recovery episode...")
            return BucketSortDirective(hold_arm=True, stuck_start=True)
        self._normal_start = False
        return self._start(link, progress)

    def _start(self, link: StudioLinkLike, progress: BucketSortProgress) -> BucketSortDirective:
        self._task = (progress.instruction or "")[:MAX_TASK_CHARS]
        self._request_id = uuid.uuid4().hex
        link.send("start_recording", {"task": self._task}, request_id=self._request_id)
        self._set_phase("starting", "Starting a Studio episode...")
        return BucketSortDirective(hold_arm=True)

    def _update_positioning(
        self, link: StudioLinkLike, state: Mapping[str, Any], progress: BucketSortProgress
    ) -> BucketSortDirective:
        """Wait for the move to the stuck pose; start there, or start normally.

        The move stops, and the cycle waits again, as soon as the episode could not start: the
        autopilot no longer leads, or Studio no longer teleoperates, has no dataset or records.

        Returns:
            Hold the arm; reset the scene again when the move disturbed the layout or timed out;
            abort the move when the episode could not start.
        """
        if not progress.leading or not _studio_ready(state):
            self._set_phase("waiting", "")
            return replace(self._update_waiting(link, state, progress), abort_stuck_start=True)
        if progress.generation != self._generation:
            self._set_phase("resetting", "The scene was reset. Resetting it again for a new episode...")
            return BucketSortDirective(hold_arm=True, reset_scene=True)
        if progress.stuck_start in {"ready", "unavailable"}:
            # Unavailable: the arm did not move, so the episode starts normally as the scene stands.
            self._stuck_starts += progress.stuck_start == "ready"
            return self._start(link, progress)
        if progress.stuck_start == "planning":
            # Planning takes up to seconds on its own thread: the timeout counts from when it is done.
            self._since = self._clock()
            return BucketSortDirective(hold_arm=True)
        if progress.stuck_start == "disturbed":
            # A start with something knocked out of place is not a clean stuck state: lay it out anew.
            message = "The move to the stuck pose disturbed the layout. Resetting the scene..."
        elif self._clock() - self._since > START_TIMEOUT_S:
            message = "The arm did not reach the stuck pose. Resetting the scene..."
        else:
            return BucketSortDirective(hold_arm=True)
        self._normal_start = True
        self._set_phase("resetting", message)
        return BucketSortDirective(hold_arm=True, reset_scene=True)

    def _update_starting(
        self, link: StudioLinkLike, state: Mapping[str, Any], _progress: BucketSortProgress
    ) -> BucketSortDirective:
        if not self._start_went_through(link):
            return BucketSortDirective()
        if state.get("is_recording"):
            self._set_phase("recording", f"Recording episode {self._counters.saved + 1}: {self._task}")
            self._success_at = self._placed_since = None
            return BucketSortDirective()
        if self._clock() - self._since > START_TIMEOUT_S:
            self._finish("error", "Studio did not start recording; is a dataset open?")
            return BucketSortDirective()
        return BucketSortDirective(hold_arm=True)

    def _update_recording(
        self, link: StudioLinkLike, state: Mapping[str, Any], progress: BucketSortProgress
    ) -> BucketSortDirective:
        if not state.get("is_recording"):
            self._finish("stopped", "Studio stopped the recording, so automatic recording stopped too.")
            return BucketSortDirective()
        if state.get("follower_source") != "teleop":
            self._finish("stopped", "Studio's follower left teleoperation, so automatic recording stopped too.")
            return BucketSortDirective()
        if progress.generation != self._generation:
            self._send_end(link, keep=False, message="The scene was reset. Discarding the episode in progress...")
        elif not progress.leading:
            self._send_end(link, keep=False, message="The autopilot stopped leading. Discarding the episode...")
        elif progress.outcome == "success":
            return self._hold_success(link, progress)
        elif progress.outcome in {"failed", "stalled"}:
            self._send_end(link, keep=False, message=f"Discarding a {progress.outcome} episode...")
        else:
            return BucketSortDirective()
        return BucketSortDirective(hold_arm=True)

    def _hold_success(self, link: StudioLinkLike, progress: BucketSortProgress) -> BucketSortDirective:
        """End a success once it has held for `SUCCESS_HOLD_S`; the arm keeps moving meanwhile.

        Returns:
            Hold the arm once the episode ends, else let it move.
        """
        now = progress.sim_time
        if self._success_at is None:
            self._success_at = now
        if not progress.placed:
            self._placed_since = None
        elif self._placed_since is None:
            self._placed_since = now
        if self._placed_since is not None and now - self._placed_since >= SUCCESS_HOLD_S:
            keep = self._options.keep == "all" or progress.clean is True
            self._send_end(
                link, keep=keep, message="Saving the episode..." if keep else "Discarding a success with a retry..."
            )
        elif now - self._success_at > SUCCESS_HOLD_TIMEOUT_S:
            self._send_end(link, keep=False, message="The object did not stay in its bucket. Discarding the episode...")
        else:
            return BucketSortDirective()
        return BucketSortDirective(hold_arm=True)

    def _update_saving(
        self, link: StudioLinkLike, _state: Mapping[str, Any], _progress: BucketSortProgress
    ) -> BucketSortDirective:
        if self._check_save(link) == "next":
            self._set_phase("waiting", "")
        return BucketSortDirective(hold_arm=self.active)
