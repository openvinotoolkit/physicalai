# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Save conveyor episodes into a Physical AI Studio dataset automatically.

Studio has no episode REST API; its recording view drives a runtime session
over ``/api/projects/{project}/runtime/ws``. A second client that sends the
same handshake (same follower and leader) attaches to the live session instead
of starting a new one, receives its state events, and may send the same
``start_recording`` / ``save_episode`` / ``discard_episode`` commands the
browser sends. This module is that client, plus the episode logic around it:

1. Find the live session for this simulation's follower: ``GET
   /api/runtime/sessions``, then the project whose robot list holds it.
2. Wait until Studio is ready: a dataset loaded, teleoperation running, and no
   recording in progress.
3. Clear the belt, ``start_recording`` with the task, and let the belt run.
4. When the conveyor episode ends, hold the feed, then save the episode (or
   discard it when it had mistakes and only perfect episodes are kept).
5. Repeat from 2 until switched off or the episode budget is used up.

The client never sends ``disconnect``: that would stop Studio's session. It
sends ``camera_ids: []`` so attaching never restarts the session for cameras.
"""

from __future__ import annotations

import http.client
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol
from urllib.parse import urlsplit

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

DEFAULT_STUDIO_URL = "http://127.0.0.1:7860"
MAX_TASK_CHARS = 200
MAX_EPISODES = 10_000
FOLLOWER_TYPES = frozenset({"MuJoCo_SO101_Follower"})
DEFAULT_TASK = "Sort the items: cracked or purple ones into reject, the others into the bin of their color."
CONNECT_TIMEOUT_S = 30.0
"""Discovery (a few 3 s requests), the 5 s websocket open and Studio's first state all fit well inside."""
START_TIMEOUT_S = 15.0
SAVE_TIMEOUT_S = 120.0

KeepPolicy = Literal["all", "perfect"]
RecorderPhase = Literal["off", "connecting", "waiting", "starting", "recording", "saving", "done", "stopped", "error"]


class StudioError(RuntimeError):
    """Studio is unreachable, or has no session this simulation can join."""


# ----------------------------------------------------------------------
# Discovery over REST
# ----------------------------------------------------------------------


def validate_studio_url(url: str) -> str:
    """Accept an ``http(s)://host[:port]`` base URL with no path, query or credentials.

    Returns:
        The URL without a trailing slash.

    Raises:
        ValueError: For any other shape.
    """
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        msg = f"Studio URL must look like http://host:port, got {url!r}"
        raise ValueError(msg)
    if parts.username or parts.password or parts.query or parts.fragment or parts.path not in {"", "/"}:
        msg = f"Studio URL must not carry a path, query or credentials: {url!r}"
        raise ValueError(msg)
    return f"{parts.scheme}://{parts.netloc}"


def _get_json(base_url: str, path: str, timeout_s: float = 3.0) -> Any:  # noqa: ANN401 - JSON
    parts = urlsplit(base_url)
    connection_type = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    conn = connection_type(parts.hostname or "", parts.port, timeout=timeout_s)
    try:
        conn.request("GET", path, headers={"Accept": "application/json"})
        response = conn.getresponse()
        body = response.read()
    except (OSError, http.client.HTTPException) as exc:
        msg = f"Studio is not reachable at {base_url}: {exc}"
        raise StudioError(msg) from exc
    finally:
        conn.close()
    if response.status != 200:  # noqa: PLR2004
        msg = f"Studio answered HTTP {response.status} for {path}"
        raise StudioError(msg)
    try:
        return json.loads(body)
    except ValueError as exc:
        msg = f"Studio answered {path} with invalid JSON"
        raise StudioError(msg) from exc


@dataclass(frozen=True)
class StudioTarget:
    """The Studio project, follower and leader of the live session to join."""

    project_id: str
    follower_id: str
    follower_name: str
    leader_id: str | None
    leader_name: str | None


def discover_session(
    base_url: str,
    owner_name: str,
    *,
    get_json: Callable[[str, str], Any] = _get_json,
) -> StudioTarget:
    """Find the live Studio session that drives this simulation's follower.

    Returns:
        Where to attach.

    Raises:
        StudioError: If Studio is unreachable or no live session drives this simulation.
    """
    sessions = {
        str(session["follower_id"]): session
        for session in get_json(base_url, "/api/runtime/sessions")
        if session.get("follower_id") and session.get("status") in {"starting", "running"}
    }
    if not sessions:
        msg = "Studio has no live robot session. Open the MuJoCo follower in Studio's recording view first."
        raise StudioError(msg)
    for project in get_json(base_url, "/api/projects"):
        project_id = str(project["id"])
        robots = get_json(base_url, f"/api/projects/{project_id}/robots")
        for robot in robots:
            session = sessions.get(str(robot.get("id")))
            payload = robot.get("payload") or {}
            if session is None or robot.get("type") not in FOLLOWER_TYPES or payload.get("name") != owner_name:
                continue
            leader_name = session.get("leader_name")
            leader_id = None
            if leader_name:
                leader_id = next(
                    (str(r["id"]) for r in robots if r.get("name") == leader_name and r.get("id") != robot.get("id")),
                    None,
                )
                if leader_id is None:
                    msg = f"Studio's session uses leader {leader_name!r}, which is not in the follower's project"
                    raise StudioError(msg)
            return StudioTarget(
                project_id=project_id,
                follower_id=str(robot["id"]),
                follower_name=str(robot.get("name", "")),
                leader_id=leader_id,
                leader_name=leader_name,
            )
    msg = f"No live Studio session drives the simulation {owner_name!r}. Open it in Studio's recording view first."
    raise StudioError(msg)


# ----------------------------------------------------------------------
# The websocket link
# ----------------------------------------------------------------------


class StudioLinkLike(Protocol):
    """What `AutoRecorder` needs from a link; `StudioLink` or a test double."""

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
        """The session the link attached to."""
        ...

    @property
    def state(self) -> dict[str, Any] | None:
        """The latest Studio state event payload."""
        ...

    def start(self) -> None:
        """Connect in the background."""
        ...

    def send(self, event: str, data: Mapping[str, Any] | None = None, request_id: str | None = None) -> None:
        """Queue one command for Studio."""
        ...

    def take_ack(self, request_id: str) -> dict[str, Any] | None:
        """Pop Studio's acknowledgement for `request_id`, once it arrived."""
        ...

    def close(self) -> None:
        """Detach from Studio; its session keeps running."""
        ...


class StudioLink:
    """Background thread that attaches to Studio's runtime websocket and relays commands and events."""

    def __init__(
        self,
        base_url: str,
        owner_name: str,
        *,
        discover: Callable[[str, str], StudioTarget] = discover_session,
        connect: Callable[[str], Any] | None = None,
    ) -> None:
        """Prepare a link; `start` connects on a background thread."""
        self._base_url = validate_studio_url(base_url)
        self._owner_name = owner_name
        self._discover = discover
        self._connect = connect
        self._lock = threading.Lock()
        self._phase = "idle"
        self._error: str | None = None
        self._target: StudioTarget | None = None
        self._state: dict[str, Any] | None = None
        self._acks: dict[str, dict[str, Any]] = {}
        self._outgoing: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def phase(self) -> str:
        """``idle``, ``connecting``, ``connected``, ``closed`` or ``error``."""
        with self._lock:
            return self._phase

    @property
    def error(self) -> str | None:
        """Why the link failed, if it did."""
        with self._lock:
            return self._error

    @property
    def target(self) -> StudioTarget | None:
        """The session this link attached to."""
        with self._lock:
            return self._target

    @property
    def state(self) -> dict[str, Any] | None:
        """The latest Studio state event payload (``is_recording``, ``dataset_loaded``, ...)."""
        with self._lock:
            return None if self._state is None else dict(self._state)

    def start(self) -> None:
        """Connect on a background thread."""
        with self._lock:
            self._phase = "connecting"
        self._thread = threading.Thread(target=self._run, name="studio-link", daemon=True)
        self._thread.start()

    def send(self, event: str, data: Mapping[str, Any] | None = None, request_id: str | None = None) -> None:
        """Queue one command for Studio."""
        message: dict[str, Any] = {"event": event}
        if data is not None:
            message["data"] = dict(data)
        if request_id is not None:
            message["request_id"] = request_id
        self._outgoing.put(message)

    def take_ack(self, request_id: str) -> dict[str, Any] | None:
        """Pop Studio's acknowledgement for `request_id`, once it arrived.

        Returns:
            ``{"request_id", "ok", "error"}``, or ``None`` while still pending.
        """
        with self._lock:
            return self._acks.pop(request_id, None)

    def close(self, timeout_s: float = 2.0) -> None:
        """Detach from Studio; its session keeps running."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_s)

    def _fail(self, message: str) -> None:
        logger.warning("Studio link: {}", message)
        with self._lock:
            self._phase, self._error = "error", message

    def _run(self) -> None:
        try:
            target = self._discover(self._base_url, self._owner_name)
        except StudioError as exc:
            self._fail(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - e.g. a response shape from another Studio version
            self._fail(f"Could not read Studio's sessions: {exc!r}")
            return
        with self._lock:
            self._target = target
        ws_url = self._base_url.replace("http", "ws", 1) + f"/api/projects/{target.project_id}/runtime/ws"
        try:
            ws = self._open(ws_url)
        except Exception as exc:  # noqa: BLE001 - refused, bad URL or handshake rejected: all end the link
            self._fail(f"Could not open {ws_url}: {exc}")
            return
        try:
            ws.send(json.dumps({"follower_id": target.follower_id, "leader_id": target.leader_id, "camera_ids": []}))
            with self._lock:
                self._phase = "connected"
            logger.info("Attached to Studio session for {!r} (leader {!r})", target.follower_name, target.leader_name)
            self._pump(ws)
        except Exception as exc:  # noqa: BLE001 - any websocket failure ends the link the same way
            self._fail(f"Studio connection lost: {exc}")
        finally:
            ws.close()
        with self._lock:
            if self._phase == "connected":
                self._phase = "closed"

    def _open(self, ws_url: str) -> Any:  # noqa: ANN401 - a websockets connection
        if self._connect is not None:
            return self._connect(ws_url)
        from websockets.sync.client import connect  # noqa: PLC0415

        return connect(ws_url, open_timeout=5.0, max_size=2**22)

    def _pump(self, ws: Any) -> None:  # noqa: ANN401 - a websockets connection
        while not self._stop.is_set():
            while True:
                try:
                    ws.send(json.dumps(self._outgoing.get_nowait()))
                except queue.Empty:
                    break
            try:
                raw = ws.recv(timeout=0.05)
            except TimeoutError:
                continue
            self._handle(json.loads(raw))

    def _handle(self, message: Mapping[str, Any]) -> None:
        event = message.get("event")
        if event == "state":
            with self._lock:
                self._state = dict(message.get("data") or {})
        elif event == "ack":
            data = dict(message.get("data") or {})
            with self._lock:
                self._acks[str(data.get("request_id"))] = data
        elif event == "error":
            logger.warning("Studio reported: {} ({})", message.get("message"), message.get("error_code"))


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

    def __post_init__(self) -> None:
        """Reject values the HTTP layer would also reject.

        Raises:
            ValueError: For an empty or overlong task, an unknown keep policy or a bad budget.
        """
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


class AutoRecorder:
    """Turn conveyor episodes into Studio episodes, one tick at a time."""

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

    def enable(self, options: RecordingOptions) -> None:
        """Start recording episodes with `options`, replacing any previous run."""
        self.disable()
        self._options = options
        self._counters = _Counters()
        self._link = self._link_factory()
        self._link.start()
        self._set_phase("connecting", "Looking for Studio's session...")

    def disable(self, message: str = "") -> None:
        """Stop recording; an episode Studio is still recording stays for the user to save or discard."""
        if self._link is not None:
            self._link.close()
        self._link = None
        if self._phase not in {"done", "stopped", "error"} or message:
            self._set_phase("off", message)

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
            self._finish("error", link.error or "Studio closed the connection.")
            return RecorderDirective()
        state = link.state
        if self._phase == "connecting":
            if link.phase == "connected" and state is not None:
                self._set_phase("waiting", "")
            elif self._clock() - self._since > CONNECT_TIMEOUT_S:
                reason = "sent no session state" if link.phase == "connected" else "did not answer in time"
                self._finish("error", f"Studio {reason}; is the robot session running?")
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
            "saved": self._counters.saved,
            "discarded": self._counters.discarded,
            "follower": target.follower_name if target is not None else None,
            "leader": target.leader_name if target is not None else None,
        }

    # -- phases ---------------------------------------------------------

    def _update_waiting(
        self, link: StudioLinkLike, state: Mapping[str, Any], episode_count: int, _last: Mapping[str, int] | None
    ) -> RecorderDirective:
        if not state.get("dataset_loaded"):
            self._message = "Waiting for Studio: open a dataset in the recording view."
        elif state.get("follower_source") != "teleop":
            self._message = "Waiting for Studio: start teleoperation (with the virtual leader for the autopilot)."
        elif state.get("is_recording"):
            self._message = "Waiting for Studio: save or discard the episode it is recording."
        else:
            self._request_id = None
            link.send("start_recording", {"task": self._options.task})
            self._episode_base = episode_count
            self._set_phase("starting", "Starting a Studio episode...")
            return RecorderDirective(hold_feed=True, clear_belt=True)
        return RecorderDirective(hold_feed=True)

    def _update_starting(
        self, _link: StudioLinkLike, state: Mapping[str, Any], episode_count: int, _last: Mapping[str, int] | None
    ) -> RecorderDirective:
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
        ack = link.take_ack(self._request_id or "")
        if ack is None:
            if self._clock() - self._since > SAVE_TIMEOUT_S:
                self._finish("error", "Studio did not confirm saving the episode.")
                return RecorderDirective()
            return RecorderDirective(hold_feed=True)
        if not ack.get("ok"):
            self._finish("error", f"Studio could not save the episode: {ack.get('error') or 'unknown error'}")
            return RecorderDirective()
        if self._pending_keep:
            self._counters.saved += 1
        else:
            self._counters.discarded += 1
        budget = self._options.max_episodes
        if budget and self._counters.saved >= budget:
            self._finish("done", f"Recorded {self._counters.saved} episodes.")
            return RecorderDirective()
        self._set_phase("waiting", "")
        return RecorderDirective(hold_feed=True)

    # -- helpers --------------------------------------------------------

    def _set_phase(self, phase: RecorderPhase, message: str) -> None:
        self._phase, self._message, self._since = phase, message, self._clock()

    def _finish(self, phase: RecorderPhase, message: str) -> None:
        logger.info("Studio recording {}: {}", phase, message)
        if self._link is not None:
            self._link.close()
        self._link = None
        self._set_phase(phase, message)
