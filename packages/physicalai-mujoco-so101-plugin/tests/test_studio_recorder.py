"""Automatic Studio episode recording: discovery, the websocket link and the episode cycle."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from websockets.sync.server import serve

from physicalai_mujoco_so101_plugin.http_server import SetStudioRecordingCommand
from physicalai_mujoco_so101_plugin.mujoco_robot import MuJoCoSO101
from physicalai_mujoco_so101_plugin.scene_registry import get_scene
from physicalai_mujoco_so101_plugin.studio_recorder import (
    CONNECT_TIMEOUT_S,
    AutoRecorder,
    RecordingOptions,
    StudioError,
    StudioLink,
    StudioTarget,
    discover_session,
    validate_studio_url,
)
from physicalai_mujoco_so101_plugin.viser_controls import _studio_markdown

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

FOLLOWER = "9fd891f9-11a3-41af-b0e0-8162c2393536"
LEADER = "6d3faa01-3324-4007-b98d-05fc73144a8d"
PROJECT = "bb3a7254-177a-4a3c-b70a-f47d193ceb80"
TARGET = StudioTarget(PROJECT, FOLLOWER, "Mary Jane", LEADER, "Autopilot")
READY = {"connected": True, "follower_source": "teleop", "dataset_loaded": True, "is_recording": False}


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------


def _studio_rest(sessions: list[dict[str, Any]]) -> Any:  # noqa: ANN401
    robots = [
        {"id": FOLLOWER, "name": "Mary Jane", "type": "MuJoCo_SO101_Follower", "payload": {"name": "mujoco-so101-follow"}},
        {"id": LEADER, "name": "Autopilot", "type": "MuJoCo_SO101_Virtual_Leader", "payload": {"http_port": 8080}},
    ]
    routes = {
        "/api/runtime/sessions": sessions,
        "/api/projects": [{"id": "other"}, {"id": PROJECT}],
        "/api/projects/other/robots": [],
        f"/api/projects/{PROJECT}/robots": robots,
    }
    return lambda _base, path: routes[path]


def test_discovers_the_live_session_and_its_leader() -> None:
    sessions = [{"follower_id": FOLLOWER, "status": "running", "leader_name": "Autopilot"}]
    target = discover_session("http://studio", "mujoco-so101-follow", get_json=_studio_rest(sessions))
    assert target == TARGET


@pytest.mark.parametrize(
    ("owner", "sessions", "match"),
    [
        ("mujoco-so101-follow", [], "no live robot session"),
        ("another-sim", [{"follower_id": FOLLOWER, "status": "running"}], "No live Studio session drives"),
        ("mujoco-so101-follow", [{"follower_id": FOLLOWER, "status": "stopped"}], "no live robot session"),
        (
            "mujoco-so101-follow",
            [{"follower_id": FOLLOWER, "status": "running", "leader_name": "Gone"}],
            "not in the follower's project",
        ),
    ],
)
def test_discovery_explains_what_is_missing(owner: str, sessions: list[dict[str, Any]], match: str) -> None:
    with pytest.raises(StudioError, match=match):
        discover_session("http://studio", owner, get_json=_studio_rest(sessions))


def test_unreachable_studio_is_a_studio_error() -> None:
    with pytest.raises(StudioError, match="not reachable"):
        discover_session("http://127.0.0.1:1", "mujoco-so101-follow")


@pytest.mark.parametrize("url", ["ftp://studio", "http://studio/api", "http://user:pw@studio", "studio:7860"])
def test_studio_url_must_be_a_bare_http_origin(url: str) -> None:
    with pytest.raises(ValueError, match="Studio URL"):
        validate_studio_url(url)
    assert validate_studio_url("http://127.0.0.1:7860/") == "http://127.0.0.1:7860"


# ----------------------------------------------------------------------
# Episode cycle against a scripted link
# ----------------------------------------------------------------------


@dataclass
class FakeLink:
    phase: str = "connected"
    error: str | None = None
    target: StudioTarget | None = TARGET
    state: dict[str, Any] | None = field(default_factory=lambda: dict(READY))
    sent: list[tuple[str, Any, str | None]] = field(default_factory=list)
    acks: dict[str, dict[str, Any]] = field(default_factory=dict)
    closed: bool = False

    def start(self) -> None:
        pass

    def send(self, event: str, data: Mapping[str, Any] | None = None, request_id: str | None = None) -> None:
        self.sent.append((event, data, request_id))

    def take_ack(self, request_id: str) -> dict[str, Any] | None:
        return self.acks.pop(request_id, None)

    def close(self) -> None:
        self.closed = True

    def ack_last(self, *, ok: bool = True, error: str | None = None) -> None:
        request_id = self.sent[-1][2]
        assert request_id is not None
        self.acks[request_id] = {"request_id": request_id, "ok": ok, "error": error}


def _recorder(options: RecordingOptions | None = None) -> tuple[AutoRecorder, FakeLink]:
    link = FakeLink()
    recorder = AutoRecorder(lambda: link)
    recorder.enable(options or RecordingOptions())
    return recorder, link


def test_cycle_starts_records_saves_and_starts_again() -> None:
    recorder, link = _recorder(RecordingOptions(task="Sort", max_episodes=2))
    assert recorder.update(0, None).hold_feed  # connecting -> waiting
    directive = recorder.update(0, None)
    assert directive.hold_feed
    assert directive.clear_belt
    assert link.sent == [("start_recording", {"task": "Sort"}, None)]
    assert recorder.update(0, None).hold_feed  # Studio has not confirmed yet
    link.state = {**READY, "is_recording": True}
    assert not recorder.update(0, None).hold_feed
    assert recorder.phase == "recording"
    assert not recorder.update(0, None).hold_feed  # the episode is running

    assert recorder.update(1, {"correct": 10, "wrong": 0, "missed": 0}).hold_feed
    assert link.sent[-1][0] == "save_episode"
    assert recorder.update(1, None).hold_feed  # waiting for the ack
    link.ack_last()
    link.state = dict(READY)
    assert recorder.update(1, None).hold_feed
    assert recorder.phase == "waiting"
    assert recorder.status()["saved"] == 1
    assert recorder.update(1, None).clear_belt  # the next episode starts

    link.state = {**READY, "is_recording": True}
    recorder.update(1, None)
    recorder.update(2, {"correct": 10, "wrong": 0, "missed": 0})
    link.ack_last()
    assert not recorder.update(2, None).hold_feed
    assert recorder.phase == "done"
    assert recorder.status()["message"] == "Recorded 2 episodes."
    assert link.closed


def test_episodes_with_mistakes_are_discarded_unless_keeping_all() -> None:
    for keep, expected in (("perfect", "discard_episode"), ("all", "save_episode")):
        recorder, link = _recorder(RecordingOptions(keep=keep))
        recorder.update(0, None)
        recorder.update(0, None)
        link.state = {**READY, "is_recording": True}
        recorder.update(0, None)
        recorder.update(1, {"correct": 9, "wrong": 1, "missed": 0})
        assert link.sent[-1][0] == expected
        link.ack_last()
        recorder.update(1, None)
        assert recorder.status()["discarded" if keep == "perfect" else "saved"] == 1


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({**READY, "dataset_loaded": False}, "open a dataset"),
        ({**READY, "follower_source": "hold"}, "start teleoperation"),
        ({**READY, "is_recording": True}, "save or discard"),
    ],
)
def test_waits_until_studio_is_ready(state: dict[str, Any], message: str) -> None:
    recorder, link = _recorder()
    link.state = state
    recorder.update(0, None)
    directive = recorder.update(0, None)
    assert directive.hold_feed
    assert not directive.clear_belt
    assert message in recorder.status()["message"]
    assert link.sent == []


def test_a_scene_jump_discards_the_episode_and_starts_a_fresh_one() -> None:
    recorder, link = _recorder()
    recorder.update(3, None)
    recorder.update(3, None)
    link.state = {**READY, "is_recording": True}
    recorder.update(3, None)  # recording from conveyor episode 3
    recorder.restart_episode("The scene was reloaded.")
    assert link.sent[-1][0] == "discard_episode"
    assert recorder.phase == "saving"
    assert recorder.status()["message"].startswith("The scene was reloaded.")
    link.ack_last()
    link.state = dict(READY)
    recorder.update(0, None)  # the reloaded conveyor counts from zero
    assert recorder.phase == "waiting"
    assert recorder.status()["discarded"] == 1
    assert recorder.update(0, None).clear_belt
    link.state = {**READY, "is_recording": True}
    recorder.update(0, None)
    recorder.update(1, {"correct": 10, "wrong": 0, "missed": 0})  # the first new episode ends it
    assert link.sent[-1][0] == "save_episode"


def test_a_scene_jump_outside_an_episode_changes_nothing() -> None:
    recorder, link = _recorder()
    recorder.update(0, None)
    recorder.update(0, None)  # starting: Studio has not begun recording yet
    recorder.restart_episode("The scene was reset.")
    assert recorder.phase == "starting"
    assert [event for event, _data, _id in link.sent] == ["start_recording"]


@pytest.mark.parametrize("link_phase", ["connected", "connecting"])
def test_gives_up_when_studio_never_sends_its_state(link_phase: str) -> None:
    """Otherwise the belt stays held forever behind an unusable link."""
    now = [0.0]
    link = FakeLink(phase=link_phase, state=None)
    recorder = AutoRecorder(lambda: link, clock=lambda: now[0])
    recorder.enable(RecordingOptions())
    assert recorder.update(0, None).hold_feed
    now[0] = CONNECT_TIMEOUT_S + 1.0
    assert not recorder.update(0, None).hold_feed
    assert recorder.phase == "error"
    assert "robot session" in recorder.status()["message"]
    assert link.closed


def test_stops_when_studio_stops_the_recording_or_the_link_fails() -> None:
    recorder, link = _recorder()
    recorder.update(0, None)
    recorder.update(0, None)
    link.state = {**READY, "is_recording": True}
    recorder.update(0, None)
    link.state = dict(READY)  # the user pressed stop in Studio
    assert not recorder.update(0, None).hold_feed
    assert recorder.phase == "stopped"

    recorder, link = _recorder()
    link.phase, link.error = "error", "Studio has no live robot session."
    assert not recorder.update(0, None).hold_feed
    assert recorder.phase == "error"
    assert recorder.status()["message"] == "Studio has no live robot session."


def test_a_failed_save_stops_with_studios_reason() -> None:
    recorder, link = _recorder()
    recorder.update(0, None)
    recorder.update(0, None)
    link.state = {**READY, "is_recording": True}
    recorder.update(0, None)
    recorder.update(1, {"correct": 10, "wrong": 0, "missed": 0})
    link.ack_last(ok=False, error="disk full")
    recorder.update(1, None)
    assert recorder.phase == "error"
    assert "disk full" in recorder.status()["message"]


@pytest.mark.parametrize("kwargs", [{"task": ""}, {"task": "x" * 201}, {"keep": "some"}, {"max_episodes": -1}])
def test_recording_options_are_bounded(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must be"):
        RecordingOptions(**kwargs)


def test_studio_markdown() -> None:
    assert _studio_markdown({"phase": "off"}) == "**Studio:** off"
    text = _studio_markdown({
        "phase": "recording",
        "follower": "Mary Jane",
        "leader": "Autopilot",
        "message": "Recording episode 3.",
        "saved": 2,
        "discarded": 1,
        "max_episodes": 5,
    })
    assert "Mary Jane with Autopilot" in text
    assert "2/5 saved, 1 discarded" in text


# ----------------------------------------------------------------------
# A fake Studio over a real websocket
# ----------------------------------------------------------------------


class FakeStudio:
    """Speaks the runtime websocket protocol the way Studio's backend does."""

    def __init__(self) -> None:
        self.handshake: dict[str, Any] | None = None
        self.received: list[dict[str, Any]] = []
        self.state = dict(READY)
        self._server = serve(self._handle, "127.0.0.1", 0)
        self.port = int(self._server.socket.getsockname()[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def _send_state(self, ws: Any) -> None:  # noqa: ANN401
        ws.send(json.dumps({"event": "state", "data": self.state}))

    def _handle(self, ws: Any) -> None:  # noqa: ANN401
        assert ws.request.path == f"/api/projects/{PROJECT}/runtime/ws"
        self.handshake = json.loads(ws.recv())
        self._send_state(ws)
        for raw in ws:
            message = json.loads(raw)
            self.received.append(message)
            event = message["event"]
            if event == "start_recording":
                self.state = {**self.state, "is_recording": True, "task": message["data"]["task"]}
                self._send_state(ws)
            elif event in {"save_episode", "discard_episode"}:
                self.state = {**self.state, "is_recording": False}
                self._send_state(ws)
                ws.send(json.dumps({"event": "ack", "data": {"request_id": message["request_id"], "ok": True}}))

    def close(self) -> None:
        self._server.shutdown()


@pytest.fixture
def studio() -> Iterator[FakeStudio]:
    server = FakeStudio()
    yield server
    server.close()


def _link(studio: FakeStudio) -> StudioLink:
    return StudioLink(f"http://127.0.0.1:{studio.port}", "mujoco-so101-follow", discover=lambda _url, _owner: TARGET)


def _wait(predicate: Any, timeout_s: float = 5.0) -> None:  # noqa: ANN401
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_link_attaches_with_the_sessions_handshake_and_relays_acks(studio: FakeStudio) -> None:
    link = _link(studio)
    link.start()
    try:
        _wait(lambda: link.state is not None)
        assert link.phase == "connected"
        assert studio.handshake == {"follower_id": FOLLOWER, "leader_id": LEADER, "camera_ids": []}
        link.send("save_episode", request_id="r1")
        _wait(lambda: studio.received)
        ack = None
        while ack is None:
            ack = link.take_ack("r1")
            time.sleep(0.01)
        assert ack["ok"] is True
    finally:
        link.close()
    assert all(message["event"] != "disconnect" for message in studio.received), "never stop Studio's session"


def test_link_reports_a_failed_discovery() -> None:
    def fail(_url: str, _owner: str) -> StudioTarget:
        raise StudioError("Studio is not reachable")

    link = StudioLink("http://127.0.0.1:1", "mujoco-so101-follow", discover=fail)
    link.start()
    _wait(lambda: link.phase == "error")
    assert link.error == "Studio is not reachable"


def _unexpected_sessions_shape(_url: str, _owner: str) -> StudioTarget:
    raise KeyError("follower_id")  # e.g. a response from another Studio version


def _found_target(_url: str, _owner: str) -> StudioTarget:
    return TARGET


def _rejected_handshake(_url: str) -> Any:  # noqa: ANN401
    raise RuntimeError("server rejected WebSocket connection: HTTP 403")  # not an OSError


@pytest.mark.parametrize(
    ("discover", "connect", "match"),
    [
        (_unexpected_sessions_shape, None, "Could not read Studio's sessions"),
        (_found_target, _rejected_handshake, "Could not open"),
    ],
    ids=["unexpected-discovery-error", "rejected-handshake"],
)
def test_link_turns_any_startup_failure_into_an_error(
    discover: Callable[[str, str], StudioTarget], connect: Callable[[str], Any] | None, match: str
) -> None:
    """Otherwise the link stays "connecting" and the recorder holds the belt forever."""
    link = StudioLink("http://127.0.0.1:1", "mujoco-so101-follow", discover=discover, connect=connect)
    link.start()
    _wait(lambda: link.phase == "error")
    assert match in (link.error or "")


def test_invalid_json_from_studio_is_a_studio_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import http.client

    class Response:
        status = 200

        def read(self) -> bytes:
            return b"<html>proxy login</html>"

    monkeypatch.setattr(http.client.HTTPConnection, "request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(http.client.HTTPConnection, "getresponse", lambda _self: Response())
    with pytest.raises(StudioError, match="invalid JSON"):
        discover_session("http://127.0.0.1:1", "mujoco-so101-follow")


@pytest.mark.slow
def test_simulation_records_an_autopilot_episode_into_studio(studio: FakeStudio) -> None:
    scene = get_scene("conveyor_sort")
    robot = MuJoCoSO101(
        model_path=str(scene.scene_xml_path),
        scene_config=asdict(scene),
        substeps=10,
        owner_name="mujoco-so101-follow",
    )
    robot.connect()
    robot._automation.recorder = AutoRecorder(lambda: _link(studio))
    try:
        robot._set_belt_speed(0.05)
        robot._automation.set_autopilot("drive")
        robot._commands.put(SetStudioRecordingCommand(options=RecordingOptions(task="Sort the belt", keep="all")))
        deadline = time.monotonic() + 60.0
        while not any(m["event"] == "save_episode" for m in studio.received):
            assert time.monotonic() < deadline, robot._automation.recorder.status()
            robot.get_observation()
        episode = robot._http_status()["episode"]
        assert episode["episode_count"] == 1
        assert sum(episode["last_episode"].values()) == 10
        assert episode["feed_held"] is True, "the belt waits while Studio saves"

        # Once Studio confirms, the next episode starts on a cleared belt.
        def next_started() -> bool:
            robot.get_observation()
            return [m["event"] for m in studio.received].count("start_recording") == 2

        _wait(next_started)
        assert [m["event"] for m in studio.received] == ["start_recording", "save_episode", "start_recording"]
        assert studio.received[0]["data"] == {"task": "Sort the belt"}
        assert robot._automation.recorder.status()["saved"] == 1
        # A fresh episode: the fake Studio confirms at once, so at most its first item is out.
        assert robot._http_status()["episode"]["spawned"] <= 1
    finally:
        robot.disconnect()
