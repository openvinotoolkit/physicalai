"""Automatic Studio episode recording: the relay to Studio's recording page and the episode cycle."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from physicalai_mujoco_plugin.http_server import SetStudioRecordingCommand
from physicalai_mujoco_plugin.robot import MuJoCoRobot
from physicalai_mujoco_plugin.scene_registry import get_scene
from physicalai_mujoco_plugin.studio_recorder import (
    CONNECT_TIMEOUT_S,
    NO_STUDIO,
    RELAY_SILENCE_S,
    AutoRecorder,
    RecordingOptions,
    RelayLink,
    StudioRelay,
    StudioTarget,
)
from physicalai_mujoco_plugin.viser_controls import _studio_markdown

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

TARGET = StudioTarget("Mary Jane", "Autopilot")
READY = {"connected": True, "follower_source": "teleop", "dataset_loaded": True, "is_recording": False}
SESSION = {"dataset": True, "recording": False, "teleop": True, "follower": "Mary Jane", "leader": "Autopilot"}
"""What Studio's recording page relays when it is ready: a dataset loaded, teleoperating, not recording."""


# ----------------------------------------------------------------------
# The relay
# ----------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_a_link_connects_on_the_first_exchange_and_fails_once_studio_falls_silent() -> None:
    clock = Clock()
    link = RelayLink(clock)
    assert link.phase == "idle"
    link.start()
    assert (link.phase, link.state, link.target) == ("connecting", None, None)

    link.exchange(SESSION, [])
    assert link.phase == "connected"
    assert link.state == {"dataset_loaded": True, "is_recording": False, "follower_source": "teleop"}
    assert link.target == TARGET

    clock.now += RELAY_SILENCE_S + 0.1
    assert link.phase == "error"
    assert link.error == NO_STUDIO
    link.close()
    assert link.phase == "closed"


def test_a_link_hands_over_studios_own_commands_and_takes_back_their_results() -> None:
    link = RelayLink()
    link.start()
    link.send("start_recording", {"task": "Sort"}, request_id="a")
    link.send("save_episode", request_id="b")
    link.send("discard_episode")

    commands = link.exchange(SESSION, [])
    assert commands[:2] == [
        {"command": "start_episode", "request_id": "a", "task": "Sort"},
        {"command": "save_episode", "request_id": "b"},
    ]
    assert commands[2]["command"] == "discard_episode"
    assert commands[2]["request_id"]  # Studio's session needs one for every command
    assert link.exchange(SESSION, []) == []  # each command goes out once

    assert link.take_ack("b") is None
    link.exchange(SESSION, [{"request_id": "b", "ok": False, "error": "disk full"}])
    assert link.take_ack("b") == {"request_id": "b", "ok": False, "error": "disk full"}
    assert link.take_ack("b") is None
    with pytest.raises(ValueError, match="relays no"):
        link.send("set_follower_source")


def test_the_relay_hands_out_the_current_run_and_drains_the_last_one() -> None:
    relay = StudioRelay()
    assert relay.exchange(SESSION, []) == {"active": False, "commands": []}

    first = relay.open()
    first.start()
    assert relay.exchange(SESSION, []) == {"active": True, "commands": []}
    first.send("discard_episode", request_id="last")
    first.close()  # switched off: the discard queued just before still goes out, once
    assert relay.exchange(SESSION, []) == {
        "active": False,
        "commands": [{"command": "discard_episode", "request_id": "last"}],
    }
    assert relay.exchange(SESSION, []) == {"active": False, "commands": []}

    second = relay.open()
    second.start()
    assert first.phase == "closed"
    second.send("save_episode", request_id="s")
    assert relay.exchange(SESSION, [])["commands"] == [{"command": "save_episode", "request_id": "s"}]


def test_a_run_stops_when_studio_never_picks_it_up() -> None:
    clock = Clock()
    relay = StudioRelay(clock)
    recorder = AutoRecorder(relay.open, clock=clock)
    recorder.enable(RecordingOptions())
    recorder.update(0, None)
    assert recorder.phase == "connecting"
    clock.now += CONNECT_TIMEOUT_S + 0.1
    recorder.update(0, None)
    assert (recorder.phase, recorder.status()["message"]) == ("error", NO_STUDIO)


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
        ({**READY, "follower_source": "hold"}, "turn teleoperation on"),
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
    assert recorder.status()["message"] == NO_STUDIO
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
# A fake Studio recording page, relaying
# ----------------------------------------------------------------------


class FakeStudio:
    """Studio's recording page: relays a simulation's automatic recording as its buttons would."""

    def __init__(self, relay: StudioRelay, session: Mapping[str, Any] = SESSION) -> None:
        self.received: list[dict[str, Any]] = []
        self.session = dict(session)
        self._relay = relay
        self._results: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            results, self._results = self._results, []
            for command in self._relay.exchange(dict(self.session), results)["commands"]:
                self.received.append(command)
                self.session["recording"] = command["command"] == "start_episode"
                self._results.append({"request_id": command["request_id"], "ok": True, "error": None})
            time.sleep(0.01)

    def commands(self) -> list[str]:
        return [command["command"] for command in self.received]

    def close(self) -> None:
        self._stop.set()
        self._thread.join()


def _wait(predicate: Any, timeout_s: float = 5.0) -> None:  # noqa: ANN401
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_a_relayed_cycle_starts_and_saves_through_studios_page() -> None:
    relay = StudioRelay()
    recorder = AutoRecorder(relay.open)
    studio = FakeStudio(relay)
    try:
        recorder.enable(RecordingOptions(task="Sort", keep="all"))

        def started() -> bool:
            recorder.update(0, None)
            return recorder.phase == "recording"

        _wait(started)
        assert studio.received == [{"command": "start_episode", "request_id": studio.received[0]["request_id"], "task": "Sort"}]
        assert recorder.status()["follower"] == "Mary Jane"

        def saved() -> bool:
            recorder.update(1, {"right": 10})
            return recorder.status()["saved"] == 1

        _wait(saved)
        assert studio.commands() == ["start_episode", "save_episode"]
    finally:
        studio.close()


@pytest.mark.slow
def test_simulation_records_an_autopilot_episode_into_studio() -> None:
    scene = get_scene("conveyor_sort")
    robot = MuJoCoRobot(
        scene=scene.scene_id,
        cameras=[],
        substeps=10,
        owner_name="mujoco-so101-follow",
    )
    robot.connect()
    studio = FakeStudio(robot._automation.studio_relay)
    try:
        robot._set_belt_speed(0.05)
        robot._automation.set_autopilot("drive")
        robot._commands.put(SetStudioRecordingCommand(options=RecordingOptions(task="Sort the belt", keep="all")))
        deadline = time.monotonic() + 60.0
        while "save_episode" not in studio.commands():
            assert time.monotonic() < deadline, robot._automation.recorder.status()
            robot.get_observation()
        episode = robot._http_status()["episode"]
        assert episode["episode_count"] == 1
        assert sum(episode["last_episode"].values()) == 10
        assert episode["feed_held"] is True, "the belt waits while Studio saves"

        # Once Studio confirms, the next episode starts on a cleared belt.
        def next_started() -> bool:
            robot.get_observation()
            return studio.commands().count("start_episode") == 2

        _wait(next_started)
        assert studio.commands() == ["start_episode", "save_episode", "start_episode"]
        assert studio.received[0]["task"] == "Sort the belt"
        assert robot._automation.recorder.status()["saved"] == 1
        # A fresh episode: the fake Studio confirms at once, so at most its first item is out.
        assert robot._http_status()["episode"]["spawned"] <= 1
    finally:
        studio.close()
        robot.disconnect()
