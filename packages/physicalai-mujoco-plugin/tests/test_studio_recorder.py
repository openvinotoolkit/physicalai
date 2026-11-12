"""Automatic Studio episode recording: the relay to Studio's recording page and the episode cycle."""

from __future__ import annotations

import dataclasses
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
    RelayContext,
    RelayLink,
    StudioRelay,
    StudioTarget,
)
from physicalai_mujoco_plugin.viser_controls import _studio_markdown

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

TARGET = StudioTarget("Mary Jane", "Autopilot")
READY = {"connected": True, "follower_source": "teleop", "dataset_loaded": True, "is_recording": False}
SESSION = {
    "session_id": "session-1",
    "dataset_id": "dataset-1",
    "dataset": True,
    "recording": False,
    "teleop": True,
    "follower": "Mary Jane",
    "leader": "Autopilot",
}
"""What Studio's recording page relays when it is ready: a dataset loaded, teleoperating, not recording."""


# ----------------------------------------------------------------------
# The relay
# ----------------------------------------------------------------------

PAGE = RelayContext("page-1", "session-1", "dataset-1")


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _ok(*request_ids: str) -> list[dict[str, Any]]:
    return [{"request_id": request_id, "ok": True, "error": None} for request_id in request_ids]


def _ids(commands: list[dict[str, Any]] | None) -> list[str]:
    return [command["request_id"] for command in commands or []]


def test_a_link_connects_on_the_first_exchange_and_fails_once_studio_falls_silent() -> None:
    clock = Clock()
    link = RelayLink(clock)
    assert link.phase == "idle"
    link.start()
    assert (link.phase, link.state, link.target) == ("connecting", None, None)

    link.exchange(PAGE, SESSION, [])
    assert link.phase == "connected"
    assert link.state == {"dataset_loaded": True, "is_recording": False, "follower_source": "teleop"}
    assert link.target == TARGET

    clock.now += RELAY_SILENCE_S + 0.1
    assert link.phase == "error"
    assert link.error == NO_STUDIO
    link.close()
    assert link.phase == "closed"


def test_a_command_comes_with_every_exchange_until_its_result_does() -> None:
    link = RelayLink()
    link.start()
    link.send("start_recording", {"task": "Sort"}, request_id="a")
    link.send("save_episode", request_id="b")

    # Each names the dataset of the run and the episode it is about, for Studio to refuse on another.
    assert link.exchange(PAGE, SESSION, []) == [
        {"command": "start_episode", "request_id": "a", "task": "Sort", "dataset_id": "dataset-1", "episode_id": "a"},
        {"command": "save_episode", "request_id": "b", "dataset_id": "dataset-1", "episode_id": "a"},
    ]
    # The reply was lost: the next exchange hands the same commands over again.
    assert _ids(link.exchange(PAGE, SESSION, [])) == ["a", "b"]
    assert _ids(link.exchange(PAGE, SESSION, [{"request_id": "b", "ok": False, "error": "disk full"}])) == ["a"]
    assert link.take_ack("b") == {"request_id": "b", "ok": False, "error": "disk full"}
    # A result sent again, or for a command never issued, is ignored and kept nowhere.
    link.exchange(PAGE, SESSION, _ok("b", "never", *[f"x{n}" for n in range(14)]))
    assert link.take_ack("b") is None and link.take_ack("never") is None
    with pytest.raises(ValueError, match="relays no"):
        link.send("set_follower_source")


def test_only_the_page_that_relays_a_run_drives_it() -> None:
    link = RelayLink()
    link.start()
    link.exchange(PAGE, SESSION, [])
    link.send("save_episode", request_id="s")

    other = RelayContext("page-2", "session-2", "dataset-1")
    assert link.exchange(other, {**SESSION, "recording": False}, _ok("s")) is None
    assert link.take_ack("s") is None
    assert link.state is not None and link.phase == "connected"  # the other page changed nothing

    assert _ids(link.exchange(PAGE, SESSION, [])) == ["s"]


@pytest.mark.parametrize("moved", [{"session_id": "session-9"}, {"dataset_id": "dataset-9"}])
def test_a_run_fails_when_its_page_moves_to_another_session_or_dataset(moved: dict[str, str]) -> None:
    link = RelayLink()
    link.start()
    link.exchange(PAGE, SESSION, [])
    link.send("discard_episode", request_id="d")

    assert link.exchange(dataclasses.replace(PAGE, **moved), SESSION, []) is None
    assert (link.phase, link.error) == ("error", "Studio's recording page moved to another session or dataset.")
    assert not link.pending  # nothing it asked for applies to the other session or dataset
    link.close()  # the recorder read why
    assert link.phase == "closed"


def test_a_long_save_keeps_the_run_alive_while_the_page_keeps_relaying() -> None:
    clock = Clock()
    relay = StudioRelay(clock)
    recorder = AutoRecorder(relay.open, clock=clock)
    recorder.enable(RecordingOptions(keep="all"))
    relay.exchange(PAGE, SESSION, [])
    recorder.update(0, None)  # Studio answered: waiting
    recorder.update(0, None)  # ready: start an episode
    start = relay.exchange(PAGE, SESSION, [])["commands"][0]
    relay.exchange(PAGE, {**SESSION, "recording": True}, _ok(start["request_id"]))
    recorder.update(0, None)
    recorder.update(1, {"right": 10})
    save = relay.exchange(PAGE, {**SESSION, "recording": True}, [])["commands"][0]
    assert recorder.phase == "saving"

    for _ in range(60):  # a 30 s save, with the page relaying meanwhile
        clock.now += 0.5
        relay.exchange(PAGE, {**SESSION, "recording": True}, [])
        recorder.update(1, None)
    assert recorder.phase == "saving"
    relay.exchange(PAGE, SESSION, _ok(save["request_id"]))
    recorder.update(1, None)
    assert (recorder.phase, recorder.status()["saved"]) == ("waiting", 1)


def test_a_start_studio_refuses_stops_the_run_with_its_reason() -> None:
    relay = StudioRelay()
    recorder = AutoRecorder(relay.open)
    recorder.enable(RecordingOptions())
    relay.exchange(PAGE, SESSION, [])
    recorder.update(0, None)  # Studio answered: waiting
    recorder.update(0, None)  # ready: start an episode
    start = relay.exchange(PAGE, SESSION, [])["commands"][0]
    relay.exchange(PAGE, SESSION, [{"request_id": start["request_id"], "ok": False, "error": "No dataset loaded"}])
    recorder.update(0, None)
    assert recorder.phase == "error"
    assert "No dataset loaded" in recorder.status()["message"]


def test_ending_a_run_cancels_what_studio_has_not_answered_but_its_discards() -> None:
    relay = StudioRelay()
    link = relay.open()
    link.start()
    relay.exchange(PAGE, SESSION, [])
    link.send("start_recording", {"task": "Sort"}, request_id="start")
    link.send("discard_episode", request_id="last")
    link.close()  # switched off before the page picked them up

    assert relay.exchange(PAGE, SESSION, []) == {
        "active": False,
        "owner": True,
        "commands": [
            {"command": "discard_episode", "request_id": "last", "dataset_id": "dataset-1", "episode_id": "start"}
        ],
    }
    assert relay.exchange(PAGE, SESSION, _ok("last"))["commands"] == []
    assert relay.exchange(PAGE, SESSION, []) == {"active": False, "owner": False, "commands": []}  # all answered


def test_a_new_run_keeps_the_last_runs_discard_for_its_page() -> None:
    relay = StudioRelay()
    first = relay.open()
    first.start()
    relay.exchange(PAGE, SESSION, [])
    first.send("discard_episode", request_id="last")
    second = relay.open()  # replaced before the page picked the discard up
    second.start()
    second.send("start_recording", {"task": "Sort"}, request_id="next")

    reply = relay.exchange(PAGE, SESSION, [])
    assert (reply["active"], _ids(reply["commands"])) == (True, ["last"])  # the new run waits for it
    assert first.phase == "closed"
    assert _ids(relay.exchange(PAGE, SESSION, _ok("last"))["commands"]) == ["next"]


def test_a_new_runs_commands_wait_until_the_page_answered_the_last_runs_discard() -> None:
    relay = StudioRelay()
    first = relay.open()
    first.start()
    relay.exchange(PAGE, SESSION, [])
    first.send("discard_episode", request_id="last")
    second = relay.open()
    second.start()
    second.send("start_recording", {"task": "Sort"}, request_id="next")

    assert _ids(relay.exchange(PAGE, SESSION, [])["commands"]) == ["last"]
    second.close()  # the new run is switched off while the page still discards
    assert _ids(relay.exchange(PAGE, SESSION, [])["commands"]) == ["last"]
    assert relay.exchange(PAGE, SESSION, _ok("last"))["commands"] == []  # its start never went out


def test_a_page_may_relay_before_it_loaded_a_dataset() -> None:
    clock = Clock()
    relay = StudioRelay(clock)
    recorder = AutoRecorder(relay.open, clock=clock)
    recorder.enable(RecordingOptions())
    bare = RelayContext("page-1", None, None)
    relay.exchange(bare, {**SESSION, "dataset": False}, [])
    recorder.update(0, None)
    recorder.update(0, None)
    assert "open a dataset" in recorder.status()["message"]

    relay.exchange(PAGE, SESSION, [])  # the session and the dataset are loaded now
    recorder.update(0, None)
    assert recorder.phase == "starting"
    other = RelayContext("page-2", "session-1", "dataset-1")
    assert relay.exchange(other, SESSION, [])["owner"] is False


def test_a_page_that_moves_to_another_dataset_stops_the_run_saying_so() -> None:
    relay = StudioRelay()
    recorder = AutoRecorder(relay.open)
    recorder.enable(RecordingOptions())
    relay.exchange(PAGE, SESSION, [])
    recorder.update(0, None)
    relay.exchange(dataclasses.replace(PAGE, dataset_id="dataset-9"), SESSION, [])
    recorder.update(0, None)

    assert recorder.phase == "error"
    assert recorder.status()["message"] == "Studio's recording page moved to another session or dataset."
    assert relay.exchange(PAGE, SESSION, [])["active"] is False


def test_a_start_answered_after_the_recording_state_is_read_and_not_kept() -> None:
    relay = StudioRelay()
    recorder = AutoRecorder(relay.open)
    recorder.enable(RecordingOptions(keep="all"))
    relay.exchange(PAGE, SESSION, [])
    recorder.update(0, None)
    for episode in range(1, 6):
        recorder.update(episode - 1, None)  # waiting: start
        start = relay.exchange(PAGE, SESSION, [])["commands"][0]
        relay.exchange(PAGE, {**SESSION, "recording": True}, [])  # the state first
        recorder.update(episode - 1, None)
        assert recorder.phase == "recording"
        relay.exchange(PAGE, {**SESSION, "recording": True}, _ok(start["request_id"]))  # then the result
        recorder.update(episode, {"right": 10})
        save = relay.exchange(PAGE, {**SESSION, "recording": True}, [])["commands"][0]
        relay.exchange(PAGE, SESSION, _ok(save["request_id"]))
        recorder.update(episode, None)
        assert recorder.status()["saved"] == episode
    assert not recorder._link._acks  # noqa: SLF001 - every result was read


@pytest.mark.parametrize("ended", [False, True], ids=["start of the run", "discard of an ended run"])
def test_a_page_that_lost_its_dataset_for_a_moment_gets_no_command_for_it(ended: bool) -> None:  # noqa: FBT001
    relay = StudioRelay()
    link = relay.open()
    link.start()
    relay.exchange(PAGE, SESSION, [])
    link.send("start_recording", {"task": "Sort"}, request_id="start")
    if ended:
        link.send("discard_episode", request_id="last")
        link.close()
    unloaded = dataclasses.replace(PAGE, dataset_id=None)  # the dataset was unloaded; another may be loading

    for page in (unloaded, dataclasses.replace(PAGE, session_id=None)):
        assert relay.exchange(page, {**SESSION, "dataset": False}, [])["commands"] == []
    assert relay.exchange(dataclasses.replace(PAGE, dataset_id="dataset-2"), SESSION, [])["commands"] == []
    assert relay.exchange(PAGE, SESSION, [])["commands"] == []  # the run is over: nothing for dataset-1 any more


def test_no_run_hands_out_nothing() -> None:
    assert StudioRelay().exchange(PAGE, SESSION, []) == {"active": False, "owner": False, "commands": []}


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
    assert [sent[:2] for sent in link.sent] == [("start_recording", {"task": "Sort"})]
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
        self._done: dict[str, dict[str, Any]] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            reply = self._relay.exchange(PAGE, dict(self.session), list(self._done.values()))
            listed = {command["request_id"] for command in reply["commands"]}
            self._done = {request_id: result for request_id, result in self._done.items() if request_id in listed}
            for command in reply["commands"]:
                if command["request_id"] in self._done:
                    continue  # run once; its result goes out until it is no longer listed
                self.received.append(command)
                self.session["recording"] = command["command"] == "start_episode"
                self._done[command["request_id"]] = {"request_id": command["request_id"], "ok": True, "error": None}
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
        assert [(command["command"], command["task"]) for command in studio.received] == [("start_episode", "Sort")]
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
