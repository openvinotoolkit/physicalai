"""Automatic Studio recording of bucket_sort episodes: the autopilot leads Studio's teleoperation."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from websockets.sync.server import serve

from physicalai_mujoco_plugin.conveyor_automation import ConveyorAutomation
from physicalai_mujoco_plugin.http_server import SetStudioRecordingCommand
from physicalai_mujoco_plugin.robot import MuJoCoRobot
from physicalai_mujoco_plugin.studio_recorder import (
    BucketSortProgress,
    BucketSortRecorder,
    RecordingOptions,
    StudioLink,
    StudioTarget,
    discover_session,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

FOLLOWER = "0b6a3c5e-5d0e-4d8e-9f0c-1c3e8b1f9a11"
LEADER = "3c1e9a7f-6b2d-4f0e-8a5c-9d7b2e4f1a33"
PROJECT = "7f0e7b51-2b7a-4f57-8d4e-2c4b1f0e6a22"
TARGET = StudioTarget(PROJECT, FOLLOWER, "Franka", LEADER, "Autopilot")
HOLDING = {"connected": True, "follower_source": "hold", "dataset_loaded": True, "is_recording": False}
TELEOP = {**HOLDING, "follower_source": "teleop"}
TASK_ID = 1
"""Stands in for the task object's id in a layout generation."""


# ----------------------------------------------------------------------
# The episode cycle against a scripted link
# ----------------------------------------------------------------------


@dataclass
class FakeLink:
    phase: str = "connected"
    error: str | None = None
    target: StudioTarget | None = TARGET
    state: dict[str, Any] | None = field(default_factory=lambda: dict(TELEOP))
    sent: list[tuple[str, Any, str | None]] = field(default_factory=list)
    acks: dict[str, dict[str, Any]] = field(default_factory=dict)
    closed: bool = False

    def start(self) -> None:
        pass

    def send(self, event: str, data: Mapping[str, Any] | None = None, request_id: str | None = None) -> None:
        self.sent.append((event, None if data is None else dict(data), request_id))

    def take_ack(self, request_id: str) -> dict[str, Any] | None:
        return self.acks.pop(request_id, None)

    def close(self) -> None:
        self.closed = True

    def ack_last(self, *, ok: bool = True) -> None:
        request_id = self.sent[-1][2]
        assert request_id is not None
        self.acks[request_id] = {"request_id": request_id, "ok": ok, "error": None}

    def events(self) -> list[str]:
        return [event for event, _data, _id in self.sent]


def _progress(
    reset: int = 0, outcome: str | None = None, *, clean: bool | None = None, leading: bool = True
) -> BucketSortProgress:
    return BucketSortProgress(
        generation=(TASK_ID, reset),
        instruction=f"put the CPU in the red bucket ({reset})" if reset else "initial",
        leading=leading,
        outcome=outcome,
        clean=clean,
    )


def _recorder(options: RecordingOptions | None = None) -> tuple[BucketSortRecorder, FakeLink]:
    link = FakeLink()
    recorder = BucketSortRecorder(lambda: link)
    recorder.enable(options or RecordingOptions())
    return recorder, link


def _start_episode(recorder: BucketSortRecorder, link: FakeLink, reset: int) -> None:
    """Drive the recorder from waiting to recording an episode that starts at layout *reset*."""
    link.state = dict(TELEOP)
    directive = recorder.update(_progress(reset - 1))
    assert directive.reset_scene, recorder.status()
    assert directive.hold_arm
    assert recorder.phase == "resetting"
    assert recorder.update(_progress(reset)).hold_arm  # the reset ran: start the Studio episode
    assert link.sent[-1][:2] == ("start_recording", {"task": f"put the CPU in the red bucket ({reset})"})
    assert recorder.update(_progress(reset)).hold_arm  # Studio has not confirmed yet
    link.state = {**TELEOP, "is_recording": True}
    assert not recorder.update(_progress(reset)).hold_arm  # the arm moves once Studio records
    assert recorder.phase == "recording"


def test_records_each_episode_with_its_instruction_while_studio_teleoperates() -> None:
    recorder, link = _recorder(RecordingOptions(task="ignored", max_episodes=2))
    assert recorder.update(_progress()).hold_arm  # connecting -> waiting

    _start_episode(recorder, link, 1)
    assert recorder.status()["task"] == "put the CPU in the red bucket (1)"
    assert "put the CPU in the red bucket (1)" in recorder.status()["message"]
    assert not recorder.update(_progress(1, "running", clean=True)).hold_arm

    assert recorder.update(_progress(1, "success", clean=True)).hold_arm
    assert link.events()[-1] == "save_episode"
    link.ack_last()
    link.state = dict(TELEOP)
    assert recorder.update(_progress(1, "success", clean=True)).hold_arm
    assert recorder.phase == "waiting"
    assert recorder.status()["saved"] == 1

    _start_episode(recorder, link, 2)
    recorder.update(_progress(2, "success", clean=True))
    link.ack_last()
    link.state = dict(TELEOP)
    assert not recorder.update(_progress(2, "success", clean=True)).hold_arm
    assert recorder.phase == "done"
    assert recorder.status()["message"] == "Recorded 2 episodes."
    assert "set_follower_source" not in link.events()  # Studio's teleoperation is the user's
    assert link.events().count("start_recording") == 2
    assert link.closed


@pytest.mark.parametrize(
    ("keep", "clean", "expected"),
    [("perfect", True, "save_episode"), ("perfect", False, "discard_episode"), ("all", False, "save_episode")],
)
def test_successes_are_kept_by_the_keep_rule(keep: str, clean: bool, expected: str) -> None:  # noqa: FBT001
    recorder, link = _recorder(RecordingOptions(keep=keep))
    recorder.update(_progress())
    _start_episode(recorder, link, 1)
    recorder.update(_progress(1, "success", clean=clean))
    assert link.events()[-1] == expected
    link.ack_last()
    recorder.update(_progress(1))
    assert recorder.status()["saved" if expected == "save_episode" else "discarded"] == 1


@pytest.mark.parametrize("outcome", ["failed", "stalled"])
def test_failed_or_stalled_episodes_are_discarded_and_a_fresh_one_starts(outcome: str) -> None:
    recorder, link = _recorder(RecordingOptions(keep="all"))
    recorder.update(_progress())
    _start_episode(recorder, link, 1)
    recorder.update(_progress(1, outcome, clean=False))
    assert link.events()[-1] == "discard_episode"
    link.ack_last()
    link.state = dict(TELEOP)
    recorder.update(_progress(1))
    assert recorder.status()["discarded"] == 1
    assert recorder.status()["saved"] == 0
    _start_episode(recorder, link, 2)  # the recorder's own reset starts the next episode


@pytest.mark.parametrize(
    ("change", "message"),
    [({"reset": 5}, "The scene was reset."), ({"leading": False}, "The autopilot stopped leading.")],
)
def test_a_reset_or_leaving_leader_mode_discards_the_episode(change: dict[str, Any], message: str) -> None:
    recorder, link = _recorder()
    recorder.update(_progress())
    _start_episode(recorder, link, 1)
    recorder.update(_progress(**{"reset": 1, "outcome": "running", **change}))
    assert link.events()[-1] == "discard_episode"
    assert recorder.status()["message"].startswith(message)


@pytest.mark.parametrize(
    ("state", "leading", "message"),
    [
        ({**TELEOP, "dataset_loaded": False}, True, "open a dataset"),
        ({**TELEOP, "is_recording": True}, True, "save or discard"),
        (TELEOP, False, "Virtual leader"),
        (HOLDING, True, "start teleoperation"),
    ],
)
def test_waits_until_studio_and_the_autopilot_are_ready(state: dict[str, Any], leading: bool, message: str) -> None:  # noqa: FBT001
    recorder, link = _recorder()
    link.state = dict(state)
    recorder.update(_progress(leading=leading))
    directive = recorder.update(_progress(leading=leading))
    assert directive.hold_arm
    assert not directive.reset_scene
    assert message in recorder.status()["message"]
    assert link.sent == []


def test_switching_off_leaves_studio_teleoperating_and_leaving_teleop_stops() -> None:
    recorder, link = _recorder()
    recorder.update(_progress())
    _start_episode(recorder, link, 1)
    recorder.disable("Switched off.")
    assert recorder.phase == "off"
    assert "set_follower_source" not in link.events()

    recorder, link = _recorder()
    recorder.update(_progress())
    _start_episode(recorder, link, 1)
    link.state = {**TELEOP, "is_recording": True, "follower_source": "hold"}  # the user stopped teleoperation
    assert not recorder.update(_progress(1, "running")).hold_arm
    assert recorder.phase == "stopped"
    assert link.events()[-1] == "start_recording"


def test_discovery_finds_a_follower_of_any_mujoco_profile() -> None:
    robots = [
        {"id": FOLLOWER, "name": "Franka", "type": "MuJoCo_FrankaResearch3_Follower", "payload": {"name": "sim"}},
        {"id": LEADER, "name": "Autopilot", "type": "MuJoCo_Virtual_Leader", "payload": {"http_port": 8080}},
    ]
    routes = {
        "/api/runtime/sessions": [{"follower_id": FOLLOWER, "status": "running", "leader_name": "Autopilot"}],
        "/api/projects": [{"id": PROJECT}],
        f"/api/projects/{PROJECT}/robots": robots,
    }
    assert discover_session("http://studio", "sim", get_json=lambda _base, path: routes[path]) == TARGET


# ----------------------------------------------------------------------
# Scene checks
# ----------------------------------------------------------------------


def _automation() -> ConveyorAutomation:
    return ConveyorAutomation(
        joint_names=("j",), unit="rad", to_units=lambda q: q, owner_name=lambda: "sim"
    )


def test_recording_needs_the_conveyor_or_bucket_sort_scene() -> None:
    automation = _automation()
    automation.attach_scene(None, rng=np.random.default_rng(0), active=True)
    assert not automation.set_studio_recording(RecordingOptions())
    assert automation.recorder.phase == "off"
    assert automation.bucket_recorder.phase == "off"
    assert automation.set_studio_recording(None)  # switching off always works


# ----------------------------------------------------------------------
# The whole loop against a fake Studio over a real websocket
# ----------------------------------------------------------------------


class FakeStudio:
    """Speaks the runtime websocket protocol, teleoperating with a leader."""

    def __init__(self) -> None:
        self.received: list[dict[str, Any]] = []
        self.state = dict(TELEOP)
        self._server = serve(self._handle, "127.0.0.1", 0)
        self.port = int(self._server.socket.getsockname()[1])
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def _send_state(self, ws: Any) -> None:  # noqa: ANN401
        ws.send(json.dumps({"event": "state", "data": self.state}))

    def _handle(self, ws: Any) -> None:  # noqa: ANN401
        json.loads(ws.recv())
        self._send_state(ws)
        for raw in ws:
            message = json.loads(raw)
            self.received.append(message)
            event = message["event"]
            if event == "set_follower_source":
                self.state = {**self.state, "follower_source": message["data"]["follower_source"]}
            elif event == "start_recording":
                self.state = {**self.state, "is_recording": True, "task": message["data"]["task"]}
            elif event in {"save_episode", "discard_episode"}:
                self.state = {**self.state, "is_recording": False}
                ws.send(json.dumps({"event": "ack", "data": {"request_id": message["request_id"], "ok": True}}))
            self._send_state(ws)

    def events(self) -> list[str]:
        return [message["event"] for message in self.received]

    def close(self) -> None:
        self._server.shutdown()


@pytest.fixture
def studio() -> Iterator[FakeStudio]:
    server = FakeStudio()
    yield server
    server.close()


def _wait(predicate: Any, timeout_s: float = 5.0) -> None:  # noqa: ANN401
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


@pytest.mark.slow
def test_simulation_records_a_led_bucket_sort_episode_into_studio(studio: FakeStudio) -> None:
    robot = MuJoCoRobot("franka_fr3", scene="bucket_sort", cameras=[], http_port=0, seed=1)
    robot.connect()
    automation = robot._automation  # noqa: SLF001
    automation.bucket_recorder = BucketSortRecorder(
        lambda: StudioLink(f"http://127.0.0.1:{studio.port}", "sim", discover=lambda _url, _owner: TARGET)
    )
    task = robot._sim.on_reset  # noqa: SLF001
    gripper = robot.joint_names.index("gripper")
    recorded: list[tuple[float, float]] = []  # (action, state) of the gripper while Studio records

    def teleop_tick() -> None:
        """One tick of Studio's teleoperation: read the leader, send it as the follower's action."""
        state = robot.get_observation().joint_positions
        action = np.asarray(automation.leader_snapshot()["joint_positions"], dtype=np.float32)
        if studio.state.get("is_recording"):
            recorded.append((float(action[gripper]), float(state[gripper])))
        robot.send_action(action)

    try:
        automation.set_autopilot("leader")
        robot._commands.put(SetStudioRecordingCommand(options=RecordingOptions(keep="all")))  # noqa: SLF001
        resets = task.reset_count
        deadline = time.monotonic() + 120.0
        while "save_episode" not in studio.events() and "discard_episode" not in studio.events():
            assert time.monotonic() < deadline, automation.status()
            teleop_tick()
        assert studio.events() == ["start_recording", "save_episode"]
        # The recorder reset the scene and recorded that episode's instruction as the task.
        assert task.reset_count == resets + 1
        assert studio.received[0]["data"] == {"task": task.instruction.text}
        assert robot._http_status()["studio"]["task"] == task.instruction.text  # noqa: SLF001
        assert robot._http_status()["episode"]["success"]  # noqa: SLF001
        assert robot._http_status()["overview_locked"]  # noqa: SLF001
        # The actions are the demonstrator's commands: the gripper closes past where the object stops it.
        squeeze = min(action - state for action, state in recorded)
        assert squeeze < -0.005, f"the recorded gripper never commanded past the fingers ({squeeze:.4f} m)"

        def next_started() -> bool:
            teleop_tick()
            return studio.events().count("start_recording") == 2

        _wait(next_started)
        assert task.reset_count == resets + 2  # a fresh layout for the next episode
        assert automation.status()["studio"]["saved"] == 1

        robot._commands.put(SetStudioRecordingCommand(options=None))  # noqa: SLF001
        robot.get_observation()
        assert "set_follower_source" not in studio.events()
    finally:
        robot.disconnect()


@pytest.mark.slow
def test_the_conveyor_scene_keeps_its_own_recorder() -> None:
    robot = MuJoCoRobot(scene="conveyor_sort", cameras=[], http_port=0)
    robot.connect()
    try:
        automation = robot._automation  # noqa: SLF001
        automation.recorder = type(automation.recorder)(FakeLink)
        assert automation.set_studio_recording(RecordingOptions(task="Sort"))
        assert automation.recorder.active
        assert automation.bucket_recorder.phase == "off"
        assert automation.studio_recorder is automation.recorder
        assert automation.status()["studio"]["task"] == "Sort"
        assert "task_source" not in automation.status()["studio"]
    finally:
        robot.disconnect()
