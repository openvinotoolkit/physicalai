"""Automatic Studio recording of bucket_sort episodes: the autopilot leads Studio's teleoperation."""

from __future__ import annotations

import dataclasses
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
from physicalai_mujoco_plugin import bucket_sort_demo as bd
from physicalai_mujoco_plugin import conveyor_automation
from physicalai_mujoco_plugin.http_server import StudioRecordingRequest
from physicalai_mujoco_plugin.studio_recorder import (
    DEFAULT_MAX_ATTEMPTS,
    MAX_ATTEMPTS,
    MAX_RECOVERY_FRACTION,
    SUCCESS_HOLD_S,
    SUCCESS_HOLD_TIMEOUT_S,
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
    reset: int = 0,
    outcome: str | None = None,
    *,
    clean: bool | None = None,
    leading: bool = True,
    placed: bool | None = None,
    sim_time: float = 0.0,
) -> BucketSortProgress:
    return BucketSortProgress(
        generation=(TASK_ID, reset),
        instruction=f"put the CPU in the red bucket ({reset})" if reset else "initial",
        leading=leading,
        outcome=outcome,
        clean=clean,
        placed=outcome == "success" if placed is None else placed,
        sim_time=sim_time,
    )


@dataclass
class Clock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now


def _recorder(options: RecordingOptions | None = None) -> tuple[BucketSortRecorder, FakeLink]:
    link = FakeLink()
    recorder = BucketSortRecorder(lambda: link, clock=Clock())
    recorder.enable(options or RecordingOptions())
    return recorder, link


def _succeed(recorder: BucketSortRecorder, link: FakeLink, reset: int, *, clean: bool = True) -> None:
    """Report a success, then let it hold until the recorder ends the episode."""
    sent = len(link.sent)
    assert not recorder.update(_progress(reset, "success", clean=clean, sim_time=10.0)).hold_arm  # the arm settles
    assert len(link.sent) == sent, "a success is not ended on its first tick"
    assert recorder.update(_progress(reset, "success", clean=clean, sim_time=10.0 + SUCCESS_HOLD_S)).hold_arm


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

    _succeed(recorder, link, 1)
    assert link.events()[-1] == "save_episode"
    link.ack_last()
    link.state = dict(TELEOP)
    assert recorder.update(_progress(1, "success", clean=True)).hold_arm
    assert recorder.phase == "waiting"
    assert recorder.status()["saved"] == 1

    _start_episode(recorder, link, 2)
    _succeed(recorder, link, 2)
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
    _succeed(recorder, link, 1, clean=clean)
    assert link.events()[-1] == expected
    link.ack_last()
    recorder.update(_progress(1))
    assert recorder.status()["saved" if expected == "save_episode" else "discarded"] == 1


@pytest.mark.parametrize(
    "fields",
    [
        {"recovery_fraction": -0.1},
        {"recovery_fraction": MAX_RECOVERY_FRACTION + 0.01},
        {"recovery_fraction": float("nan")},
        {"max_attempts": 0},
        {"max_attempts": MAX_ATTEMPTS + 1},
    ],
)
def test_recovery_share_and_attempts_are_validated(fields: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=next(iter(fields))):
        RecordingOptions(**fields)
    with pytest.raises(ValueError, match=next(iter(fields))):
        StudioRecordingRequest(enabled=True, **fields)


def test_recovery_and_attempts_default_to_the_demonstrators_behaviour() -> None:
    options = RecordingOptions()
    assert options.recovery_fraction == 0.0
    assert options.max_attempts == DEFAULT_MAX_ATTEMPTS == bd.DEFAULT_MAX_ATTEMPTS
    request = StudioRecordingRequest(enabled=True, recovery_fraction=0.15, max_attempts=3)
    assert (request.recovery_fraction, request.max_attempts) == (0.15, 3)


@pytest.mark.parametrize(("fraction", "stuck"), [(0.0, False), (MAX_RECOVERY_FRACTION, True)])
def test_stuck_starts_follow_the_recovery_share(fraction: float, stuck: bool) -> None:  # noqa: FBT001
    """A draw of 0 is below any share but 0: share 0 never starts stuck, the largest share (drawn 0) always does."""
    link = FakeLink()
    recorder = BucketSortRecorder(lambda: link, clock=Clock(), chance=lambda: 0.0)
    recorder.enable(RecordingOptions(keep="all", recovery_fraction=fraction))
    recorder.update(_progress())
    for reset in range(1, 6):
        link.state = dict(TELEOP)
        assert recorder.update(_progress(reset - 1)).reset_scene
        directive = recorder.update(_progress(reset))
        assert directive.stuck_start is stuck and directive.hold_arm
        if stuck:
            assert recorder.phase == "positioning"
            moving = dataclasses.replace(_progress(reset), stuck_start="moving")
            assert recorder.update(moving).hold_arm and recorder.phase == "positioning"
            assert link.events().count("start_recording") == reset - 1  # not before the arm is there
            recorder.update(dataclasses.replace(_progress(reset), stuck_start="ready"))
        assert link.sent[-1][:2] == ("start_recording", {"task": f"put the CPU in the red bucket ({reset})"})
        link.state = {**TELEOP, "is_recording": True}
        assert not recorder.update(_progress(reset)).hold_arm
        recorder.update(_progress(reset, "failed", clean=False))
        link.ack_last()
        link.state = dict(TELEOP)
        recorder.update(_progress(reset))
    assert recorder.status()["stuck_starts"] == (5 if stuck else 0)


@pytest.mark.parametrize("state", ["disturbed", "unavailable"])
def test_a_stuck_start_that_disturbed_the_layout_resets_for_a_normal_start(state: str) -> None:
    link = FakeLink()
    recorder = BucketSortRecorder(lambda: link, clock=Clock(), chance=lambda: 0.0)
    recorder.enable(RecordingOptions(recovery_fraction=MAX_RECOVERY_FRACTION))
    recorder.update(_progress())
    link.state = dict(TELEOP)
    assert recorder.update(_progress(0)).reset_scene
    assert recorder.update(_progress(1)).stuck_start
    directive = recorder.update(dataclasses.replace(_progress(1), stuck_start=state))
    if state == "unavailable":  # the arm did not move: a normal start, as the scene stands
        assert not directive.reset_scene and link.events() == ["start_recording"]
        return
    assert directive.reset_scene and recorder.phase == "resetting" and "start_recording" not in link.events()
    directive = recorder.update(_progress(2))
    assert not directive.stuck_start and link.events() == ["start_recording"]  # a normal start, this once
    assert recorder.status()["stuck_starts"] == 0


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


def test_a_success_that_does_not_hold_is_not_saved() -> None:
    recorder, link = _recorder(RecordingOptions(keep="all"))
    recorder.update(_progress())
    _start_episode(recorder, link, 1)
    recorder.update(_progress(1, "success", clean=True, sim_time=10.0))
    recorder.update(_progress(1, "success", clean=True, placed=False, sim_time=10.5))  # it bounced out
    assert not recorder.update(_progress(1, "success", clean=True, sim_time=11.1)).hold_arm  # the hold restarted
    assert link.events()[-1] == "start_recording"
    hold_timed_out = 10.0 + SUCCESS_HOLD_TIMEOUT_S + 0.1
    assert recorder.update(_progress(1, "success", clean=True, placed=False, sim_time=hold_timed_out)).hold_arm
    assert link.events()[-1] == "discard_episode"
    assert recorder.status()["message"].startswith("The object did not stay in its bucket.")


def test_disabling_with_discard_discards_the_episode_studio_records() -> None:
    recorder, link = _recorder()
    recorder.update(_progress())
    _start_episode(recorder, link, 1)
    recorder.disable("The scene is gone.", discard=True)
    assert link.events()[-1] == "discard_episode"
    assert link.closed

    recorder, link = _recorder()
    recorder.update(_progress())
    recorder.disable("Switched off.", discard=True)  # nothing recording: nothing to discard
    assert "discard_episode" not in link.events()


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


@pytest.fixture
def bucket_robot() -> Iterator[MuJoCoRobot]:
    robot = MuJoCoRobot("franka_fr3", scene="bucket_sort", cameras=[], http_port=0, seed=1)
    robot.connect()
    yield robot
    robot.disconnect()


def _recording_with_fake_link(robot: MuJoCoRobot) -> FakeLink:
    """Switch bucket_sort recording on against a scripted Studio that teleoperates and records."""
    link = FakeLink()
    automation = robot._automation  # noqa: SLF001
    automation.bucket_recorder = BucketSortRecorder(lambda: link)
    automation.set_autopilot("leader")
    assert automation.set_studio_recording(RecordingOptions(keep="all"))
    return link


def test_a_reset_publishes_the_reset_pose_before_the_next_tick(bucket_robot: MuJoCoRobot) -> None:
    robot = bucket_robot
    _recording_with_fake_link(robot)
    robot.get_observation()
    robot.send_action(np.asarray(robot._automation.leader_snapshot()["joint_positions"]) + 10.0)  # noqa: SLF001
    robot.get_observation()
    robot._reset()  # noqa: SLF001 - as the recorder's reset request runs it, at the end of a tick
    published = np.asarray(robot._automation.leader_snapshot()["joint_positions"])  # noqa: SLF001
    np.testing.assert_allclose(published, robot._targets_to_units(robot._arm_targets()), atol=1e-6)  # noqa: SLF001


def test_leaving_bucket_sort_discards_the_episode_studio_records(bucket_robot: MuJoCoRobot) -> None:
    robot = bucket_robot
    link = _recording_with_fake_link(robot)
    recorder = robot._automation.bucket_recorder  # noqa: SLF001
    for _ in range(200):
        if recorder.phase == "starting":
            link.state = {**TELEOP, "is_recording": True}
        robot.get_observation()
        if recorder.phase == "recording":
            break
    assert recorder.phase == "recording", recorder.status()
    robot._automation.attach_scene(None, rng=np.random.default_rng(0), active=True)  # noqa: SLF001 - a scene switch
    assert link.events()[-1] == "discard_episode"
    assert recorder.phase == "off"


@pytest.mark.slow
@pytest.mark.parametrize("success_dwell_s", [None, 0.5])
def test_simulation_records_a_led_bucket_sort_episode_into_studio(
    studio: FakeStudio, success_dwell_s: float | None
) -> None:
    robot = MuJoCoRobot("franka_fr3", scene="bucket_sort", cameras=[], http_port=0, seed=1)
    robot.connect()
    if success_dwell_s is not None:  # shorter than the recorder's success hold: the recorder still saves
        robot._episode_auto_reset.set_dwell(success_dwell_s)  # noqa: SLF001
    automation = robot._automation  # noqa: SLF001
    automation.bucket_recorder = BucketSortRecorder(
        lambda: StudioLink(f"http://127.0.0.1:{studio.port}", "sim", discover=lambda _url, _owner: TARGET)
    )
    task = robot._sim.on_reset  # noqa: SLF001
    gripper = robot.joint_names.index("gripper")
    recorded: list[tuple[float, float]] = []  # (action, state) of the gripper while Studio records
    first_frames: list[tuple[np.ndarray, np.ndarray]] = []  # each episode's (action, state) arm joints, frame 0
    recording = [False]

    def teleop_tick() -> None:
        """One tick of Studio's teleoperation: read the leader, send it as the follower's action."""
        state = robot.get_observation().joint_positions
        action = np.asarray(automation.leader_snapshot()["joint_positions"], dtype=np.float32)
        if studio.state.get("is_recording"):
            recorded.append((float(action[gripper]), float(state[gripper])))
            if not recording[0]:
                first_frames.append((action[:gripper].copy(), state[:gripper].copy()))
        recording[0] = bool(studio.state.get("is_recording"))
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
        _wait(lambda: (teleop_tick(), len(first_frames) == 2)[1])
        # Both episodes start at the reset pose, with the action where the arm is. Without holding the
        # reset pose, Studio's last action of episode 1 pulls the arm back to where that episode ended.
        (action1, state1), (action2, state2) = first_frames
        assert np.max(np.abs(state2 - state1)) < 2.0, (state1, state2)  # degrees
        assert np.max(np.abs(action1 - state1)) < 2.0
        assert np.max(np.abs(action2 - state2)) < 2.0

        robot._commands.put(SetStudioRecordingCommand(options=None))  # noqa: SLF001
        robot.get_observation()
        assert "set_follower_source" not in studio.events()
    finally:
        robot.disconnect()


def _stuck_recorder(robot: MuJoCoRobot, studio: FakeStudio, **fields: Any) -> BucketSortRecorder:  # noqa: ANN401
    """Record with every episode starting stuck (a chance of 0 is below any share but 0)."""
    automation = robot._automation  # noqa: SLF001
    automation.bucket_recorder = BucketSortRecorder(
        lambda: StudioLink(f"http://127.0.0.1:{studio.port}", "sim", discover=lambda _url, _owner: TARGET),
        chance=lambda: 0.0,
    )
    automation.set_autopilot("leader")
    options = RecordingOptions(keep="all", recovery_fraction=MAX_RECOVERY_FRACTION, max_attempts=3, **fields)
    robot._commands.put(SetStudioRecordingCommand(options=options))  # noqa: SLF001
    return automation.bucket_recorder


@pytest.mark.slow
def test_an_episode_that_starts_stuck_is_recorded_from_the_stuck_pose(studio: FakeStudio) -> None:
    """The move to the stuck pose is not recorded: frame 0 is there, and the episode recovers and is saved."""
    robot = MuJoCoRobot("franka_fr3", scene="bucket_sort", cameras=[], http_port=0, seed=1)
    robot.connect()
    try:
        recorder = _stuck_recorder(robot, studio)
        automation = robot._automation  # noqa: SLF001
        gripper = robot.joint_names.index("gripper")
        home = robot.get_observation().joint_positions.copy()
        frames: list[tuple[np.ndarray, np.ndarray]] = []  # (action, state) while Studio records
        deadline = time.monotonic() + 120.0
        while "save_episode" not in studio.events() and "discard_episode" not in studio.events():
            assert time.monotonic() < deadline, automation.status()
            state = robot.get_observation().joint_positions
            action = np.asarray(automation.leader_snapshot()["joint_positions"], dtype=np.float64)
            if studio.state.get("is_recording"):
                frames.append((action, np.asarray(state, dtype=np.float64)))
            robot.send_action(action.astype(np.float32))
        assert studio.events() == ["start_recording", "save_episode"], automation.status()
        assert recorder.status()["stuck_starts"] == 1
        assert automation.autopilot.demonstrator.max_attempts == 3  # noqa: PLR2004
        action, state = frames[0]
        np.testing.assert_allclose(action[:gripper], state[:gripper], atol=0.5)  # degrees: no jump at the start
        assert np.abs(state[:gripper] - home[:gripper]).max() > 5.0  # noqa: PLR2004 - it started elsewhere than home
        assert state[gripper] <= bd.STUCK_GRIP[1] + 0.002 and action[gripper] <= bd.STUCK_GRIP[1]  # closed, empty
        squeeze = min(a[gripper] - s_[gripper] for a, s_ in frames)
        assert squeeze < -0.005, f"the recorded gripper never commanded past the fingers ({squeeze:.4f} m)"
    finally:
        robot.disconnect()


@pytest.mark.slow
@pytest.mark.parametrize("lost", ["autopilot off", "teleop stopped"])
def test_losing_the_leader_or_teleop_mid_move_stops_the_stuck_start(lost: str) -> None:
    """The unrecorded move stops at once: no more targets from it, no recording, client actions count again."""
    robot = MuJoCoRobot("franka_fr3", scene="bucket_sort", cameras=[], http_port=0, seed=1)
    robot.connect()
    try:
        link = FakeLink(state=dict(TELEOP))
        automation = robot._automation  # noqa: SLF001
        automation.bucket_recorder = BucketSortRecorder(lambda: link, chance=lambda: 0.0)
        automation.set_autopilot("leader")
        options = RecordingOptions(keep="all", recovery_fraction=MAX_RECOVERY_FRACTION)
        robot._commands.put(SetStudioRecordingCommand(options=options))  # noqa: SLF001
        for _ in range(500):
            robot.get_observation()
            stuck = automation._stuck  # noqa: SLF001
            if stuck is not None and stuck.state == "moving" and stuck.tick > 5:  # noqa: PLR2004
                break
        else:
            pytest.fail(f"no stuck start began: {automation.status()}")
        if lost == "autopilot off":
            automation.set_autopilot("off")
        else:
            link.state = {**TELEOP, "follower_source": "hold"}
        robot.get_observation()
        recorder = automation.bucket_recorder
        assert recorder.phase == "waiting" and recorder.status()["message"].startswith("Waiting for")
        assert automation._stuck is None and not automation.drives_arm  # noqa: SLF001
        held = robot._arm_targets().copy()  # noqa: SLF001
        for _ in range(20):
            robot.get_observation()
        np.testing.assert_array_equal(robot._arm_targets(), held)  # noqa: SLF001 - the move wrote nothing more
        assert "start_recording" not in link.events()
        action = robot.get_observation().joint_positions.astype(np.float32)
        action[0] += 5.0  # degrees: a client moves the base joint
        robot.send_action(action)
        for _ in range(10):
            robot.get_observation()
        assert robot._arm_targets()[0] == pytest.approx(np.radians(action[0]), abs=1e-4)  # noqa: SLF001
    finally:
        robot.disconnect()


@pytest.mark.slow
def test_a_stuck_start_that_moves_the_layout_falls_back_to_a_normal_start(studio: FakeStudio, monkeypatch) -> None:
    monkeypatch.setattr(conveyor_automation, "STUCK_DISTURBANCE", -1.0)  # every move counts as disturbing
    robot = MuJoCoRobot("franka_fr3", scene="bucket_sort", cameras=[], http_port=0, seed=1)
    robot.connect()
    try:
        recorder = _stuck_recorder(robot, studio)
        task = robot._sim.on_reset  # noqa: SLF001
        resets = task.reset_count
        home = robot.get_observation().joint_positions.copy()
        deadline = time.monotonic() + 30.0
        while "start_recording" not in studio.events():
            assert time.monotonic() < deadline, robot._automation.status()  # noqa: SLF001
            robot.get_observation()
        assert task.reset_count == resets + 2  # the episode's reset, then one more after the disturbing move
        assert recorder.status()["stuck_starts"] == 0
        _wait(lambda: studio.state.get("is_recording"))
        start = robot.get_observation().joint_positions
        np.testing.assert_allclose(start, home, atol=0.5)  # a normal start: the arm at its reset pose
    finally:
        robot.disconnect()


@pytest.mark.slow
@pytest.mark.parametrize(
    ("profile", "grippers", "per_finger"),
    [("franka_panda", ("actuator8",), True), ("franka_duo", ("left_gripper", "right_gripper"), False)],
)
def test_the_other_franka_robots_record_through_the_same_leader(
    studio: FakeStudio,
    profile: str,
    grippers: tuple[str, ...],
    per_finger: bool,  # noqa: FBT001
) -> None:
    """The recorder is the FR3's: the leader carries the profile's channels, whichever arm works."""
    robot = MuJoCoRobot(profile, scene="bucket_sort", cameras=[], http_port=0, seed=1)
    robot.connect()
    automation = robot._automation  # noqa: SLF001
    automation.bucket_recorder = BucketSortRecorder(
        lambda: StudioLink(f"http://127.0.0.1:{studio.port}", "sim", discover=lambda _url, _owner: TARGET)
    )
    columns = [robot.joint_names.index(name) for name in grippers]
    squeeze = 0.0
    try:
        automation.set_autopilot("leader")
        robot._commands.put(SetStudioRecordingCommand(options=RecordingOptions(keep="all")))  # noqa: SLF001
        deadline = time.monotonic() + 120.0
        while "save_episode" not in studio.events() and "discard_episode" not in studio.events():
            assert time.monotonic() < deadline, automation.status()
            state = robot.get_observation().joint_positions
            leader = automation.leader_snapshot()
            assert leader["joint_names"] == list(robot.joint_names)
            action = np.asarray(leader["joint_positions"], dtype=np.float32)
            if studio.state.get("is_recording"):
                squeeze = min(squeeze, *(float(action[c] - state[c]) for c in columns))
            robot.send_action(action)
        assert studio.events() == ["start_recording", "save_episode"]
        assert robot._http_status()["episode"]["success"]  # noqa: SLF001
        # The Panda's actuator8 channel is one finger's travel, half the hand's opening.
        assert squeeze < (-0.0025 if per_finger else -0.005), squeeze
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
