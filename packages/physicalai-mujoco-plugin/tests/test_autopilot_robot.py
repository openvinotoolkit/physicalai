"""Autopilot episodes applied on the simulation thread, with real MuJoCo models and no display.

Physical AI Studio is played by the test: it starts episodes in the robot's episode book (as
``/autopilot/v1`` does), and each tick reads the leader (as ``GET /leader`` does, a sign of life) and,
when it teleoperates, sends the leader's targets back as the follower's action.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.request
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from physicalai_studio_plugin.autopilot import LEADER_TIMEOUT_S, Episode, EpisodeOptions, EpisodeRequest

from physicalai_mujoco_plugin import control, conveyor_automation
from physicalai_mujoco_plugin.autopilot import STOPPED_READING_REASON, STUDIO_ABORT_REASON
from physicalai_mujoco_plugin.conveyor_automation import CONVEYOR_TASK, DEFAULT_MAX_ATTEMPTS
from physicalai_mujoco_plugin.http_server import (
    HomeCommand,
    ReplayCommand,
    ResetCommand,
    SetAutopilotCommand,
    SetAutoResetCommand,
    SetBeltSpeedCommand,
    SetObjectPoseCommand,
    SetOverviewCommand,
    SetSeedCommand,
    ShutdownCommand,
    StopReplayCommand,
    SwitchSceneCommand,
    build_app,
)
from physicalai_mujoco_plugin.robot import MuJoCoRobot

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from physicalai_mujoco_plugin.autopilot import EpisodeBook


def _book(robot: MuJoCoRobot) -> EpisodeBook:
    return robot._automation.book  # noqa: SLF001


def _read(robot: MuJoCoRobot) -> dict[str, object]:
    """Read the leader as ``GET /leader`` does: a sign of life."""
    return robot._automation.read_leader()  # noqa: SLF001


def _teleop(robot: MuJoCoRobot) -> np.ndarray:
    """One tick of Studio following the leader: measure, read the leader, send its targets back."""
    state = robot.get_observation().joint_positions
    robot.send_action(np.asarray(_read(robot)["joint_positions"], dtype=np.float32))
    return state


def _watch(robot: MuJoCoRobot) -> None:
    """One tick of Studio reading the leader without sending anything: nobody follows."""
    robot.get_observation()
    _read(robot)


def _until(robot: MuJoCoRobot, done: Callable[[], bool], *, timeout_s: float, step: Callable | None = None) -> None:
    deadline = time.monotonic() + timeout_s
    while not done():
        assert time.monotonic() < deadline, robot._http_status()["autopilot_episode"]  # noqa: SLF001
        (step or _teleop)(robot)


def _phase(robot: MuJoCoRobot, episode: Episode) -> str:
    return _book(robot).get(episode.id).phase


def _ready(robot: MuJoCoRobot, options: EpisodeOptions | None = None, *, timeout_s: float = 30.0) -> Episode:
    episode = _book(robot).create(EpisodeRequest(options=options or EpisodeOptions()))
    _until(robot, lambda: _phase(robot, episode) == "ready", timeout_s=timeout_s)
    return _book(robot).get(episode.id)


def _running(robot: MuJoCoRobot) -> Episode:
    episode = _ready(robot)
    _book(robot).begin(episode.id)
    _teleop(robot)
    assert _phase(robot, episode) == "running"
    return episode


@pytest.fixture
def conveyor_robot() -> Iterator[MuJoCoRobot]:
    robot = MuJoCoRobot(scene="conveyor_sort", cameras=[], substeps=10)
    robot.connect()
    robot._set_belt_speed(0.10)  # noqa: SLF001 - the fastest belt, for a short test
    conveyor = robot._automation.conveyor  # noqa: SLF001
    assert conveyor is not None
    conveyor._config.items_per_episode = 2  # noqa: SLF001
    yield robot
    robot.disconnect()


@pytest.fixture
def bucket_robot() -> Iterator[MuJoCoRobot]:
    robot = MuJoCoRobot("franka_fr3", scene="bucket_sort", cameras=[], seed=1)
    robot.connect()
    yield robot
    robot.disconnect()


# ----------------------------------------------------------------------
# What the simulation reports
# ----------------------------------------------------------------------


def test_the_conveyor_and_bucket_sort_scenes_have_an_autopilot_and_others_say_why_not(
    conveyor_robot: MuJoCoRobot,
) -> None:
    info = _book(conveyor_robot).info()
    assert (info.supported, info.scene, info.options) == (True, "conveyor_sort", [])
    assert info.joint_names == conveyor_robot.joint_names
    robot = MuJoCoRobot("so101", scene="single_pick_place", substeps=1, cameras=[])
    robot.connect()
    try:
        info = _book(robot).info()
        assert info.supported is False
        assert "no autopilot" in (info.reason or "")
    finally:
        robot.disconnect()


def test_the_leader_route_carries_the_episode_id() -> None:
    from physicalai_mujoco_plugin.studio_catalog import autopilot_leader  # noqa: PLC0415

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    robot = MuJoCoRobot(scene="conveyor_sort", cameras=[], substeps=10, http_port=port)
    robot.connect()
    leader = autopilot_leader(None, f"http://127.0.0.1:{port}")  # type: ignore[arg-type] - as Studio builds it
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        robot.get_observation()
        with opener.open(f"http://127.0.0.1:{port}/leader", timeout=5) as response:
            sample = json.loads(response.read())
        assert (sample["episode_id"], sample["phase"]) == (None, None)
        episode = _book(robot).create(EpisodeRequest())
        with opener.open(f"http://127.0.0.1:{port}/leader", timeout=5) as response:
            sample = json.loads(response.read())
        assert (sample["episode_id"], sample["phase"]) == (episode.id, "preparing")
        leader.connect()
        _until(robot, lambda: leader.get_observation().phase == "ready", timeout_s=10.0, step=MuJoCoRobot.get_observation)
        assert leader.episode_id == episode.id
        assert leader.get_observation().episode_id == episode.id
    finally:
        leader.disconnect()
        robot.disconnect()


# ----------------------------------------------------------------------
# The lifecycle, for each outcome
# ----------------------------------------------------------------------


def test_a_conveyor_episode_nobody_follows_ends_as_a_failure_and_gives_the_scene_back(
    conveyor_robot: MuJoCoRobot,
) -> None:
    robot = conveyor_robot
    automation = robot._automation  # noqa: SLF001
    conveyor = automation.conveyor
    conveyor.set_active(False)  # the user paused the belt before the episode
    robot.get_observation()
    episode = _book(robot).create(EpisodeRequest())
    _until(robot, lambda: _phase(robot, episode) == "ready", timeout_s=10.0, step=_watch)
    ready = _book(robot).get(episode.id)
    assert ready.task == CONVEYOR_TASK
    assert conveyor.feed_held
    assert automation.drives_arm  # the arm holds its start pose until begin
    spawned = conveyor.status()["spawned"]
    for _ in range(20):  # ready: the belt does not feed
        _watch(robot)
    assert conveyor.status()["spawned"] == spawned
    _book(robot).begin(episode.id)
    _watch(robot)
    assert conveyor.active  # the feed runs for the episode
    assert not conveyor.feed_held
    assert automation.autopilot.mode == "leader"
    _until(robot, lambda: _phase(robot, episode) == "ended", timeout_s=240.0, step=_watch)
    ended = _book(robot).get(episode.id)
    assert (ended.outcome, ended.clean) == ("failure", False), ended
    assert "missed" in (ended.reason or "")
    # The scene settings from before the episode are back.
    assert not conveyor.active
    assert not conveyor.feed_held
    assert automation.autopilot.mode == "off"
    assert not automation.drives_arm
    # The leader keeps returning the demonstrator's last target.
    last = _read(robot)["joint_positions"]
    for _ in range(10):
        _watch(robot)
    assert _read(robot)["joint_positions"] == last


def test_a_conveyor_episode_that_takes_too_long_stalls(
    conveyor_robot: MuJoCoRobot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(conveyor_automation.EPISODE_LIMIT_S, "conveyor", 1.0)
    robot = conveyor_robot
    episode = _running(robot)
    _until(robot, lambda: _phase(robot, episode) == "ended", timeout_s=30.0)
    ended = _book(robot).get(episode.id)
    assert (ended.outcome, ended.reason) == ("stalled", "The episode ran out of time (1 s simulated).")


@pytest.mark.slow
def test_a_bucket_sort_episode_is_prepared_led_and_succeeds(bucket_robot: MuJoCoRobot) -> None:
    robot = bucket_robot
    automation = robot._automation  # noqa: SLF001
    task = robot._sim.on_reset  # noqa: SLF001
    helper = robot._episode_auto_reset  # noqa: SLF001
    robot.get_observation()
    resets = task.reset_count
    episode = _ready(robot, EpisodeOptions(max_attempts=3))
    assert task.reset_count == resets + 1  # a fresh layout
    assert episode.task == task.instruction.text
    assert automation.autopilot.demonstrator.max_attempts == 3  # noqa: PLR2004
    assert helper._resets_suspended  # noqa: SLF001 - the scene's own success reset waits
    # The leader publishes where the reset left the arm; client actions are ignored until begin.
    published = np.asarray(_read(robot)["joint_positions"])
    np.testing.assert_allclose(published, robot._targets_to_units(robot._arm_targets()), atol=1e-4)  # noqa: SLF001
    gripper = robot.joint_names.index("gripper")
    recorded: list[tuple[float, float]] = []
    _book(robot).begin(episode.id)
    deadline = time.monotonic() + 150.0
    while _phase(robot, episode) != "ended":
        assert time.monotonic() < deadline, automation.status()
        state = _teleop(robot)
        action = np.asarray(_read(robot)["joint_positions"])
        recorded.append((float(action[gripper]), float(state[gripper])))
    ended = _book(robot).get(episode.id)
    assert ended.outcome == "success", ended
    assert ended.clean is not None
    assert robot._http_status()["episode"]["success"]  # noqa: SLF001
    # The actions are the demonstrator's commands: the gripper closes past where the object stops it.
    squeeze = min(action - state for action, state in recorded)
    assert squeeze < -0.005, f"the gripper never commanded past the fingers ({squeeze:.4f} m)"
    # Ended: the demonstrator stops, the leader keeps its last target, the scene settings are back.
    phase = automation.autopilot.status()["phase"]
    last = _read(robot)["joint_positions"]
    for _ in range(20):
        _teleop(robot)
    assert automation.autopilot.status()["phase"] == phase
    assert _read(robot)["joint_positions"] == last
    assert not helper._resets_suspended  # noqa: SLF001
    assert automation._max_attempts == DEFAULT_MAX_ATTEMPTS  # noqa: SLF001
    assert automation.autopilot.mode == "off"


@pytest.mark.slow
def test_a_bucket_sort_episode_can_start_stuck(bucket_robot: MuJoCoRobot) -> None:
    """With a recovery share, the arm moves to a stuck pose before ``ready``; the episode begins there."""
    from physicalai_mujoco_plugin import bucket_sort_demo as bd  # noqa: PLC0415

    robot = bucket_robot
    automation = robot._automation  # noqa: SLF001
    automation._chance = lambda: 0.0  # noqa: SLF001 - below any share above 0
    gripper = robot.joint_names.index("gripper")
    home = robot.get_observation().joint_positions.copy()
    _ready(robot, EpisodeOptions(recovery_fraction=0.5), timeout_s=60.0)
    assert automation.status()["autopilot_episode"]["stuck_starts"] == 1
    state = _teleop(robot)
    assert np.abs(state[:gripper] - home[:gripper]).max() > 5.0  # noqa: PLR2004 - degrees: it waits elsewhere
    assert state[gripper] <= bd.STUCK_GRIP[1] + 0.002  # closed and empty
    action = np.asarray(_read(robot)["joint_positions"])
    np.testing.assert_allclose(action[:gripper], state[:gripper], atol=0.5)  # no jump when it begins


def test_studio_aborts_a_running_episode_and_the_demonstrator_stops_at_once(bucket_robot: MuJoCoRobot) -> None:
    robot = bucket_robot
    automation = robot._automation  # noqa: SLF001
    episode = _ready(robot)
    _book(robot).begin(episode.id)
    for _ in range(30):
        _teleop(robot)
    aborted = _book(robot).abort(episode.id)
    assert (aborted.phase, aborted.outcome, aborted.reason) == ("ended", "aborted", STUDIO_ABORT_REASON)
    last = _read(robot)["joint_positions"]
    robot.get_observation()
    assert automation.autopilot.mode == "off"
    for _ in range(10):
        _teleop(robot)
    assert _read(robot)["joint_positions"] == last  # nothing more from the demonstrator


def test_a_new_post_replaces_the_episode_in_progress(conveyor_robot: MuJoCoRobot) -> None:
    robot = conveyor_robot
    automation = robot._automation  # noqa: SLF001
    first = _running(robot)
    second = _book(robot).create(EpisodeRequest())
    assert second.id != first.id
    _until(robot, lambda: _phase(robot, second) == "ready", timeout_s=10.0)
    assert automation.conveyor.feed_held  # laid out anew for the second episode
    assert automation.autopilot.mode == "leader"
    assert robot._http_status()["autopilot_episode"]["episode"]["id"] == second.id  # noqa: SLF001


# ----------------------------------------------------------------------
# Liveness
# ----------------------------------------------------------------------


def test_one_second_without_a_leader_read_aborts_the_episode(conveyor_robot: MuJoCoRobot) -> None:
    robot = conveyor_robot
    automation = robot._automation  # noqa: SLF001
    automation.conveyor._config.items_per_episode = 1000  # noqa: SLF001 - it must not end by itself meanwhile
    episode = _running(robot)
    alive_until = time.monotonic() + 1.5 * LEADER_TIMEOUT_S
    while time.monotonic() < alive_until:  # leader reads keep it going past the timeout
        _teleop(robot)
    assert _phase(robot, episode) == "running"
    started = time.monotonic()
    while _book(robot).summary()["episode"]["phase"] != "ended":  # type: ignore[index]
        robot.get_observation()  # the simulation ticks, nobody reads the leader
        assert time.monotonic() - started < 5.0  # noqa: PLR2004
    assert time.monotonic() - started >= LEADER_TIMEOUT_S - 0.05
    ended = _book(robot).get(episode.id)
    assert (ended.outcome, ended.reason) == ("aborted", STOPPED_READING_REASON)
    robot.get_observation()  # the next tick gives the scene back
    assert not automation.conveyor.feed_held
    assert automation.autopilot.mode == "off"


# ----------------------------------------------------------------------
# Every other change to the scene ends the episode first, then applies
# ----------------------------------------------------------------------


def _object_joint(robot: MuJoCoRobot) -> str:
    return str(robot._http_status()["objects"][0]["joint"])  # noqa: SLF001


def _replay_frames(robot: MuJoCoRobot) -> np.ndarray:
    return np.tile(robot.get_observation().joint_positions, (3, 1))


CHANGES: dict[str, tuple[Callable[[MuJoCoRobot], object], str, Callable[[MuJoCoRobot], bool]]] = {
    "reset": (lambda _r: ResetCommand(), "The scene was reset.", lambda r: not r._automation.drives_arm),  # noqa: SLF001
    "switch": (
        lambda _r: SwitchSceneCommand(scene_id="conveyor_sort"),
        "The scene was switched.",
        lambda r: r._current_scene_id == "conveyor_sort",  # noqa: SLF001
    ),
    "home": (lambda _r: HomeCommand(), "The arm was sent home.", lambda r: not r._automation.drives_arm),  # noqa: SLF001
    "seed": (lambda _r: SetSeedCommand(seed=7), "The seed changed.", lambda r: r._seed == 7),  # noqa: SLF001, PLR2004
    "auto_reset": (
        lambda _r: SetAutoResetCommand(enabled=False),
        "The episode auto-reset settings changed.",
        lambda r: not r._automation.conveyor.active,  # noqa: SLF001 - not undone by the episode's belt restore
    ),
    "belt_speed": (
        lambda _r: SetBeltSpeedCommand(speed=0.02),
        "The belt speed changed.",
        lambda r: r._automation.belt_speed == pytest.approx(0.02),  # noqa: SLF001
    ),
    "object_pose": (
        lambda r: SetObjectPoseCommand(joint=_object_joint(r), position=(0.0, 0.3, 0.2)),
        "An object was moved.",
        lambda r: True,  # noqa: ARG005
    ),
    "autopilot": (
        lambda _r: SetAutopilotCommand(mode="drive"),
        "The autopilot mode changed.",
        lambda r: r._automation.autopilot.mode == "drive",  # noqa: SLF001 - not switched off after it
    ),
    "overview": (
        lambda _r: SetOverviewCommand(style="front"),
        "The overview camera moved.",
        lambda r: r._overview == "front" or "front" not in r._overview_styles(),  # noqa: SLF001
    ),
    "replay": (
        lambda r: ReplayCommand(joint_positions=_replay_frames(r), fps=30.0),
        "A replay started.",
        lambda r: r._replay is not None and not r._automation.book.info().supported,  # noqa: SLF001
    ),
    "replay_stop": (lambda _r: StopReplayCommand(), "A replay was stopped.", lambda r: r._replay is None),  # noqa: SLF001
    "shutdown": (lambda _r: ShutdownCommand(), "The simulation is shutting down.", lambda _r: True),
}


@pytest.mark.parametrize("source", ["http", "viewer"])
@pytest.mark.parametrize("change", list(CHANGES))
def test_a_change_ends_the_episode_first_then_applies(conveyor_robot: MuJoCoRobot, change: str, source: str) -> None:
    robot = conveyor_robot
    make, reason, applied = CHANGES[change]
    episode = _running(robot)
    command = make(robot)
    with patch.object(control, "_signal_owner_shutdown") as shutdown:
        if source == "viewer":
            robot._submit_command(command)  # noqa: SLF001 - as the viewer panel does
        else:
            robot._commands.put(command)  # noqa: SLF001 - as every mutating HTTP route does
        robot.get_observation()
    ended = _book(robot).get(episode.id)
    assert (ended.phase, ended.outcome, ended.reason) == ("ended", "aborted", reason)
    assert applied(robot), change
    assert shutdown.called == (change == "shutdown")


def test_a_mutating_route_ends_the_episode_over_http(conveyor_robot: MuJoCoRobot) -> None:
    robot = conveyor_robot
    episode = _running(robot)
    app = build_app(
        service_name="sim",
        buffers={},
        commands=robot._commands,  # noqa: SLF001
        get_status=robot._http_status,  # noqa: SLF001
        autopilot=_book(robot),
    )
    assert TestClient(app).post("/conveyor/belt-speed", json={"speed": 0.05}).status_code == 200
    robot.get_observation()
    ended = _book(robot).get(episode.id)
    assert (ended.outcome, ended.reason) == ("aborted", "The belt speed changed.")
    assert robot._automation.belt_speed == pytest.approx(0.05)  # noqa: SLF001


def test_the_native_viewer_reset_ends_the_episode(conveyor_robot: MuJoCoRobot) -> None:
    robot = conveyor_robot
    episode = _running(robot)
    robot._last_sim_time = float(robot._sim.data.time) + 10.0  # noqa: SLF001 - the viewer rewound the clock
    robot._handle_viewer_reset()  # noqa: SLF001
    assert _book(robot).get(episode.id).reason == "The native viewer reset the scene."
    assert not robot._automation.drives_arm  # noqa: SLF001 - manual control again


def test_a_scene_reset_of_its_own_ends_the_episode(conveyor_robot: MuJoCoRobot) -> None:
    robot = conveyor_robot
    episode = _ready(robot)
    robot._run_scene_reset()  # noqa: SLF001 - not the episode's own reset
    assert _book(robot).get(episode.id).reason == "The scene was reset."


def test_disconnecting_ends_the_episode() -> None:
    robot = MuJoCoRobot(scene="conveyor_sort", cameras=[], substeps=10)
    robot.connect()
    try:
        episode = _ready(robot)
    finally:
        robot.disconnect()
    assert _book(robot).get(episode.id).reason == "The simulation stopped."


def test_a_manual_virtual_leader_still_follows_the_autopilot_without_an_episode(conveyor_robot: MuJoCoRobot) -> None:
    """POST /autopilot leader and the catalog's MuJoCo_Virtual_Leader still lead a teleoperated follower."""
    robot = conveyor_robot
    automation = robot._automation  # noqa: SLF001
    robot._commands.put(SetAutopilotCommand(mode="leader"))  # noqa: SLF001
    robot.get_observation()
    first = np.asarray(_read(robot)["joint_positions"])
    _until(
        robot,
        lambda: not np.allclose(np.asarray(_read(robot)["joint_positions"]), first, atol=0.5),
        timeout_s=60.0,
    )
    assert automation.autopilot.mode == "leader"
    assert _read(robot)["episode_id"] is None
