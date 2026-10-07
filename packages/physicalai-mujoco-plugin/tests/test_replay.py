"""Kinematic replay (``POST /replay``) on the driver's control tick."""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING
from unittest.mock import patch

import mujoco
import numpy as np
import pytest
from fastapi.testclient import TestClient

from physicalai_mujoco_plugin.http_server import (
    HomeCommand,
    ReplayCommand,
    ResetCommand,
    StopReplayCommand,
    SwitchSceneCommand,
    build_app,
)
from physicalai_mujoco_plugin.replay import Replay
from physicalai_mujoco_plugin.robot import MuJoCoRobot

if TYPE_CHECKING:
    from collections.abc import Iterator

FRAMES = 40


def _state(robot: MuJoCoRobot) -> np.ndarray:
    """The full integration state (time, qpos, qvel, act, warm start, ctrl, ...)."""
    model, data = robot._model, robot._data  # noqa: SLF001
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    return state


def _record(robot: MuJoCoRobot, frames: int = FRAMES) -> np.ndarray:
    """Drive every joint along a sine sweep and record the observed joint positions."""
    home = robot.get_observation().joint_positions.astype(np.float64)
    phases = np.linspace(0.0, np.pi, len(home))
    recorded = []
    for tick in range(frames):
        robot.send_action(home + 15.0 * np.sin(0.2 * tick + phases))
        recorded.append(robot.get_observation().joint_positions)
    return np.array(recorded)


def _tick_fps(robot: MuJoCoRobot) -> float:
    """The frame rate that shows one frame per control tick."""
    return 1.0 / (robot._substeps * float(robot._model.opt.timestep))  # noqa: SLF001


def _client(robot: MuJoCoRobot) -> TestClient:
    """The driver's HTTP app, without starting a server."""
    return TestClient(
        build_app(
            service_name="test",
            buffers={},
            commands=robot._commands,  # noqa: SLF001
            get_status=robot._http_status,  # noqa: SLF001
            check_replay=robot._replay_problem,  # noqa: SLF001
        )
    )


def _so101(unit: str = "normalized") -> MuJoCoRobot:
    robot = MuJoCoRobot("so101", scene="single_pick_place", unit=unit, cameras=[])  # type: ignore[arg-type]
    robot.connect()
    return robot


@pytest.fixture
def robot() -> Iterator[MuJoCoRobot]:
    robot = _so101()
    yield robot
    robot.disconnect()


@pytest.mark.parametrize("unit", ["normalized", "degrees"])
def test_replay_of_a_recorded_so101_episode_matches_the_recorded_joints(unit: str) -> None:
    robot = _so101(unit)
    try:
        recorded = _record(robot)
        for _ in range(10):  # move on, so the replay has to put the joints back
            robot.get_observation()
        before = _state(robot)
        client = _client(robot)
        fps = _tick_fps(robot)

        response = client.post("/replay", json={"joint_positions": recorded.tolist(), "fps": fps, "base": None})

        assert response.status_code == 200, response.text
        assert response.json()["frames"] == FRAMES
        for index, row in enumerate(recorded):
            observed = robot.get_observation().joint_positions
            np.testing.assert_allclose(observed, row, rtol=0, atol=1e-4, err_msg=f"frame {index}")
            assert client.get("/health").json()["replay"] == {
                "active": True, "finished": False, "frame": index, "frames": FRAMES, "fps": fps, "unsupported_joints": [],
            }  # fmt: skip
        assert robot._data.time == before[0]  # noqa: SLF001 - kinematic: physics never stepped

        # Past the last frame the replay holds it, and actions stay ignored.
        robot.send_action(np.zeros(len(robot.joint_names)))
        np.testing.assert_allclose(robot.get_observation().joint_positions, recorded[-1], rtol=0, atol=1e-4)
        assert client.get("/health").json()["replay"]["finished"] is True

        assert client.post("/replay/stop").status_code == 200
        robot._substeps = 0  # noqa: SLF001 - drain the stop without stepping afterwards
        robot.get_observation()

        np.testing.assert_array_equal(_state(robot), before)
        assert client.get("/health").json()["replay"] == {"active": False, "unsupported_joints": []}
    finally:
        robot.disconnect()


def test_actions_apply_again_after_the_replay(robot) -> None:
    recorded = _record(robot, frames=5)
    robot._submit_command(ReplayCommand(joint_positions=recorded, fps=30.0))  # noqa: SLF001
    robot.get_observation()
    target = recorded[0] + 10.0
    robot.send_action(target)
    robot._submit_command(StopReplayCommand())  # noqa: SLF001
    robot.get_observation()

    robot.send_action(target)
    for _ in range(50):
        observed = robot.get_observation().joint_positions

    np.testing.assert_allclose(observed, target, atol=1.0)


def test_a_new_replay_keeps_the_state_from_before_the_first(robot) -> None:
    frames = np.tile(robot.get_observation().joint_positions + 5.0, (3, 1))
    before = _state(robot)
    robot._submit_command(ReplayCommand(joint_positions=frames, fps=30.0))  # noqa: SLF001
    robot.get_observation()
    robot._submit_command(ReplayCommand(joint_positions=frames - 10.0, fps=30.0))  # noqa: SLF001
    robot.get_observation()
    np.testing.assert_allclose(robot.get_observation().joint_positions, frames[0] - 10.0, atol=1e-4)

    robot._submit_command(StopReplayCommand())  # noqa: SLF001
    robot._substeps = 0  # noqa: SLF001
    robot.get_observation()

    np.testing.assert_array_equal(_state(robot), before)


@pytest.mark.parametrize(("command", "handler"), [(ResetCommand(), "_run_scene_reset"), (HomeCommand(), "_go_home")])
def test_reset_and_home_end_the_replay_first(robot, command, handler: str) -> None:
    frames = np.tile(robot.get_observation().joint_positions + 5.0, (3, 1))
    before = _state(robot)
    robot._submit_command(ReplayCommand(joint_positions=frames, fps=30.0))  # noqa: SLF001
    robot.get_observation()
    robot._substeps = 0  # noqa: SLF001
    seen: list[np.ndarray] = []
    robot._submit_command(command)  # noqa: SLF001

    with patch.object(robot, handler, side_effect=lambda: seen.append(_state(robot))):
        robot.get_observation()

    np.testing.assert_array_equal(seen[0], before)
    assert robot._replay is None  # noqa: SLF001


def test_a_scene_switch_drops_the_replay(robot) -> None:
    frames = np.tile(robot.get_observation().joint_positions, (3, 1))
    robot._submit_command(ReplayCommand(joint_positions=frames, fps=30.0))  # noqa: SLF001
    robot.get_observation()

    robot._submit_command(SwitchSceneCommand(scene_id="yahtzee"))  # noqa: SLF001
    robot.get_observation()

    assert robot._current_scene_id == "yahtzee"  # noqa: SLF001
    assert robot._http_status()["replay"] == {"active": False, "unsupported_joints": []}  # noqa: SLF001


def test_frames_of_the_wrong_width_are_refused_on_the_sim_thread(robot) -> None:
    robot._submit_command(ReplayCommand(joint_positions=np.zeros((3, 5)), fps=30.0))  # noqa: SLF001
    robot.get_observation()

    assert robot._replay is None  # noqa: SLF001


def test_disconnect_drops_the_replay(robot) -> None:
    robot._submit_command(ReplayCommand(joint_positions=np.zeros((3, 6)), fps=30.0))  # noqa: SLF001
    robot.get_observation()
    robot.disconnect()

    assert robot._replay is None  # noqa: SLF001
    assert robot._http_status()["replay"] == {"active": False, "unsupported_joints": []}  # noqa: SLF001


def test_panel_state_carries_the_replay_status(robot) -> None:
    robot._submit_command(ReplayCommand(joint_positions=np.zeros((3, 6)), fps=30.0))  # noqa: SLF001
    robot.get_observation()

    assert robot._panel_state().replay == {  # noqa: SLF001
        "active": True, "finished": False, "frame": 0, "frames": 3, "fps": 30.0, "unsupported_joints": [],
    }  # fmt: skip


def test_the_native_viewer_reset_ends_the_replay_before_the_scene_reset(robot) -> None:
    frames = np.tile(robot.get_observation().joint_positions + 5.0, (3, 1))
    before = _state(robot)
    robot._submit_command(ReplayCommand(joint_positions=frames, fps=30.0))  # noqa: SLF001
    robot.get_observation()
    robot._last_sim_time = float(robot._data.time)  # noqa: SLF001
    mujoco.mj_resetData(robot._model, robot._data)  # noqa: SLF001 - what the native viewer's reset button does
    seen: list[tuple[np.ndarray, object]] = []

    with patch.object(robot, "_run_scene_reset", side_effect=lambda: seen.append((_state(robot), robot._replay))):  # noqa: SLF001
        robot._handle_viewer_reset()  # noqa: SLF001

    state, replay = seen[0]
    np.testing.assert_array_equal(state, before)
    assert replay is None


_TENDON_ARM = """<mujoco><compiler angle="radian"/><option timestep="0.002"/><worldbody>
  <body name="link"><joint name="hinge" axis="0 1 0" range="-2 2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  <body name="link2" pos="0.3 0 0"><joint name="elbow" axis="0 1 0" range="-2 2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/></body></body></worldbody>
  <tendon><fixed name="finger"><joint joint="hinge" coef="1"/></fixed></tendon>
  <actuator><position name="elbow" joint="elbow" kp="20"/><position name="finger" tendon="finger" kp="20"/></actuator>
  </mujoco>"""


def test_a_robot_with_a_tendon_channel_refuses_replay(tmp_path) -> None:
    path = tmp_path / "tendon_arm.xml"
    path.write_text(_TENDON_ARM)
    robot = MuJoCoRobot("ur5e", model_path=str(path), unit="degrees", cameras=[])
    robot.connect()
    try:
        # A trajectory recorded by driving this very model.
        recorded = []
        for tick in range(20):
            robot.send_action(np.array([10.0 * np.sin(0.2 * tick), 0.3 * np.sin(0.2 * tick)]))
            recorded.append(robot.get_observation().joint_positions)
        client = _client(robot)

        response = client.post("/replay", json={"joint_positions": np.array(recorded).tolist(), "fps": 50.0})

        assert response.status_code == 409
        assert "Cannot replay this robot: finger drive no hinge or slide joint directly" in response.json()["detail"]
        replay = client.get("/health").json()["replay"]
        assert (replay["active"], replay["unsupported_joints"]) == (False, ["finger"])
        assert robot._commands.empty()  # noqa: SLF001
        # The sim thread refuses it too, should a command get past the HTTP check.
        robot._submit_command(ReplayCommand(joint_positions=np.array(recorded, dtype=np.float64), fps=50.0))  # noqa: SLF001
        robot.get_observation()
        assert robot._replay is None  # noqa: SLF001
    finally:
        robot.disconnect()


def test_the_simulation_continues_as_if_the_replay_never_ran() -> None:
    def run(*, replay: bool) -> tuple[np.ndarray, np.ndarray]:
        robot = MuJoCoRobot("so101", scene="single_pick_place", cameras=[], seed=1)
        robot.connect()
        try:
            observed = []
            for tick in range(40):
                if replay and tick == 20:
                    robot._submit_command(ReplayCommand(joint_positions=np.zeros((5, 6)), fps=50.0))  # noqa: SLF001
                    for _ in range(8):
                        robot.get_observation()
                    robot._submit_command(StopReplayCommand())  # noqa: SLF001
                    substeps, robot._substeps = robot._substeps, 0  # noqa: SLF001
                    robot.get_observation()
                    robot._substeps = substeps  # noqa: SLF001
                robot.send_action(20.0 * np.sin(0.1 * tick + np.arange(6)))
                observed.append(robot.get_observation().joint_positions)
            return np.array(observed), _state(robot)
        finally:
            robot.disconnect()

    observed, state = run(replay=True)
    expected_observed, expected_state = run(replay=False)

    np.testing.assert_array_equal(observed, expected_observed)
    np.testing.assert_array_equal(state, expected_state)


class TestReplayClock:
    @staticmethod
    def _shown(fps: float, ticks: int, frames: int = 4, control_dt: float = 0.02) -> list[tuple[int, bool]]:
        replay = Replay(np.arange(frames, dtype=np.float64).reshape(-1, 1), fps, np.zeros(1))
        shown = []
        for _ in range(ticks):
            row = replay.next_frame(control_dt)
            assert row[0] == replay.frame
            shown.append((replay.frame, replay.finished))
        return shown

    def test_one_frame_per_tick_at_the_tick_rate(self) -> None:
        assert self._shown(50.0, 5) == [(0, False), (1, False), (2, False), (3, False), (3, True)]

    def test_a_slower_recording_holds_each_frame(self) -> None:
        assert [frame for frame, _ in self._shown(25.0, 6)] == [0, 0, 1, 1, 2, 2]

    def test_a_faster_recording_skips_frames(self) -> None:
        assert self._shown(100.0, 3) == [(0, False), (2, False), (3, True)]

    def test_status_never_reads_a_torn_frame_and_finished_pair(self) -> None:
        # Another thread can read the status between any two attribute writes of next_frame.
        # A replay faster than the tick rate jumps from frame 0 straight to "finished", so a
        # reader must see the held last frame with it, never the previous frame.
        statuses: list[dict[str, object]] = []

        class ObservedReplay(Replay):
            def __setattr__(self, name: str, value: object) -> None:
                super().__setattr__(name, value)
                if "_ticks" in self.__dict__:  # constructed
                    statuses.append(self.status())

        replay = ObservedReplay(np.zeros((3, 1)), 1000.0, np.zeros(1))
        for _ in range(2):
            replay.next_frame(0.01)  # 10 frames per tick

        assert statuses
        assert all(status["frame"] == 2 for status in statuses if status["finished"])
        assert replay.status()["finished"] is True

    def test_frame_times_survive_rounding(self) -> None:
        # 3 ticks of 1/30 s at 30 fps is frame 3, although 3 * (1/30) * 30 rounds below 3.
        assert [frame for frame, _ in self._shown(30.0, 4, frames=10, control_dt=1 / 30)] == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("unit", "value", "message"),
    [
        # A normalized position past the range would be clamped to it: 150 would show as 100.
        ("normalized", 150.0, "shoulder_pan = 150.0 is outside its normalized range"),
        # A joint 1e40 degrees out would overflow the float32 observation.
        ("degrees", 1e40, "shoulder_pan = 1e+40 puts shoulder_pan beyond 1e+06 model units"),
    ],
)
def test_positions_the_robot_cannot_show_unchanged_are_refused(unit: str, value: float, message: str) -> None:
    robot = _so101(unit)
    try:
        client = _client(robot)
        frames = [[0.0] * 6, [value, 0.0, 0.0, 0.0, 0.0, 0.0]]

        response = client.post("/replay", json={"joint_positions": frames, "fps": 30.0})

        assert response.status_code == 400
        assert response.json()["detail"].startswith(f"Cannot replay joint_positions: frame 1: {message}")
        assert robot._commands.empty()  # noqa: SLF001
        # The sim thread refuses it too, should a command get past the HTTP check.
        robot._submit_command(ReplayCommand(joint_positions=np.array(frames), fps=30.0))  # noqa: SLF001
        observation = robot.get_observation()
        assert robot._replay is None  # noqa: SLF001
        assert np.isfinite(observation.joint_positions).all()
    finally:
        robot.disconnect()


def test_positions_at_the_ends_of_the_normalized_range_replay_unchanged() -> None:
    robot = _so101("normalized")
    try:
        frames = [[-100.0] * 5 + [0.0], [100.0] * 6]  # the gripper spans 0..100
        response = _client(robot).post("/replay", json={"joint_positions": frames, "fps": 1.0})
        observation = robot.get_observation()

        assert response.status_code == 200, response.text
        np.testing.assert_allclose(observation.joint_positions, frames[0], atol=1e-3)
    finally:
        robot.disconnect()


def test_a_frame_one_robot_cannot_show_stops_the_whole_replay_before_anything_moves() -> None:
    robot = MuJoCoRobot("so101", scene="garment_fold", bimanual=True, cameras=[])
    robot.connect()
    try:
        before = robot._data.qpos.copy()  # noqa: SLF001
        # Fine for the left arm in every frame; the right arm's gripper is out of range in frame 2.
        frames = np.zeros((3, 12))
        frames[2, 11] = 150.0
        client = _client(robot)

        response = client.post("/replay", json={"joint_positions": frames.tolist(), "fps": 30.0})

        assert response.status_code == 400
        assert "frame 2: right_gripper = 150.0 is outside its normalized range" in response.json()["detail"]
        robot._submit_command(ReplayCommand(joint_positions=frames, fps=30.0))  # noqa: SLF001
        robot._drain_commands()  # noqa: SLF001
        assert robot._replay is None  # noqa: SLF001
        np.testing.assert_array_equal(robot._data.qpos, before)  # noqa: SLF001
    finally:
        robot.disconnect()


def test_a_server_without_the_robots_check_refuses_replays() -> None:
    commands: queue.Queue = queue.Queue()
    status = {"connected": True, "joint_names": ["a"], "replay": {"active": False, "unsupported_joints": []}}
    client = TestClient(build_app(service_name="test", buffers={}, commands=commands, get_status=lambda: status))

    response = client.post("/replay", json={"joint_positions": [[0.0]], "fps": 30.0})

    assert response.status_code == 409
    assert response.json()["detail"] == "This server cannot check replay positions"
    assert commands.empty()
