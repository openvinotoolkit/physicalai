"""Autopilot and virtual leader on the bundled conveyor_sort model (real MuJoCo, no display)."""

from dataclasses import asdict

import numpy as np
import pytest

from physicalai_mujoco_so101_plugin.autopilot import Autopilot
from physicalai_mujoco_so101_plugin.http_server import SetAutopilotCommand
from physicalai_mujoco_so101_plugin.mujoco_robot import MuJoCoSO101
from physicalai_mujoco_so101_plugin.scene_registry import get_scene
from physicalai_mujoco_so101_plugin.viser_controls import _autopilot_markdown

TICK_S = 0.02


@pytest.fixture
def robot() -> MuJoCoSO101:
    scene = get_scene("conveyor_sort")
    sim = MuJoCoSO101(model_path=str(scene.scene_xml_path), scene_config=asdict(scene), substeps=10)
    sim.connect()
    sim._set_belt_speed(0.03)
    yield sim
    sim.disconnect()


def score(sim: MuJoCoSO101) -> dict[str, int]:
    return dict(sim._http_status()["episode"]["score"])


def test_autopilot_is_only_available_with_a_conveyor() -> None:
    autopilot = Autopilot()
    autopilot.bind(None, None)
    autopilot.set_mode("drive")
    assert autopilot.mode == "off"
    assert autopilot.step(object(), TICK_S) is None
    with pytest.raises(ValueError, match="Unknown autopilot mode"):
        autopilot.set_mode("fly")  # type: ignore[arg-type]


@pytest.mark.slow
def test_drive_mode_sorts_items_and_ignores_client_actions(robot: MuJoCoSO101) -> None:
    robot._commands.put(SetAutopilotCommand(mode="drive"))
    robot._drain_commands()
    assert robot._http_status()["autopilot"]["mode"] == "drive"
    hold = np.zeros(6, dtype=np.float32)
    for _ in range(round(25.0 / TICK_S)):
        robot.get_observation()
        robot.send_action(hold)  # a client holding the arm must not stop the autopilot
    assert score(robot)["correct"] >= 1
    assert score(robot)["wrong"] == 0


@pytest.mark.slow
def test_leader_mode_publishes_targets_that_sort_when_sent_back(robot: MuJoCoSO101) -> None:
    robot._automation.set_autopilot("leader")
    start = np.array(robot._data.ctrl[list(robot._ctrl_indices)])
    for _ in range(20):
        robot.get_observation()
    np.testing.assert_allclose(robot._data.ctrl[list(robot._ctrl_indices)], start)  # the autopilot moved nothing
    first = robot._automation.leader_snapshot()
    assert first["mode"] == "leader"
    assert first["unit"] == "normalized"
    assert first["joint_names"] == list(robot.JOINT_ORDER)

    # Close the loop the way Studio's teleoperation does: read the leader, send it as the action.
    for _ in range(round(25.0 / TICK_S)):
        robot.get_observation()
        robot.send_action(np.asarray(robot._automation.leader_snapshot()["joint_positions"], dtype=np.float32))
    assert robot._automation.leader_snapshot()["seq"] > first["seq"]
    assert score(robot)["correct"] >= 1
    assert score(robot)["wrong"] == 0


def test_leader_echoes_the_arm_targets_when_the_autopilot_is_off(robot: MuJoCoSO101) -> None:
    robot.get_observation()
    leader = np.asarray(robot._automation.leader_snapshot()["joint_positions"], dtype=np.float32)
    robot.send_action(leader)
    before = np.array(robot._data.ctrl[list(robot._ctrl_indices)])
    for _ in range(10):
        robot.get_observation()
        robot.send_action(np.asarray(robot._automation.leader_snapshot()["joint_positions"], dtype=np.float32))
    np.testing.assert_allclose(robot._data.ctrl[list(robot._ctrl_indices)], before, atol=1e-5)


def test_feed_hold_stops_the_belt_without_touching_the_user_pause(robot: MuJoCoSO101) -> None:
    conveyor = robot._episode_auto_reset
    conveyor.set_feed_hold(True)
    for _ in range(round(8.0 / TICK_S)):
        robot.get_observation()
    status = robot._http_status()["episode"]
    assert status["spawned"] == 0
    assert status["phase"] == "held"
    assert status["active"] is True
    assert status["lights"]["red"] is True
    conveyor.set_feed_hold(False)
    for _ in range(round(1.0 / TICK_S)):
        robot.get_observation()
    assert robot._http_status()["episode"]["spawned"] == 1


def test_switching_to_a_scene_without_a_belt_turns_the_autopilot_off(robot: MuJoCoSO101) -> None:
    robot._automation.set_autopilot("drive")
    assert robot._switch_to_scene("single_pick_place")
    assert robot._http_status()["autopilot"] == {"available": False, "mode": "off", "phase": None, "target": None}
    robot._automation.set_autopilot("drive")  # ignored: nothing to drive
    assert robot._automation.autopilot.mode == "off"
    assert robot._switch_to_scene("conveyor_sort")
    assert robot._automation.autopilot.mode == "off"  # "drive" does not come back with the belt
    assert not robot._automation.drives_arm


def test_autopilot_markdown() -> None:
    assert _autopilot_markdown({"mode": "off"}) == "**Autopilot:** off"
    assert _autopilot_markdown({"mode": "drive", "phase": "carry", "target": "item_cube_red"}) == (
        "**Autopilot:** carry (item_cube_red)"
    )


@pytest.mark.parametrize("jump", ["reload", "reset"])
def test_a_reload_or_reset_discards_the_studio_episode_in_progress(robot: MuJoCoSO101, jump: str) -> None:
    from physicalai_mujoco_so101_plugin.studio_recorder import AutoRecorder, RecordingOptions

    sent: list[str] = []

    class Link:
        phase, error, target = "connected", None, None
        state = {"dataset_loaded": True, "follower_source": "teleop", "is_recording": True}

        def start(self) -> None: ...
        def close(self) -> None: ...
        def take_ack(self, _request_id: str) -> None: ...

        def send(self, event: str, _data: object = None, request_id: str | None = None) -> None:
            sent.append(event)

    recorder = AutoRecorder(Link)
    robot._automation.recorder = recorder
    recorder.enable(RecordingOptions())
    recorder._phase = "recording"  # mid-episode
    if jump == "reload":
        assert robot._switch_to_scene("conveyor_sort")
    else:
        robot._run_scene_reset()
    assert sent == ["discard_episode"]
    assert recorder.phase == "saving"
