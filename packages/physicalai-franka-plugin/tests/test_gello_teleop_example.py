# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for examples/gello_teleop.py (no hardware: fake pylibfranka and GELLO values)."""

from __future__ import annotations

import gc
import importlib.util
import multiprocessing
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from _fr3_fakes import START, _fake_pylibfranka, _FakeControl


_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "gello_teleop.py"


@pytest.fixture(scope="module")
def teleop():
    spec = importlib.util.spec_from_file_location("gello_teleop_example", _EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gello_obs(*arms: tuple[float, float]) -> SimpleNamespace:
    """Each arm: (value for all 7 joints in degrees, trigger)."""
    values = np.concatenate([[*[joint] * 7, trigger] for joint, trigger in arms])
    return SimpleNamespace(joint_positions=values.astype(np.float32))


def test_mapping_arm_only_drops_the_triggers(teleop) -> None:
    mapping = teleop.GelloToFranka(2, gripper=False)
    action = mapping(_gello_obs((10.0, 1.0), (-20.0, 0.0)))
    assert action.shape == (14,)
    assert np.allclose(action[:7], 10.0)
    assert np.allclose(action[7:], -20.0)


def test_mapping_gripper_has_hysteresis(teleop) -> None:
    mapping = teleop.GelloToFranka(1, gripper=True)
    widths = [mapping(_gello_obs((0.0, trigger)))[7] for trigger in (1.0, 0.5, 0.2, 0.5, 0.8)]
    open_m = teleop.GRIPPER_OPEN_M
    assert widths == [open_m, open_m, 0.0, 0.0, open_m]


def _gello_joints(joints: list[float], trigger: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(joint_positions=np.array([*joints, trigger], dtype=np.float32))


def test_follow_joints_moves_only_those_and_holds_the_rest(teleop) -> None:
    mapping = teleop.GelloToFranka(1, gripper=False, catch_up_deg_s=25.0, fps=25.0, follow_joints=[0])
    home = np.array([0.0, 0.0, 0.0, -90.0, 0.0, 90.0, 0.0])
    mapping.start(home[None, :])
    gello = [3.0, 40.0, -50.0, -120.0, 60.0, 30.0, 170.0]  # every joint far off; only J1 may follow
    for _ in range(5):
        action = mapping(_gello_joints(gello))
    assert mapping.aligned  # the 3 deg J1 gap is within one 1 deg/tick step after 3 ticks
    assert action[0] == pytest.approx(3.0)
    assert np.allclose(action[1:7], home[1:])
    gello[0] = -10.0
    assert mapping(_gello_joints(gello))[0] == pytest.approx(-10.0)


def test_joint_list_parses_one_based_and_rejects_out_of_range(teleop) -> None:
    assert teleop._joint_list("1") == [0]
    assert teleop._joint_list("3,1") == [0, 2]
    for bad in ("0", "8", ""):
        with pytest.raises(teleop.argparse.ArgumentTypeError):
            teleop._joint_list(bad)


def test_a_wrapping_joint_is_held_while_the_others_track_then_resumes(teleop) -> None:
    mapping = teleop.GelloToFranka(1, gripper=False)
    pose = [10.0, 0.0, 0.0, -90.0, 0.0, 90.0, 150.0]
    mapping(_gello_joints(pose))
    # GELLO wrist turned past 180 deg: J7 flips from +154 (clipped) to -154 while J1 moves on.
    for step in range(20):  # held as long as it stays wrapped; never stops the run
        action = mapping(_gello_joints([11.0 + step, 0.0, 0.0, -90.0, 0.0, 90.0, -154.0]))
        assert action[6] == pytest.approx(150.0)
        assert action[0] == pytest.approx(11.0 + step)
    resumed = mapping(_gello_joints([30.0, 0.0, 0.0, -90.0, 0.0, 90.0, 152.0]))  # turned back
    assert resumed[6] == pytest.approx(152.0)


def test_catch_up_moves_slowly_to_the_gello_then_tracks_and_unlocks_the_hand(teleop) -> None:
    mapping = teleop.GelloToFranka(1, gripper=True, catch_up_deg_s=10.0, fps=25.0)
    mapping.start(np.zeros(7))
    step = 10.0 / 25.0
    first = mapping(_gello_obs((30.0, 0.0)))  # trigger pressed during the catch-up: ignored
    assert np.allclose(first[:7], step)
    assert first[7] == teleop.GRIPPER_OPEN_M
    actions = [mapping(_gello_obs((30.0, 0.0))) for _ in range(200)]
    assert np.all(np.diff([a[0] for a in [first, *actions]]) <= step + 1e-9)
    assert mapping.aligned
    assert np.allclose(actions[-1][:7], 30.0)
    assert actions[-1][7] == 0.0  # hand closes once aligned
    live = mapping(_gello_obs((45.0, 1.0)))  # now tracks directly (below the glitch threshold)
    assert np.allclose(live[:7], 45.0)


def test_catch_up_follows_a_gello_that_moves_meanwhile(teleop) -> None:
    mapping = teleop.GelloToFranka(1, gripper=False, catch_up_deg_s=25.0, fps=25.0)
    mapping.start(np.zeros(7))
    for target in (5.0, 10.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0):
        action = mapping(_gello_obs((target, 1.0)))
    assert mapping.aligned
    assert np.allclose(action, 8.0)


def test_teleop_source_starts_the_catch_up_from_the_first_robot_observation(teleop) -> None:
    pytest.importorskip("physicalai.runtime")
    leader = SimpleNamespace(is_connected=lambda: True, get_observation=lambda: _gello_obs((20.0, 1.0), (-20.0, 1.0)))
    mapping = teleop.GelloToFranka(2, gripper=True, catch_up_deg_s=25.0, fps=25.0)
    source = teleop.make_teleop_source(leader, mapping, gripper=True)
    robot_state = SimpleNamespace(joint_positions=np.array([*[10.0] * 7, 0.08, *[-10.0] * 7, 0.08]))
    action = source.update(robot_state, {}, 0)
    assert np.allclose(action[:7], 11.0)
    assert np.allclose(action[8:15], -11.0)
    assert action[7] == action[15] == teleop.GRIPPER_OPEN_M


class _FlakyLeader:
    """GELLO leader whose reads fail on demand."""

    def __init__(self) -> None:
        self.fail = False
        self.connects = 0
        self.joint_names = ["j"] * 8

    def connect(self) -> None:
        self.connects += 1

    def disconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        return True

    def get_observation(self):
        if self.fail:
            raise ConnectionError("Failed to read GELLO positions")
        return _gello_joints([float(self.connects)] * 7)


def test_resilient_leader_holds_the_last_pose_reopens_and_recovers(teleop) -> None:
    flaky = _FlakyLeader()
    leader = teleop.ResilientLeader(flaky, max_outage_s=1.0, reconnect_after=3)
    leader.connect()
    good = leader.get_observation()
    flaky.fail = True
    for _ in range(3):
        assert leader.get_observation() is good  # held, no exception
    assert flaky.connects == 2  # reopened after 3 failures
    flaky.fail = False
    assert leader.get_observation() is not good
    assert leader.joint_names == ["j"] * 8  # attributes are delegated


def test_resilient_leader_gives_up_after_the_outage_limit(teleop) -> None:
    flaky = _FlakyLeader()
    leader = teleop.ResilientLeader(flaky, max_outage_s=0.05)
    leader.connect()
    leader.get_observation()
    flaky.fail = True
    leader.get_observation()
    time.sleep(0.1)
    with pytest.raises(ConnectionError):
        leader.get_observation()


def test_resilient_leader_raises_without_any_good_read(teleop) -> None:
    flaky = _FlakyLeader()
    flaky.fail = True
    leader = teleop.ResilientLeader(flaky)
    with pytest.raises(ConnectionError):
        leader.get_observation()


def _park_setup(teleop, monkeypatch, answers):
    controls = []

    class FakeFollower(teleop.FrankaFR3):
        def __init__(self, host, **kwargs):
            super().__init__(host, **kwargs)
            control = _FakeControl(START)
            controls.append(control)
            self._mp_context = multiprocessing.get_context("fork")
            self._pylibfranka = _fake_pylibfranka(control)

    monkeypatch.setattr(teleop, "FrankaFR3", FakeFollower)
    replies = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))
    goal = np.degrees(START) + np.array([4.0, -3.0, 2.0, 3.0, -2.0, 2.0, -6.0])
    leader = SimpleNamespace(get_observation=lambda: SimpleNamespace(joint_positions=np.append(goal, 1.0)))
    args = SimpleNamespace(
        arm_cpus="",
        realtime_config="kIgnore",
        dynamics_factor=0.3,
        catch_up_deg_s=20.0,
        gripper_force_n=30.0,
        joint_limits="hardware",
        limit_margin_deg=5.0,
        control="position",  # the fake robot only moves under position commands
    )
    return args, leader, goal


def test_park_moves_each_joint_to_the_gello_pose(teleop, monkeypatch, capsys) -> None:
    pytest.importorskip("physicalai_franka_plugin")
    args, leader, goal = _park_setup(teleop, monkeypatch, [""] * 7)
    teleop.park(args, ["left"], ["10.0.0.2"], leader)
    out = capsys.readouterr().out
    parked = out.split("left arm parked at")[1].split()[:7]
    assert np.allclose([float(v) for v in parked], goal, atol=0.6)


def test_park_skips_joints_on_request(teleop, monkeypatch, capsys) -> None:
    pytest.importorskip("physicalai_franka_plugin")
    args, leader, goal = _park_setup(teleop, monkeypatch, ["s"] * 7)
    teleop.park(args, ["left"], ["10.0.0.2"], leader)
    parked = capsys.readouterr().out.split("left arm parked at")[1].split()[:7]
    assert np.allclose([float(v) for v in parked], np.degrees(START), atol=0.1)


def test_park_home_retraces_in_reverse_order(teleop, monkeypatch, capsys) -> None:
    pytest.importorskip("physicalai_franka_plugin")
    args, leader, goal = _park_setup(teleop, monkeypatch, [""] * 7)
    args.home = True
    monkeypatch.setattr(teleop, "HOME_DEG", goal)  # a nearby "home" keeps the fake moves short
    prompts = []
    replies = iter([""] * 7)
    monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt) or next(replies))
    teleop.park(args, ["left"], ["10.0.0.2"], leader)
    out = capsys.readouterr().out
    assert [int(p.split("J")[1].split(":")[0]) - 1 for p in prompts] == list(teleop.PARK_ORDER[::-1])
    parked = out.split("left arm parked at")[1].split()[:7]
    assert np.allclose([float(v) for v in parked], goal, atol=0.6)


def test_park_pose_moves_to_given_angles_not_the_gello(teleop, monkeypatch, capsys) -> None:
    pytest.importorskip("physicalai_franka_plugin")
    args, leader, gello = _park_setup(teleop, monkeypatch, [""] * 7)
    pose = np.degrees(START) + np.array([-3.0, 2.0, -4.0, 2.0, 3.0, -2.0, 5.0])
    args.pose = list(pose)
    teleop.park(args, ["left"], ["10.0.0.2"], leader)
    parked = [float(v) for v in capsys.readouterr().out.split("left arm parked at")[1].split()[:7]]
    assert np.allclose(parked, pose, atol=0.6)
    assert not np.allclose(parked, gello, atol=0.6)
    args.pose = [0.0] * 6
    with pytest.raises(SystemExit, match="7 joint angles"):
        teleop.park(args, ["left"], ["10.0.0.2"], leader)


def test_align_waits_for_each_joint_to_settle(teleop, capsys) -> None:
    robot = np.array([10.0, -20.0, 30.0, -90.0, 5.0, 90.0, 179.0])
    reads = []

    def observation():
        reads.append(None)
        gello = robot.copy()
        gello[0] += max(0.0, 30.0 - 3.0 * len(reads))  # J1 starts 30 deg off and converges
        gello[6] = -178.0  # J7 within tolerance only across the +-180 wrap
        return SimpleNamespace(joint_positions=np.append(gello, 1.0))

    leader = SimpleNamespace(get_observation=observation)
    teleop.align(leader, robot, ["right"], tol_deg=5.0, period_s=0.0)
    out = capsys.readouterr().out
    for joint in range(1, 8):
        assert f"right J{joint}: aligned" in out
    assert "turn so it decreases" in out  # J1 read high first
    # J1 is outside the band for reads 1-8 (30 - 3 * 8 = 6 deg), then 8 stable reads; the other six need 8 each
    assert len(reads) == 8 + teleop.ALIGN_STABLE_TICKS + 6 * teleop.ALIGN_STABLE_TICKS


def test_align_bar_and_wrap(teleop) -> None:
    assert teleop._wrap_deg(181.0) == pytest.approx(-179.0)
    assert teleop._wrap_deg(-190.0) == pytest.approx(170.0)
    centre = teleop._align_bar(0.0, 5.0)
    assert centre[len(centre) // 2] == "#"
    assert teleop._align_bar(90.0, 5.0)[-2] == "#"  # clipped at the right edge
    assert teleop._align_bar(-90.0, 5.0)[1] == "#"
