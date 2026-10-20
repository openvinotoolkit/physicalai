# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FR3 follower: tracker, shared buffer, 1 kHz arm process and FrankaFR3 (fake pylibfranka)."""

from __future__ import annotations

import gc
import importlib.util
import multiprocessing
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from physicalai.config import Config

from physicalai_franka_plugin import robot as franka
from physicalai_franka_plugin.constants import (
    FR3_GRIPPER_WIDTH_RANGE_M,
    FR3_JOINT_MAX_ACCELERATION_RAD_S2,
    FR3_JOINT_MAX_EFFORT_NM,
    FR3_JOINT_MAX_JERK_RAD_S3,
    FR3_JOINT_MAX_TORQUE_RATE_NM_S,
    FR3_JOINT_MAX_VELOCITY_RAD_S,
)
from physicalai_franka_plugin.robot import FrankaFR3


def _load_fakes():
    # Loaded by path: CI runs pytest with --import-mode=importlib, which keeps tests/ off sys.path.
    spec = importlib.util.spec_from_file_location("_fr3_fakes", Path(__file__).with_name("_fr3_fakes.py"))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_fakes = _load_fakes()
START = _fakes.START
_fake_pylibfranka = _fakes._fake_pylibfranka
_FakeControl = _fakes._FakeControl
_SlowGripper = _fakes._SlowGripper




class TestFrankaFR3Construction:
    def test_defaults(self) -> None:
        robot = FrankaFR3("172.16.0.2")

        assert robot.host == "172.16.0.2"
        assert robot.gripper_enabled is True
        assert robot.realtime_config == "kEnforce"
        assert robot.goal_time == pytest.approx(0.1)
        assert robot.control_mode == "impedance"
        assert robot.joint_names == [*FrankaFR3.JOINT_ORDER, "gripper"]
        assert robot.device_ids == ()
        assert robot.limits_deg[3, 0] == pytest.approx(np.degrees(-3.0770) + 5.0)  # hardware - 5 deg

    def test_arm_only_joint_layout(self) -> None:
        robot = FrankaFR3("172.16.0.2", gripper_enabled=False)

        assert robot.joint_names == FrankaFR3.JOINT_ORDER

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"host": ""}, "host must be a non-empty"),
            ({"realtime_config": "kSometimes"}, "Invalid realtime_config"),
            ({"goal_time": 0.0}, "goal_time must be positive"),
            ({"control_mode": "velocity"}, "Invalid control_mode"),
            ({"dynamics_factor": 0.0}, "dynamics_factor"),
            ({"gripper_force_n": 0.0}, "gripper_force_n"),
            ({"joint_limits": "nope"}, "joint limits"),
        ],
    )
    def test_invalid_parameters_raise(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            FrankaFR3(**{"host": "172.16.0.2", **kwargs})

    def test_exports_recipe(self) -> None:
        robot = FrankaFR3("172.16.0.2", gripper_enabled=False)

        assert Config.from_instance(robot)["init_args"] == {
            "host": "172.16.0.2",
            "gripper_enabled": False,
        }

    def test_connect_requires_pylibfranka(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "pylibfranka", None)  # makes the import fail even where installed
        robot = FrankaFR3("172.16.0.2")

        with pytest.raises(ImportError, match="pylibfranka"):
            robot.connect()

    def test_send_action_and_observation_require_a_running_loop(self) -> None:
        robot = FrankaFR3("172.16.0.2", gripper_enabled=False)

        with pytest.raises(ConnectionError, match="not running"):
            robot.get_observation()
        with pytest.raises(ConnectionError, match="not running"):
            robot.send_action(np.zeros(7))
        robot.disconnect()  # no-op without connect


def _simulate(target_fn, seconds: float, factor: float = 0.3):
    dt = 0.001
    tracker = franka.JerkLimitedTracker(
        np.zeros(7),
        factor * np.asarray(FR3_JOINT_MAX_VELOCITY_RAD_S),
        factor * np.asarray(FR3_JOINT_MAX_ACCELERATION_RAD_S2),
        factor * np.asarray(FR3_JOINT_MAX_JERK_RAD_S3),
    )
    positions = np.array([tracker.step(target_fn(k * dt), dt) for k in range(int(seconds / dt))])
    velocity = np.diff(positions, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    return positions, velocity, acceleration, jerk


@pytest.mark.parametrize("factor", [0.3, 1.0])
def test_tracker_respects_limits_for_a_large_step(factor: float) -> None:
    positions, velocity, acceleration, jerk = _simulate(lambda _t: np.full(7, 1.0), 6.0, factor)
    margin = 1.0 + 1e-6
    assert np.all(np.abs(velocity) <= factor * np.asarray(FR3_JOINT_MAX_VELOCITY_RAD_S) * margin)
    assert np.all(np.abs(acceleration) <= factor * np.asarray(FR3_JOINT_MAX_ACCELERATION_RAD_S2) * margin)
    assert np.all(np.abs(jerk) <= factor * np.asarray(FR3_JOINT_MAX_JERK_RAD_S3) * margin)
    assert np.max(positions) < 1.0 + np.radians(0.1)  # no meaningful overshoot
    assert np.allclose(positions[-1], 1.0, atol=1e-4)


def test_tracker_smooths_a_25hz_zero_order_hold_stream() -> None:
    def gello(t: float) -> np.ndarray:  # GELLO sampled at 25 Hz, held between samples
        return np.full(7, 0.3 * np.sin(2 * np.pi * 1.5 * (np.floor(t * 25) / 25)))

    _positions, velocity, acceleration, jerk = _simulate(gello, 4.0)
    assert np.all(np.abs(velocity) <= 0.3 * np.asarray(FR3_JOINT_MAX_VELOCITY_RAD_S) * (1 + 1e-6))
    assert np.all(np.abs(acceleration) <= 0.3 * np.asarray(FR3_JOINT_MAX_ACCELERATION_RAD_S2) * (1 + 1e-6))
    assert np.all(np.abs(jerk) <= 0.3 * np.asarray(FR3_JOINT_MAX_JERK_RAD_S3) * (1 + 1e-6))


def test_tracker_holds_still_at_the_target() -> None:
    tracker = franka.JerkLimitedTracker(np.ones(7), np.ones(7), np.ones(7), np.ones(7) * 100)
    for _ in range(100):
        command = tracker.step(np.ones(7), 0.001)
    assert np.array_equal(command, np.ones(7))


def test_arm_loop_starts_at_q_d_and_moves_within_the_limits() -> None:
    ctx = multiprocessing.get_context("fork")
    control = _FakeControl(START)
    target = franka.SeqBuffer.create(ctx, 7)
    state = franka.SeqBuffer.create(ctx, franka._STATE_SIZE)  # noqa: SLF001
    stop, ready, errors = threading.Event(), threading.Event(), ctx.Queue()
    loop = threading.Thread(
        target=franka._arm_process,  # noqa: SLF001 - run the process body in a thread to inspect it
        args=("10.0.0.2", "kIgnore", 0.3, 80, True, target.raw, state.raw, stop, ready, errors),
        kwargs={"pylibfranka": _fake_pylibfranka(control), "control_mode": "position"},
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(franka.signal, "signal", lambda *a: None)  # only allowed in the main thread
        patch.setattr(franka, "_set_realtime_priority", lambda priority: None)
        patch.setattr(franka, "_isolate_process", lambda cpu: None)
        loop.start()
        assert ready.wait(2.0)
        target.write(START + np.radians(5.0))
        time.sleep(1.5)
        stop.set()
        loop.join(2.0)
    commands = np.array(control.commands)
    assert np.allclose(commands[0], START)  # first command equals the desired pose: no jump
    assert (
        np.max(np.abs(np.diff(commands, axis=0))) <= 0.3 * np.asarray(FR3_JOINT_MAX_VELOCITY_RAD_S).max() * 0.001 * 1.01
    )
    assert np.allclose(commands[-1], START + np.radians(5.0), atol=np.radians(0.5))
    assert state.read()[franka._STATE_ALIVE] == 0.0  # noqa: SLF001 - cleared on exit


def test_arm_loop_impedance_sends_franka_gello_torques_within_limits() -> None:
    ctx = multiprocessing.get_context("fork")
    control = _FakeControl(START)  # the fake robot stays at START, so the error keeps growing
    target = franka.SeqBuffer.create(ctx, 7)
    state = franka.SeqBuffer.create(ctx, franka._STATE_SIZE)  # noqa: SLF001
    stop, ready, errors = threading.Event(), threading.Event(), ctx.Queue()
    loop = threading.Thread(
        target=franka._arm_process,  # noqa: SLF001
        args=("10.0.0.2", "kIgnore", 0.3, 80, True, target.raw, state.raw, stop, ready, errors),
        kwargs={"pylibfranka": _fake_pylibfranka(control), "control_mode": "impedance"},
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(franka.signal, "signal", lambda *a: None)
        patch.setattr(franka, "_set_realtime_priority", lambda priority: None)
        patch.setattr(franka, "_isolate_process", lambda cpu: None)
        loop.start()
        assert ready.wait(2.0)
        goal = START.copy()
        goal[0] += 0.5  # J1 only, far enough that the torque saturates
        target.write(goal)
        deadline = time.monotonic() + 5.0  # the tracker needs ~1.5 s at dynamics_factor 0.3 to saturate J1
        while time.monotonic() < deadline and not (control.torques and control.torques[-1][0] >= 87.0 - 1e-6):
            time.sleep(0.05)
        stop.set()
        loop.join(2.0)
    assert not control.commands  # no joint-position commands in impedance mode
    torques = np.array(control.torques)
    assert np.allclose(torques[0], 0.0)  # starts at the measured pose: no torque jump
    assert np.all(torques[-1, 0] > 0.0) and np.allclose(torques[:, 1:], 0.0)  # pulls J1 the right way only
    assert np.max(np.abs(np.diff(torques, axis=0))) <= FR3_JOINT_MAX_TORQUE_RATE_NM_S * 0.001 + 1e-9
    assert np.all(np.abs(torques) <= np.asarray(FR3_JOINT_MAX_EFFORT_NM) + 1e-9)
    assert torques[-1, 0] == pytest.approx(
        np.asarray(FR3_JOINT_MAX_EFFORT_NM)[0]
    )  # 240 Nm/rad x 0.5 rad, clipped at 87 Nm
    assert gc.isenabled()  # the loop gives garbage collection back on exit


def _process_follower(control: _FakeControl, **kwargs):
    kwargs.setdefault("gripper_enabled", False)
    kwargs.setdefault("control_mode", "position")  # the fake robot only moves under position commands
    follower = franka.FrankaFR3("10.0.0.2", realtime_config="kIgnore", **kwargs)
    follower._mp_context = multiprocessing.get_context("fork")  # noqa: SLF001 - children see the fakes
    follower._pylibfranka = _fake_pylibfranka(control)  # noqa: SLF001
    follower._gripper_factory = _SlowGripper  # noqa: SLF001
    return follower


def test_process_follower_tracks_and_drives_the_hand_from_its_own_processes() -> None:
    follower = _process_follower(_FakeControl(START), gripper_enabled=True)
    follower.connect()
    try:
        assert follower.joint_names[-1] == "gripper"
        target_deg = np.degrees(START) + 5.0
        follower.send_action(np.append(target_deg, 0.0))  # grasp: holds the gripper child's GIL
        time.sleep(1.5)
        closed = follower.get_observation().joint_positions
        follower.send_action(np.append(target_deg, FR3_GRIPPER_WIDTH_RANGE_M[1]))
        time.sleep(0.8)
        opened = follower.get_observation().joint_positions
    finally:
        follower.disconnect()
    assert np.allclose(closed[:7], target_deg, atol=0.5)
    assert closed[7] == pytest.approx(0.03)
    assert opened[7] == pytest.approx(FR3_GRIPPER_WIDTH_RANGE_M[1])
    assert not follower.is_connected()


def test_process_follower_reports_a_reflex_from_the_arm_process() -> None:
    control = _FakeControl(START)
    control.fail_after = 200  # ~0.2 s, then the fake robot aborts with a reflex
    follower = _process_follower(control)
    follower.connect()
    try:
        time.sleep(0.6)
        with pytest.raises(ConnectionError, match="communication_constraints_violation"):
            follower.get_observation()
        with pytest.raises(ConnectionError):
            follower.send_action(np.degrees(START))
    finally:
        follower.disconnect()


def test_process_follower_connect_failure_raises_with_the_reason() -> None:
    follower = franka.FrankaFR3("10.0.0.2", realtime_config="kIgnore")
    follower._mp_context = multiprocessing.get_context("fork")  # noqa: SLF001
    follower._pylibfranka = _fake_pylibfranka(_FakeControl(START), fail_connect="Connection to FCI refused")  # noqa: SLF001
    with pytest.raises(ConnectionError, match="FCI refused"):
        follower.connect()
    assert not follower.is_connected()


def test_seq_buffer_round_trips_and_never_returns_a_torn_read() -> None:
    buffer = franka.SeqBuffer.create(multiprocessing.get_context("fork"), 3)
    buffer.write(np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(buffer.read(), [1.0, 2.0, 3.0])
    view = np.frombuffer(buffer.raw, dtype=np.float64)
    view[0] += 1.0  # a writer is mid-update
    assert buffer.read(attempts=5) is None  # the reader gives up instead of blocking or tearing


def _robot_view(missed: set[int], *, coast: bool, ms: int = 3000, factor: float = 0.3) -> np.ndarray:
    """Positions the robot executes each 1 ms when our command packets for ``missed`` cycles are lost.

    Missed cycles are extrapolated by the robot with constant acceleration. Our loop only
    learns about them from the next state's duration (``periods`` > 1).
    """
    tracker = franka.JerkLimitedTracker(
        np.zeros(7),
        factor * np.asarray(FR3_JOINT_MAX_VELOCITY_RAD_S),
        factor * np.asarray(FR3_JOINT_MAX_ACCELERATION_RAD_S2),
        factor * np.asarray(FR3_JOINT_MAX_JERK_RAD_S3),
    )
    target = np.full(7, 0.5)
    executed = [np.zeros(7), np.zeros(7)]
    periods = 1
    for k in range(ms):
        if k in missed:
            q1, q0 = executed[-1], executed[-2]
            velocity = q1 - q0
            acceleration = velocity - (q0 - executed[-3]) if len(executed) > 2 else 0.0  # noqa: PLR2004
            executed.append(q1 + velocity + acceleration)  # constant acceleration
            periods += 1
            continue
        command = tracker.advance(target, periods) if coast else tracker.step(target, periods * 0.001)
        executed.append(command)
        periods = 1
    return np.array(executed)


@pytest.mark.parametrize("coast", [True, False])
def test_missed_packets_keep_the_robots_acceleration_continuous_only_when_coasting(coast: bool) -> None:
    rng = np.random.default_rng(1)
    missed = {int(k) for k in rng.choice(np.arange(100, 2500), size=60, replace=False)} | {300, 301, 302, 303}
    executed = _robot_view(missed, coast=coast)
    acceleration = np.diff(executed, n=2, axis=0) / 1e-6
    jump = np.max(
        np.abs(np.diff(acceleration, axis=0)) / (np.asarray(FR3_JOINT_MAX_JERK_RAD_S3) * 1e-3)
    )  # fraction of FR3 limit
    if coast:
        assert jump < 0.5  # well inside the robot's acceleration-discontinuity check
    else:
        assert jump > 1.0  # the old single big step violates it (the reflex seen on hardware)


def test_prepare_connects_without_starting_control_until_connect() -> None:
    follower = _process_follower(_FakeControl(START))
    follower.prepare()
    try:
        time.sleep(0.3)
        assert not follower.is_connected()  # process and robot connection up, but no 1 kHz loop yet
        assert follower._arm.is_alive()  # noqa: SLF001
        follower.connect()
        assert follower.is_connected()
        assert np.allclose(follower.get_observation().joint_positions, np.degrees(START), atol=1e-3)
    finally:
        follower.disconnect()


def test_loop_overruns_survive_disconnect_and_a_clean_stop_saves_the_last_cycles() -> None:
    follower = _process_follower(_FakeControl(START))
    dump = Path(f"/tmp/franka-{follower.host}-last.npz")
    dump.unlink(missing_ok=True)
    follower.connect()
    time.sleep(0.2)
    follower.disconnect()  # the count is reported after this
    history = np.load(dump)["history"]
    assert history.shape == (franka._HISTORY, 23)  # robot dt, command, measured, wall-clock dt, torque
    assert np.isfinite(history[-1]).all()
    published = np.zeros(franka._STATE_SIZE)
    published[franka._STATE_OVERRUNS] = 14
    follower._state.write(published)  # noqa: SLF001
    assert follower.loop_overruns == 14
