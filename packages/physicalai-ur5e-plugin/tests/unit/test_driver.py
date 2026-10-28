# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import sys
import subprocess
import threading
import time

import numpy as np
import pytest

from physicalai.config import Config
from physicalai.robot import Robot
from physicalai.robot.interface import RobotObservation
from physicalai_ur5e_plugin import JOINT_NAMES, SUPPORTED_MODELS, UniversalRobot
from physicalai_ur5e_plugin.driver import _Trajectory
from physicalai_ur5e_plugin.profiles import joint_limits


def names(sdk):
    return [name for name, _ in sdk.calls]


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for fake SDK state")
        time.sleep(0.002)


def test_import_and_construction_without_vendor_or_studio(monkeypatch):
    monkeypatch.setitem(sys.modules, "rtde_control", None)
    monkeypatch.setitem(sys.modules, "rtde_receive", None)
    monkeypatch.setitem(sys.modules, "physicalai_studio_plugin", None)
    importlib.reload(importlib.import_module("physicalai_ur5e_plugin"))
    robot = UniversalRobot(ip="192.0.2.10")
    assert isinstance(robot, Robot)
    assert not robot.is_connected()
    assert robot.device_ids == ("tcp:192.0.2.10",)
    assert robot.joint_names == list(JOINT_NAMES)
    robot.joint_names.clear()
    assert len(robot.joint_names) == 6
    robot.disconnect()


@pytest.mark.parametrize("model", SUPPORTED_MODELS)
def test_config_round_trip(model):
    robot = UniversalRobot(ip="192.0.2.10", model=model, max_velocity=12.0, control_mode="external_control")
    config = Config.from_instance(robot)
    wire = json.loads(json.dumps(config.to_dict()))
    restored = Config.from_dict(wire).instantiate()
    assert restored.model == model
    assert restored.device_ids == robot.device_ids
    assert not restored.is_connected()
    assert Config.from_instance(restored).to_dict() == wire
    assert wire["class_path"] == "physicalai_ur5e_plugin.UniversalRobot"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ip": ""},
        {"ip": "robot.local"},
        {"ip": "http://192.0.2.10"},
        {"ip": "0.0.0.0"},
        {"ip": "224.0.0.1"},
        {"ip": "255.255.255.255"},
        {"ip": "192.0.2.10;whoami"},
        {"model": "ur20"},
        {"frequency": 0},
        {"frequency": 501},
        {"frequency": float("nan")},
        {"max_velocity": 0},
        {"gain": 2001},
        {"lookahead_time": 0.01},
        {"command_timeout": 0.01},
        {"external_control_port": 65536},
        {"control_mode": "freedrive"},
    ],
)
def test_invalid_constructor(kwargs):
    with pytest.raises(ValueError):
        UniversalRobot(**({"ip": "192.0.2.10"} | kwargs))


def test_disconnected_behavior(make_robot):
    robot = make_robot()
    with pytest.raises(ConnectionError):
        robot.get_observation()
    with pytest.raises(ConnectionError):
        robot.send_action(np.zeros(6))


def test_lifecycle_and_observation(sdk, make_robot):
    sdk.q = np.deg2rad([1, 2, 3, 4, 5, 6])
    robot = make_robot()
    robot.connect()
    robot.connect()
    assert names(sdk).count("control init") == 1
    assert "servo" not in names(sdk)
    observation = robot.get_observation()
    assert isinstance(observation, RobotObservation)
    np.testing.assert_allclose(observation.joint_positions, [1, 2, 3, 4, 5, 6])
    assert observation.timestamp <= time.monotonic()
    assert observation.sensor_data is observation.images is None
    np.testing.assert_array_equal(observation.state, observation.joint_positions)
    observation.joint_positions[:] = 99
    assert robot.get_observation().joint_positions[0] == pytest.approx(1)
    robot.disconnect()
    robot.disconnect()
    assert "servo stop" not in names(sdk)
    assert names(sdk)[-3:] == ["script stop", "control close", "receive close"]
    assert not robot.is_connected()
    assert len(sdk.thread_ids) == 1
    assert threading.get_ident() not in sdk.thread_ids


def test_servo_interpolation_units_and_goal_time(sdk, make_robot):
    robot = make_robot(max_velocity=10)
    robot.connect()
    action = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    robot.send_action(action, goal_time=0.1)
    action[:] = 0
    assert sdk.servo_event.wait(1)
    wait_for(lambda: len(sdk.servos) >= 5)
    samples = np.vstack([np.zeros(6), *sdk.servos])
    assert np.max(np.abs(np.diff(samples, axis=0))) <= np.deg2rad(10) / 500 + 1e-12
    assert sdk.servos[0][-1] > 0
    servo = next(args for name, args in sdk.calls if name == "servo")
    assert servo[1:] == (0.0, 0.0, 0.002, 0.1, 300.0)
    robot.disconnect()
    assert names(sdk)[-4:] == ["servo stop", "script stop", "control close", "receive close"]


def test_trajectory_does_not_skip_on_wall_clock_delay():
    trajectory = _Trajectory(np.zeros(6), np.ones(6), duration=1)
    for _ in range(10):
        result = trajectory.step(0.002)
    np.testing.assert_allclose(result, np.full(6, 0.02))
    for _ in range(500):
        result = trajectory.step(0.002)
    np.testing.assert_array_equal(result, np.ones(6))


@pytest.mark.parametrize("action", [np.zeros(5), np.zeros(7), np.zeros((6, 1)), np.full(6, np.nan), np.full(6, np.inf)])
def test_invalid_actions(sdk, make_robot, action):
    robot = make_robot()
    robot.connect()
    with pytest.raises(ValueError, match="shape"):
        robot.send_action(action)
    assert not sdk.servos


@pytest.mark.parametrize("goal_time", [0, -1, float("nan"), float("inf")])
def test_invalid_goal_time(make_robot, goal_time):
    robot = make_robot()
    robot.connect()
    with pytest.raises(ValueError, match="goal_time"):
        robot.send_action(np.zeros(6), goal_time=goal_time)


def test_target_jump_and_joint_limits(sdk, make_robot):
    robot = make_robot()
    robot.connect()
    with pytest.raises(ValueError, match="max_relative_target"):
        robot.send_action(np.full(6, 11))
    with pytest.raises(ValueError, match="joint limits"):
        robot.send_action(np.array([0, 0, 181, 0, 0, 0]))
    assert not sdk.servos


def test_continuous_ur3e_wrist(sdk, make_robot):
    assert joint_limits("ur3e")["upper"][-1] is None
    sdk.q[-1] = np.deg2rad(720)
    robot = make_robot(model="ur3e")
    robot.connect()
    robot.send_action(np.array([0, 0, 0, 0, 0, 721]))
    assert sdk.servo_event.wait(1)


def test_command_timeout_latches_and_stops(sdk, make_robot):
    robot = make_robot(command_timeout=0.1)
    robot.connect()
    robot.send_action(np.ones(6))
    assert sdk.servo_event.wait(1)
    wait_for(lambda: "receive close" in names(sdk))
    assert not robot.is_connected()
    with pytest.raises(RuntimeError, match="command_timeout"):
        robot.get_observation()
    with pytest.raises(ConnectionError, match="Disconnect"):
        robot.connect()
    with pytest.raises(RuntimeError, match="command_timeout"):
        robot.disconnect()
    count = len(sdk.servos)
    robot.connect()
    assert len(sdk.servos) == count  # old target must not resume
    robot.disconnect()


@pytest.mark.parametrize("failure", ["safety", "network", "program", "stale", "invalid_state", "backwards_time"])
def test_faults_close_both_interfaces(sdk, make_robot, failure):
    robot = make_robot(command_timeout=0.1)
    robot.connect()
    if failure == "safety":
        sdk.safety_mode = 3
    elif failure == "network":
        sdk.connected = False
    elif failure == "program":
        sdk.running = False
    elif failure == "stale":
        sdk.frozen_timestamp = time.monotonic()
    elif failure == "backwards_time":
        sdk.frozen_timestamp = 0
    else:
        sdk.q[:] = np.nan
    wait_for(lambda: "receive close" in names(sdk))
    assert "control close" in names(sdk)
    assert not robot.is_connected()
    with pytest.raises(RuntimeError):
        robot.send_action(np.zeros(6))


@pytest.mark.parametrize("failure", ["target_rejected", "servo_rejected", "stop_rejected"])
def test_sdk_false_results_are_not_ignored(sdk, make_robot, failure):
    robot = make_robot()
    robot.connect()
    sdk.safety_accepts = failure != "target_rejected"
    sdk.servo_accepts = failure != "servo_rejected"
    sdk.stop_accepts = failure != "stop_rejected"
    robot.send_action(np.ones(6))
    if failure == "stop_rejected":
        assert sdk.servo_event.wait(1)
        with pytest.raises(RuntimeError, match="servoStop"):
            robot.disconnect()
    else:
        wait_for(lambda: "receive close" in names(sdk))
        with pytest.raises(RuntimeError, match="rejected"):
            robot.get_observation()
    assert "control close" in names(sdk) and "receive close" in names(sdk)


def test_partial_connect_cleanup(sdk, make_robot):
    sdk.fail_control = True
    robot = make_robot()
    with pytest.raises(ConnectionError, match="control connection failed"):
        robot.connect()
    wait_for(lambda: "receive close" in names(sdk))


def test_watchdog_setup_failure(sdk, make_robot):
    sdk.watchdog_accepts = False
    robot = make_robot()
    with pytest.raises(ConnectionError, match="setWatchdog"):
        robot.connect()
    assert "script stop" in names(sdk)
    assert "receive close" in names(sdk)


def test_external_control_flags(sdk, make_robot):
    robot = make_robot(control_mode="external_control", external_control_port=50003)
    robot.connect()
    args = next(args for name, args in sdk.calls if name == "control init")
    assert args == ("192.0.2.10", 500.0, 2, 50003)


def test_idle_connection_kicks_watchdog_without_servoing(sdk, make_robot):
    robot = make_robot(command_timeout=0.1)
    robot.connect()
    wait_for(lambda: "kick watchdog" in names(sdk))
    assert not sdk.servos


def test_fresh_process_import_does_not_load_sdk_or_studio():
    code = """
import sys
from physicalai_ur5e_plugin import UniversalRobot
from physicalai.config import Config
robot = UniversalRobot(ip='192.0.2.10')
Config.from_instance(robot).instantiate()
assert not any(name.startswith(('rtde_', 'physicalai_studio_plugin')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_replacement_targets_preserve_velocity_bound(sdk, make_robot):
    robot = make_robot(max_velocity=10)
    robot.connect()
    robot.send_action(np.ones(6))
    wait_for(lambda: len(sdk.servos) >= 5)
    robot.send_action(-np.ones(6))
    wait_for(lambda: sdk.servos[-1][0] < 0)
    robot.disconnect()
    samples = np.vstack([np.zeros(6), *sdk.servos])
    assert np.max(np.abs(np.diff(samples, axis=0))) <= np.deg2rad(10) / 500 + 1e-12


def test_tracking_error_stops_servo(sdk, make_robot):
    robot = make_robot()
    robot.connect()
    robot.send_action(np.ones(6))
    assert sdk.servo_event.wait(1)
    sdk.q = np.full(6, np.deg2rad(-20))
    wait_for(lambda: "receive close" in names(sdk))
    with pytest.raises(RuntimeError, match="tracking error"):
        robot.get_observation()
    assert "servo stop" in names(sdk)


def test_stale_action_is_not_sent_after_slow_safety_query(sdk, make_robot):
    robot = make_robot(command_timeout=0.1)
    robot.connect()
    sdk.safety_delay = 0.15
    robot.send_action(np.ones(6))
    wait_for(lambda: "receive close" in names(sdk))
    assert not sdk.servos
    with pytest.raises(RuntimeError, match="command_timeout"):
        robot.get_observation()


def test_watchdog_kick_failure_latches_fault(sdk, make_robot):
    robot = make_robot(command_timeout=0.1)
    robot.connect()
    sdk.kick_accepts = False
    wait_for(lambda: "receive close" in names(sdk))
    with pytest.raises(RuntimeError, match="kickWatchdog"):
        robot.get_observation()


def test_disconnect_never_closes_sdk_from_another_thread(sdk, make_robot, monkeypatch):
    gate = threading.Event()
    sdk.period_gate = gate
    robot = make_robot()
    robot.connect()
    assert sdk.period_entered.wait(1)
    original_join = robot._thread.join
    monkeypatch.setattr(robot._thread, "join", lambda timeout: original_join(timeout=0.01))
    try:
        with pytest.raises(TimeoutError, match="owner did not stop"):
            robot.disconnect()
        assert "script stop" not in names(sdk)
        assert "control close" not in names(sdk)
        assert "receive close" not in names(sdk)
    finally:
        gate.set()
        original_join(timeout=2)
    robot.disconnect()
    assert names(sdk)[-3:] == ["script stop", "control close", "receive close"]
    assert len(sdk.thread_ids) == 1


def test_cleanup_attempts_receive_close_after_control_close_raises(sdk, make_robot, monkeypatch):
    robot = make_robot()
    robot.connect()

    def fail_close():
        sdk.record("control close failure")
        raise OSError("close failed")

    monkeypatch.setattr(sdk.control, "disconnect", fail_close)
    with pytest.raises(RuntimeError, match="close failed"):
        robot.disconnect()
    assert names(sdk)[-2:] == ["control close failure", "receive close"]


def test_reduced_safety_mode_is_allowed(sdk, make_robot):
    sdk.safety_mode = 2
    robot = make_robot()
    robot.connect()
    assert robot.is_connected()
