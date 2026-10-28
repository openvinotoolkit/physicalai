# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Deterministic fake RTDE bindings; no hardware/network access."""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest


class FakeSDK:
    def __init__(self) -> None:
        self.q = np.zeros(6)
        self.calls = []
        self.thread_ids = set()
        self.servos = []
        self.servo_event = threading.Event()
        self.safety_mode = 1
        self.robot_mode = 7
        self.running = True
        self.connected = True
        self.frozen_timestamp = None
        self.safety_accepts = True
        self.servo_accepts = True
        self.watchdog_accepts = True
        self.fail_control = False
        self.stop_accepts = True
        self.kick_accepts = True
        self.safety_delay = 0.0
        self.period_gate = None
        self.period_entered = threading.Event()
        self.control = None
        self.receive = None

    def record(self, name, *args):
        self.thread_ids.add(threading.get_ident())
        self.calls.append((name, args))

    def control_class(self):
        sdk = self

        class Control:
            FLAGS_DEFAULT = 1
            FLAG_USE_EXT_UR_CAP = 2

            def __init__(self, ip, frequency, *, flags, ur_cap_port):
                sdk.record("control init", ip, frequency, flags, ur_cap_port)
                if sdk.fail_control:
                    raise OSError("control connection failed")
                sdk.control = self
                self.dt = 1.0 / frequency

            def isConnected(self):
                return sdk.connected

            def isProgramRunning(self):
                return sdk.running

            def setWatchdog(self, frequency):
                sdk.record("watchdog", frequency)
                return sdk.watchdog_accepts

            def kickWatchdog(self):
                sdk.record("kick watchdog")
                return sdk.kick_accepts

            def initPeriod(self):
                return time.monotonic()

            def waitPeriod(self, cycle):
                sdk.period_entered.set()
                if sdk.period_gate is not None:
                    sdk.period_gate.wait(timeout=2.0)
                time.sleep(max(0, cycle + self.dt - time.monotonic()))

            def isJointsWithinSafetyLimits(self, positions):
                sdk.record("safety check", positions)
                time.sleep(sdk.safety_delay)
                return sdk.safety_accepts

            def servoJ(self, positions, speed, acceleration, dt, lookahead, gain):
                sdk.record("servo", positions, speed, acceleration, dt, lookahead, gain)
                sdk.q = np.array(positions)
                sdk.servos.append(sdk.q.copy())
                sdk.servo_event.set()
                return sdk.servo_accepts

            def servoStop(self):
                sdk.record("servo stop")
                return sdk.stop_accepts

            def stopScript(self):
                sdk.record("script stop")

            def disconnect(self):
                sdk.record("control close")

        return Control

    def receive_class(self):
        sdk = self

        class Receive:
            def __init__(self, ip, frequency):
                sdk.record("receive init", ip, frequency)
                sdk.receive = self

            def isConnected(self):
                return sdk.connected

            def getActualQ(self):
                return sdk.q.tolist()

            def getSafetyMode(self):
                return sdk.safety_mode

            def getRobotMode(self):
                return sdk.robot_mode

            def getTimestamp(self):
                return sdk.frozen_timestamp if sdk.frozen_timestamp is not None else time.monotonic()

            def disconnect(self):
                sdk.record("receive close")

        return Receive


@pytest.fixture
def sdk(monkeypatch):
    fake = FakeSDK()
    monkeypatch.setitem(sys.modules, "rtde_control", SimpleNamespace(RTDEControlInterface=fake.control_class()))
    monkeypatch.setitem(sys.modules, "rtde_receive", SimpleNamespace(RTDEReceiveInterface=fake.receive_class()))
    return fake


@pytest.fixture
def make_robot(sdk):
    from physicalai_ur5e_plugin import UniversalRobot

    robots = []

    def make(**kwargs):
        robot = UniversalRobot(ip="192.0.2.10", **kwargs)
        robots.append(robot)
        return robot

    yield make
    for robot in robots:
        with contextlib.suppress(RuntimeError):
            robot.disconnect()
