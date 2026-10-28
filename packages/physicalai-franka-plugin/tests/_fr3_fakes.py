# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fake pylibfranka pieces shared by the FR3 follower tests (imported from the tests directory)."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np


class _FakeControl:
    def __init__(self, q_d: np.ndarray) -> None:
        self.q_d = q_d.copy()
        self.commands: list[np.ndarray] = []
        self.torques: list[np.ndarray] = []
        self.read_times: list[float] = []
        self.fail_after: int | None = None

    def readOnce(self):  # noqa: N802 - pylibfranka name
        time.sleep(0.001)
        self.read_times.append(time.monotonic())
        if self.fail_after is not None and len(self.read_times) > self.fail_after:
            raise RuntimeError(
                'Move command aborted: motion aborted by reflex! ["communication_constraints_violation"]'
            )
        state = SimpleNamespace(
            q=self.q_d.copy(), q_d=self.q_d.copy(), dq=np.zeros(7), tau_J=np.zeros(7), tau_ext_hat_filtered=np.zeros(7)
        )
        return state, SimpleNamespace(to_sec=lambda: 0.001)

    def writeOnce(self, command) -> None:  # noqa: N802 - pylibfranka name
        if hasattr(command, "tau_J"):  # torque mode: the fake robot does not move
            self.torques.append(np.asarray(command.tau_J, dtype=np.float64))
            return
        self.q_d = np.asarray(command.q, dtype=np.float64)
        self.commands.append(self.q_d.copy())


def _hold_gil(seconds: float) -> None:
    """Block like pylibfranka does: a C-level loop that never releases the GIL."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        sum(range(10_000_000))  # one C call, ~100 ms with the GIL held


class _SlowGripper:
    """Fake Franka Hand whose commands hold the GIL for 0.3 s, like pylibfranka's."""

    def __init__(self) -> None:
        self.width = 0.08

    def grasp(self, *args) -> bool:
        _hold_gil(0.3)
        self.width = 0.03
        return True

    def move(self, width: float, speed: float) -> bool:
        _hold_gil(0.3)
        self.width = width
        return True

    def read_once(self):
        return SimpleNamespace(width=self.width)

    def stop(self) -> None:
        pass


def _fake_pylibfranka(control: _FakeControl, *, fail_connect: str | None = None) -> SimpleNamespace:
    def make_robot(host, config):
        if fail_connect:
            raise RuntimeError(fail_connect)
        return SimpleNamespace(
            read_once=lambda: SimpleNamespace(q=control.q_d.copy(), current_errors=[]),
            set_collision_behavior=lambda *a: None,
            set_joint_impedance=lambda *a: None,
            start_joint_position_control=lambda mode: control,
            start_torque_control=lambda: control,
            stop=lambda: None,
            automatic_error_recovery=lambda: None,
        )

    return SimpleNamespace(
        Robot=make_robot,
        RealtimeConfig=SimpleNamespace(kEnforce=0, kIgnore=1),
        ControllerMode=SimpleNamespace(JointImpedance=0),
        JointPositions=lambda q: SimpleNamespace(q=list(q)),
        Torques=lambda tau: SimpleNamespace(tau_J=list(tau)),
    )


START = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5])
