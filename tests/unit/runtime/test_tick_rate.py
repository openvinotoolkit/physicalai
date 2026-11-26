# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Control-loop rate when ``time.sleep`` wakes up late, as it does on macOS."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from physicalai.runtime import RobotRuntime
from physicalai.runtime.interpolation import LinearInterpolator
from tests.unit.runtime.conftest import FakeActionSource, FakeRobotObservation

FPS = 30.0


class _LateClock:
    """Fake ``time`` module: every sleep wakes ``oversleep_s`` late, every tick costs ``work_s``."""

    def __init__(self, *, oversleep_s: float, work_s: float, late_sleeps: dict[int, float] | None = None) -> None:
        self.now = 100.0
        self.oversleep_s = oversleep_s
        self.work_s = work_s
        self.late_sleeps = late_sleeps or {}  # sleep call index -> extra lateness
        self.sleeps: list[float] = []

    def perf_counter(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds + self.oversleep_s + self.late_sleeps.get(len(self.sleeps), 0.0)
        self.sleeps.append(seconds)


def _run(clock: _LateClock, *, cycles: int, multiplier: int = 1, stall: dict[int, float] | None = None) -> np.ndarray:
    """Run ``cycles`` control cycles and return the fake time of every command sent to the robot.

    ``stall`` maps a loop tick index to extra time spent reading that tick's observation.
    """
    sends: list[float] = []
    reads = 0
    stall = stall or {}
    robot = MagicMock()
    robot.joint_names = ["a", "b"]

    def observe() -> FakeRobotObservation:
        nonlocal reads
        clock.now += clock.work_s + stall.get(reads - 1, 0.0)  # read 0 is the initial state, before the loop
        reads += 1
        return FakeRobotObservation(joint_positions=np.zeros(2))

    robot.get_observation.side_effect = observe
    robot.send_action.side_effect = lambda *_a, **_k: sends.append(clock.now)
    runtime = RobotRuntime(
        robot=robot,
        action_source=FakeActionSource(),
        fps=FPS,
        interpolator=LinearInterpolator(multiplier=multiplier),
    )
    runtime._connected = True  # noqa: SLF001
    with patch("physicalai.runtime.core.time", clock):
        runtime.run(duration_s=cycles / FPS)
    return np.array(sends)


@pytest.mark.parametrize("multiplier", [1, 3])
def test_late_wakeups_do_not_slow_the_commands(multiplier: int) -> None:
    clock = _LateClock(oversleep_s=0.003, work_s=0.002)
    sends = _run(clock, cycles=300, multiplier=multiplier)

    # Every command, interpolated ones included, sits on the control-tick grid (fps * multiplier).
    # The first command is sent before the schedule starts, so the grid starts at the second one.
    tick_period = 1 / (FPS * multiplier)
    grid = sends[1] + np.arange(len(sends) - 1) * tick_period
    assert len(sends) == 1 + 299 * multiplier  # the first action completes at once, then full cycles
    hz = (len(sends) - 2) / (sends[-1] - sends[1])
    assert hz == pytest.approx(FPS * multiplier, rel=0.005), f"commands went out at {hz:.2f} Hz"
    assert np.abs(sends[1:] - grid).max() < 1e-9


def test_one_late_wakeup_does_not_shift_later_commands() -> None:
    # Main re-anchored every cycle at the late wake-up, so every later command moved by 10 ms.
    clock = _LateClock(oversleep_s=0.0, work_s=0.002, late_sleeps={20: 0.010})
    sends = _run(clock, cycles=60)

    grid = sends[0] + np.arange(len(sends)) / FPS
    assert sends[21] - grid[21] == pytest.approx(0.010, abs=1e-9)  # the late command itself
    assert np.abs(sends[22:] - grid[22:]).max() < 1e-9


def test_a_stalled_tick_is_followed_by_one_overdue_command_then_the_normal_cadence() -> None:
    """What the schedule guarantees after an overrun; main behaved the same, this guards it.

    The command after a stalled tick is already overdue, so it goes out as soon as its own tick's
    work is done (2 ms here). There is no burst beyond that one: from then on commands are a full
    period apart again.
    """
    clock = _LateClock(oversleep_s=0.003, work_s=0.002)
    sends = _run(clock, cycles=60, stall={20: 0.2})

    gaps = np.diff(sends)
    assert gaps[19] > 0.2  # the stalled tick's command
    assert gaps[20] == pytest.approx(clock.work_s, abs=1e-9)  # the one overdue command
    others = np.delete(gaps, [19, 20])
    assert others.min() > 0.9 / FPS
