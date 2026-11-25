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

    def __init__(self, *, oversleep_s: float, work_s: float) -> None:
        self.now = 100.0
        self.oversleep_s = oversleep_s
        self.work_s = work_s
        self.sleeps: list[float] = []

    def perf_counter(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds + self.oversleep_s


def _run(clock: _LateClock, *, cycles: int, multiplier: int = 1, stall: dict[int, float] | None = None) -> list[float]:
    """Run ``cycles`` control cycles and return the fake time at the start of every tick."""
    starts: list[float] = []
    stall = stall or {}
    robot = MagicMock()
    robot.joint_names = ["a", "b"]

    def observe() -> FakeRobotObservation:
        starts.append(clock.now)
        clock.now += clock.work_s + stall.get(len(starts) - 2, 0.0)  # key: loop tick index
        return FakeRobotObservation(joint_positions=np.zeros(2))

    robot.get_observation.side_effect = observe
    runtime = RobotRuntime(
        robot=robot,
        action_source=FakeActionSource(),
        fps=FPS,
        interpolator=LinearInterpolator(multiplier=multiplier),
    )
    runtime._connected = True  # noqa: SLF001
    with patch("physicalai.runtime.core.time", clock):
        runtime.run(duration_s=cycles / FPS)
    return starts[1:]  # the first observation is the initial-state read before the loop


@pytest.mark.parametrize("multiplier", [1, 3])
def test_late_wakeups_do_not_slow_the_loop(multiplier: int) -> None:
    clock = _LateClock(oversleep_s=0.003, work_s=0.002)
    starts = _run(clock, cycles=300, multiplier=multiplier)

    cycle_starts = starts[::multiplier]
    hz = (len(cycle_starts) - 1) / (cycle_starts[-1] - cycle_starts[0])
    assert hz == pytest.approx(FPS, rel=0.005), f"loop ran at {hz:.2f} Hz"


def test_a_stalled_tick_restarts_the_schedule_instead_of_bursting() -> None:
    clock = _LateClock(oversleep_s=0.003, work_s=0.002)
    starts = _run(clock, cycles=60, stall={20: 0.2})

    gaps = np.diff(starts)
    assert gaps[20] == pytest.approx(0.2 + 0.002, abs=1e-9)  # the stalled tick itself
    # No catch-up: every tick after the stall still gets most of a period.
    assert gaps[21:].min() > 0.9 / FPS
