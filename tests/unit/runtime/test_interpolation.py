# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for action interpolation in physicalai.runtime."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from physicalai.config import Config
from physicalai.runtime import LifecycleEvent, LinearInterpolator, RobotRuntime, TickEvent

from tests.unit.runtime.conftest import FakeRobotObservation


def _a(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


class TestLinearInterpolator:
    def test_rejects_multiplier_below_one(self) -> None:
        with pytest.raises(ValueError, match="multiplier"):
            LinearInterpolator(multiplier=0)

    def test_get_before_add_raises(self) -> None:
        with pytest.raises(RuntimeError, match="before add"):
            LinearInterpolator(multiplier=2).get(0)

    def test_first_action_is_sent_as_is_in_one_substep(self) -> None:
        interp = LinearInterpolator(multiplier=3)
        first = _a(1.0)

        interp.add(first)

        np.testing.assert_array_equal(interp.get(0), first)
        assert interp.needs_new_action()

    def test_add_copies_action(self) -> None:
        interp = LinearInterpolator(multiplier=2)
        action = _a(1.0)

        interp.add(action)
        action[0] = 99.0

        np.testing.assert_array_equal(interp.get(0), _a(1.0))

    def test_emits_linear_points_then_exact_target(self) -> None:
        interp = LinearInterpolator(multiplier=4)
        interp.add(_a(0.0, 0.0))
        interp.get(0)
        target = _a(4.0, 8.0)

        interp.add(target)
        points = [interp.get(i) for i in range(4)]

        np.testing.assert_allclose(points[:3], [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        np.testing.assert_array_equal(points[3], target)
        assert points[0].dtype == np.float32
        assert interp.needs_new_action()

    def test_late_substep_skips_to_target(self) -> None:
        interp = LinearInterpolator(multiplier=4)
        interp.add(_a(0.0))
        interp.get(0)
        target = _a(4.0)
        interp.add(target)

        np.testing.assert_array_equal(interp.get(5), target)
        assert interp.needs_new_action()

    def test_reset_drops_previous_action(self) -> None:
        interp = LinearInterpolator(multiplier=2)
        interp.add(_a(0.0))
        interp.get(0)
        interp.reset()
        target = _a(10.0)

        interp.add(target)

        np.testing.assert_array_equal(interp.get(0), target)

    def test_config_round_trip(self) -> None:
        config = Config.from_instance(LinearInterpolator(multiplier=3))

        rebuilt = config.instantiate()

        assert isinstance(rebuilt, LinearInterpolator)
        assert rebuilt.multiplier == 3


class _FakeClock:
    """Monotonic clock that only moves when the code under test sleeps or works."""

    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class _SequenceSource:
    """Action source returning ``actions`` in order and advancing ``clock`` by ``work_s`` per update."""

    actions: list[np.ndarray]
    clock: _FakeClock | None = None
    work_s: float = 0.0
    update_steps: list[int] = field(default_factory=list)
    update_times: list[float] = field(default_factory=list)

    def connect(self, *, bus: Any, session_id: str) -> None: ...

    def update(self, robot_state: Any, camera_frames: Mapping[str, Any], step: int) -> np.ndarray:
        self.update_steps.append(step)
        if self.clock is not None:
            self.update_times.append(self.clock.now)
            self.clock.now += self.work_s
        return self.actions[min(len(self.update_steps), len(self.actions)) - 1]

    def disconnect(self) -> None: ...


@dataclass
class _Recorder:
    events: list[TickEvent] = field(default_factory=list)
    lifecycle: list[LifecycleEvent] = field(default_factory=list)

    def on_tick(self, event: TickEvent) -> None:
        self.events.append(event)

    def on_lifecycle(self, event: LifecycleEvent) -> None:
        self.lifecycle.append(event)


def _make_robot() -> MagicMock:
    robot = MagicMock()
    robot.get_observation.return_value = FakeRobotObservation(joint_positions=_a(0.0))
    return robot


def _sent(robot: MagicMock) -> list[float]:
    return [float(call.args[0][0]) for call in robot.send_action.call_args_list]


@contextmanager
def _patched_time(clock: _FakeClock) -> Iterator[None]:
    with patch("physicalai.runtime.core.time") as mock_time:
        mock_time.perf_counter.side_effect = clock.perf_counter
        mock_time.sleep.side_effect = clock.sleep
        mock_time.time.return_value = 0.0
        yield


def _make_runtime(source: _SequenceSource, robot: MagicMock, *, fps: float, multiplier: int, **kwargs: Any) -> RobotRuntime:
    runtime = RobotRuntime(
        robot=robot,
        action_source=source,
        fps=fps,
        interpolator=LinearInterpolator(multiplier=multiplier),
        **kwargs,
    )
    runtime._connected = True  # noqa: SLF001
    return runtime


class TestRuntimeInterpolation:
    def test_source_is_queried_once_per_cycle(self) -> None:
        robot = _make_robot()
        camera = MagicMock()
        recorder = _Recorder()
        source = _SequenceSource(actions=[_a(0.0), _a(3.0), _a(6.0)])
        runtime = _make_runtime(
            source, robot, fps=10.0, multiplier=3, cameras={"cam": camera}, callbacks=[recorder]
        )

        with _patched_time(_FakeClock()):
            steps = runtime.run(duration_s=0.3)

        # The first action has nothing to interpolate from, so its cycle is one tick.
        assert steps == 7
        assert _sent(robot) == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert source.update_steps == [0, 1, 4]
        assert [e.source_updated for e in recorder.events] == [True, True, False, False, True, False, False]
        assert camera.read_latest.call_count == 3
        assert all(not e.camera_frames for e in recorder.events if not e.source_updated)

    def test_goal_time_follows_control_rate(self) -> None:
        robot = _make_robot()
        runtime = _make_runtime(_SequenceSource(actions=[_a(0.0)]), robot, fps=10.0, multiplier=3)

        with _patched_time(_FakeClock()):
            runtime.run(duration_s=0.1)

        assert robot.send_action.call_args.kwargs["goal_time"] == pytest.approx(3 / 30.0)

    def test_ticks_are_evenly_spaced_when_on_time(self) -> None:
        clock = _FakeClock()
        robot = _make_robot()
        send_times: list[float] = []
        robot.send_action.side_effect = lambda *_args, **_kwargs: send_times.append(clock.now)
        runtime = _make_runtime(_SequenceSource(actions=[_a(0.0), _a(4.0)]), robot, fps=10.0, multiplier=4)

        with _patched_time(clock):
            runtime.run(duration_s=0.2)

        assert send_times == pytest.approx([0.0, 0.1, 0.125, 0.15, 0.175])

    def test_slow_update_skips_overdue_substeps_and_keeps_cycle_rate(self) -> None:
        clock = _FakeClock()
        robot = _make_robot()
        # 60 ms of inference in a 100 ms cycle split into 25 ms ticks.
        source = _SequenceSource(actions=[_a(0.0), _a(4.0), _a(8.0)], clock=clock, work_s=0.06)
        runtime = _make_runtime(source, robot, fps=10.0, multiplier=4)

        with _patched_time(clock):
            runtime.run(duration_s=0.3)

        assert source.update_times == pytest.approx([0.0, 0.1, 0.2])
        # Substeps 0 and 1 were due during the update, so each cycle sends only the 3/4 point and the target.
        assert _sent(robot) == pytest.approx([0.0, 3.0, 4.0, 7.0, 8.0])

    def test_interpolator_is_reset_between_runs(self) -> None:
        robot = _make_robot()
        runtime = _make_runtime(_SequenceSource(actions=[_a(5.0)]), robot, fps=10.0, multiplier=2)

        with _patched_time(_FakeClock()):
            runtime.run(duration_s=0.1)
        runtime._action_source = _SequenceSource(actions=[_a(9.0)])  # noqa: SLF001
        robot.send_action.reset_mock()
        with _patched_time(_FakeClock()):
            runtime.run(duration_s=0.1)

        assert _sent(robot) == pytest.approx([9.0])

    def test_in_place_action_transform_does_not_corrupt_interpolation(self) -> None:
        class _DoubleInPlace:
            def on_action_ready(self, *, action: np.ndarray, step: int) -> np.ndarray:  # noqa: PLR6301
                action *= 2
                return action

        robot = _make_robot()
        runtime = _make_runtime(
            _SequenceSource(actions=[_a(4.0)]), robot, fps=10.0, multiplier=2, callbacks=[_DoubleInPlace()]
        )

        with _patched_time(_FakeClock()):
            runtime.run(duration_s=0.3)

        assert _sent(robot) == pytest.approx([8.0] * 5)

    def test_start_event_reports_multiplier(self) -> None:
        recorder = _Recorder()
        runtime = _make_runtime(
            _SequenceSource(actions=[_a(0.0)]), _make_robot(), fps=10.0, multiplier=3, callbacks=[recorder]
        )

        with _patched_time(_FakeClock()):
            runtime.run(duration_s=0.1)

        start = next(e for e in recorder.lifecycle if e.event == "start")
        assert start.metadata["fps"] == 10.0
        assert start.metadata["interpolation_multiplier"] == 3
