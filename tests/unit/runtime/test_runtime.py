# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for physicalai.runtime.runtime."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from physicalai.runtime._action_queue import ChunkedActionQueue as ActionQueue, ChunkedActionQueue
from physicalai.runtime.execution import SyncExecution, WorkerDiedError
from physicalai.runtime.runtime import PolicyRuntime, RunStats
from physicalai.robot.interface import RobotObservation
from physicalai.inference.model import InferenceModel

from physicalai.capture import Frame


@dataclass
class FakeRobotObservation:
    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None
    images: dict | None

    @property
    def state(self) -> np.ndarray:
        return self.joint_positions


def _make_mock_robot(joint_positions: np.ndarray | None = None) -> MagicMock:
    robot = MagicMock()
    if joint_positions is None:
        joint_positions = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    robot.get_observation.return_value = FakeRobotObservation(
        joint_positions=joint_positions,
        timestamp=time.monotonic(),
        sensor_data=None,
        images=None,
    )
    return robot


def _make_mock_model(chunk_size: int = 4, action_dim: int = 3) -> MagicMock:
    model = MagicMock()
    model.predict_action_chunk.return_value = np.random.randn(chunk_size, action_dim).astype(np.float32)
    return model


def _make_runtime(**kwargs: Any) -> PolicyRuntime:
    """Create a PolicyRuntime with _connected=True for testing."""
    runtime = PolicyRuntime(**kwargs)
    runtime._connected = True  # noqa: SLF001
    return runtime


def _exhaustible_side_effect(
    initial_chunks: list[np.ndarray],
    action_dim: int = 2,
) -> Callable[[Any], np.ndarray]:
    """Return *initial_chunks* in order, then empty arrays forever.

    Prevents StopIteration when SyncExecution refills more times than
    the test expected.
    """
    it = iter(initial_chunks)
    empty = np.empty((0, action_dim), dtype=np.float32)
    return lambda _obs: next(it, empty)


class TestPolicyRuntime:
    def test_full_loop_with_duration(self) -> None:
        robot = _make_mock_robot()
        model = _make_mock_model(chunk_size=20, action_dim=3)
        execution = SyncExecution()
        queue=ChunkedActionQueue()

        runtime = _make_runtime(
            robot=robot,
            model=model,
            execution=execution,
            fps=10.0,
            action_queue=queue,
        )

        with patch("physicalai.runtime.runtime.time") as mock_time:
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            stats = runtime.run(duration_s=0.5)

        assert stats.steps == 5
        assert robot.send_action.call_count >= 5

    def test_hold_fallback_when_queue_empty(self) -> None:
        robot = _make_mock_robot()
        chunk = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        model = _make_mock_model()
        model.predict_action_chunk.side_effect = _exhaustible_side_effect([chunk], action_dim=2)

        execution = SyncExecution()
        queue=ChunkedActionQueue()

        runtime = _make_runtime(
            robot=robot,
            model=model,
            execution=execution,
            fps=10.0,
            action_queue=queue,
        )

        with patch("physicalai.runtime.runtime.time") as mock_time:
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            stats = runtime.run(duration_s=0.4)

        assert stats.steps == 4
        assert robot.send_action.call_count == 4

    def test_worker_died_error_propagation(self) -> None:
        robot = _make_mock_robot()
        model = _make_mock_model(chunk_size=4)

        execution = MagicMock()
        execution.start = MagicMock()
        execution.warmup = MagicMock()
        execution.maybe_request.side_effect = WorkerDiedError("dead")
        execution.stop = MagicMock()

        queue=ChunkedActionQueue()
        queue.push_chunk(np.random.randn(4, 3).astype(np.float32))

        runtime = _make_runtime(
            robot=robot,
            model=model,
            execution=execution,
            fps=10.0,
            action_queue=queue,
        )

        with patch("physicalai.runtime.runtime.time") as mock_time, pytest.raises(WorkerDiedError, match="dead"):
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            runtime.run(duration_s=1.0)

    def test_shutdown_does_not_disconnect(self) -> None:
        robot = _make_mock_robot()
        model = _make_mock_model()
        execution = SyncExecution()

        runtime = _make_runtime(
            robot=robot,
            model=model,
            execution=execution,
            fps=10.0,
        )

        with patch("physicalai.runtime.runtime.time") as mock_time:
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            runtime.run(duration_s=0.1)

        robot.disconnect.assert_not_called()

    def test_run_raises_if_not_connected(self) -> None:
        robot = _make_mock_robot()
        model = _make_mock_model()
        execution = SyncExecution()

        runtime = PolicyRuntime(
            robot=robot,
            model=model,
            execution=execution,
            fps=10.0,
        )

        with pytest.raises(RuntimeError, match="connect"):
            runtime.run(duration_s=1.0)


class TestRuntimeCallback:
    def test_before_send_action_called(self) -> None:
        robot = _make_mock_robot()
        model = _make_mock_model(chunk_size=10)
        execution = SyncExecution()
        callback = MagicMock()
        callback.before_send_action.return_value = None

        runtime = _make_runtime(
            robot=robot,
            model=model,
            execution=execution,
            fps=10.0,
            callbacks=[callback],
        )

        with patch("physicalai.runtime.runtime.time") as mock_time:
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            runtime.run(duration_s=0.2)

        assert callback.before_send_action.call_count == 2

    def test_callback_raises_does_not_crash_loop(self) -> None:
        robot = _make_mock_robot()
        model = _make_mock_model(chunk_size=10)
        execution = SyncExecution()
        bad_callback = MagicMock()
        bad_callback.before_send_action.side_effect = RuntimeError("oops")

        runtime = _make_runtime(
            robot=robot,
            model=model,
            execution=execution,
            fps=10.0,
            callbacks=[bad_callback],
        )

        with patch("physicalai.runtime.runtime.time") as mock_time:
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            stats = runtime.run(duration_s=0.3)

        assert stats.steps == 3

    def test_on_hold_called_when_queue_empty(self) -> None:
        robot = _make_mock_robot()
        chunk = np.array([[1.0, 2.0]], dtype=np.float32)
        model = _make_mock_model()
        model.predict_action_chunk.side_effect = _exhaustible_side_effect([chunk], action_dim=2)

        execution = SyncExecution()
        callback = MagicMock()
        callback.before_send_action.return_value = None
        callback.on_hold.return_value = None

        runtime = _make_runtime(
            robot=robot,
            model=model,
            execution=execution,
            fps=10.0,
            callbacks=[callback],
        )

        with patch("physicalai.runtime.runtime.time") as mock_time:
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            runtime.run(duration_s=0.3)

        assert callback.on_hold.call_count >= 1


class TestLowPassFilterCallback:
    def test_low_pass_filtering_values(self) -> None:
        from physicalai.runtime.runtime import LowPassFilterCallback

        cb = LowPassFilterCallback(alpha=0.6)

        # First step: initialize
        act1 = np.array([1.0, 2.0], dtype=np.float32)
        res1 = cb.before_send_action(action=act1, step=0)
        assert np.allclose(res1, act1)

        # Second step: verify formula y_t = alpha * x_t + (1 - alpha) * y_t-1
        # y_1 = 0.6 * [3.0, 4.0] + 0.4 * [1.0, 2.0] = [1.8 + 0.4, 2.4 + 0.8] = [2.2, 3.2]
        act2 = np.array([3.0, 4.0], dtype=np.float32)
        res2 = cb.before_send_action(action=act2, step=1)
        assert np.allclose(res2, np.array([2.2, 3.2], dtype=np.float32))

    def test_low_pass_invalid_alpha(self) -> None:
        from physicalai.runtime.runtime import LowPassFilterCallback

        with pytest.raises(ValueError, match="alpha"):
            LowPassFilterCallback(alpha=0.0)

        with pytest.raises(ValueError, match="alpha"):
            LowPassFilterCallback(alpha=1.1)


class TestRunStats:
    def test_fields_populated(self) -> None:
        stats = RunStats(steps=10, total_pops=8, total_holds=2, inference_count=3)
        assert stats.steps == 10
        assert stats.total_pops == 8
        assert stats.total_holds == 2
        assert stats.inference_count == 3


class _ConfigFakeRobot:
    """Minimal Robot-protocol stub usable as a YAML ``class_path`` target."""

    def __init__(self, port: str = "/dev/null") -> None:
        self.port = port

    @property
    def joint_names(self) -> list[str]:
        return ["j0", "j1"]

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool:
        return True

    def get_observation(self) -> RobotObservation:
        return FakeRobotObservation(
            joint_positions=np.zeros(2, dtype=np.float32),
            timestamp=time.monotonic(),
            sensor_data=None,
            images=None,
        )

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None: ...


class _ConfigFakeModel(InferenceModel):
    """Minimal InferenceModel subclass that skips export-dir filesystem access."""

    def __init__(self, export_dir: str = "/tmp/fake") -> None:  # noqa: S108
        self.export_dir = export_dir  # type: ignore[assignment]


_FAKE_ROBOT_PATH = f"{__name__}._ConfigFakeRobot"
_SYNC_EXECUTION_PATH = "physicalai.runtime.execution.SyncExecution"
_MODEL_PATH = f"{__name__}._ConfigFakeModel"


def _minimal_yaml(*, fps: float = 30.0, include_run_block: bool = False) -> str:
    body = (
        "runtime:\n"
        f"  fps: {fps}\n"
        "  robot:\n"
        f"    class_path: {_FAKE_ROBOT_PATH}\n"
        "    init_args:\n"
        "      port: /dev/null\n"
        "  model:\n"
        f"    class_path: {_MODEL_PATH}\n"
        "    init_args:\n"
        "      export_dir: /tmp/fake\n"
        "  execution:\n"
        f"    class_path: {_SYNC_EXECUTION_PATH}\n"
    )
    if include_run_block:
        body += "run:\n  duration_s: 5\n"
    return body


class TestFromConfig:
    """``PolicyRuntime.from_config`` — YAML/JSON loader symmetric to the CLI."""

    def test_loads_minimal_yaml(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "runtime.yaml"
        cfg_path.write_text(_minimal_yaml())

        runtime = PolicyRuntime.from_config(cfg_path)

        assert isinstance(runtime, PolicyRuntime)
        assert runtime._fps == 30.0  # noqa: SLF001
        assert isinstance(runtime.robot, _ConfigFakeRobot)
        assert runtime.robot.port == "/dev/null"

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "runtime.yaml"
        cfg_path.write_text(_minimal_yaml(fps=15.0))

        runtime = PolicyRuntime.from_config(str(cfg_path))

        assert runtime._fps == 15.0  # noqa: SLF001

    def test_ignores_run_block(self, tmp_path: Path) -> None:
        """The CLI's ``run:`` block parses but is dropped — caller passes duration to run()."""
        cfg_path = tmp_path / "runtime.yaml"
        cfg_path.write_text(_minimal_yaml(include_run_block=True))

        runtime = PolicyRuntime.from_config(cfg_path)

        assert isinstance(runtime, PolicyRuntime)
        # Runtime carries no record of run.duration_s; only its constructor args.
        assert not hasattr(runtime, "duration_s")

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "runtime.yaml"
        # No model block — schema must reject.
        cfg_path.write_text(
            "runtime:\n"
            "  fps: 30\n"
            "  robot:\n"
            f"    class_path: {_FAKE_ROBOT_PATH}\n"
            "  execution:\n"
            f"    class_path: {_SYNC_EXECUTION_PATH}\n",
        )
        with pytest.raises(SystemExit):
            PolicyRuntime.from_config(cfg_path)

    def test_returns_disconnected_runtime(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "runtime.yaml"
        cfg_path.write_text(_minimal_yaml())

        runtime = PolicyRuntime.from_config(cfg_path)

        assert runtime._connected is False  # noqa: SLF001
