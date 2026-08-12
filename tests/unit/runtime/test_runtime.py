# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for physicalai.runtime.core."""

from __future__ import annotations

import json
import multiprocessing as mp
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from physicalai.runtime import AsyncCallback, AsyncExecution, ChunkedActionQueue as ActionQueue, ChunkedActionQueue, Execution, JsonlCallback, LifecycleEvent, PolicySource, RobotRuntime, StopSignal, SyncExecution, WorkerDiedError
from physicalai.runtime._callback_bus import _CallbackBus
from physicalai.robot.interface import RobotObservation
from physicalai.inference.model import InferenceModel
from physicalai.inference.constants import IMAGES, STATE, TASK

from physicalai.capture import Frame

from tests.unit.runtime.conftest import FakeActionSource


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


def _make_runtime(
    *,
    robot: MagicMock | None = None,
    model: MagicMock | None = None,
    execution: Any = None,
    action_queue: ActionQueue | None = None,
    task: str | None = None,
    fps: float = 10.0,
    cameras: dict | None = None,
    callbacks: Any = (),
) -> tuple[RobotRuntime, PolicySource]:
    """Create a RobotRuntime + PolicySource pair with _connected=True for testing.

    Returns:
        Tuple ``(runtime, policy_source)`` so tests can inspect the action
        source directly (e.g. ``policy_source.action_queue``).
    """
    policy_source = PolicySource(
        model=model or _make_mock_model(),
        execution=execution or SyncExecution(),
        action_queue=action_queue,
        task=task,
    )
    runtime = RobotRuntime(
        robot=robot or _make_mock_robot(),
        action_source=policy_source,
        fps=fps,
        cameras=cameras,
        callbacks=callbacks,
    )
    runtime._connected = True  # noqa: SLF001
    return runtime, policy_source


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


class TestRobotRuntimeWithPolicySource:
    def test_full_loop_with_duration(self) -> None:
        robot = _make_mock_robot()
        model = _make_mock_model(chunk_size=20, action_dim=3)
        execution = SyncExecution()
        queue = ChunkedActionQueue()

        runtime, _policy_source = _make_runtime(
            robot=robot,
            model=model,
            execution=execution,
            fps=10.0,
            action_queue=queue,
        )

        with patch("physicalai.runtime.core.time") as mock_time:
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            steps = runtime.run(duration_s=0.5)

        assert steps == 5
        assert robot.send_action.call_count >= 5

    def test_hold_fallback_when_queue_empty(self) -> None:
        robot = _make_mock_robot()
        chunk = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        model = _make_mock_model()
        model.predict_action_chunk.side_effect = _exhaustible_side_effect([chunk], action_dim=2)

        execution = SyncExecution()
        queue = ChunkedActionQueue()

        runtime, _policy_source = _make_runtime(
            robot=robot,
            model=model,
            execution=execution,
            fps=10.0,
            action_queue=queue,
        )

        with patch("physicalai.runtime.core.time") as mock_time:
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            steps = runtime.run(duration_s=0.4)

        assert steps == 4
        assert robot.send_action.call_count == 4

    def test_worker_died_error_propagation(self) -> None:
        robot = _make_mock_robot()
        model = _make_mock_model(chunk_size=4)

        execution = MagicMock()
        execution.start = MagicMock()
        execution.warmup = MagicMock()
        execution.maybe_request.side_effect = WorkerDiedError("dead")
        execution.stop = MagicMock()

        queue = ChunkedActionQueue()
        queue.push_chunk(np.random.randn(4, 3).astype(np.float32))

        runtime, _policy_source = _make_runtime(
            robot=robot,
            model=model,
            execution=execution,
            fps=10.0,
            action_queue=queue,
        )

        with patch("physicalai.runtime.core.time") as mock_time, pytest.raises(WorkerDiedError, match="dead"):
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            runtime.run(duration_s=1.0)

    def test_execution_start_failure_does_not_call_stop(self) -> None:
        execution = MagicMock()
        execution.start.side_effect = RuntimeError("previous worker is busy")
        runtime, _source = _make_runtime(execution=execution)

        with pytest.raises(RuntimeError, match="previous worker is busy"):
            runtime.run(duration_s=1.0)

        execution.stop.assert_not_called()

    def test_shutdown_does_not_disconnect(self) -> None:
        robot = _make_mock_robot()
        model = _make_mock_model()
        execution = SyncExecution()

        runtime, _policy_source = _make_runtime(
            robot=robot,
            model=model,
            execution=execution,
            fps=10.0,
        )

        with patch("physicalai.runtime.core.time") as mock_time:
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            runtime.run(duration_s=0.1)

        robot.disconnect.assert_not_called()

    def test_run_raises_if_not_connected(self) -> None:
        robot = _make_mock_robot()
        model = _make_mock_model()
        execution = SyncExecution()
        policy_source = PolicySource(model=model, execution=execution)

        runtime = RobotRuntime(
            robot=robot,
            action_source=policy_source,
            fps=10.0,
        )

        with pytest.raises(RuntimeError, match="connect"):
            runtime.run(duration_s=1.0)

    def test_failed_connect_can_be_retried_directly(self) -> None:
        robot = _make_mock_robot()
        robot.connect.side_effect = [ConnectionError("not ready"), None]
        runtime = RobotRuntime(robot=robot, action_source=FakeActionSource(), fps=10.0)

        with pytest.raises(ConnectionError, match="not ready"):
            runtime.connect()

        runtime.connect()
        assert robot.connect.call_count == 2

    def test_disconnect_after_failed_connect_is_terminal(self) -> None:
        robot = _make_mock_robot()
        robot.connect.side_effect = ConnectionError("not ready")
        runtime = RobotRuntime(robot=robot, action_source=FakeActionSource(), fps=10.0)

        with pytest.raises(ConnectionError, match="not ready"):
            runtime.connect()
        runtime.disconnect()

        with pytest.raises(RuntimeError, match="cannot be connected again"):
            runtime.connect()


class TestGenericActionSource:
    """Loop mechanics using a bare ActionSource double (no PolicySource specifics)."""

    def test_update_called_with_two_params_and_step(self) -> None:
        robot = _make_mock_robot(np.array([0.5, 0.6], dtype=np.float32))
        action_source = FakeActionSource(next_action=np.array([1.0, 2.0], dtype=np.float32))
        runtime = RobotRuntime(robot=robot, action_source=action_source, fps=10.0)
        runtime._connected = True  # noqa: SLF001

        with patch("physicalai.runtime.core.time") as mock_time:
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            steps = runtime.run(duration_s=0.3)

        assert steps == 3
        assert len(action_source.calls) == 3
        for robot_state, camera_frames, step in action_source.calls:
            np.testing.assert_allclose(robot_state.joint_positions, [0.5, 0.6], rtol=1e-6)
            assert camera_frames == {}
            assert isinstance(step, int)

    def test_connect_receives_bus_and_session_id(self) -> None:
        robot = _make_mock_robot()
        action_source = FakeActionSource(next_action=np.zeros(3, dtype=np.float32))
        runtime = RobotRuntime(robot=robot, action_source=action_source, fps=10.0)
        runtime._connected = True  # noqa: SLF001

        with patch("physicalai.runtime.core.time") as mock_time:
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            runtime.run(duration_s=0.1)

        assert action_source.connected
        assert action_source.bus is not None
        assert action_source.session_id != ""

    def test_disconnect_called_on_shutdown_no_drain(self) -> None:
        """No queue-drain concept — disconnect() is called and that's it."""
        robot = _make_mock_robot()
        action_source = FakeActionSource(next_action=np.zeros(3, dtype=np.float32))
        runtime = RobotRuntime(robot=robot, action_source=action_source, fps=10.0)
        runtime._connected = True  # noqa: SLF001

        with patch("physicalai.runtime.core.time") as mock_time:
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            steps = runtime.run(duration_s=0.2)

        assert action_source.disconnected
        # No drain: send count == step count exactly, nothing extra flushed.
        assert robot.send_action.call_count == steps

    def test_run_returns_plain_int(self) -> None:
        robot = _make_mock_robot()
        action_source = FakeActionSource(next_action=np.zeros(3, dtype=np.float32))
        runtime = RobotRuntime(robot=robot, action_source=action_source, fps=10.0)
        runtime._connected = True  # noqa: SLF001

        with patch("physicalai.runtime.core.time") as mock_time:
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            result = runtime.run(duration_s=0.2)

        assert type(result) is int  # noqa: E721 — must be a plain int, not RunStats


class TestRuntimeCallback:
    def test_on_action_ready_called(self) -> None:
        robot = _make_mock_robot()
        model = _make_mock_model(chunk_size=10)
        execution = SyncExecution()
        callback = MagicMock()
        callback.on_action_ready.side_effect = lambda *, action, step: action  # noqa: ARG005

        runtime, _policy_source = _make_runtime(
            robot=robot,
            model=model,
            execution=execution,
            fps=10.0,
            callbacks=[callback],
        )

        with patch("physicalai.runtime.core.time") as mock_time:
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            runtime.run(duration_s=0.2)

        assert callback.on_action_ready.call_count == 2

    def test_callback_raises_stops_loop_and_propagates(self) -> None:
        """A failed action-transform can't be trusted, so it ends the run instead of being silently skipped."""
        robot = _make_mock_robot()
        model = _make_mock_model(chunk_size=10)
        execution = SyncExecution()
        bad_callback = MagicMock()
        bad_callback.on_action_ready.side_effect = RuntimeError("oops")

        runtime, _policy_source = _make_runtime(
            robot=robot,
            model=model,
            execution=execution,
            fps=10.0,
            callbacks=[bad_callback],
        )

        with patch("physicalai.runtime.core.time") as mock_time, pytest.raises(RuntimeError, match="oops"):
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            runtime.run(duration_s=0.3)

        robot.send_action.assert_not_called()

    def test_on_action_ready_must_return_valid_action_no_none_sentinel(self) -> None:
        """A callback that raises leaves the bus's running result untouched (isolated), never None."""
        robot = _make_mock_robot()
        model = _make_mock_model(chunk_size=10)
        execution = SyncExecution()
        callback = MagicMock()
        callback.on_action_ready.side_effect = lambda *, action, step: action * 2  # noqa: ARG005

        runtime, _policy_source = _make_runtime(
            robot=robot,
            model=model,
            execution=execution,
            fps=10.0,
            callbacks=[callback],
        )

        with patch("physicalai.runtime.core.time") as mock_time:
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            runtime.run(duration_s=0.1)

        sent_action = robot.send_action.call_args[0][0]
        assert sent_action is not None


class TestLowPassFilterCallback:
    def test_low_pass_filtering_values(self) -> None:
        from physicalai.runtime import LowPassFilterCallback

        cb = LowPassFilterCallback(alpha=0.6)

        # First step: initialize
        act1 = np.array([1.0, 2.0], dtype=np.float32)
        res1 = cb.on_action_ready(action=act1, step=0)
        assert np.allclose(res1, act1)

        # Second step: verify formula y_t = alpha * x_t + (1 - alpha) * y_t-1
        # y_1 = 0.6 * [3.0, 4.0] + 0.4 * [1.0, 2.0] = [1.8 + 0.4, 2.4 + 0.8] = [2.2, 3.2]
        act2 = np.array([3.0, 4.0], dtype=np.float32)
        res2 = cb.on_action_ready(action=act2, step=1)
        assert np.allclose(res2, np.array([2.2, 3.2], dtype=np.float32))

    def test_low_pass_invalid_alpha(self) -> None:
        from physicalai.runtime import LowPassFilterCallback

        with pytest.raises(ValueError, match="alpha"):
            LowPassFilterCallback(alpha=0.0)

        with pytest.raises(ValueError, match="alpha"):
            LowPassFilterCallback(alpha=1.1)


class _ConfigFakeRobot:
    """Minimal Robot-protocol stub usable as a YAML ``class_path`` target."""

    def __init__(self, port: str = "/dev/null") -> None:
        self.port = port

    @property
    def joint_names(self) -> list[str]:
        return ["j0", "j1"]

    @property
    def device_ids(self) -> tuple[str, ...]:
        return (f"fake:{self.port}",)

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
_SYNC_EXECUTION_PATH = "physicalai.runtime.SyncExecution"
_MODEL_PATH = f"{__name__}._ConfigFakeModel"
_POLICY_SOURCE_PATH = "physicalai.runtime.PolicySource"


def _minimal_yaml(*, fps: float = 30.0, include_run_block: bool = False) -> str:
    body = (
        "runtime:\n"
        f"  fps: {fps}\n"
        "  robot:\n"
        f"    class_path: {_FAKE_ROBOT_PATH}\n"
        "    init_args:\n"
        "      port: /dev/null\n"
        "  action_source:\n"
        f"    class_path: {_POLICY_SOURCE_PATH}\n"
        "    init_args:\n"
        "      model:\n"
        f"        class_path: {_MODEL_PATH}\n"
        "        init_args:\n"
        "          export_dir: /tmp/fake\n"
        "      execution:\n"
        f"        class_path: {_SYNC_EXECUTION_PATH}\n"
    )
    if include_run_block:
        body += "run:\n  duration_s: 5\n"
    return body


class TestFromConfig:
    """``RobotRuntime.from_config`` — YAML/JSON loader symmetric to the CLI."""

    def test_loads_minimal_yaml(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "runtime.yaml"
        cfg_path.write_text(_minimal_yaml())

        runtime = RobotRuntime.from_config(cfg_path)

        assert isinstance(runtime, RobotRuntime)
        assert runtime._fps == 30.0  # noqa: SLF001
        assert isinstance(runtime.robot, _ConfigFakeRobot)
        assert runtime.robot.port == "/dev/null"
        assert isinstance(runtime.action_source, PolicySource)

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "runtime.yaml"
        cfg_path.write_text(_minimal_yaml(fps=15.0))

        runtime = RobotRuntime.from_config(str(cfg_path))

        assert runtime._fps == 15.0  # noqa: SLF001

    def test_ignores_run_block(self, tmp_path: Path) -> None:
        """The CLI's ``run:`` block parses but is dropped — caller passes duration to run()."""
        cfg_path = tmp_path / "runtime.yaml"
        cfg_path.write_text(_minimal_yaml(include_run_block=True))

        runtime = RobotRuntime.from_config(cfg_path)

        assert isinstance(runtime, RobotRuntime)
        # Runtime carries no record of run.duration_s; only its constructor args.
        assert not hasattr(runtime, "duration_s")

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "runtime.yaml"
        # No action_source block — schema must reject (it's a required constructor arg).
        cfg_path.write_text(
            "runtime:\n"
            "  fps: 30\n"
            "  robot:\n"
            f"    class_path: {_FAKE_ROBOT_PATH}\n",
        )
        with pytest.raises(SystemExit):
            RobotRuntime.from_config(cfg_path)

    def test_empty_yaml_fails_via_parser_not_type_error(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "runtime.yaml"
        cfg_path.write_text("", encoding="utf-8")

        with pytest.raises(SystemExit):
            RobotRuntime.from_config(cfg_path)

    def test_returns_disconnected_runtime(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "runtime.yaml"
        cfg_path.write_text(_minimal_yaml())

        runtime = RobotRuntime.from_config(cfg_path)

        assert runtime._connected is False  # noqa: SLF001


def _make_frame(value: int = 0) -> Frame:
    return Frame(data=np.full((2, 2, 3), value, dtype=np.uint8), timestamp=0.0, sequence=0)


class TestPolicySourceModelInput:
    """Covers image-key layout in PolicySource._to_model_input / to_model_input."""

    @staticmethod
    def _policy_source(**kwargs: Any) -> PolicySource:
        defaults: dict[str, Any] = {
            "model": _make_mock_model(),
            "execution": SyncExecution(),
        }
        defaults.update(kwargs)
        return PolicySource(**defaults)

    def test_single_image_uses_bare_images_key(self) -> None:
        """A single image input is placed under the bare ``images`` key."""
        robot_obs = FakeRobotObservation(
            joint_positions=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            timestamp=0.0,
            sensor_data=None,
            images=None,
        )
        frame = _make_frame(1)
        policy_source = self._policy_source()

        model_input = policy_source.to_model_input(robot_obs, {"cam0": frame})

        assert IMAGES in model_input
        assert f"{IMAGES}.cam0" not in model_input
        np.testing.assert_array_equal(model_input[IMAGES], frame.data[np.newaxis])
        np.testing.assert_array_equal(model_input[STATE], np.array([robot_obs.state], dtype=np.float32))

    def test_multiple_images_use_namespaced_keys(self) -> None:
        """Multiple image inputs are placed under ``images.<name>`` keys."""
        robot_obs = FakeRobotObservation(
            joint_positions=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            timestamp=0.0,
            sensor_data=None,
            images=None,
        )
        frame0 = _make_frame(1)
        frame1 = _make_frame(2)
        policy_source = self._policy_source()

        model_input = policy_source.to_model_input(robot_obs, {"cam0": frame0, "cam1": frame1})

        assert IMAGES not in model_input
        np.testing.assert_array_equal(model_input[f"{IMAGES}.cam0"], frame0.data[np.newaxis])
        np.testing.assert_array_equal(model_input[f"{IMAGES}.cam1"], frame1.data[np.newaxis])

    def test_single_robot_embedded_image_uses_bare_images_key(self) -> None:
        """A single robot-embedded image also uses the bare ``images`` key."""
        frame = _make_frame(3)
        robot_obs = FakeRobotObservation(
            joint_positions=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            timestamp=0.0,
            sensor_data=None,
            images={"wrist": frame},
        )
        policy_source = self._policy_source()

        model_input = policy_source.to_model_input(robot_obs, {})

        assert IMAGES in model_input
        np.testing.assert_array_equal(model_input[IMAGES], frame.data[np.newaxis])

    def test_robot_and_camera_images_combine_to_namespaced_keys(self) -> None:
        """A robot-embedded image plus a camera frame yield namespaced keys."""
        embedded = _make_frame(4)
        camera = _make_frame(5)
        robot_obs = FakeRobotObservation(
            joint_positions=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            timestamp=0.0,
            sensor_data=None,
            images={"wrist": embedded},
        )
        policy_source = self._policy_source()

        model_input = policy_source.to_model_input(robot_obs, {"cam0": camera})

        assert IMAGES not in model_input
        np.testing.assert_array_equal(model_input[f"{IMAGES}.wrist"], embedded.data[np.newaxis])
        np.testing.assert_array_equal(model_input[f"{IMAGES}.cam0"], camera.data[np.newaxis])

    def test_no_images_omits_image_keys(self) -> None:
        """With no images, no ``images`` keys are present."""
        robot_obs = FakeRobotObservation(
            joint_positions=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            timestamp=0.0,
            sensor_data=None,
            images=None,
        )
        policy_source = self._policy_source()

        model_input = policy_source.to_model_input(robot_obs, {})

        assert IMAGES not in model_input
        assert not any(key.startswith(f"{IMAGES}.") for key in model_input)

    def test_task_included_when_set(self) -> None:
        """The task string is forwarded when configured on the action source."""
        robot_obs = FakeRobotObservation(
            joint_positions=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            timestamp=0.0,
            sensor_data=None,
            images=None,
        )
        policy_source = self._policy_source(task="pick the cube")

        model_input = policy_source.to_model_input(robot_obs, {"cam0": _make_frame(1)})

        assert model_input[TASK] == ["pick the cube"]


@dataclass
class _LifecycleRecorder:
    """Captures lifecycle events and final callback disposal."""

    events: list[str] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False

    def on_lifecycle(self, event: LifecycleEvent) -> None:
        self.events.append(event.event)
        self.metadata.append(event.metadata)

    def close(self) -> None:
        self.closed = True

    @property
    def shutdown_metadata(self) -> dict[str, Any]:
        return next(m for e, m in zip(self.events, self.metadata, strict=True) if e == "shutdown")


@dataclass
class _InterruptOnceOnShutdown:
    """Raises ``KeyboardInterrupt`` from ``on_lifecycle``, once, on shutdown."""

    fired: bool = False

    def on_lifecycle(self, event: LifecycleEvent) -> None:
        if event.event == "shutdown" and not self.fired:
            self.fired = True
            raise KeyboardInterrupt


@dataclass
class _StopBeforeSend:
    """Requests a stop from ``on_action_ready``, i.e. before the tick's send.

    Stopping here makes the "in-flight action is still sent" guarantee
    falsifiable: bailing out between the check and the send would drop it.
    """

    step: int
    runtime: RobotRuntime | None = None

    def on_action_ready(self, *, action: np.ndarray, step: int) -> np.ndarray:
        if step >= self.step:
            assert self.runtime is not None
            self.runtime.stop()
        return action


@dataclass
class _RaisingActionSource:
    """Action source whose ``update()`` raises, to drive the exit paths."""

    exc: BaseException
    connect_exc: BaseException | None = None
    disconnect_exc: BaseException | None = None

    def connect(self, *, bus: Any, session_id: str) -> None:
        if self.connect_exc is not None:
            raise self.connect_exc

    def update(self, robot_state: Any, camera_frames: Any, step: int) -> np.ndarray:
        raise self.exc

    def disconnect(self) -> None:
        if self.disconnect_exc is not None:
            raise self.disconnect_exc


class _CountdownStopSignal:
    """Stop signal that trips after *polls* checks; implements only ``is_set()``."""

    def __init__(self, polls: int) -> None:
        self.polls = polls
        self.calls = 0

    def is_set(self) -> bool:
        self.calls += 1
        return self.calls > self.polls


@contextmanager
def _frozen_time() -> Iterator[MagicMock]:
    """Patch ``core.time`` so ticks are instantaneous and deterministic."""
    with patch("physicalai.runtime.core.time") as mock_time:
        mock_time.perf_counter.return_value = 0.0
        mock_time.time.return_value = 0.0
        mock_time.sleep = MagicMock()
        yield mock_time


def _stop_runtime(**kwargs: Any) -> tuple[RobotRuntime, MagicMock]:
    """Build a connected runtime, defaulting to a trivial action source."""
    robot = kwargs.pop("robot", None) or _make_mock_robot()
    source = kwargs.pop("action_source", None) or FakeActionSource(next_action=np.zeros(3, dtype=np.float32))
    fps = kwargs.pop("fps", 10.0)
    runtime = RobotRuntime(robot=robot, action_source=source, fps=fps, **kwargs)
    runtime._connected = True  # noqa: SLF001
    return runtime, robot


class TestCooperativeStop:
    """``stop()`` / ``stop_event`` — the cooperative exit path."""

    def test_stop_mid_run_still_sends_the_in_flight_action(self) -> None:
        stopper = _StopBeforeSend(step=2)
        runtime, robot = _stop_runtime(callbacks=[stopper])
        stopper.runtime = runtime

        with _frozen_time():
            steps = runtime.run(duration_s=100.0)

        assert steps == 3
        assert robot.send_action.call_count == 3  # step 2's send still happened
        assert runtime.last_run_reason == "stop_requested"

    def test_stop_before_run_is_remembered_then_cleared(self) -> None:
        """One path end to end: the loop's stop check at step 0.

        A ``stop()`` arriving before ``run()`` is remembered, repeating it
        changes nothing, and it wins over an equally-satisfied ``duration_s``.
        The next ``run()`` is unaffected.
        """
        runtime, robot = _stop_runtime()

        runtime.stop()
        runtime.stop()
        assert runtime.last_run_reason is None

        with _frozen_time():
            stopped = runtime.run(duration_s=0.0)  # both exits satisfied
            assert stopped == 0
            assert type(stopped) is int  # noqa: E721 — not np.int64
            robot.send_action.assert_not_called()
            assert runtime.last_run_reason == "stop_requested"

            resumed = runtime.run(duration_s=0.3)

        assert resumed == 3
        assert robot.send_action.call_count == 3
        assert runtime.last_run_reason == "duration_elapsed"

    @pytest.mark.parametrize("make_event", [threading.Event, mp.Event], ids=["threading", "multiprocessing"])
    def test_external_stop_event(self, make_event: Callable[[], Any]) -> None:
        """Factory, not instance, so no mp semaphore is created at collection."""
        runtime, robot = _stop_runtime()
        stop_event = make_event()
        stop_event.set()

        with _frozen_time():
            steps = runtime.run(duration_s=100.0, stop_event=stop_event)

        assert steps == 0
        robot.send_action.assert_not_called()
        assert runtime.last_run_reason == "stop_requested"

    def test_duck_typed_stop_signal_needs_only_is_set(self) -> None:
        runtime, robot = _stop_runtime()
        signal = _CountdownStopSignal(polls=3)

        with _frozen_time():
            steps = runtime.run(duration_s=100.0, stop_event=signal)

        assert steps == 3
        assert robot.send_action.call_count == 3
        assert runtime.last_run_reason == "stop_requested"

    def test_stop_signal_is_not_runtime_checkable(self) -> None:
        """The runtime does no isinstance anywhere; the protocol must not invite it."""
        with pytest.raises(TypeError):
            isinstance(threading.Event(), StopSignal)  # type: ignore[misc]  # noqa: S101

    def test_stop_from_another_thread_ends_a_live_loop(self) -> None:
        """The headline claim, against a real loop and real time."""
        first_tick = threading.Event()

        class _Signal:
            def on_action_sent(self, *, action: np.ndarray, step: int) -> None:  # noqa: ARG002
                first_tick.set()

        runtime, _robot = _stop_runtime(fps=50.0, callbacks=[_Signal()])

        def stopper() -> None:
            first_tick.wait(timeout=5.0)
            runtime.stop()

        thread = threading.Thread(target=stopper)
        thread.start()
        try:
            steps = runtime.run(duration_s=30.0)
        finally:
            thread.join(timeout=5.0)

        assert runtime.last_run_reason == "stop_requested"
        assert 1 <= steps < 500  # ended on the stop, far short of duration_s


class TestStopFlagLifecycle:
    """The stop flag must never survive a run, however that run ended."""

    def test_action_source_connect_failure_does_not_poison_runtime(self) -> None:
        source = _RaisingActionSource(exc=RuntimeError("unused"), connect_exc=RuntimeError("model load failed"))
        recorder = _LifecycleRecorder()
        runtime, robot = _stop_runtime(action_source=source, callbacks=[recorder])

        runtime.stop()
        with _frozen_time(), pytest.raises(RuntimeError, match="model load failed"):
            runtime.run(duration_s=1.0)

        assert runtime.last_run_reason == "error"
        assert recorder.shutdown_metadata["reason"] == "error"
        assert not recorder.closed

        runtime._action_source = FakeActionSource(next_action=np.zeros(3, dtype=np.float32))  # noqa: SLF001
        with _frozen_time():
            assert runtime.run(duration_s=0.3) == 3
        assert robot.send_action.call_count == 3

    @pytest.mark.parametrize("raise_from", ["disconnect", "on_lifecycle"], ids=["disconnect", "on_lifecycle"])
    def test_base_exception_during_teardown_still_completes_shutdown(self, raise_from: str) -> None:
        """A Ctrl+C mid-teardown must not skip the event, the flush, or the clear.

        The bus isolates ``Exception`` but lets ``BaseException`` through, so
        each teardown step needs its own ``finally``.
        """
        recorder = _LifecycleRecorder()
        callbacks: list[Any] = [recorder]
        source: Any = FakeActionSource(next_action=np.zeros(3, dtype=np.float32))
        if raise_from == "disconnect":
            source = _RaisingActionSource(exc=RuntimeError("unused"), disconnect_exc=KeyboardInterrupt())
        else:
            callbacks.append(_InterruptOnceOnShutdown())

        runtime, robot = _stop_runtime(action_source=source, callbacks=callbacks)
        runtime.stop()

        with _frozen_time(), pytest.raises(KeyboardInterrupt):
            runtime.run(duration_s=1.0)

        assert recorder.shutdown_metadata["reason"] == "stop_requested"
        assert not recorder.closed
        assert runtime.last_run_reason == "stop_requested"

        # The flag did not survive into the next session.
        runtime._action_source = FakeActionSource(next_action=np.zeros(3, dtype=np.float32))  # noqa: SLF001
        with _frozen_time():
            assert runtime.run(duration_s=0.3) == 3
        assert robot.send_action.call_count == 3

    def test_not_connected_run_leaves_previous_reason_untouched(self) -> None:
        runtime, _robot = _stop_runtime()
        with _frozen_time():
            runtime.run(duration_s=0.1)
        assert runtime.last_run_reason == "duration_elapsed"

        runtime._connected = False  # noqa: SLF001
        with pytest.raises(RuntimeError, match="connect"):
            runtime.run(duration_s=0.1)

        assert runtime.last_run_reason == "duration_elapsed"

    def test_stop_during_teardown_does_not_leak_into_next_run(self) -> None:
        """Why the flag clears at the *end* of shutdown: clearing it first would
        let a stop arriving mid-teardown zero-step the following session.
        """

        class _StopOnDisconnect:
            runtime: RobotRuntime | None = None

            def connect(self, *, bus: Any, session_id: str) -> None: ...

            def update(self, robot_state: Any, camera_frames: Any, step: int) -> np.ndarray:
                return np.zeros(3, dtype=np.float32)

            def disconnect(self) -> None:
                assert self.runtime is not None
                self.runtime.stop()

        source = _StopOnDisconnect()
        runtime, _robot = _stop_runtime(action_source=source)
        source.runtime = runtime

        with _frozen_time():
            assert runtime.run(duration_s=0.2) == 2
            assert runtime.run(duration_s=0.3) == 3

        assert runtime.last_run_reason == "duration_elapsed"


def _fake_source() -> FakeActionSource:
    return FakeActionSource(next_action=np.zeros(3, dtype=np.float32))


class TestRunReason:
    """All four exit reasons, on both the property and the shutdown event."""

    @pytest.mark.parametrize(
        ("make_source", "stop_first", "expected"),
        [
            (_fake_source, False, "duration_elapsed"),
            (_fake_source, True, "stop_requested"),
            (lambda: _RaisingActionSource(exc=KeyboardInterrupt()), False, "interrupted"),
        ],
        ids=["duration_elapsed", "stop_requested", "interrupted"],
    )
    def test_reason_on_normal_return(self, make_source: Callable[[], Any], stop_first: bool, expected: str) -> None:
        recorder = _LifecycleRecorder()
        runtime, _robot = _stop_runtime(action_source=make_source(), callbacks=[recorder])
        if stop_first:
            runtime.stop()

        with _frozen_time():
            runtime.run(duration_s=0.2)

        assert runtime.last_run_reason == expected
        assert recorder.shutdown_metadata["reason"] == expected

    @pytest.mark.parametrize(
        ("via", "exc_type", "message"),
        [
            ("source", ConnectionError, "link down"),
            ("source", WorkerDiedError, "dead"),
            ("callback", RuntimeError, "filter blew up"),
        ],
        ids=["source_connection_error", "source_worker_died", "callback_raises"],
    )
    def test_error_reason_on_propagating_exception(
        self,
        via: str,
        exc_type: type[Exception],
        message: str,
    ) -> None:
        """A crashed run is not mislabelled as a normal exit."""
        recorder = _LifecycleRecorder()
        callbacks: list[Any] = [recorder]
        source: Any = _fake_source()
        if via == "source":
            source = _RaisingActionSource(exc=exc_type(message))
        else:
            bad = MagicMock()
            bad.on_action_ready.side_effect = exc_type(message)
            callbacks.append(bad)

        runtime, _robot = _stop_runtime(action_source=source, callbacks=callbacks)

        with _frozen_time(), pytest.raises(exc_type, match=message):
            runtime.run(duration_s=100.0)

        assert runtime.last_run_reason == "error"
        assert recorder.shutdown_metadata["reason"] == "error"

    def test_reason_is_none_while_run_in_flight(self) -> None:
        seen: list[Any] = []

        class _ReasonProbe:
            def on_action_sent(self, *, action: np.ndarray, step: int) -> None:  # noqa: ARG002
                seen.append(runtime.last_run_reason)

        runtime, _robot = _stop_runtime(callbacks=[_ReasonProbe()])

        with _frozen_time():
            runtime.run(duration_s=0.1)
            assert runtime.last_run_reason == "duration_elapsed"
            runtime.run(duration_s=0.1)

        assert seen == [None, None]  # never the prior run's reason


class TestStopEventNotInConfigSchema:
    def test_from_config_rejects_stop_event_as_unknown_option(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Asserts the *reason* for rejection, which is what the skip controls.

        Without ``skip={"stop_event"}`` the key is a known option and fails type
        validation instead, so asserting on bare ``SystemExit`` covers nothing.
        """
        cfg_path = tmp_path / "runtime.yaml"
        cfg_path.write_text(_minimal_yaml() + "run:\n  stop_event: something\n")

        with pytest.raises(SystemExit):
            RobotRuntime.from_config(cfg_path)

        assert "does not accept option 'stop_event'" in capsys.readouterr().err


class TestPolicySourceRerun:
    """A runtime carrying a real ``PolicySource`` must survive being run twice."""

    @staticmethod
    def _model(chunk_rows: int = 5, action_dim: int = 3) -> MagicMock:
        model = MagicMock()
        model.predict_action_chunk.side_effect = lambda _obs: np.zeros((chunk_rows, action_dim), dtype=np.float32)
        return model

    def test_queue_counters_describe_one_run_not_all_runs(self) -> None:
        """``connect()`` calls ``reset()``, so the counters are per-session.

        The reference docs point callers at ``action_queue.total_pops`` for run
        stats, and ``_reset_session()`` zeroes its own counters each run — the
        queue has to agree.
        """
        source = PolicySource(model=self._model(), execution=SyncExecution())
        runtime, robot = _stop_runtime(action_source=source)

        with _frozen_time():
            first = runtime.run(duration_s=0.5)
            assert source.action_queue.total_pops == first

            second = runtime.run(duration_s=0.5)

        assert first == second == 5
        assert robot.send_action.call_count == 10
        assert source.action_queue.total_pops == second  # not first + second

    def test_previous_queue_contents_are_never_replayed(self) -> None:
        """A stopped run leaves unsent actions behind; they must be dropped.

        They were computed from observations that are now stale, so ``connect()``
        discards them instead of resuming mid-chunk.
        """
        tag = {"value": 1.0}
        model = MagicMock()
        model.predict_action_chunk.side_effect = lambda _obs: np.full((8, 3), tag["value"], dtype=np.float32)
        source = PolicySource(model=model, execution=SyncExecution())
        runtime, robot = _stop_runtime(action_source=source)

        with _frozen_time():
            runtime.run(duration_s=0.2)
        assert source.action_queue.remaining > 0

        tag["value"] = 2.0
        robot.send_action.reset_mock()
        with _frozen_time():
            runtime.run(duration_s=0.2)

        sent = [float(call[0][0][0]) for call in robot.send_action.call_args_list]
        assert sent
        assert all(value == 2.0 for value in sent), f"run 1 actions replayed: {sent}"

    def test_rerun_with_async_execution_keeps_inferring(self) -> None:
        """The second run must be driven by fresh inference, not a frozen action.

        Regression guard for two faults that made a re-run *look* fine:
        ``PolicySource`` skipping warmup on an emptied queue, and
        ``AsyncExecution`` spawning a worker that saw a stale stop flag and
        exited. Together: a full-length run with zero inference.

        Real time, since the point is that a background thread runs.
        """
        execution = AsyncExecution()
        source = PolicySource(model=self._model(chunk_rows=5), execution=execution)
        runtime, _robot = _stop_runtime(action_source=source, fps=50.0)

        steps_first = runtime.run(duration_s=0.6)
        steps_second = runtime.run(duration_s=0.6)

        assert steps_first == steps_second == 30
        # Both counters restart each run, so read them directly.
        assert execution.inference_count >= 2, "no background inference in the second run"
        holds = source.action_queue.total_holds
        assert holds < steps_second // 3, f"queue starved: {holds} holds"


class TestPolicySourceEpisodeReset:
    @staticmethod
    def _observation(value: float) -> FakeRobotObservation:
        return FakeRobotObservation(
            joint_positions=np.full(3, value, dtype=np.float32),
            timestamp=0.0,
            sensor_data=None,
            images=None,
        )

    def test_explicit_warmup_seeds_first_update_without_repeating_inference(self) -> None:
        model = MagicMock()
        model.predict_action_chunk.side_effect = lambda obs: np.repeat(obs[STATE], 4, axis=0)
        source = PolicySource(model=model, execution=SyncExecution())
        source.connect(bus=_CallbackBus([]), session_id="session")
        try:
            source.warmup(source.to_model_input(self._observation(2.0), {}))
            action = source.update(self._observation(3.0), {}, step=0)

            np.testing.assert_array_equal(action, np.full(3, 2.0, dtype=np.float32))
            model.predict_action_chunk.assert_called_once()
        finally:
            source.disconnect()

    def test_reset_reseeds_from_current_observation(self) -> None:
        model = MagicMock()
        model.predict_action_chunk.side_effect = lambda obs: np.repeat(obs[STATE], 4, axis=0)
        source = PolicySource(model=model, execution=SyncExecution())
        source.connect(bus=_CallbackBus([]), session_id="session")
        try:
            first = source.update(self._observation(1.0), {}, step=0)
            np.testing.assert_array_equal(first, np.full(3, 1.0, dtype=np.float32))
            assert source.action_queue.remaining > 0

            resets_after_connect = model.reset.call_count
            source.reset(reset_model=False)
            assert source.action_queue.remaining == 0
            second = source.update(self._observation(9.0), {}, step=1)

            np.testing.assert_array_equal(second, np.full(3, 9.0, dtype=np.float32))
            assert model.reset.call_count == resets_after_connect
        finally:
            source.disconnect()

    def test_custom_execution_runs_but_rejects_episode_reset(self) -> None:
        class LegacyExecution(Execution):
            def start(self, model: Any, action_queue: Any) -> None:
                self.model = model
                self.queue = action_queue

            def maybe_request(self, observation: dict[str, Any]) -> None:
                pass

            def warmup(self, sample_observation: dict[str, Any]) -> None:
                self.queue.push_chunk(self.model.predict_action_chunk(sample_observation))

            def stop(self) -> None:
                pass

            @property
            def chunk_size(self) -> int:
                return 1

        model = MagicMock()
        model.predict_action_chunk.return_value = np.ones((1, 3), dtype=np.float32)
        source = PolicySource(model=model, execution=LegacyExecution())

        source.connect(bus=_CallbackBus([]), session_id="session")
        try:
            action = source.update(self._observation(1.0), {}, step=0)
            np.testing.assert_array_equal(action, np.ones(3, dtype=np.float32))
            remaining = source.action_queue.remaining
            with pytest.raises(RuntimeError, match="does not support safe episode reset"):
                source.reset()
            assert source.action_queue.remaining == remaining
        finally:
            source.disconnect()

    def test_duck_typed_execution_with_reset_is_supported(self) -> None:
        class DuckExecution:
            def set_bus(self, bus: Any, session_id: str) -> None:
                pass

            def start(self, model: Any, action_queue: Any) -> None:
                self.model = model
                self.queue = action_queue

            def reset(self, *, reset_model: bool = True) -> None:
                if reset_model:
                    self.model.reset()

            def maybe_request(self, observation: dict[str, Any]) -> None:
                pass

            def warmup(self, sample_observation: dict[str, Any]) -> None:
                self.queue.push_chunk(self.model.predict_action_chunk(sample_observation))

            def stop(self) -> None:
                pass

        model = MagicMock()
        model.predict_action_chunk.return_value = np.ones((1, 3), dtype=np.float32)
        source = PolicySource(model=model, execution=DuckExecution())  # type: ignore[arg-type]

        source.connect(bus=_CallbackBus([]), session_id="session")
        try:
            source.reset()
            assert model.reset.call_count == 2
        finally:
            source.disconnect()

    def test_reset_and_warmup_require_connection(self) -> None:
        source = PolicySource(model=MagicMock(), execution=SyncExecution())

        with pytest.raises(RuntimeError, match=r"reset\(\) requires connect"):
            source.reset()
        with pytest.raises(RuntimeError, match=r"warmup\(\) requires connect"):
            source.warmup({STATE: np.zeros((1, 3), dtype=np.float32)})


class TestRuntimeCallbackReuse:
    def test_jsonl_callback_records_two_runs_and_closes_on_disconnect(self, tmp_path: Path) -> None:
        path = tmp_path / "runs.jsonl"
        callback = JsonlCallback(path)
        robot = _make_mock_robot()
        robot.joint_names = ["joint_1", "joint_2", "joint_3"]
        runtime, _robot = _stop_runtime(robot=robot, callbacks=[callback])

        with _frozen_time():
            runtime.run(duration_s=0.1)
            runtime.run(duration_s=0.1)

        records = [json.loads(line) for line in path.read_text().splitlines()]
        starts = [record for record in records if record.get("event") == "start"]
        shutdowns = [record for record in records if record.get("event") == "shutdown"]
        assert len(starts) == len(shutdowns) == 2
        assert starts[0]["session_id"] != starts[1]["session_id"]

        runtime.disconnect()
        assert callback._file.closed  # noqa: SLF001
        with pytest.raises(RuntimeError, match="cannot be connected again"):
            runtime.connect()

    def test_async_callback_delivers_two_runs_before_final_close(self) -> None:
        inner = MagicMock(spec=["on_tick", "on_lifecycle", "close"])
        callback = AsyncCallback(inner)
        runtime, _robot = _stop_runtime(callbacks=[callback])

        with _frozen_time():
            runtime.run(duration_s=0.1)
            runtime.run(duration_s=0.1)

        lifecycle_events = [call.args[0].event for call in inner.on_lifecycle.call_args_list]
        assert lifecycle_events == ["start", "shutdown", "start", "shutdown"]
        assert inner.on_tick.call_count == 2
        assert callback._thread.is_alive()  # noqa: SLF001
        inner.close.assert_not_called()

        runtime.disconnect()
        assert not callback._thread.is_alive()  # noqa: SLF001
        inner.close.assert_called_once()
        runtime.disconnect()
        inner.close.assert_called_once()

    def test_concurrent_disconnect_is_idempotent(self) -> None:
        callback = MagicMock(spec=["close"])
        runtime, _robot = _stop_runtime(callbacks=[callback])
        errors: list[BaseException] = []

        def disconnect() -> None:
            try:
                runtime.disconnect()
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=disconnect) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)

        assert errors == []
        callback.close.assert_called_once()

    @pytest.mark.parametrize("operation", ["connect", "disconnect", "run"])
    def test_lifecycle_operation_rejected_while_run_active(self, operation: str) -> None:
        first_tick = threading.Event()
        release = threading.Event()

        class BlockingSource(FakeActionSource):
            def update(self, robot_state: Any, camera_frames: Any, step: int) -> np.ndarray:
                first_tick.set()
                assert release.wait(timeout=5.0)
                return np.zeros(3, dtype=np.float32)

        runtime, _robot = _stop_runtime(action_source=BlockingSource())
        run_thread = threading.Thread(target=runtime.run, kwargs={"duration_s": 0.1})
        run_thread.start()
        assert first_tick.wait(timeout=5.0)
        try:
            with pytest.raises(RuntimeError, match="active"):
                getattr(runtime, operation)()
        finally:
            release.set()
            run_thread.join(timeout=5.0)
