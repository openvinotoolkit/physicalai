# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for physicalai.runtime.core."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from physicalai.runtime import ChunkedActionQueue as ActionQueue, ChunkedActionQueue, PolicySource, RobotRuntime, SyncExecution, WorkerDiedError
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

