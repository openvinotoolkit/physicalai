# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from physicalai.capture import Frame
from physicalai.capture.errors import CaptureError
from physicalai.runtime.execution import SyncExecution
from physicalai.runtime.runtime import (
    PolicyRuntime,
    _MAX_OBS_RETRIES,
    _MAX_SEND_RETRIES,
    _WARMUP_RETRIES,
)


@dataclass
class FakeRobotObservation:
    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None
    images: dict | None

    @property
    def state(self) -> np.ndarray:
        return self.joint_positions


def _make_obs(positions: np.ndarray | None = None) -> FakeRobotObservation:
    if positions is None:
        positions = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    return FakeRobotObservation(
        joint_positions=positions,
        timestamp=time.monotonic(),
        sensor_data=None,
        images=None,
    )


def _make_mock_robot(obs: FakeRobotObservation | None = None) -> MagicMock:
    robot = MagicMock()
    robot.get_observation.return_value = obs or _make_obs()
    return robot


def _make_mock_model(chunk_size: int = 10, action_dim: int = 3) -> MagicMock:
    model = MagicMock()
    model.predict_action_chunk.return_value = np.random.randn(chunk_size, action_dim).astype(np.float32)
    return model


def _make_runtime(
    robot: MagicMock | None = None,
    model: MagicMock | None = None,
    cameras: dict | None = None,
    fps: float = 10.0,
) -> PolicyRuntime:
    return PolicyRuntime(
        robot=robot or _make_mock_robot(),
        model=model or _make_mock_model(),
        execution=SyncExecution(),
        fps=fps,
        cameras=cameras or {},
    )


class TestResilientObserve:
    def test_transient_observe_error_retries_then_succeeds(self) -> None:
        obs = _make_obs()
        robot = _make_mock_robot()
        robot.get_observation.side_effect = [ConnectionError("flake"), obs]

        rt = _make_runtime(robot=robot)

        with patch("physicalai.runtime.runtime.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.perf_counter.return_value = 0.0
            mock_time.time.return_value = 0.0
            result = rt._resilient_observe()

        assert result is not None
        assert robot.get_observation.call_count == 2
        assert rt._stale_obs_ticks == 0

    def test_sustained_observe_error_uses_stale_fallback(self) -> None:
        robot = _make_mock_robot()
        robot.get_observation.side_effect = ConnectionError("down")

        rt = _make_runtime(robot=robot)
        rt._last_robot_obs = _make_obs()

        with patch("physicalai.runtime.runtime.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.perf_counter.return_value = 0.0
            mock_time.time.return_value = 0.0
            result = rt._resilient_observe()

        assert result is not None
        assert robot.get_observation.call_count == _MAX_OBS_RETRIES
        assert rt._stale_obs_ticks == 1
        assert rt._consecutive_error_ticks == 1

    def test_max_consecutive_errors_raises(self) -> None:
        robot = _make_mock_robot()
        robot.get_observation.side_effect = ConnectionError("down")

        rt = _make_runtime(robot=robot, fps=10.0)
        rt._last_robot_obs = _make_obs()
        rt._consecutive_error_ticks = rt._max_consecutive_error_ticks - 1

        with patch("physicalai.runtime.runtime.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            with pytest.raises(ConnectionError, match="Exceeded max consecutive"):
                rt._resilient_observe()

    def test_no_stale_obs_raises_immediately(self) -> None:
        robot = _make_mock_robot()
        robot.get_observation.side_effect = OSError("USB gone")

        rt = _make_runtime(robot=robot)
        assert rt._last_robot_obs is None

        with patch("physicalai.runtime.runtime.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            with pytest.raises(ConnectionError, match="no stale observation"):
                rt._resilient_observe()

    def test_fatal_error_propagates(self) -> None:
        robot = _make_mock_robot()
        robot.get_observation.side_effect = ValueError("bad joint config")

        rt = _make_runtime(robot=robot)

        with pytest.raises(ValueError, match="bad joint config"):
            rt._resilient_observe()

        assert robot.get_observation.call_count == 1


class TestResilientObserveCameras:
    def test_camera_capture_error_uses_stale_frame(self) -> None:
        stale_frame = Frame(data=np.zeros((480, 640, 3), dtype=np.uint8), timestamp=0.0, sequence=0)
        camera = MagicMock()
        camera.read_latest.side_effect = CaptureError("timeout")

        rt = _make_runtime(cameras={"cam0": camera})
        rt._last_camera_frames["cam0"] = stale_frame

        with patch("physicalai.runtime.runtime.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.perf_counter.return_value = 0.0
            mock_time.time.return_value = 0.0
            _robot_obs, camera_frames = rt._resilient_observe()

        assert "cam0" in camera_frames
        assert camera_frames["cam0"] is stale_frame

    def test_camera_first_read_fails_raises(self) -> None:
        camera = MagicMock()
        camera.read_latest.side_effect = CaptureError("no device")

        rt = _make_runtime(cameras={"cam0": camera})

        with patch("physicalai.runtime.runtime.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.perf_counter.return_value = 0.0
            mock_time.time.return_value = 0.0
            with pytest.raises(CaptureError, match="no device"):
                rt._resilient_observe()


class TestResilientSend:
    def test_resilient_send_retries(self) -> None:
        robot = _make_mock_robot()
        robot.send_action.side_effect = [ConnectionError("flake"), None]

        rt = _make_runtime(robot=robot)
        action = np.zeros(3, dtype=np.float32)

        with patch("physicalai.runtime.runtime.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            rt._resilient_send(action)

        assert robot.send_action.call_count == 2
        assert rt._transient_errors == 0
        assert rt._consecutive_error_ticks == 0

    def test_resilient_send_all_retries_fail_skips_tick(self) -> None:
        robot = _make_mock_robot()
        robot.send_action.side_effect = OSError("USB gone")

        rt = _make_runtime(robot=robot)
        action = np.zeros(3, dtype=np.float32)

        with patch("physicalai.runtime.runtime.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            rt._resilient_send(action)

        assert robot.send_action.call_count == _MAX_SEND_RETRIES
        assert rt._transient_errors == 1


class TestWarmupWithRetry:
    def test_warmup_retries_on_connection_error(self) -> None:
        obs = _make_obs()
        robot = _make_mock_robot(obs)
        robot.get_observation.side_effect = [ConnectionError(), ConnectionError(), obs]
        model = _make_mock_model()

        execution = MagicMock()
        rt = PolicyRuntime(
            robot=robot,
            model=model,
            execution=execution,
            fps=10.0,
        )

        with patch("physicalai.runtime.runtime.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.perf_counter.return_value = 0.0
            mock_time.time.return_value = 0.0
            rt._warmup_with_retry()

        assert execution.warmup.called

    def test_warmup_exhausted_raises(self) -> None:
        robot = _make_mock_robot()
        robot.get_observation.side_effect = ConnectionError("down")

        rt = _make_runtime(robot=robot)

        with patch("physicalai.runtime.runtime.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            with pytest.raises(ConnectionError, match=f"Warmup failed after {_WARMUP_RETRIES}"):
                rt._warmup_with_retry()


class TestShutdownDrain:
    def test_shutdown_drain_uses_resilient_send(self) -> None:
        robot = _make_mock_robot()
        robot.send_action.side_effect = OSError("USB gone during drain")
        model = _make_mock_model(chunk_size=20)

        rt = _make_runtime(robot=robot, model=model)
        rt._action_queue.push_chunk(np.ones((5, 3), dtype=np.float32))

        with patch("physicalai.runtime.runtime.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.perf_counter.return_value = 0.0
            mock_time.time.return_value = 0.0
            rt._shutdown(step=10)

        assert rt._transient_errors > 0


class TestRunStatsWithFaults:
    def test_run_stats_includes_fault_metrics(self) -> None:
        obs = _make_obs()
        robot = _make_mock_robot(obs)

        call_count = [0]

        def get_obs_with_loop_errors():
            call_count[0] += 1
            # Call 1: warmup (_build_model_input)
            # Call 2: first tick (_resilient_observe) — sets _last_robot_obs
            if call_count[0] <= 2:
                return obs
            # Calls 3..5: second tick retries all fail — uses stale fallback
            if call_count[0] <= 2 + _MAX_OBS_RETRIES:
                raise ConnectionError("flake")
            return obs

        robot.get_observation.side_effect = get_obs_with_loop_errors
        robot.send_action.return_value = None

        rt = _make_runtime(robot=robot)
        rt._connected = True

        with patch("physicalai.runtime.runtime.time") as mock_time:
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            stats = rt.run(duration_s=0.3)

        assert stats.stale_obs_ticks >= 1
        assert stats.steps == 3
