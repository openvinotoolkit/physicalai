# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from physicalai.capture import Frame
from physicalai.capture.errors import CaptureError
from physicalai.runtime import PolicySource, RobotRuntime, SyncExecution
from physicalai.runtime.core import (
    _MAX_OBS_RETRIES,
    _MAX_SEND_RETRIES,
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
) -> RobotRuntime:
    policy_source = PolicySource(model=model or _make_mock_model(), execution=SyncExecution())
    return RobotRuntime(
        robot=robot or _make_mock_robot(),
        action_source=policy_source,
        fps=fps,
        cameras=cameras or {},
    )


class TestResilientObserve:
    def test_transient_observe_error_retries_then_succeeds(self) -> None:
        obs = _make_obs()
        robot = _make_mock_robot()
        robot.get_observation.side_effect = [ConnectionError("flake"), obs]

        rt = _make_runtime(robot=robot)

        with patch("physicalai.runtime.core.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.perf_counter.return_value = 0.0
            mock_time.time.return_value = 0.0
            result = rt._read_observation()

        assert result is not None
        assert robot.get_observation.call_count == 2
        assert rt._stale_obs_ticks == 0

    def test_sustained_observe_error_uses_stale_fallback(self) -> None:
        robot = _make_mock_robot()
        robot.get_observation.side_effect = ConnectionError("down")

        rt = _make_runtime(robot=robot)
        rt._last_robot_obs = _make_obs()

        with patch("physicalai.runtime.core.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.perf_counter.return_value = 0.0
            mock_time.time.return_value = 0.0
            result = rt._read_observation()

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

        with patch("physicalai.runtime.core.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            with pytest.raises(ConnectionError, match="Exceeded max consecutive"):
                rt._read_observation()

    def test_no_stale_obs_raises_immediately(self) -> None:
        robot = _make_mock_robot()
        robot.get_observation.side_effect = OSError("USB gone")

        rt = _make_runtime(robot=robot)
        assert rt._last_robot_obs is None

        with patch("physicalai.runtime.core.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            with pytest.raises(ConnectionError, match="no stale observation"):
                rt._read_observation()

    def test_fatal_error_propagates(self) -> None:
        robot = _make_mock_robot()
        robot.get_observation.side_effect = ValueError("bad joint config")

        rt = _make_runtime(robot=robot)

        with pytest.raises(ValueError, match="bad joint config"):
            rt._read_observation()

        assert robot.get_observation.call_count == 1


class TestResilientObserveCameras:
    def test_camera_capture_error_uses_stale_frame(self) -> None:
        stale_frame = Frame(data=np.zeros((480, 640, 3), dtype=np.uint8), timestamp=0.0, sequence=0)
        camera = MagicMock()
        camera.read_latest.side_effect = CaptureError("timeout")

        rt = _make_runtime(cameras={"cam0": camera})
        rt._last_camera_frames["cam0"] = stale_frame

        with patch("physicalai.runtime.core.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.perf_counter.return_value = 0.0
            mock_time.time.return_value = 0.0
            _robot_obs, camera_frames = rt._read_observation()

        assert "cam0" in camera_frames
        assert camera_frames["cam0"] is stale_frame

    def test_camera_first_read_fails_raises(self) -> None:
        camera = MagicMock()
        camera.read_latest.side_effect = CaptureError("no device")

        rt = _make_runtime(cameras={"cam0": camera})

        with patch("physicalai.runtime.core.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.perf_counter.return_value = 0.0
            mock_time.time.return_value = 0.0
            with pytest.raises(CaptureError, match="no device"):
                rt._read_observation()


class TestResilientSend:
    def test_resilient_send_retries(self) -> None:
        robot = _make_mock_robot()
        robot.send_action.side_effect = [ConnectionError("flake"), None]

        rt = _make_runtime(robot=robot)
        action = np.zeros(3, dtype=np.float32)

        with patch("physicalai.runtime.core.time") as mock_time:
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

        with patch("physicalai.runtime.core.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            rt._resilient_send(action)

        assert robot.send_action.call_count == _MAX_SEND_RETRIES
        assert rt._transient_errors == 1


class TestRunReturnsStepsWithFaults:
    def test_run_returns_step_count_despite_stale_reads(self) -> None:
        """No RunStats — run() returns plain steps; stale reads are visible via
        TickEvent.stale_obs (see TestStaleObsEventFlag), not an aggregate return field.
        """
        obs = _make_obs()
        robot = _make_mock_robot(obs)

        call_count = [0]

        def get_obs_with_loop_errors():
            call_count[0] += 1
            # Reads are pull-based: the robot is read only on ticks that request
            # inference. With request_threshold=1.0 the queue is below threshold
            # every tick after the first pop, so each in-loop tick reads once.
            # Call 1: warmup. Call 2: first read tick — sets _last_robot_obs.
            if call_count[0] <= 2:
                return obs
            # Calls 3..5: next read tick retries all fail — uses stale fallback.
            if call_count[0] <= 2 + _MAX_OBS_RETRIES:
                raise ConnectionError("flake")
            return obs

        robot.get_observation.side_effect = get_obs_with_loop_errors
        robot.send_action.return_value = None

        policy_source = PolicySource(model=_make_mock_model(), execution=SyncExecution(request_threshold=1.0))
        rt = RobotRuntime(robot=robot, action_source=policy_source, fps=10.0)
        rt._connected = True

        with patch("physicalai.runtime.core.time") as mock_time:
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            steps = rt.run(duration_s=0.3)

        assert rt._stale_obs_ticks >= 1
        assert steps == 3


class TestStaleObsEventFlag:
    def test_stale_read_with_successful_send_reports_stale(self) -> None:
        obs = _make_obs()
        robot = _make_mock_robot(obs)

        call_count = [0]

        def get_obs_with_one_stale_tick():
            call_count[0] += 1
            # Call 1: warmup. Call 2: first read tick (sets _last_robot_obs).
            if call_count[0] <= 2:
                return obs
            # Calls 3..5: next read tick fails all retries -> stale fallback used.
            if call_count[0] <= 2 + _MAX_OBS_RETRIES:
                raise ConnectionError("flake")
            return obs

        robot.get_observation.side_effect = get_obs_with_one_stale_tick
        robot.send_action.return_value = None  # send always succeeds

        events: list = []
        callback = MagicMock()
        callback.on_tick.side_effect = events.append
        callback.on_action_ready.side_effect = lambda *, action, step: action  # noqa: ARG005

        policy_source = PolicySource(model=_make_mock_model(), execution=SyncExecution(request_threshold=1.0))
        rt = RobotRuntime(robot=robot, action_source=policy_source, fps=10.0, callbacks=[callback])
        rt._connected = True

        with patch("physicalai.runtime.core.time") as mock_time:
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            rt.run(duration_s=0.3)

        # Despite every send succeeding, the tick whose robot read fell back to a
        # stale observation must report stale_obs=True (it is derived from the
        # per-tick read, not the send-reset error counter).
        assert any(e.stale_obs for e in events)

