# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from physicalai.runtime.events import InferenceEvent, LifecycleEvent, TickEvent


@pytest.fixture()
def mock_rerun() -> MagicMock:
    """Provide a mock rerun module injected into sys.modules."""
    rr = MagicMock()
    rr.__name__ = "rerun"
    rr.Scalar = MagicMock(side_effect=lambda v: ("Scalar", v))
    rr.Image = MagicMock(side_effect=lambda d: ("Image", d))
    rr.TextLog = MagicMock(side_effect=lambda t: ("TextLog", t))
    return rr


@pytest.fixture()
def _patch_rerun(mock_rerun: MagicMock) -> Any:
    """Patch rerun in sys.modules so RerunCallback can import it."""
    with patch.dict(sys.modules, {"rerun": mock_rerun}):
        yield


@pytest.fixture()
def make_callback(_patch_rerun: Any, mock_rerun: MagicMock) -> Any:
    """Factory that creates a RerunCallback with the mocked rerun module."""
    from physicalai.runtime.callbacks import RerunCallback

    def _factory(**kwargs: Any) -> RerunCallback:
        defaults: dict[str, Any] = {"mode": "spawn"}
        defaults.update(kwargs)
        return RerunCallback(**defaults)

    return _factory


def _lifecycle_start(session_id: str = "sess-1", fps: int = 30) -> LifecycleEvent:
    return LifecycleEvent(
        session_id=session_id,
        timestamp=1000.0,
        event="start",
        metadata={"fps": fps, "cameras": []},
    )


def _tick(step: int = 0, dof: int = 7) -> TickEvent:
    return TickEvent(
        session_id="sess-1",
        step=step,
        timestamp=1000.0 + step * (1 / 30),
        joint_positions=np.arange(dof, dtype=np.float64),
        action_sent=np.ones(dof, dtype=np.float64),
        queue_remaining=5,
        loop_duration_s=0.033,
        sleep_time_s=0.0,
        stale_obs=False,
    )


def _inference(horizon: int = 50, dof: int = 7) -> InferenceEvent:
    return InferenceEvent(
        session_id="sess-1",
        timestamp=1001.0,
        latency_s=0.05,
        offset=0,
        chunk=np.zeros((horizon, dof), dtype=np.float32),
    )


@pytest.mark.usefixtures("_patch_rerun")
class TestRerunCallbackConstruction:
    def test_construction_succeeds_with_mocked_rerun(self, make_callback: Any) -> None:
        cb = make_callback()
        assert cb is not None

    def test_mode_save_requires_save_path(self, make_callback: Any) -> None:
        with pytest.raises(ValueError, match="save_path"):
            make_callback(mode="save")

    def test_mode_save_with_path_succeeds(self, make_callback: Any) -> None:
        cb = make_callback(mode="save", save_path="/tmp/test.rrd")
        assert cb is not None

    def test_missing_rerun_raises_import_error(self) -> None:
        with patch.dict(sys.modules, {"rerun": None}):
            from physicalai.runtime.callbacks import RerunCallback

            with pytest.raises((ImportError, ModuleNotFoundError)):
                RerunCallback(mode="spawn")


@pytest.mark.usefixtures("_patch_rerun")
class TestRerunCallbackLifecycle:
    def test_init_rerun_spawn(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start("my-session"))

        mock_rerun.init.assert_called_once_with(application_id="physicalai-runtime", recording_id="my-session")
        mock_rerun.spawn.assert_called_once()
        mock_rerun.save.assert_not_called()

    def test_init_rerun_save(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback(mode="save", save_path="/tmp/out.rrd")
        cb.on_lifecycle(_lifecycle_start())

        mock_rerun.save.assert_called_once_with("/tmp/out.rrd")
        mock_rerun.spawn.assert_not_called()

    def test_lifecycle_marker_logged(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())

        mock_rerun.log.assert_called()
        calls = [c for c in mock_rerun.log.call_args_list if "lifecycle" in str(c)]
        assert len(calls) > 0

    def test_second_start_does_not_reinitialize(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        cb.on_lifecycle(_lifecycle_start())

        assert mock_rerun.init.call_count == 1

    def test_fps_extracted_from_metadata(self, make_callback: Any) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start(fps=60))
        assert cb._fps == 60


@pytest.mark.usefixtures("_patch_rerun")
class TestRerunCallbackTick:
    def test_tick_logs_joint_scalars(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        mock_rerun.reset_mock()

        cb.on_tick(_tick(step=1, dof=3))

        log_calls = mock_rerun.log.call_args_list
        joint_calls = [c for c in log_calls if "robot/joint/" in str(c.args[0])]
        assert len(joint_calls) == 3

    def test_tick_logs_action_scalars(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        mock_rerun.reset_mock()

        cb.on_tick(_tick(step=1, dof=4))

        log_calls = mock_rerun.log.call_args_list
        action_calls = [c for c in log_calls if "robot/action/" in str(c.args[0])]
        assert len(action_calls) == 4

    def test_tick_logs_runtime_metrics(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        mock_rerun.reset_mock()

        cb.on_tick(_tick(step=1))

        log_calls = mock_rerun.log.call_args_list
        paths = [c.args[0] for c in log_calls]
        assert "runtime/queue_remaining" in paths
        assert "runtime/loop_duration_s" in paths
        assert "runtime/sleep_time_s" in paths
        assert "runtime/stale_obs" in paths

    def test_tick_updates_last_step(self, make_callback: Any) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        cb.on_tick(_tick(step=42))
        assert cb._last_step == 42

    def test_tick_sets_timelines(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        mock_rerun.reset_mock()

        event = _tick(step=5)
        cb.on_tick(event)

        mock_rerun.set_time_sequence.assert_called_with("step", 5)
        mock_rerun.set_time_seconds.assert_called_with("wall", event.timestamp)

    def test_none_joint_positions_skipped(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        mock_rerun.reset_mock()

        event = TickEvent(
            session_id="sess-1",
            step=1,
            timestamp=1000.0,
            joint_positions=None,
            action_sent=None,
            queue_remaining=5,
            loop_duration_s=0.033,
            sleep_time_s=0.0,
            stale_obs=False,
        )
        cb.on_tick(event)

        log_calls = mock_rerun.log.call_args_list
        joint_calls = [c for c in log_calls if "robot/joint/" in str(c.args[0])]
        assert len(joint_calls) == 0


@pytest.mark.usefixtures("_patch_rerun")
class TestRerunCallbackInference:
    def test_ghost_trajectory_logs_at_future_steps(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start(fps=30))

        cb.on_tick(_tick(step=10))
        mock_rerun.reset_mock()

        cb.on_inference(_inference(horizon=5, dof=3))

        set_time_calls = mock_rerun.set_time_sequence.call_args_list
        step_values = [c.args[1] for c in set_time_calls if c.args[0] == "step"]
        assert step_values == [11, 12, 13, 14, 15]

    def test_ghost_trajectory_logs_predicted_scalars(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        cb.on_tick(_tick(step=0))
        mock_rerun.reset_mock()

        cb.on_inference(_inference(horizon=2, dof=3))

        log_calls = mock_rerun.log.call_args_list
        predicted_calls = [c for c in log_calls if "robot/predicted/" in str(c.args[0])]
        assert len(predicted_calls) == 2 * 3

    def test_ghost_trajectory_wall_time_spacing(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start(fps=10))
        cb.on_tick(_tick(step=0))
        mock_rerun.reset_mock()

        event = _inference(horizon=3, dof=1)
        cb.on_inference(event)

        wall_calls = mock_rerun.set_time_seconds.call_args_list
        wall_values = [c.args[1] for c in wall_calls if c.args[0] == "wall"]
        expected = [event.timestamp + k / 10 for k in range(3)]
        assert wall_values == pytest.approx(expected)


@pytest.mark.usefixtures("_patch_rerun")
class TestRerunCallbackImageDecimation:
    def test_decimation_skips_non_nth_ticks(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback(image_decimation=3)
        cb.on_lifecycle(_lifecycle_start())

        for step in range(6):
            mock_rerun.reset_mock()
            cb.on_tick(_tick(step=step))

    def test_decimation_default_is_3(self, make_callback: Any) -> None:
        cb = make_callback()
        assert cb._image_decimation == 3


@pytest.mark.usefixtures("_patch_rerun")
class TestRerunCallbackCameraSubscribers:
    def test_non_shared_camera_warns(self, make_callback: Any, mock_rerun: MagicMock, caplog: Any) -> None:
        fake_camera = MagicMock()
        fake_camera.__class__.__name__ = "FakeCamera"

        cb = make_callback(cameras={"top": fake_camera})
        cb.on_lifecycle(_lifecycle_start())

        assert cb._camera_subscribers == {}

    def test_shared_camera_subscriber_logs_frames(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback(image_decimation=1)
        cb.on_lifecycle(_lifecycle_start())

        mock_sub = MagicMock()
        mock_frame = MagicMock()
        mock_frame.data = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_sub.read_latest.return_value = mock_frame
        cb._camera_subscribers = {"top": mock_sub}

        mock_rerun.reset_mock()
        cb.on_tick(_tick(step=0))

        log_calls = mock_rerun.log.call_args_list
        image_calls = [c for c in log_calls if "camera/top" in str(c.args[0])]
        assert len(image_calls) == 1

    def test_close_disconnects_subscribers(self, make_callback: Any) -> None:
        cb = make_callback()
        mock_sub = MagicMock()
        cb._camera_subscribers = {"cam1": mock_sub, "cam2": MagicMock()}

        cb.close()

        mock_sub.disconnect.assert_called_once()
        assert cb._camera_subscribers == {}

    def test_close_handles_disconnect_error(self, make_callback: Any) -> None:
        cb = make_callback()
        mock_sub = MagicMock()
        mock_sub.disconnect.side_effect = RuntimeError("connection lost")
        cb._camera_subscribers = {"cam1": mock_sub}

        cb.close()
        assert cb._camera_subscribers == {}


@pytest.mark.usefixtures("_patch_rerun")
class TestRerunCallbackLifecycleMarker:
    def test_lifecycle_logs_text(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        mock_rerun.reset_mock()

        event = LifecycleEvent(
            session_id="sess-1",
            timestamp=2000.0,
            event="shutdown",
            metadata={"reason": "done"},
        )
        cb.on_lifecycle(event)

        log_calls = mock_rerun.log.call_args_list
        lifecycle_calls = [c for c in log_calls if "runtime/lifecycle/shutdown" in str(c.args[0])]
        assert len(lifecycle_calls) == 1
