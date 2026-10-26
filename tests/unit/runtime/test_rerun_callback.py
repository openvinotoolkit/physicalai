# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from concurrent.futures import Future
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from physicalai.runtime.callbacks.gpu_stats import GpuStatsSample
from physicalai.runtime.events import InferenceEvent, LifecycleEvent, MetricsEvent, TickEvent
from tests.unit.runtime.conftest import FakeRobotObservation


@pytest.fixture()
def mock_rerun() -> MagicMock:
    """Provide a mock rerun module injected into sys.modules."""
    rr = MagicMock()
    rr.__name__ = "rerun"
    rr.Scalars = MagicMock(side_effect=lambda *a, **kw: ("Scalars", a, kw))
    rr.Scalars.columns = MagicMock(return_value=MagicMock())
    rr.Image = MagicMock(side_effect=lambda d: ("Image", d))
    rr.TextLog = MagicMock(side_effect=lambda t: ("TextLog", t))
    rr.StateChange = MagicMock(side_effect=lambda *a, **kw: ("StateChange", a, kw))
    rr.Clear = MagicMock(side_effect=lambda *a, **kw: ("Clear", a, kw))
    rr.TimeColumn = MagicMock(side_effect=lambda *a, **kw: ("TimeColumn", a, kw))
    return rr


@pytest.fixture()
def mock_psutil() -> MagicMock:
    """Provide deterministic OS counters without requiring the observer extra."""
    psutil = MagicMock()
    psutil.Error = RuntimeError
    psutil.cpu_percent.return_value = 42.5
    psutil.cpu_count.return_value = 8
    psutil.virtual_memory.return_value.percent = 62.5
    psutil.virtual_memory.return_value.total = 16 * 2**30
    psutil.virtual_memory.return_value.available = 6 * 2**30
    psutil.Process.return_value.cpu_percent.return_value = 125.0
    psutil.Process.return_value.memory_info.return_value.rss = 256 * 2**20
    return psutil


@pytest.fixture()
def mock_gpu_executor() -> Any:
    executor = MagicMock()
    executor.submit.return_value.done.return_value = False
    with patch("physicalai.runtime.callbacks.rerun.ThreadPoolExecutor", return_value=executor):
        yield executor


@pytest.fixture()
def _patch_rerun(mock_rerun: MagicMock, mock_psutil: MagicMock, mock_gpu_executor: MagicMock) -> Any:
    """Patch optional observer dependencies so RerunCallback can import them."""
    with patch.dict(sys.modules, {"rerun": mock_rerun, "psutil": mock_psutil}):
        yield


@pytest.fixture()
def make_callback(_patch_rerun: Any, mock_rerun: MagicMock) -> Any:
    """Factory that creates a RerunCallback with the mocked rerun module."""
    from physicalai.runtime import RerunCallback

    def _factory(**kwargs: Any) -> RerunCallback:
        defaults: dict[str, Any] = {"mode": "spawn"}
        defaults.update(kwargs)
        return RerunCallback(**defaults)

    return _factory


def _lifecycle_start(session_id: str = "sess-1", fps: int = 30, multiplier: int = 1) -> LifecycleEvent:
    return LifecycleEvent(
        session_id=session_id,
        timestamp=1000.0,
        event="start",
        metadata={"fps": fps, "interpolation_multiplier": multiplier, "cameras": []},
    )


def _tick(
    step: int = 0,
    dof: int = 7,
    camera_frames: dict[str, Any] | None = None,
    *,
    source_updated: bool = True,
) -> TickEvent:
    return TickEvent(
        session_id="sess-1",
        step=step,
        timestamp=1000.0 + step * (1 / 30),
        robot_state=FakeRobotObservation(
            joint_positions=np.arange(dof, dtype=np.float64),
        ),
        camera_frames=camera_frames or {},
        action_sent=np.ones(dof, dtype=np.float64),
        loop_duration_s=0.033,
        sleep_time_s=0.0,
        stale_obs=False,
        source_updated=source_updated,
    )


def _fake_frame() -> Any:
    """A minimal stand-in for a Frame — just needs a ``.data`` array."""
    frame = MagicMock()
    frame.data = np.zeros((480, 640, 3), dtype=np.uint8)
    return frame


def _metrics(step: int = 0, *, queue_remaining: float = 5.0) -> MetricsEvent:
    return MetricsEvent(
        session_id="sess-1",
        step=step,
        timestamp=1000.0 + step * (1 / 30),
        values={"queue_remaining": queue_remaining},
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
            from physicalai.runtime import RerunCallback

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
        lifecycle_calls = [c for c in mock_rerun.log.call_args_list if "lifecycle" in str(c)]
        assert len(lifecycle_calls) > 0

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
        joint_calls = [c for c in log_calls if c.args[0] == "robot/joints"]
        assert len(joint_calls) == 1

    def test_tick_logs_action_scalars(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        mock_rerun.reset_mock()

        cb.on_tick(_tick(step=1, dof=4))

        log_calls = mock_rerun.log.call_args_list
        action_calls = [c for c in log_calls if c.args[0] == "robot/actions"]
        assert len(action_calls) == 1

    def test_tick_logs_runtime_metrics(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        mock_rerun.reset_mock()

        cb.on_tick(_tick(step=1))

        log_calls = mock_rerun.log.call_args_list
        paths = [c.args[0] for c in log_calls]
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

        mock_rerun.set_time.assert_any_call("step", sequence=5)
        mock_rerun.set_time.assert_any_call("wall", timestamp=event.timestamp)

    def test_none_action_sent_skipped(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        mock_rerun.reset_mock()

        event = TickEvent(
            session_id="sess-1",
            step=1,
            timestamp=1000.0,
            robot_state=FakeRobotObservation(
                joint_positions=np.arange(7, dtype=np.float64),
            ),
            camera_frames={},
            action_sent=None,
            loop_duration_s=0.033,
            sleep_time_s=0.0,
            stale_obs=False,
        )
        cb.on_tick(event)

        log_calls = mock_rerun.log.call_args_list
        action_calls = [c for c in log_calls if c.args[0] == "robot/actions"]
        assert len(action_calls) == 0


class TestRerunCallbackMetrics:
    """Queue depth moved from on_tick to the optional on_metrics hook (MetricsEvent)."""

    def test_on_metrics_logs_queue_remaining(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        mock_rerun.reset_mock()

        cb.on_metrics(_metrics(step=3, queue_remaining=7.0))

        log_calls = mock_rerun.log.call_args_list
        queue_calls = [c for c in log_calls if c.args[0] == "queue/remaining"]
        assert len(queue_calls) == 1

    def test_on_metrics_ignores_unknown_keys(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        mock_rerun.reset_mock()

        cb.on_metrics(MetricsEvent(session_id="sess-1", step=0, timestamp=1000.0, values={"other": 1.0}))

        log_calls = mock_rerun.log.call_args_list
        queue_calls = [c for c in log_calls if c.args[0] == "queue/remaining"]
        assert len(queue_calls) == 0


@pytest.mark.usefixtures("_patch_rerun")
class TestRerunCallbackInference:
    def test_send_columns_called_with_prediction_steps(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start(fps=30))

        cb.on_tick(_tick(step=10))
        mock_rerun.reset_mock()

        cb.on_inference(_inference(horizon=5, dof=3))

        # Predictions are logged via send_columns, not individual log calls.
        mock_rerun.send_columns.assert_called_once()
        call_args = mock_rerun.send_columns.call_args
        assert call_args.args[0] == "robot/predicted"

    @pytest.mark.parametrize(("multiplier", "expected"), [(1, [11, 12, 13]), (3, [13, 16, 19])])
    def test_prediction_steps_strided_by_multiplier(
        self, make_callback: Any, mock_rerun: MagicMock, multiplier: int, expected: list[int]
    ) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start(fps=30, multiplier=multiplier))
        cb.on_tick(_tick(step=10))

        cb.on_inference(_inference(horizon=3, dof=2))

        step_col = mock_rerun.send_columns.call_args.kwargs["indexes"][0]
        assert step_col[1] == ("step",)
        np.testing.assert_array_equal(step_col[2]["sequence"], expected)

    def test_inference_logs_queue_spike(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        cb.on_tick(_tick(step=0))
        mock_rerun.reset_mock()

        cb.on_inference(_inference(horizon=5, dof=3))

        log_calls = mock_rerun.log.call_args_list
        queue_calls = [c for c in log_calls if c.args[0] == "queue/inference"]
        assert len(queue_calls) == 1
        mock_rerun.log.assert_any_call("inference/latency_ms", ("Scalars", (50.0,), {}))

    def test_inference_resets_time_to_current_step(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start(fps=10))
        cb.on_tick(_tick(step=5))
        mock_rerun.reset_mock()

        cb.on_inference(_inference(horizon=3, dof=1))

        # After send_columns, time is reset to current step for queue/inference log.
        set_time_calls = mock_rerun.set_time.call_args_list
        step_values = [c.kwargs["sequence"] for c in set_time_calls if c.args[0] == "step" and "sequence" in c.kwargs]
        assert 5 in step_values

    def test_stale_predictions_cleared(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start(fps=30))
        cb.on_tick(_tick(step=5))
        cb.on_inference(_inference(horizon=5, dof=3))
        mock_rerun.reset_mock()

        # Second inference should clear old predictions first.
        cb.on_tick(_tick(step=10))
        cb.on_inference(_inference(horizon=5, dof=3))

        log_calls = mock_rerun.log.call_args_list
        clear_calls = [c for c in log_calls if c.args[0] == "robot/predicted"]
        assert len(clear_calls) >= 1  # At least the Clear call

    def test_pred_horizon_tracked(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start(fps=30))
        cb.on_tick(_tick(step=0))
        cb.on_inference(_inference(horizon=50, dof=7))
        assert cb._pred_horizon == 50


class TestRerunCallbackSystemStats:
    @pytest.fixture()
    def clock(self) -> Any:
        with patch("physicalai.runtime.callbacks.rerun.time.monotonic", return_value=10.0) as clock:
            yield clock

    def test_primes_cpu_counters_without_logging(
        self, make_callback: Any, mock_rerun: MagicMock, mock_psutil: MagicMock, clock: MagicMock
    ) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        cb.on_tick(_tick())

        mock_psutil.cpu_percent.assert_called_once_with(interval=None)
        mock_psutil.Process.return_value.cpu_percent.assert_called_once_with(interval=None)
        assert not any(c.args[0] == "runtime/monitoring" for c in mock_rerun.log.call_args_list)

    def test_logs_system_table_at_most_once_per_second(
        self, make_callback: Any, mock_rerun: MagicMock, mock_psutil: MagicMock, clock: MagicMock
    ) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        mock_rerun.reset_mock()
        cb.on_tick(_tick())
        clock.return_value = 10.9
        cb.on_tick(_tick(step=1))
        mock_psutil.virtual_memory.assert_not_called()
        assert not any(c.args[0].startswith("runtime/utilization/cpu/") for c in mock_rerun.log.call_args_list)

        clock.return_value = 11.0
        cb.on_tick(_tick(step=2))
        clock.return_value = 11.5
        cb.on_tick(_tick(step=3))
        mock_psutil.virtual_memory.assert_called_once_with()
        stats_calls = [c for c in mock_rerun.log.call_args_list if c.args[0] == "runtime/monitoring"]
        assert len(stats_calls) == 1
        mock_rerun.log.assert_any_call("runtime/utilization/cpu/host", ("Scalars", (42.5,), {}))
        mock_rerun.log.assert_any_call("runtime/utilization/cpu/process", ("Scalars", (15.625,), {}))
        assert len([c for c in mock_rerun.log.call_args_list if c.args[0] == "runtime/utilization/cpu/host"]) == 1
        md = mock_rerun.TextDocument.call_args.args[0]
        assert "42.5%" in md
        assert "62.5% (10.00 / 16.00 GiB)" in md
        assert "| **Process CPU** | 15.6% |" in md
        assert "one core" not in md
        assert "256.0 MiB" in md
        assert "Waiting for inference" in md
        assert mock_rerun.TextDocument.call_args.kwargs["media_type"] == mock_rerun.MediaType.MARKDOWN

        clock.return_value = 12.0
        cb.on_tick(_tick(step=4))
        assert mock_psutil.virtual_memory.call_count == 2

    @pytest.mark.parametrize(
        ("cores", "raw_cpu", "expected"),
        [(8, 800.0, "100.0%"), (8, 100.0, "12.5%"), (1, 100.0, "100.0%"), (None, 100.0, "100.0%")],
    )
    def test_process_cpu_is_normalized_across_all_cores(
        self,
        make_callback: Any,
        mock_rerun: MagicMock,
        mock_psutil: MagicMock,
        clock: MagicMock,
        cores: int | None,
        raw_cpu: float,
        expected: str,
    ) -> None:
        mock_psutil.cpu_count.return_value = cores
        mock_psutil.Process.return_value.cpu_percent.return_value = raw_cpu
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        cb.on_tick(_tick())
        clock.return_value += 1.0
        cb.on_tick(_tick(step=1))

        md = mock_rerun.TextDocument.call_args.args[0]
        assert f"| **Process CPU** | {expected} |" in md
        assert "| **Host CPU** | 42.5% |" in md

    def test_session_start_reprimes_cpu_counters(
        self, make_callback: Any, mock_psutil: MagicMock, clock: MagicMock
    ) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        cb.on_tick(_tick())
        cb.on_lifecycle(_lifecycle_start("sess-2"))
        cb.on_tick(_tick())

        assert mock_psutil.Process.call_count == 2
        assert mock_psutil.cpu_percent.call_count == 2
        mock_psutil.virtual_memory.assert_not_called()

    @pytest.mark.parametrize("error", [OSError("unavailable"), RuntimeError("access denied")])
    @pytest.mark.parametrize("during_priming", [True, False])
    def test_counter_failure_does_not_break_ticks(
        self,
        make_callback: Any,
        mock_psutil: MagicMock,
        clock: MagicMock,
        error: Exception,
        during_priming: bool,
    ) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        if during_priming:
            mock_psutil.Process.side_effect = error
        else:
            cb.on_tick(_tick())
            clock.return_value = 11.0
            mock_psutil.virtual_memory.side_effect = error
        cb.on_tick(_tick(step=1))
        assert cb._process is None

        mock_psutil.Process.side_effect = None
        mock_psutil.virtual_memory.side_effect = None
        clock.return_value += 1.0
        cb.on_tick(_tick(step=2))
        assert cb._process is not None
        clock.return_value += 1.0
        cb.on_tick(_tick(step=3))
        assert cb._last_step == 3

    def test_monitoring_preserves_both_sections_on_updates(
        self, make_callback: Any, mock_rerun: MagicMock, mock_psutil: MagicMock, clock: MagicMock
    ) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        cb.on_tick(_tick())
        cb.on_inference(_inference())
        md = mock_rerun.TextDocument.call_args.args[0]
        assert "Waiting for system sample" in md
        assert "| **Last (round-trip)** | 50.0 ms |" in md

        clock.return_value += 1.0
        cb.on_tick(_tick(step=1))
        md = mock_rerun.TextDocument.call_args.args[0]
        assert "Waiting for system sample" not in md
        assert "Waiting for inference" not in md
        assert md.startswith("### Inference Latency\n\n| Metric | Value |")
        assert "| **Host CPU** | 42.5% |" in md
        assert "\n\n### System\n\n| Metric | Value |" in md
        assert md.index("### Inference Latency") < md.index("### System") < md.index("### GPU")
        assert "| **Last (round-trip)** | 50.0 ms |" in md
        assert "| Samples | 1 |" in md
        assert md.count("| Metric | Value |") == 3
        assert "\n\n### GPU\n\n| Metric | Value |" in md

        mock_psutil.virtual_memory.reset_mock()
        cb.on_inference(_inference())
        md = mock_rerun.TextDocument.call_args.args[0]
        assert "| **Host CPU** | 42.5% |" in md
        assert "| Samples | 2 |" in md
        mock_psutil.virtual_memory.assert_not_called()
        paths = [c.args[0] for c in mock_rerun.log.call_args_list]
        assert "runtime/monitoring" in paths
        assert "system/stats" not in paths
        assert "inference/stats" not in paths

    def test_monitoring_clears_both_sections_on_session_start(
        self, make_callback: Any, mock_rerun: MagicMock, clock: MagicMock
    ) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        cb.on_tick(_tick())
        cb.on_inference(_inference())
        clock.return_value += 1.0
        cb.on_tick(_tick(step=1))

        cb.on_lifecycle(_lifecycle_start("sess-2"))
        cb.on_inference(_inference())
        md = mock_rerun.TextDocument.call_args.args[0]
        assert "Waiting for system sample" in md
        assert "Host CPU" not in md
        assert "| Samples | 1 |" in md

        cb.on_lifecycle(_lifecycle_start("sess-3"))
        cb.on_tick(_tick())
        clock.return_value += 1.0
        cb.on_tick(_tick(step=1))
        md = mock_rerun.TextDocument.call_args.args[0]
        assert "| **Host CPU** | 42.5% |" in md
        assert "Waiting for inference" in md
        assert "| **Last (round-trip)** |" not in md

    def test_gpu_sampling_runs_in_background_without_queueing_queries(
        self, make_callback: Any, mock_rerun: MagicMock, mock_gpu_executor: MagicMock, clock: MagicMock
    ) -> None:
        from physicalai.runtime.callbacks.gpu_stats import collect_gpu_stats

        future: Future[GpuStatsSample] = Future()
        mock_gpu_executor.submit.return_value = future
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        cb.on_tick(_tick())
        mock_gpu_executor.submit.assert_called_once_with(collect_gpu_stats)

        clock.return_value += 1.0
        cb.on_tick(_tick(step=1))
        mock_gpu_executor.submit.assert_called_once()
        assert "Waiting for GPU sample" in mock_rerun.TextDocument.call_args.args[0]

        future.set_result(GpuStatsSample("| **NVIDIA GPU 0** | |\n| Utilization | 65% |", {"nvidia_0": 65.0}))
        clock.return_value += 1.0
        cb.on_tick(_tick(step=2))
        assert mock_gpu_executor.submit.call_count == 2
        assert "NVIDIA GPU 0" in mock_rerun.TextDocument.call_args.args[0]
        assert "65%" in mock_rerun.TextDocument.call_args.args[0]
        mock_rerun.log.assert_any_call("runtime/utilization/gpu/nvidia_0", ("Scalars", (65.0,), {}))

    def test_gpu_stats_still_update_when_system_counters_fail(
        self,
        make_callback: Any,
        mock_rerun: MagicMock,
        mock_psutil: MagicMock,
        mock_gpu_executor: MagicMock,
        clock: MagicMock,
    ) -> None:
        future: Future[GpuStatsSample] = Future()
        mock_gpu_executor.submit.return_value = future
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        cb.on_tick(_tick())
        future.set_result(GpuStatsSample("| **Intel GPU 0** | |\n| Utilization | 30% |", {"intel_0": 30.0}))
        mock_psutil.virtual_memory.side_effect = OSError("unavailable")
        clock.return_value += 1.0
        cb.on_tick(_tick(step=1))
        assert "Intel GPU 0" in mock_rerun.TextDocument.call_args.args[0]
        mock_rerun.log.assert_any_call("runtime/utilization/gpu/intel_0", ("Scalars", (30.0,), {}))

    def test_session_start_discards_previous_gpu_query(
        self, make_callback: Any, mock_rerun: MagicMock, mock_gpu_executor: MagicMock, clock: MagicMock
    ) -> None:
        old: Future[GpuStatsSample] = Future()
        old.set_running_or_notify_cancel()
        new: Future[GpuStatsSample] = Future()
        mock_gpu_executor.submit.side_effect = [old, new]
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        cb.on_tick(_tick())
        cb.on_lifecycle(_lifecycle_start("sess-2"))
        cb.on_tick(_tick())
        old.set_result(GpuStatsSample("| Old session | Stale GPU sample |", {"nvidia_0": 99.0}))
        clock.return_value += 1.0
        cb.on_tick(_tick(step=1))
        md = mock_rerun.TextDocument.call_args.args[0]
        assert "Stale GPU sample" not in md
        assert "Waiting for GPU sample" in md
        assert not any(c.args[0].startswith("runtime/utilization/gpu/") for c in mock_rerun.log.call_args_list)

    def test_unavailable_gpu_metrics_do_not_create_zero_history_samples(
        self, make_callback: Any, mock_rerun: MagicMock, mock_gpu_executor: MagicMock, clock: MagicMock
    ) -> None:
        future: Future[GpuStatsSample] = Future()
        mock_gpu_executor.submit.return_value = future
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        cb.on_tick(_tick())
        future.set_result(GpuStatsSample("| NVIDIA | Unavailable |"))
        clock.return_value += 1.0
        cb.on_tick(_tick(step=1))
        assert "Unavailable" in mock_rerun.TextDocument.call_args.args[0]
        assert not any(c.args[0].startswith("runtime/utilization/gpu/") for c in mock_rerun.log.call_args_list)

    def test_close_releases_gpu_worker_without_waiting(
        self, make_callback: Any, mock_gpu_executor: MagicMock, clock: MagicMock
    ) -> None:
        cb = make_callback()
        cb.close()
        mock_gpu_executor.shutdown.assert_not_called()
        cb.on_lifecycle(_lifecycle_start())
        cb.on_tick(_tick())
        cb.close()
        cb.close()
        mock_gpu_executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
        assert cb._gpu_executor is None
        assert cb._gpu_future is None

    @pytest.mark.parametrize("with_camera", [False, True])
    def test_blueprint_includes_one_monitoring_view(
        self, make_callback: Any, mock_rerun: MagicMock, with_camera: bool
    ) -> None:
        blueprint = MagicMock()
        mock_rerun.blueprint = blueprint
        with patch.dict(sys.modules, {"rerun.blueprint": blueprint}):
            cb = make_callback()
            cb.on_lifecycle(_lifecycle_start())
            cb.on_tick(_tick(camera_frames={"top": _fake_frame()} if with_camera else {}))

        assert blueprint.TextDocumentView.call_args_list
        assert all(
            c.kwargs == {"origin": "/runtime/monitoring", "name": "Monitoring"}
            for c in blueprint.TextDocumentView.call_args_list
        )
        blueprint.Horizontal.assert_called_with(
            blueprint.Vertical.return_value, blueprint.Tabs.return_value, column_shares=[3, 1]
        )
        sidebar = [c for c in blueprint.Tabs.call_args_list if c.kwargs.get("active_tab") == "Monitoring"]
        assert sidebar
        assert len(sidebar[-1].args) == 3
        blueprint.TimeSeriesView.assert_any_call(origin="/runtime/utilization/cpu", name="CPU %")
        blueprint.TimeSeriesView.assert_any_call(origin="/runtime/utilization/gpu", name="GPU %")
        assert blueprint.Grid.called is with_camera
        mock_rerun.send_blueprint.assert_called()


@pytest.mark.usefixtures("_patch_rerun")
class TestRerunCallbackImageDecimation:
    def test_decimation_skips_non_nth_ticks(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback(image_decimation=3)
        cb.on_lifecycle(_lifecycle_start())

        image_logged_at: list[int] = []
        for step in range(6):
            mock_rerun.reset_mock()
            cb.on_tick(_tick(step=step, camera_frames={"top": _fake_frame()}))
            log_calls = mock_rerun.log.call_args_list
            if any("camera/top" in str(c.args[0]) for c in log_calls):
                image_logged_at.append(step)

        assert image_logged_at == [0, 3]

    def test_decimation_counts_source_updated_ticks_with_interpolation(
        self, make_callback: Any, mock_rerun: MagicMock
    ) -> None:
        # Multiplier 3: the first action completes immediately, so source ticks land at 0, 1, 4, 7, ...
        source_updated = [True, True, False, False, True, False, False, True, False, False]
        source_updated += [True, False, False, True, False, False, True]
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())

        image_logged_at: list[int] = []
        for step, updated in enumerate(source_updated):
            mock_rerun.reset_mock()
            frames = {"top": _fake_frame()} if updated else None
            cb.on_tick(_tick(step=step, camera_frames=frames, source_updated=updated))
            if any("camera/top" in str(c.args[0]) for c in mock_rerun.log.call_args_list):
                image_logged_at.append(step)

        assert image_logged_at == [0, 7, 16]

    def test_decimation_counter_resets_on_new_session(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())
        cb.on_tick(_tick(step=0, camera_frames={"top": _fake_frame()}))

        cb.on_lifecycle(_lifecycle_start(session_id="sess-2"))
        mock_rerun.reset_mock()
        cb.on_tick(_tick(step=0, camera_frames={"top": _fake_frame()}))

        assert any("camera/top" in str(c.args[0]) for c in mock_rerun.log.call_args_list)

    def test_decimation_default_is_3(self, make_callback: Any) -> None:
        cb = make_callback()
        assert cb._image_decimation == 3


@pytest.mark.usefixtures("_patch_rerun")
class TestRerunCallbackCameraFrames:
    """RerunCallback takes no Camera reference — images come solely from TickEvent."""

    def test_discovers_camera_names_from_first_tick(self, make_callback: Any) -> None:
        cb = make_callback()
        cb.on_lifecycle(_lifecycle_start())

        assert cb._camera_names is None
        cb.on_tick(_tick(step=0, camera_frames={"top": _fake_frame()}))
        assert cb._camera_names == ["top"]

    def test_logs_frames_directly_from_tick_event(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback(image_decimation=1)
        cb.on_lifecycle(_lifecycle_start())

        mock_rerun.reset_mock()
        cb.on_tick(_tick(step=0, camera_frames={"top": _fake_frame()}))

        log_calls = mock_rerun.log.call_args_list
        image_calls = [c for c in log_calls if "camera/top" in str(c.args[0])]
        assert len(image_calls) == 1

    def test_no_camera_names_when_log_images_false(self, make_callback: Any, mock_rerun: MagicMock) -> None:
        cb = make_callback(log_images=False)
        cb.on_lifecycle(_lifecycle_start())

        mock_rerun.reset_mock()
        cb.on_tick(_tick(step=0, camera_frames={"top": _fake_frame()}))

        assert cb._camera_names == []
        image_calls = [c for c in mock_rerun.log.call_args_list if "camera/top" in str(c.args[0])]
        assert len(image_calls) == 0


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
