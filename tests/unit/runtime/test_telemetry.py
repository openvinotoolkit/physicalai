# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from physicalai.runtime._telemetry import TelemetryEmitter, _decode_numpy, _encode_numpy


class TestNumpyEncoding:
    def test_encode_numpy_float32(self) -> None:
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        encoded = _encode_numpy(arr)
        assert encoded["__np__"] is True
        assert encoded["dtype"] == "float32"
        assert encoded["shape"] == [2, 2]
        assert isinstance(encoded["data"], bytes)

    def test_encode_preserves_shape(self) -> None:
        arr = np.zeros((3, 4, 5), dtype=np.float64)
        encoded = _encode_numpy(arr)
        assert encoded["shape"] == [3, 4, 5]
        assert encoded["dtype"] == "float64"

    def test_roundtrip(self) -> None:
        arr = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        decoded = _decode_numpy(_encode_numpy(arr))
        np.testing.assert_array_equal(arr, decoded)


class TestTelemetryEmitterNoOp:
    def test_emitter_noop_without_zenoh(self) -> None:
        with patch.dict("sys.modules", {"zenoh": None, "msgpack": None}):
            e = TelemetryEmitter.__new__(TelemetryEmitter)
            e._session_id = "test"
            e._session = None
            e._msgpack = None
            e._enabled = False

        assert not e.enabled

    def test_noop_emit_methods(self) -> None:
        e = TelemetryEmitter.__new__(TelemetryEmitter)
        e._session_id = "test"
        e._session = None
        e._msgpack = None
        e._enabled = False

        e.emit_lifecycle("test_event", foo="bar")
        e.emit_tick(
            step=0,
            timestamp=0.0,
            joint_positions=None,
            action_sent=None,
            queue_remaining=0,
            loop_duration_s=0.033,
            sleep_time_s=0.001,
        )
        e.emit_inference(latency_s=0.1, offset=3, chunk=np.zeros((5, 3)))
        e.close()

    def test_session_id_auto_generated(self) -> None:
        e = TelemetryEmitter.__new__(TelemetryEmitter)
        e._session_id = None  # type: ignore[assignment]
        e._session = None
        e._msgpack = None
        e._enabled = False

        assert e._session_id is None


class TestTelemetryEmitterWithMock:
    def _make_emitter(self) -> tuple[TelemetryEmitter, MagicMock]:
        mock_session = MagicMock()
        mock_msgpack = MagicMock()
        mock_msgpack.packb.return_value = b"\x80"

        e = TelemetryEmitter.__new__(TelemetryEmitter)
        e._session_id = "abc123"
        e._session = mock_session
        e._msgpack = mock_msgpack
        e._enabled = True
        return e, mock_session

    def test_emit_tick_publishes(self) -> None:
        e, session = self._make_emitter()
        e.emit_tick(
            step=42,
            timestamp=1.0,
            joint_positions=np.zeros(3),
            action_sent=np.ones(3),
            queue_remaining=5,
            loop_duration_s=0.033,
            sleep_time_s=0.001,
        )
        session.put.assert_called_once()
        topic = session.put.call_args[0][0]
        assert topic == "physicalai/rt/abc123/tick"

    def test_emit_lifecycle_publishes(self) -> None:
        e, session = self._make_emitter()
        e.emit_lifecycle("start", fps=30)
        session.put.assert_called_once()
        topic = session.put.call_args[0][0]
        assert topic == "physicalai/rt/abc123/lifecycle"

    def test_emit_inference_publishes(self) -> None:
        e, session = self._make_emitter()
        e.emit_inference(latency_s=0.05, offset=2, chunk=np.zeros((10, 6)))
        session.put.assert_called_once()
        topic = session.put.call_args[0][0]
        assert topic == "physicalai/rt/abc123/inference"

    def test_topic_prefix_format(self) -> None:
        e, session = self._make_emitter()
        e.emit_lifecycle("test")
        topic = session.put.call_args[0][0]
        assert topic.startswith("physicalai/rt/abc123/")

    def test_close_closes_session(self) -> None:
        e, session = self._make_emitter()
        e.close()
        session.close.assert_called_once()
        assert not e.enabled


class TestTelemetryInRuntime:
    def test_runtime_emits_lifecycle_start_and_shutdown(self) -> None:
        from physicalai.runtime.execution import SyncExecution
        from physicalai.runtime.runtime import PolicyRuntime

        robot = MagicMock()
        obs = MagicMock()
        obs.joint_positions = np.zeros(6)
        obs.timestamp = 0.0
        obs.sensor_data = None
        obs.images = None
        robot.get_observation.return_value = obs

        model = MagicMock()
        model.predict_action_chunk.return_value = np.zeros((10, 6))

        telemetry = MagicMock()
        rt = PolicyRuntime(
            robot=robot,
            model=model,
            execution=SyncExecution(),
            fps=10.0,
            telemetry=telemetry,
        )

        with patch("physicalai.runtime.runtime.time") as mock_time:
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            rt.run(duration_s=0.1)

        lifecycle_calls = telemetry.emit_lifecycle.call_args_list
        events = [c.args[0] if c.args else c.kwargs.get("event") for c in lifecycle_calls]
        assert "start" in events
        assert "shutdown" in events

    def test_runtime_emits_tick_events(self) -> None:
        from physicalai.runtime.execution import SyncExecution
        from physicalai.runtime.runtime import PolicyRuntime

        robot = MagicMock()
        obs = MagicMock()
        obs.joint_positions = np.zeros(3)
        obs.timestamp = 0.0
        obs.sensor_data = None
        obs.images = None
        robot.get_observation.return_value = obs

        model = MagicMock()
        model.predict_action_chunk.return_value = np.zeros((10, 3))

        telemetry = MagicMock()
        rt = PolicyRuntime(
            robot=robot,
            model=model,
            execution=SyncExecution(),
            fps=10.0,
            telemetry=telemetry,
        )

        with patch("physicalai.runtime.runtime.time") as mock_time:
            mock_time.perf_counter.return_value = 0.0
            mock_time.sleep = MagicMock()
            rt.run(duration_s=0.2)

        assert telemetry.emit_tick.call_count == 2
