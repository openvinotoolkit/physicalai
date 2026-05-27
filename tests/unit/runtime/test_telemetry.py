# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from physicalai.runtime._callback_bus import _CallbackBus
from physicalai.runtime._telemetry import TelemetryEmitter, _decode_numpy, _encode_numpy
from physicalai.runtime.callbacks import AsyncCallback, ConsoleCallback, JsonlCallback
from physicalai.runtime.events import InferenceEvent, LifecycleEvent, TickEvent
from tests.unit.runtime.conftest import FakeRobotObservation


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

    def test_close_closes_session(self) -> None:
        e, session = self._make_emitter()
        e.close()
        session.close.assert_called_once()
        assert not e.enabled


class TestCallbackBus:
    def _make_tick_event(self, step: int = 0) -> TickEvent:
        return TickEvent(
            session_id="test",
            step=step,
            timestamp=0.0,
            robot_observation=FakeRobotObservation(joint_positions=np.zeros(3)),
            camera_frames={},
            action_sent=np.zeros(3),
            queue_remaining=5,
            loop_duration_s=0.03,
            sleep_time_s=0.003,
            stale_obs=False,
        )

    def _make_inference_event(self) -> InferenceEvent:
        return InferenceEvent(
            session_id="test",
            timestamp=0.0,
            latency_s=0.1,
            offset=3,
            chunk=np.zeros((10, 3)),
        )

    def _make_lifecycle_event(self, event: str = "start") -> LifecycleEvent:
        return LifecycleEvent(
            session_id="test",
            timestamp=0.0,
            event=event,
            metadata={"fps": 30},
        )

    def test_emit_tick_dispatches_to_callback(self) -> None:
        cb = MagicMock()
        bus = _CallbackBus([cb])
        event = self._make_tick_event()
        bus.emit_tick(event)
        cb.on_tick.assert_called_once_with(event)

    def test_emit_lifecycle_dispatches(self) -> None:
        cb = MagicMock()
        bus = _CallbackBus([cb])
        event = self._make_lifecycle_event()
        bus.emit_lifecycle(event)
        cb.on_lifecycle.assert_called_once_with(event)

    def test_emit_inference_queues_for_drain(self) -> None:
        cb = MagicMock()
        bus = _CallbackBus([cb])
        event = self._make_inference_event()
        bus.emit_inference(event)
        cb.on_inference.assert_not_called()
        bus.emit_tick(self._make_tick_event())
        cb.on_inference.assert_called_once_with(event)

    def test_invoke_before_send_action_chains(self) -> None:
        cb1 = MagicMock()
        cb1.before_send_action.return_value = np.ones(3)
        cb2 = MagicMock()
        cb2.before_send_action.return_value = None

        bus = _CallbackBus([cb1, cb2])
        original = np.zeros(3)
        result = bus.invoke_before_send_action(action=original, step=0)

        np.testing.assert_array_equal(result, np.ones(3))
        cb2.before_send_action.assert_called_once()
        passed = cb2.before_send_action.call_args[1]["action"]
        np.testing.assert_array_equal(passed, np.ones(3))

    def test_invoke_on_hold_dispatches(self) -> None:
        cb = MagicMock()
        bus = _CallbackBus([cb])
        bus.invoke_on_hold(step=5, holds=3)
        cb.on_hold.assert_called_once_with(step=5, holds=3)

    def test_callback_exception_isolated(self) -> None:
        bad_cb = MagicMock()
        bad_cb.on_tick.side_effect = RuntimeError("oops")
        good_cb = MagicMock()
        bus = _CallbackBus([bad_cb, good_cb])
        bus.emit_tick(self._make_tick_event())
        good_cb.on_tick.assert_called_once()

    def test_close_calls_close_on_callbacks(self) -> None:
        cb = MagicMock()
        bus = _CallbackBus([cb])
        bus.close()
        cb.close.assert_called_once()

    def test_missing_methods_skipped(self) -> None:
        class MinimalCallback:
            pass

        bus = _CallbackBus([MinimalCallback()])
        bus.emit_tick(self._make_tick_event())
        bus.emit_lifecycle(self._make_lifecycle_event())
        bus.invoke_on_hold(step=0, holds=1)


class TestConsoleCallback:
    def test_throttles_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        cb = ConsoleCallback(throttle_steps=5)
        for i in range(10):
            cb.on_tick(
                TickEvent(
                    session_id="t",
                    step=i,
                    timestamp=0.0,
                    robot_observation=FakeRobotObservation(joint_positions=np.zeros(3)),
                    camera_frames={},
                    action_sent=np.zeros(3),
                    queue_remaining=5,
                    loop_duration_s=0.03,
                    sleep_time_s=0.003,
                    stale_obs=False,
                )
            )
        captured = capsys.readouterr()
        lines = [l for l in captured.out.strip().split("\n") if l]
        assert len(lines) == 2


class TestJsonlCallback:
    def test_writes_events(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        cb = JsonlCallback(path)

        cb.on_tick(
            TickEvent(
                session_id="s1",
                step=0,
                timestamp=1.0,
                robot_observation=FakeRobotObservation(
                    joint_positions=np.array([0.1, 0.2]),
                ),
                camera_frames={},
                action_sent=np.array([0.3, 0.4]),
                queue_remaining=5,
                loop_duration_s=0.03,
                sleep_time_s=0.003,
                stale_obs=False,
            )
        )
        cb.on_lifecycle(
            LifecycleEvent(
                session_id="s1",
                timestamp=1.0,
                event="start",
                metadata={"fps": 30},
            )
        )
        cb.close()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        tick_record = json.loads(lines[0])
        assert tick_record["type"] == "tick"
        assert tick_record["step"] == 0
        lifecycle_record = json.loads(lines[1])
        assert lifecycle_record["type"] == "lifecycle"
        assert lifecycle_record["event"] == "start"


class TestAsyncCallback:
    def test_dispatches_events_asynchronously(self) -> None:
        inner = MagicMock(spec=["on_tick", "on_inference", "on_lifecycle", "close"])
        called = threading.Event()
        inner.on_lifecycle.side_effect = lambda e: called.set()
        cb = AsyncCallback(inner, max_queue=64)
        event = LifecycleEvent(session_id="t", timestamp=0.0, event="start", metadata={})
        cb.on_lifecycle(event)
        assert called.wait(timeout=2.0), "on_lifecycle not called within timeout"
        inner.on_lifecycle.assert_called_once_with(event)
        cb.close()

    def test_close_joins_thread(self) -> None:
        inner = MagicMock(spec=["on_tick", "on_inference", "on_lifecycle", "close"])
        cb = AsyncCallback(inner)
        cb.close()
        assert not cb._thread.is_alive()
        inner.close.assert_called_once()
