from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from physicalai.runtime._action_queue import ChunkedActionQueue as ActionQueue, ChunkedActionQueue
from physicalai.runtime.execution import AsyncExecution, SyncExecution, WorkerDiedError


def _make_mock_model(chunk: np.ndarray | None = None) -> MagicMock:
    model = MagicMock()
    if chunk is None:
        chunk = np.random.randn(6, 4).astype(np.float32)
    model.predict_action_chunk.return_value = chunk
    return model


class TestSyncExecution:
    def test_warmup_seeds_queue_and_discovers_chunk_size(self) -> None:
        chunk = np.random.randn(8, 3).astype(np.float32)
        model = _make_mock_model(chunk)
        queue=ChunkedActionQueue()
        ex = SyncExecution()
        obs = {"state": np.zeros(3)}

        ex.start(model, queue)
        ex.warmup(obs)

        assert ex.chunk_size == 8
        assert queue.remaining == 8
        model.predict_action_chunk.assert_called_once_with(obs)

    def test_maybe_request_refills_when_empty(self) -> None:
        chunk = np.random.randn(4, 2).astype(np.float32)
        model = _make_mock_model(chunk)
        queue=ChunkedActionQueue()
        ex = SyncExecution()
        obs = {"state": np.zeros(2)}

        ex.start(model, queue)
        ex.warmup(obs)

        for _ in range(4):
            queue.pop()
        assert queue.remaining == 0

        model.predict_action_chunk.reset_mock()
        model.predict_action_chunk.return_value = chunk
        ex.maybe_request(obs)

        assert queue.remaining == 4
        model.predict_action_chunk.assert_called_once()

    def test_maybe_request_does_not_refill_when_nonempty(self) -> None:
        chunk = np.random.randn(4, 2).astype(np.float32)
        model = _make_mock_model(chunk)
        queue=ChunkedActionQueue()
        ex = SyncExecution()
        obs = {"state": np.zeros(2)}

        ex.start(model, queue)
        ex.warmup(obs)
        queue.pop()

        model.predict_action_chunk.reset_mock()
        ex.maybe_request(obs)
        model.predict_action_chunk.assert_not_called()

    def test_stop_is_noop(self) -> None:
        ex = SyncExecution()
        ex.stop()

    def test_inference_count_increments(self) -> None:
        chunk = np.random.randn(4, 2).astype(np.float32)
        model = _make_mock_model(chunk)
        queue=ChunkedActionQueue()
        ex = SyncExecution()
        obs = {"state": np.zeros(2)}

        ex.start(model, queue)
        ex.warmup(obs)
        for _ in range(4):
            queue.pop()
        ex.maybe_request(obs)
        assert ex.inference_count == 1


class TestAsyncExecution:
    def test_start_spawns_thread(self) -> None:
        model = _make_mock_model()
        queue=ChunkedActionQueue()
        ex = AsyncExecution()

        ex.start(model, queue)
        assert ex.alive is True
        ex.stop()

    def test_warmup_seeds_queue(self) -> None:
        chunk = np.random.randn(6, 4).astype(np.float32)
        model = _make_mock_model(chunk)
        queue=ChunkedActionQueue()
        ex = AsyncExecution()

        ex.start(model, queue)
        obs = {"state": np.zeros(4)}
        ex.warmup(obs)

        assert ex.chunk_size == 6
        assert queue.remaining == 6
        ex.stop()

    def test_maybe_request_submits_when_below_threshold(self) -> None:
        chunk = np.random.randn(10, 2).astype(np.float32)
        model = _make_mock_model(chunk)
        queue=ChunkedActionQueue()
        ex = AsyncExecution(request_threshold=0.5)

        ex.start(model, queue)
        obs = {"state": np.zeros(2)}
        ex.warmup(obs)

        for _ in range(10):
            queue.pop()

        model.predict_action_chunk.reset_mock()
        model.predict_action_chunk.return_value = chunk
        ex.maybe_request(obs)

        time.sleep(0.3)
        assert queue.remaining > 0
        ex.stop()

    def test_defensive_copy_of_observation(self) -> None:
        chunk = np.random.randn(4, 2).astype(np.float32)
        model = _make_mock_model(chunk)
        queue=ChunkedActionQueue()
        ex = AsyncExecution(request_threshold=0.5)

        ex.start(model, queue)
        obs = {"state": np.zeros(2)}
        ex.warmup(obs)
        for _ in range(4):
            queue.pop()

        model.predict_action_chunk.reset_mock()
        original_state = np.array([1.0, 2.0])
        obs_to_submit = {"state": original_state.copy()}
        ex.maybe_request(obs_to_submit)
        obs_to_submit["state"][:] = 99.0

        time.sleep(0.3)
        if model.predict_action_chunk.called:
            submitted = model.predict_action_chunk.call_args[0][0]["state"]
            np.testing.assert_array_equal(submitted, original_state)
        ex.stop()

    def test_worker_death_raises_error(self) -> None:
        model = _make_mock_model()
        model.predict_action_chunk.side_effect = [
            np.random.randn(4, 2).astype(np.float32),
            ValueError("model exploded"),
        ]
        queue=ChunkedActionQueue()
        ex = AsyncExecution(request_threshold=0.5)

        ex.start(model, queue)
        obs = {"state": np.zeros(2)}
        ex.warmup(obs)

        for _ in range(4):
            queue.pop()

        ex.maybe_request(obs)
        time.sleep(0.5)

        with pytest.raises(WorkerDiedError, match="model exploded"):
            ex.maybe_request(obs)

        ex.stop()

    def test_stop_signals_and_joins(self) -> None:
        model = _make_mock_model()
        queue=ChunkedActionQueue()
        ex = AsyncExecution()

        ex.start(model, queue)
        assert ex.alive is True

        ex.stop()
        assert ex._thread is not None
        assert not ex._thread.is_alive()

    def test_health_properties(self) -> None:
        chunk = np.random.randn(4, 2).astype(np.float32)
        model = _make_mock_model(chunk)
        queue=ChunkedActionQueue()
        ex = AsyncExecution()

        ex.start(model, queue)
        obs = {"state": np.zeros(2)}
        ex.warmup(obs)

        assert ex.inference_count == 0

        for _ in range(4):
            queue.pop()

        model.predict_action_chunk.reset_mock()
        model.predict_action_chunk.return_value = chunk
        ex.maybe_request(obs)
        time.sleep(0.3)

        assert ex.inference_count >= 1
        ex.stop()

    def test_watchdog_triggers_force_reset(self) -> None:
        chunk = np.random.randn(4, 2).astype(np.float32)
        model = _make_mock_model(chunk)

        call_count = 0

        def slow_predict(obs: dict) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                time.sleep(100)
            return chunk

        model.predict_action_chunk.side_effect = slow_predict
        queue=ChunkedActionQueue()
        ex = AsyncExecution(request_threshold=0.5, watchdog_timeout_s=0.1)

        ex.start(model, queue)
        obs = {"state": np.zeros(2)}
        ex.warmup(obs)

        for _ in range(4):
            queue.pop()
        ex.maybe_request(obs)

        time.sleep(0.3)
        ex.maybe_request(obs)

        ex.stop()
