from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from physicalai.runtime import AsyncExecution, ChunkedActionQueue as ActionQueue, ChunkedActionQueue, SyncExecution, WorkerDiedError


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

    def test_maybe_request_skips_inference_when_queue_full(self) -> None:
        chunk = np.random.randn(8, 2).astype(np.float32)
        model = _make_mock_model(chunk)
        queue = ChunkedActionQueue()
        ex = SyncExecution(request_threshold=0.5)
        obs = {"state": np.zeros(2)}

        ex.start(model, queue)
        ex.warmup(obs)

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

    def test_maybe_request_skips_inference_when_queue_full(self) -> None:
        chunk = np.random.randn(10, 2).astype(np.float32)
        model = _make_mock_model(chunk)
        queue = ChunkedActionQueue()
        ex = AsyncExecution(request_threshold=0.5)

        ex.start(model, queue)
        obs = {"state": np.zeros(2)}
        ex.warmup(obs)

        model.predict_action_chunk.reset_mock()
        ex.maybe_request(obs)

        model.predict_action_chunk.assert_not_called()
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
        inference_started = threading.Event()
        release_inference = threading.Event()

        def slow_predict(obs: dict) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                inference_started.set()
                assert release_inference.wait(timeout=1.0)
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

        assert inference_started.wait(timeout=1.0)
        with patch.object(ex, "_force_reset", wraps=ex._force_reset) as force_reset:
            time.sleep(0.3)
            ex.maybe_request(obs)
            force_reset.assert_called_once()
        release_inference.set()
        ex.stop()

    def test_reset_discards_in_flight_result_without_restarting_worker(self) -> None:
        chunk = np.ones((4, 2), dtype=np.float32)
        entered = threading.Event()
        release = threading.Event()
        calls = 0

        def predict(_obs: dict[str, Any]) -> np.ndarray:
            nonlocal calls
            calls += 1
            if calls == 2:
                entered.set()
                assert release.wait(timeout=5.0)
            return chunk

        model = _make_mock_model(chunk)
        model.predict_action_chunk.side_effect = predict
        queue = ChunkedActionQueue()
        ex = AsyncExecution(request_threshold=0.5)
        ex.start(model, queue)
        ex.warmup({"state": np.zeros(2)})
        for _ in range(4):
            queue.pop()

        worker = ex._thread  # noqa: SLF001
        ex.maybe_request({"state": np.ones(2)})
        assert entered.wait(timeout=5.0)

        reset_done = threading.Event()

        def reset() -> None:
            ex.reset()
            reset_done.set()

        reset_thread = threading.Thread(target=reset)
        reset_thread.start()
        try:
            assert not reset_done.wait(timeout=0.05)
            release.set()
            assert reset_done.wait(timeout=5.0)
            assert queue.remaining == 0
            assert ex._thread is worker  # noqa: SLF001
            assert worker is not None and worker.is_alive()
            model.reset.assert_called_once()
        finally:
            release.set()
            reset_thread.join(timeout=5.0)
            ex.stop()

    def test_reset_skips_dequeued_request_that_has_not_entered_model(self) -> None:
        model = _make_mock_model()
        queue = ChunkedActionQueue()
        ex = AsyncExecution()
        ex.start(model, queue)
        ex._threshold_count = 1  # noqa: SLF001

        ex._model_lock.acquire()  # noqa: SLF001
        reset_thread: threading.Thread | None = None
        try:
            ex.maybe_request({"state": np.zeros(4, dtype=np.float32)})
            deadline = time.monotonic() + 5.0
            while not ex._running_inference and time.monotonic() < deadline:  # noqa: SLF001
                time.sleep(0.001)
            assert ex._running_inference  # noqa: SLF001

            incarnation = ex._incarnation  # noqa: SLF001
            reset_thread = threading.Thread(target=ex.reset)
            reset_thread.start()
            while ex._incarnation == incarnation and time.monotonic() < deadline:  # noqa: SLF001
                time.sleep(0.001)
            assert ex._incarnation > incarnation  # noqa: SLF001
        finally:
            ex._model_lock.release()  # noqa: SLF001
            if reset_thread is not None:
                reset_thread.join(timeout=5.0)
            ex.stop()

        model.predict_action_chunk.assert_not_called()
        model.reset.assert_called_once()
        assert queue.remaining == 0

    def test_reset_cancels_explicit_warmup(self) -> None:
        chunk = np.ones((4, 2), dtype=np.float32)
        entered = threading.Event()
        release = threading.Event()

        def predict(_obs: dict[str, Any]) -> np.ndarray:
            entered.set()
            assert release.wait(timeout=5.0)
            return chunk

        model = _make_mock_model(chunk)
        model.predict_action_chunk.side_effect = predict
        queue = ChunkedActionQueue()
        ex = AsyncExecution()
        ex.start(model, queue)
        warmup_error: list[BaseException] = []

        def warmup() -> None:
            try:
                ex.warmup({"state": np.zeros(2, dtype=np.float32)})
            except BaseException as exc:
                warmup_error.append(exc)

        warmup_thread = threading.Thread(target=warmup)
        warmup_thread.start()
        assert entered.wait(timeout=5.0)
        reset_thread = threading.Thread(target=ex.reset, kwargs={"reset_model": False})
        reset_thread.start()
        try:
            release.set()
            warmup_thread.join(timeout=5.0)
            reset_thread.join(timeout=5.0)

            assert len(warmup_error) == 1
            assert "cancelled by reset" in str(warmup_error[0])
            assert queue.remaining == 0
        finally:
            release.set()
            warmup_thread.join(timeout=5.0)
            reset_thread.join(timeout=5.0)
            ex.stop()


class TestRTCExecutionObsSlot:
    def test_worker_releases_obs_lock_before_idle_wait(self) -> None:
        from physicalai.runtime import RTCActionQueue, RTCExecution
        from physicalai.runtime.execution import rtc

        model = _rtc_model(chunk_size=20, action_dim=3)
        queue = RTCActionQueue()
        ex = RTCExecution(chunk_size=20, max_action_dim=3, fps=30.0)
        idle_wait_observed = threading.Event()
        lock_was_available = False

        def inspect_idle_wait(_seconds: float) -> None:
            nonlocal lock_was_available
            lock_was_available = ex._obs_lock.acquire(blocking=False)  # noqa: SLF001
            if lock_was_available:
                ex._obs_lock.release()  # noqa: SLF001
            idle_wait_observed.set()
            ex._stop_event.set()  # noqa: SLF001

        with patch.object(rtc.time, "sleep", side_effect=inspect_idle_wait):
            ex.start(model, queue)
            try:
                assert idle_wait_observed.wait(timeout=5.0)
            finally:
                ex.stop()

        assert lock_was_available

    def test_worker_clears_obs_slot_after_consuming_warmup_sample(self) -> None:
        from physicalai.runtime import RTCActionQueue, RTCExecution

        chunk_size = 20
        action_dim = 3

        model = MagicMock()
        model.chunk_size = chunk_size
        model.postprocessors = []
        model.return_value = {"action": np.random.randn(1, chunk_size, action_dim).astype(np.float32)}

        queue = RTCActionQueue()
        ex = RTCExecution(chunk_size=chunk_size, max_action_dim=action_dim, fps=30.0)
        ex.start(model, queue)
        try:
            ex.warmup({"state": np.zeros(action_dim, dtype=np.float32)})

            # warmup() blocks until the worker produced the first chunk, which means
            # it has already consumed the warmup observation. The slot must be cleared
            # so a later below-threshold refill cannot reuse the stale warmup sample.
            with ex._obs_lock:  # noqa: SLF001
                assert ex._obs_slot is None  # noqa: SLF001
        finally:
            ex.stop()

    def test_reset_rearms_warmup_without_restarting_worker(self) -> None:
        from physicalai.runtime import RTCActionQueue, RTCExecution

        model = _rtc_model(chunk_size=20, action_dim=3)
        queue = RTCActionQueue()
        ex = RTCExecution(chunk_size=20, max_action_dim=3, fps=30.0)
        ex.start(model, queue)
        try:
            ex.warmup({"state": np.zeros(3, dtype=np.float32)})
            worker = ex._thread  # noqa: SLF001
            assert queue.remaining == 20

            queue.reset()
            ex.reset(reset_model=False)
            ex.warmup({"state": np.ones(3, dtype=np.float32)})

            assert queue.remaining == 20
            assert ex._thread is worker  # noqa: SLF001
            assert worker is not None and worker.is_alive()
            assert model.call_count == 2
        finally:
            ex.stop()

    def test_reset_discards_in_flight_chunk(self) -> None:
        from physicalai.runtime import RTCActionQueue, RTCExecution

        entered = threading.Event()
        release = threading.Event()
        model = _rtc_model(chunk_size=20, action_dim=3)

        def predict(_inputs: dict[str, Any]) -> dict[str, np.ndarray]:
            entered.set()
            assert release.wait(timeout=5.0)
            return {"action": np.ones((1, 20, 3), dtype=np.float32)}

        model.side_effect = predict
        queue = RTCActionQueue()
        ex = RTCExecution(chunk_size=20, max_action_dim=3, fps=30.0)
        ex.start(model, queue)
        worker = ex._thread  # noqa: SLF001
        with ex._obs_lock:  # noqa: SLF001
            observation = {"state": np.zeros(3, dtype=np.float32)}
            ex._obs_slot = (observation, ex._incarnation, None)  # noqa: SLF001
        assert entered.wait(timeout=5.0)

        reset_done = threading.Event()

        def reset() -> None:
            ex.reset()
            reset_done.set()

        reset_thread = threading.Thread(target=reset)
        reset_thread.start()
        try:
            assert not reset_done.wait(timeout=0.05)
            release.set()
            assert reset_done.wait(timeout=5.0)
            assert queue.remaining == 0
            assert ex._thread is worker  # noqa: SLF001
            assert worker is not None and worker.is_alive()
            model.reset.assert_called_once()
        finally:
            release.set()
            reset_thread.join(timeout=5.0)
            ex.stop()

    def test_reset_skips_dequeued_chunk_that_has_not_entered_model(self) -> None:
        from physicalai.runtime import RTCActionQueue, RTCExecution

        model = _rtc_model(chunk_size=20, action_dim=3)
        queue = RTCActionQueue()
        ex = RTCExecution(chunk_size=20, max_action_dim=3, fps=30.0)
        ex.start(model, queue)

        ex._model_lock.acquire()  # noqa: SLF001
        reset_thread: threading.Thread | None = None
        try:
            with ex._obs_lock:  # noqa: SLF001
                observation = {"state": np.zeros(3, dtype=np.float32)}
                ex._obs_slot = (observation, ex._incarnation, None)  # noqa: SLF001
            deadline = time.monotonic() + 5.0
            while ex._obs_slot is not None and time.monotonic() < deadline:  # noqa: SLF001
                time.sleep(0.001)
            assert ex._obs_slot is None  # noqa: SLF001

            incarnation = ex._incarnation  # noqa: SLF001
            reset_thread = threading.Thread(target=ex.reset)
            reset_thread.start()
            while ex._incarnation == incarnation and time.monotonic() < deadline:  # noqa: SLF001
                time.sleep(0.001)
            assert ex._incarnation > incarnation  # noqa: SLF001
        finally:
            ex._model_lock.release()  # noqa: SLF001
            if reset_thread is not None:
                reset_thread.join(timeout=5.0)
            ex.stop()

        model.assert_not_called()
        model.reset.assert_called_once()
        assert queue.remaining == 0

    def test_reset_cancels_warmup_waiting_for_model_lock(self) -> None:
        from physicalai.runtime import RTCActionQueue, RTCExecution

        model = _rtc_model(chunk_size=20, action_dim=3)
        queue = RTCActionQueue()
        ex = RTCExecution(chunk_size=20, max_action_dim=3, fps=30.0)
        ex.start(model, queue)
        ex._model_lock.acquire()  # noqa: SLF001
        warmup_error: list[BaseException] = []

        def warmup() -> None:
            try:
                ex.warmup({"state": np.zeros(3, dtype=np.float32)})
            except BaseException as exc:
                warmup_error.append(exc)

        warmup_thread = threading.Thread(target=warmup)
        warmup_thread.start()
        try:
            deadline = time.monotonic() + 5.0
            while ex._warmup_signal is None and time.monotonic() < deadline:  # noqa: SLF001
                time.sleep(0.001)
            assert ex._warmup_signal is not None  # noqa: SLF001
            while ex._obs_slot is not None and time.monotonic() < deadline:  # noqa: SLF001
                time.sleep(0.001)
            assert ex._obs_slot is None  # noqa: SLF001

            ex.reset(reset_model=False)
            warmup_thread.join(timeout=1.0)

            assert not warmup_thread.is_alive()
            assert len(warmup_error) == 1
            assert isinstance(warmup_error[0], RuntimeError)
            assert "cancelled by reset" in str(warmup_error[0])
            model.assert_not_called()
        finally:
            ex._model_lock.release()  # noqa: SLF001
            warmup_thread.join(timeout=5.0)
            ex.stop()

    def test_reset_cancels_warmup_during_inference(self) -> None:
        from physicalai.runtime import RTCActionQueue, RTCExecution

        entered = threading.Event()
        release = threading.Event()
        model = _rtc_model(chunk_size=20, action_dim=3)

        def predict(_inputs: dict[str, Any]) -> dict[str, np.ndarray]:
            entered.set()
            assert release.wait(timeout=5.0)
            return {"action": np.ones((1, 20, 3), dtype=np.float32)}

        model.side_effect = predict
        queue = RTCActionQueue()
        ex = RTCExecution(chunk_size=20, max_action_dim=3, fps=30.0)
        ex.start(model, queue)
        warmup_error: list[BaseException] = []

        def warmup() -> None:
            try:
                ex.warmup({"state": np.zeros(3, dtype=np.float32)})
            except BaseException as exc:
                warmup_error.append(exc)

        warmup_thread = threading.Thread(target=warmup)
        warmup_thread.start()
        assert entered.wait(timeout=5.0)

        reset_thread = threading.Thread(target=ex.reset, kwargs={"reset_model": False})
        reset_thread.start()
        try:
            warmup_thread.join(timeout=1.0)
            assert not warmup_thread.is_alive()
            assert len(warmup_error) == 1
            assert "cancelled by reset" in str(warmup_error[0])
        finally:
            release.set()
            reset_thread.join(timeout=5.0)
            warmup_thread.join(timeout=5.0)
            ex.stop()

        assert queue.remaining == 0

    def test_start_cancels_existing_warmup_before_refusing_active_worker(self) -> None:
        from physicalai.runtime import RTCActionQueue, RTCExecution
        from physicalai.runtime.execution import rtc

        entered = threading.Event()
        release = threading.Event()
        model = _rtc_model(chunk_size=20, action_dim=3)

        def predict(_inputs: dict[str, Any]) -> dict[str, np.ndarray]:
            entered.set()
            assert release.wait(timeout=5.0)
            return {"action": np.ones((1, 20, 3), dtype=np.float32)}

        model.side_effect = predict
        queue = RTCActionQueue()
        ex = RTCExecution(chunk_size=20, max_action_dim=3, fps=30.0)
        ex.start(model, queue)
        warmup_error: list[BaseException] = []

        def warmup() -> None:
            try:
                ex.warmup({"state": np.zeros(3, dtype=np.float32)})
            except BaseException as exc:
                warmup_error.append(exc)

        warmup_thread = threading.Thread(target=warmup)
        warmup_thread.start()
        assert entered.wait(timeout=5.0)

        start_error: list[BaseException] = []

        def restart() -> None:
            try:
                with patch.object(rtc, "_STRAGGLER_GRACE_S", 0.05):
                    ex.start(model, queue)
            except BaseException as exc:
                start_error.append(exc)

        start_thread = threading.Thread(target=restart)
        start_thread.start()
        try:
            warmup_thread.join(timeout=1.0)
            start_thread.join(timeout=1.0)
            assert not warmup_thread.is_alive()
            assert len(warmup_error) == 1
            assert "cancelled by restart" in str(warmup_error[0])
            assert len(start_error) == 1
            assert isinstance(start_error[0], RuntimeError)
        finally:
            release.set()
            warmup_thread.join(timeout=5.0)
            start_thread.join(timeout=5.0)
            ex.stop()

    def test_stop_cancels_warmup_before_joining_worker(self) -> None:
        from physicalai.runtime import RTCActionQueue, RTCExecution

        entered = threading.Event()
        release = threading.Event()
        model = _rtc_model(chunk_size=20, action_dim=3)

        def predict(_inputs: dict[str, Any]) -> dict[str, np.ndarray]:
            entered.set()
            assert release.wait(timeout=5.0)
            return {"action": np.ones((1, 20, 3), dtype=np.float32)}

        model.side_effect = predict
        queue = RTCActionQueue()
        ex = RTCExecution(chunk_size=20, max_action_dim=3, fps=30.0)
        ex.start(model, queue)
        warmup_error: list[BaseException] = []

        def warmup() -> None:
            try:
                ex.warmup({"state": np.zeros(3, dtype=np.float32)})
            except BaseException as exc:
                warmup_error.append(exc)

        warmup_thread = threading.Thread(target=warmup)
        warmup_thread.start()
        assert entered.wait(timeout=5.0)

        stop_thread = threading.Thread(target=ex.stop)
        stop_thread.start()
        try:
            warmup_thread.join(timeout=1.0)
            assert not warmup_thread.is_alive()
            assert len(warmup_error) == 1
            assert "cancelled because execution stopped" in str(warmup_error[0])
        finally:
            release.set()
            warmup_thread.join(timeout=5.0)
            stop_thread.join(timeout=5.0)


def _blocking_model(entered: threading.Event, release: threading.Event, rows: int = 6) -> MagicMock:
    """Model whose inference blocks until *release* is set."""

    def blocking(_obs: object) -> np.ndarray:
        entered.set()
        release.wait(timeout=30.0)
        return np.zeros((rows, 4), dtype=np.float32)

    model = MagicMock()
    model.predict_action_chunk.side_effect = blocking
    return model


def _rtc_model(chunk_size: int = 20, action_dim: int = 3) -> MagicMock:
    model = MagicMock()
    model.chunk_size = chunk_size
    model.postprocessors = []
    model.return_value = {"action": np.random.randn(1, chunk_size, action_dim).astype(np.float32)}
    return model


class TestRestartAfterStop:
    """``start()`` must hand back a usable execution after a previous ``stop()``.

    A stopped runtime can be run again, which calls ``stop()`` then ``start()``
    on the same execution. Per-run state left behind here makes the second run
    misbehave silently rather than fail.
    """

    def test_async_clears_stale_per_run_state(self) -> None:
        model = _make_mock_model()
        queue = ChunkedActionQueue()
        ex = AsyncExecution()
        ex.start(model, queue)
        ex.stop()

        # Leftovers from the finished run.
        ex._death_cause = RuntimeError("died previously")  # noqa: SLF001
        ex._inference_count = 5  # noqa: SLF001
        with ex._lock:  # noqa: SLF001
            stale = {"state": np.full(4, 99.0, dtype=np.float32)}
            ex._obs_slot = (stale, ex._incarnation)  # noqa: SLF001
            ex._running_inference = True  # noqa: SLF001

        ex.start(model, queue)
        try:
            assert ex._death_cause is None  # noqa: SLF001
            assert ex.inference_count == 0
            assert ex._busy is False  # derived from _obs_slot and _running_inference  # noqa: SLF001
            ex.maybe_request({"state": np.zeros(4, dtype=np.float32)})  # must not raise WorkerDiedError
        finally:
            ex.stop()

    def test_rtc_clears_stale_per_run_state(self) -> None:
        from physicalai.runtime import RTCActionQueue, RTCExecution

        model = _rtc_model()
        queue = RTCActionQueue()
        ex = RTCExecution(chunk_size=20, max_action_dim=3, fps=30.0)
        ex.start(model, queue)
        ex.stop()

        ex._death_cause = RuntimeError("died previously")  # noqa: SLF001
        with ex._obs_lock:  # noqa: SLF001
            stale = {"state": np.full(3, 99.0, dtype=np.float32)}
            ex._obs_slot = (stale, ex._incarnation, None)  # noqa: SLF001

        ex.start(model, queue)
        try:
            assert ex._death_cause is None  # noqa: SLF001
            assert ex.inference_count == 0
            with ex._obs_lock:  # noqa: SLF001
                assert ex._obs_slot is None  # noqa: SLF001
            # warmup() blocks on a per-request signal created after restart.
            ex.warmup({"state": np.zeros(3, dtype=np.float32)})
        finally:
            ex.stop()

    def test_sync_inference_count_is_per_run(self) -> None:
        model = _make_mock_model()
        queue = ChunkedActionQueue()
        ex = SyncExecution()

        ex.start(model, queue)
        ex.warmup({"state": np.zeros(4, dtype=np.float32)})
        for _ in range(6):
            queue.pop()
        ex.maybe_request({"state": np.zeros(4, dtype=np.float32)})
        assert ex.inference_count == 1

        ex.start(model, queue)
        assert ex.inference_count == 0

    def test_rtc_resets_stat_but_not_the_cold_start_gate(self) -> None:
        """The public stat restarts; the compilation-warmup gate does not.

        Compilation is paid once per process, so a second run must not discard
        its latency samples as though they were cold. Asserted on whether
        ``on_reset()`` fires, so rewiring the gate to the per-run count is caught.
        """
        from physicalai.runtime import RTCActionQueue, RTCExecution

        tracker = MagicMock()
        tracker.compute_delay.return_value = 0
        model = _rtc_model()
        queue = RTCActionQueue()
        ex = RTCExecution(
            chunk_size=20,
            max_action_dim=3,
            fps=30.0,
            latency_tracker=tracker,
            warmup_inferences=1,
        )

        ex.start(model, queue)
        try:
            ex.warmup({"state": np.zeros(3, dtype=np.float32)})
            lifetime = ex._lifetime_inferences  # noqa: SLF001
            assert ex.inference_count >= 1
            assert tracker.on_reset.call_count == 1
        finally:
            ex.stop()

        tracker.on_reset.reset_mock()
        ex.start(model, queue)
        try:
            assert ex.inference_count == 0
            assert ex._lifetime_inferences == lifetime  # noqa: SLF001
            queue.clear()
            ex.warmup({"state": np.zeros(3, dtype=np.float32)})
            tracker.on_reset.assert_not_called()
        finally:
            ex.stop()


def _async_setup() -> tuple[Any, MagicMock, Any]:
    return AsyncExecution(), _make_mock_model(), ChunkedActionQueue()


def _rtc_setup() -> tuple[Any, MagicMock, Any]:
    from physicalai.runtime import RTCActionQueue, RTCExecution

    return RTCExecution(chunk_size=20, max_action_dim=3, fps=30.0), _rtc_model(), RTCActionQueue()


class TestStopTimeoutStraggler:
    """``stop()`` only joins with a timeout, so a worker can outlive it.

    Such a straggler must not be revived by a later ``start()``, must discard the
    result it was computing, and must not run concurrently with a new worker
    through the same unsynchronised ``InferenceModel``.
    """

    @staticmethod
    def _timed_out_stop(ex: Any) -> None:
        """Call ``stop()`` with ``join`` neutered, simulating a timeout fast."""
        with patch.object(threading.Thread, "join", lambda _self, timeout=None: None):  # noqa: ARG005
            ex.stop()

    @pytest.mark.parametrize("setup", [_async_setup, _rtc_setup], ids=["async", "rtc"])
    def test_outgoing_runs_stop_flag_survives_a_new_start(self, setup: Callable[[], tuple[Any, MagicMock, Any]]) -> None:
        """Each run owns its stop event, so a new run cannot un-stop the old one.

        Clearing a shared event here is what previously revived a straggler.
        """
        ex, model, queue = setup()
        ex.start(model, queue)
        outgoing = ex._stop_event  # noqa: SLF001
        ex.stop()
        assert outgoing.is_set()

        ex.start(model, queue)
        try:
            assert outgoing.is_set(), "outgoing run's flag was cleared — its worker would be revived"
            assert ex._stop_event is not outgoing  # noqa: SLF001
            assert not ex._stop_event.is_set()  # noqa: SLF001
        finally:
            ex.stop()

    @pytest.mark.parametrize("kind", ["async", "rtc"], ids=["async", "rtc"])
    def test_straggler_discards_the_result_it_was_computing(self, kind: str, caplog: pytest.LogCaptureFixture) -> None:
        """A result finished after its run ended describes a stale observation."""
        entered, release = threading.Event(), threading.Event()
        obs = {"state": np.zeros(3 if kind == "rtc" else 4, dtype=np.float32)}

        if kind == "async":
            ex, _model, queue = _async_setup()
            model = _blocking_model(entered, release)
            ex.start(model, queue)
            ex._threshold_count = 1  # noqa: SLF001 — submit on an empty queue
            ex.maybe_request(obs)
        else:
            ex, model, queue = _rtc_setup()

            def blocking(*_args: object, **_kwargs: object) -> dict[str, np.ndarray]:
                entered.set()
                release.wait(timeout=30.0)
                return {"action": np.zeros((1, 20, 3), dtype=np.float32)}

            model.side_effect = blocking
            ex.start(model, queue)
            with ex._obs_lock:  # noqa: SLF001
                ex._obs_slot = (dict(obs), ex._incarnation, None)  # noqa: SLF001

        assert entered.wait(timeout=5.0), "worker never entered inference"
        straggler = ex._thread  # noqa: SLF001
        assert straggler is not None

        self._timed_out_stop(ex)
        assert straggler.is_alive()
        # A worker outliving the join is reported, not silently abandoned.
        assert "did not exit within" in caplog.text

        release.set()
        straggler.join(timeout=5.0)
        assert not straggler.is_alive()
        # The discard happens before the counter and the push, so both stay put.
        assert ex.inference_count == 0
        assert queue.remaining == 0

    def test_start_refuses_while_straggler_holds_the_model(self) -> None:
        """Two threads through one InferenceModel is worse than a failed resume."""
        from physicalai.runtime.execution import async_execution

        entered, release = threading.Event(), threading.Event()
        model = _blocking_model(entered, release)
        queue = ChunkedActionQueue()

        ex = AsyncExecution()
        ex.start(model, queue)
        ex._threshold_count = 1  # noqa: SLF001
        ex.maybe_request({"state": np.zeros(4, dtype=np.float32)})
        assert entered.wait(timeout=5.0)
        straggler = ex._thread  # noqa: SLF001
        assert straggler is not None

        self._timed_out_stop(ex)
        try:
            with (
                patch.object(async_execution, "_STRAGGLER_GRACE_S", 0.05),
                pytest.raises(RuntimeError, match="not safe for concurrent use"),
            ):
                ex.start(model, queue)
        finally:
            release.set()
            straggler.join(timeout=5.0)
