from __future__ import annotations

import threading
import time
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


class TestRTCExecutionObsSlot:
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


class TestRestartAfterStop:
    """``start()`` must hand back a usable execution after a previous ``stop()``.

    A runtime can be stopped and run again, which calls ``stop()`` then
    ``start()`` on the same execution object. Per-session state left over from
    the previous run has to be cleared here, or the second run misbehaves
    silently rather than failing loudly.
    """

    def test_async_revives_worker_after_stop(self) -> None:
        """The worker must actually infer again, not exit on a stale stop flag."""
        chunk = np.random.randn(6, 4).astype(np.float32)
        model = _make_mock_model(chunk)
        queue = ChunkedActionQueue()

        ex = AsyncExecution()
        ex.start(model, queue)
        ex.warmup({"state": np.zeros(4, dtype=np.float32)})
        ex.stop()

        # Second session on the same object.
        queue.clear()
        ex.start(model, queue)
        try:
            assert ex._thread is not None  # noqa: SLF001
            assert ex._thread.is_alive(), "worker exited immediately on a stale _stop_event"  # noqa: SLF001
            # start() begins a fresh count, so any inference below is this session's.
            assert ex.inference_count == 0

            ex.warmup({"state": np.zeros(4, dtype=np.float32)})
            for _ in range(len(chunk)):
                queue.pop()
            ex.maybe_request({"state": np.zeros(4, dtype=np.float32)})

            deadline = time.perf_counter() + 5.0
            while ex.inference_count == 0 and time.perf_counter() < deadline:
                time.sleep(0.01)

            assert ex.inference_count >= 1, "worker never ran inference in the second session"
        finally:
            ex.stop()

    def test_async_start_clears_stale_death_cause(self) -> None:
        """A worker death in run 1 must not fail run 2."""
        model = _make_mock_model()
        queue = ChunkedActionQueue()
        ex = AsyncExecution()
        ex.start(model, queue)
        ex.stop()
        ex._death_cause = RuntimeError("died in the previous session")  # noqa: SLF001

        ex.start(model, queue)
        try:
            assert ex._death_cause is None  # noqa: SLF001
            ex.maybe_request({"state": np.zeros(4, dtype=np.float32)})  # must not raise
        finally:
            ex.stop()

    def test_async_start_drops_stale_observation(self) -> None:
        """A leftover observation must not be inferred on as if it were current."""
        model = _make_mock_model()
        queue = ChunkedActionQueue()
        ex = AsyncExecution()
        ex.start(model, queue)
        ex.stop()
        with ex._lock:  # noqa: SLF001
            ex._obs_slot = {"state": np.full(4, 99.0, dtype=np.float32)}  # noqa: SLF001
            ex._running_inference = True  # noqa: SLF001

        ex.start(model, queue)
        try:
            with ex._lock:  # noqa: SLF001
                assert ex._obs_slot is None  # noqa: SLF001
                assert ex._running_inference is False  # noqa: SLF001
            # _busy is derived from both, so a submission is possible again.
            assert ex._busy is False  # noqa: SLF001
        finally:
            ex.stop()

    def test_rtc_start_clears_stale_death_cause_and_observation(self) -> None:
        from physicalai.runtime import RTCActionQueue, RTCExecution

        chunk_size, action_dim = 20, 3
        model = MagicMock()
        model.chunk_size = chunk_size
        model.postprocessors = []
        model.return_value = {"action": np.random.randn(1, chunk_size, action_dim).astype(np.float32)}

        queue = RTCActionQueue()
        ex = RTCExecution(chunk_size=chunk_size, max_action_dim=action_dim, fps=30.0)
        ex.start(model, queue)
        ex.stop()
        ex._death_cause = RuntimeError("died in the previous session")  # noqa: SLF001
        with ex._obs_lock:  # noqa: SLF001
            ex._obs_slot = {"state": np.full(action_dim, 99.0, dtype=np.float32)}  # noqa: SLF001

        ex.start(model, queue)
        try:
            assert ex._death_cause is None  # noqa: SLF001
            with ex._obs_lock:  # noqa: SLF001
                assert ex._obs_slot is None  # noqa: SLF001
            # warmup() blocks on _first_chunk_ready, which start() also reset.
            ex.warmup({"state": np.zeros(action_dim, dtype=np.float32)})
        finally:
            ex.stop()


class TestInferenceCountIsPerRun:
    """``inference_count`` describes one run, matching the queue's counters.

    ``docs/reference/runtime-api.md`` points callers at both
    ``action_queue.total_pops`` and ``execution.inference_count`` for run
    stats, so the two must agree on scope.
    """

    def test_sync_resets_on_start(self) -> None:
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

    def test_async_resets_on_start(self) -> None:
        model = _make_mock_model()
        queue = ChunkedActionQueue()
        ex = AsyncExecution()
        ex.start(model, queue)
        ex._inference_count = 5  # noqa: SLF001 — stand in for a completed run
        ex.stop()

        ex.start(model, queue)
        try:
            assert ex.inference_count == 0
        finally:
            ex.stop()

    def test_rtc_resets_stat_but_not_the_cold_start_gate(self) -> None:
        """The public stat restarts; the compilation-warmup gate does not.

        Model compilation is paid once per process, so a second run must not
        re-discard its latency samples as though they were cold. Asserted on
        the gated behaviour — whether ``on_reset()`` fires — not just on the
        counters, so wiring the gate back to the per-run count is caught.
        """
        from physicalai.runtime import RTCActionQueue, RTCExecution

        chunk_size, action_dim = 20, 3
        model = MagicMock()
        model.chunk_size = chunk_size
        model.postprocessors = []
        model.return_value = {"action": np.random.randn(1, chunk_size, action_dim).astype(np.float32)}

        tracker = MagicMock()
        tracker.compute_delay.return_value = 0
        queue = RTCActionQueue()
        ex = RTCExecution(
            chunk_size=chunk_size,
            max_action_dim=action_dim,
            fps=30.0,
            latency_tracker=tracker,
            warmup_inferences=1,
        )

        ex.start(model, queue)
        try:
            ex.warmup({"state": np.zeros(action_dim, dtype=np.float32)})
            assert ex.inference_count >= 1
            lifetime_after_first = ex._lifetime_inferences  # noqa: SLF001
            assert lifetime_after_first >= 1
            assert tracker.on_reset.call_count == 1, "cold-start discard should fire once"
        finally:
            ex.stop()

        tracker.on_reset.reset_mock()
        ex.start(model, queue)
        try:
            # Public stat restarts...
            assert ex.inference_count == 0
            # ...while the lifetime count carries on, keeping the gate closed.
            assert ex._lifetime_inferences == lifetime_after_first  # noqa: SLF001

            queue.clear()
            ex.warmup({"state": np.zeros(action_dim, dtype=np.float32)})
            assert ex.inference_count >= 1
            tracker.on_reset.assert_not_called()
        finally:
            ex.stop()
