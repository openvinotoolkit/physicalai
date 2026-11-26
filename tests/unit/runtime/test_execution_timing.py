# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Timing of action chunks against the tick their observation was read on.

The fake models here return chunks whose action ``k`` holds ``obs_tick + k``:
the tick that action is meant for. A test then pops actions tick by tick and
checks each one against the tick it is sent on.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from physicalai.runtime import AsyncExecution, ChunkedActionQueue, RTCActionQueue, RTCExecution

if TYPE_CHECKING:
    from collections.abc import Callable

    from physicalai.inference.model import InferenceModel

CHUNK = 20


def _wait_for(predicate: Any, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return False


class _GatedModel:
    """Returns ``obs_tick + k`` per action; each call after the first waits for ``release``."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def predict_action_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        self.calls += 1
        if self.calls > 1:
            self.entered.set()
            self.release.wait(timeout=5.0)
            self.release.clear()
        tick = float(np.asarray(obs["state"]).reshape(-1)[0])
        return (tick + np.arange(CHUNK, dtype=np.float32))[:, None].repeat(2, axis=1)

    def reset(self) -> None:
        pass


class _RTCModel:
    """The RTC side of :class:`_GatedModel`: called with the RTC inputs, which it records.

    Each call after the first waits for ``release``, which stays set once given.
    """

    chunk_size = CHUNK
    manifest = SimpleNamespace(model_extra={})

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.rtc_inputs: list[dict[str, Any]] = []

    def _gate(self) -> None:
        self.calls += 1
        if self.calls > 1:
            self.entered.set()
            self.release.wait(timeout=5.0)

    def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        self.rtc_inputs.append({k: np.array(v) for k, v in inputs.items() if k != "state"})
        self._gate()
        tick = float(np.asarray(inputs["state"]).reshape(-1)[0])
        return {"action": (tick + np.arange(self.chunk_size, dtype=np.float32))[None, :, None].repeat(2, axis=2)}

    def reset(self) -> None:
        pass


def _obs(tick: int) -> dict[str, Any]:
    return {"state": np.array([[tick, tick]], dtype=np.float32)}


class _Driver:
    """A control loop that requests if the queue is low, then sends one action, as ``PolicySource`` does."""

    def __init__(self, threshold: float, queue: ChunkedActionQueue | None = None) -> None:
        self.model, self.queue = _GatedModel(), queue or ChunkedActionQueue()
        self.ex = AsyncExecution(request_threshold=threshold)
        self.count = max(1, int(CHUNK * threshold))
        self.tick = 0

    def __enter__(self) -> _Driver:
        self.ex.start(cast("InferenceModel", self.model), self.queue)
        self.ex.warmup(_obs(0))
        return self

    def __exit__(self, *_: object) -> None:
        self.model.release.set()
        self.ex.stop()

    def request(self) -> None:
        """Tick until a re-plan is requested; the model then waits for :meth:`deliver`."""
        self.model.entered.clear()
        requested = False
        while not requested:
            requested = self.queue.remaining < self.count
            self.ex.maybe_request(_obs(self.tick))
            self.queue.pop()
            self.tick += 1
        assert self.model.entered.wait(timeout=2.0), "no re-plan was requested"

    def deliver(self, ticks_in_flight: int) -> None:
        """Keep the inference in flight for ``ticks_in_flight`` more ticks, then let it return."""
        for _ in range(ticks_in_flight):
            self.queue.pop()
            self.tick += 1
        done = self.ex.inference_count + 1
        self.model.release.set()
        assert _wait_for(lambda: self.ex.inference_count == done and not self.ex._busy)  # noqa: SLF001

    def replan(self, ticks_in_flight: int, times: int = 1) -> None:
        for _ in range(times):
            self.request()
            self.deliver(ticks_in_flight)


def _dry_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "ran dry" in r.getMessage()]


class TestAsyncThresholdZero:
    def test_requests_again_once_the_queue_is_empty(self) -> None:
        """request_threshold=0 means "play the whole chunk", as in SyncExecution, not "never ask again"."""
        model, queue = _GatedModel(), ChunkedActionQueue()
        model.release.set()
        ex = AsyncExecution(request_threshold=0.0)
        ex.start(cast("InferenceModel", model), queue)
        try:
            ex.warmup(_obs(0))
            for _ in range(CHUNK):
                assert queue.pop() is not None
            ex.maybe_request(_obs(CHUNK))
            assert _wait_for(lambda: queue.remaining > 0), "no second chunk was requested"
        finally:
            ex.stop()

    def test_holding_for_the_chunk_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """At threshold 0 the robot holds during inference by choice, so there is nothing to warn about."""
        with caplog.at_level("WARNING"), _Driver(0.0) as d:
            d.replan(ticks_in_flight=5)
        assert d.queue.total_holds > 0
        assert not _dry_warnings(caplog)


class TestAsyncAlignment:
    def test_actions_sent_during_inference_are_dropped_from_the_new_chunk(self) -> None:
        with _Driver(0.5) as d:
            d.replan(ticks_in_flight=4)
            sent = d.queue.pop()
        assert d.queue.total_holds == 0
        assert sent is not None
        assert sent[0] == d.tick, "the first action of the new chunk is not the one meant for this tick"


class TestAsyncThresholdShorterThanLatency:
    """A threshold that cannot cover the latency must not stutter silently."""

    def test_warns_when_the_queue_ran_dry_during_inference(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"), _Driver(0.25) as d:
            d.replan(ticks_in_flight=7)
        assert d.queue.total_holds > 0
        assert len(_dry_warnings(caplog)) == 1
        assert "The request left 4 action(s), but inference took about 8 ticks" in _dry_warnings(caplog)[0]

    def test_warns_at_a_threshold_that_rounds_to_one_action(self, caplog: pytest.LogCaptureFixture) -> None:
        """0.05 of 20 is one action, which requests on an empty queue like 0 does, but was not asked for."""
        with caplog.at_level("WARNING"), _Driver(0.05) as d:
            d.replan(ticks_in_flight=3)
        assert len(_dry_warnings(caplog)) == 1

    def test_suggested_threshold_covers_back_to_back_inferences(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"), _Driver(0.25) as d:
            d.replan(ticks_in_flight=7)
        match = re.search(r"at least (\d\.\d\d)", _dry_warnings(caplog)[0])
        assert match is not None
        caplog.clear()

        with caplog.at_level("WARNING"), _Driver(float(match.group(1))) as d:
            d.replan(ticks_in_flight=7, times=4)
        assert d.queue.total_holds == 0
        assert not _dry_warnings(caplog)

    def test_says_when_no_threshold_can_cover_back_to_back_inferences(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Twelve ticks of a 20-action chunk: the first request has room, the trimmed chunk after it does not."""
        with caplog.at_level("WARNING"), _Driver(0.73) as d:
            d.replan(ticks_in_flight=11, times=2)
        warnings = _dry_warnings(caplog)
        assert len(warnings) == 1
        assert "The request left 8 action(s)" in warnings[0]
        assert "no request_threshold avoids this" in warnings[0]

    def test_a_slow_log_handler_does_not_stall_the_control_loop(self) -> None:
        """The warning is logged after the chunk is queued and the lock is released."""

        class _BlockingHandler(logging.Handler):
            def __init__(self) -> None:
                super().__init__(level=logging.WARNING)
                self.entered = threading.Event()
                self.proceed = threading.Event()

            def emit(self, record: logging.LogRecord) -> None:
                if "ran dry" in record.getMessage():
                    self.entered.set()
                    self.proceed.wait(timeout=5.0)

        handler = _BlockingHandler()
        module_logger = logging.getLogger(AsyncExecution.__module__)
        module_logger.addHandler(handler)
        try:
            with _Driver(0.25) as d:
                d.request()
                for _ in range(7):
                    d.queue.pop()
                d.model.release.set()
                assert handler.entered.wait(timeout=2.0)
                control_tick = threading.Thread(target=d.ex.maybe_request, args=(_obs(d.tick),), daemon=True)
                control_tick.start()
                control_tick.join(timeout=1.0)
                assert not control_tick.is_alive(), "maybe_request waited for the log handler"
                assert d.queue.remaining > 0, "the chunk waited for the log handler"

                while d.queue.remaining >= d.count:  # play down to the next request
                    d.queue.pop()
                d.ex.maybe_request(_obs(d.tick))
                assert d.ex._obs_slot is not None, "the request was turned away while the handler ran"  # noqa: SLF001
                d.model.release.set()
                handler.proceed.set()
                assert _wait_for(lambda: d.ex.inference_count == 2), "the request never ran"
        finally:
            handler.proceed.set()
            module_logger.removeHandler(handler)

    def test_counts_a_hold_that_lands_just_before_the_chunk(self, caplog: pytest.LogCaptureFixture) -> None:
        """The control thread does not take the execution's lock, so a pop can land between reading and pushing."""

        class _HoldBeforePush(ChunkedActionQueue):
            """Lands a hold just before each re-planned chunk is pushed, as a racing control tick would."""

            def __init__(self) -> None:
                super().__init__()
                self.pushes = 0

            def push_chunk(self, chunk: np.ndarray, offset: int = 0) -> None:
                self.pushes += 1
                if self.pushes > 1 and self.remaining == 0:
                    assert self.pop() is None
                super().push_chunk(chunk, offset=offset)

        with caplog.at_level("WARNING"), _Driver(0.25, queue=_HoldBeforePush()) as d:
            d.replan(ticks_in_flight=3)  # the request left 4 actions; inference used exactly 4 ticks
        assert d.queue.total_holds == 1
        assert len(_dry_warnings(caplog)) == 1

    def test_queue_reset_during_inference_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """A reset restarts the queue's counters, so holds after it say nothing about the threshold."""
        with caplog.at_level("WARNING"), _Driver(0.25) as d:
            d.request()
            d.queue.reset()
            d.deliver(ticks_in_flight=5)
        assert not _dry_warnings(caplog)


class _GatedQueue(RTCActionQueue):
    """An RTC queue whose worker-side reads wait for ``release`` once ``armed``.

    The control thread (the test) is never held. This fixes the order of the
    worker's queue snapshot against the control thread's pop: the test
    publishes, lets the worker reach the snapshot, pops, then releases it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.control = threading.current_thread()
        self.armed = False
        self.reached = threading.Event()
        self.release = threading.Event()

    def _gate(self) -> None:
        if self.armed and threading.current_thread() is not self.control:
            self.reached.set()
            assert self.release.wait(timeout=5.0), "test never released the worker"

    def get_left_over(self) -> np.ndarray | None:
        self._gate()
        return super().get_left_over()

    def get_action_index(self) -> int:
        self._gate()
        return super().get_action_index()

    def snapshot_left_over(self, from_index: int | None = None) -> tuple[int, np.ndarray | None]:
        self._gate()
        return super().snapshot_left_over(from_index)


class TestRTCAlignment:
    def _run_until_second_chunk(self, **kwargs: Any) -> tuple[_RTCModel, RTCActionQueue, int, int]:
        """Warm up, then tick until the worker takes a request; the worker reads the queue after that tick's pop."""
        model, queue = _RTCModel(), _GatedQueue()
        ex = RTCExecution(chunk_size=CHUNK, fps=30.0, **kwargs)
        ex.start(cast("InferenceModel", model), queue)
        try:
            ex.warmup(_obs(0))
            queue.armed = True
            deadline = time.monotonic() + 5.0
            tick, request_tick = 0, -1
            while request_tick < 0:
                assert time.monotonic() < deadline, "the worker never took a request"
                ex.maybe_request(_obs(tick))
                if queue.below_threshold(ex.queue_threshold):
                    assert queue.reached.wait(timeout=2.0), "the worker did not reach its queue snapshot"
                    request_tick = tick
                queue.pop()  # this tick's action goes out before the worker reads the queue
                tick += 1
            queue.release.set()
            assert model.entered.wait(timeout=2.0)
            for _ in range(3):
                queue.pop()
                tick += 1
            model.release.set()
            assert _wait_for(lambda: ex.inference_count == 2)
        finally:
            queue.release.set()
            model.release.set()
            ex.stop()
        return model, queue, tick, request_tick

    def test_new_chunk_starts_at_the_action_meant_for_the_current_tick(self) -> None:
        _model, queue, tick, _ = self._run_until_second_chunk(execution_horizon=5)
        sent = queue.pop()
        assert sent is not None
        assert sent[0] == tick

    def test_left_over_starts_at_the_action_sent_with_the_observation(self) -> None:
        """Index 0 of the tail and index 0 of the new chunk must stand for the same tick."""
        model, _queue, _tick, request_tick = self._run_until_second_chunk(execution_horizon=5)
        left_over = model.rtc_inputs[1]["prev_chunk_left_over"][0]
        assert left_over[0, 0] == request_tick

    def test_horizon_never_reaches_past_the_left_over_tail(self) -> None:
        model, _queue, _tick, _ = self._run_until_second_chunk(execution_horizon=10, queue_threshold=4)
        inputs = model.rtc_inputs[1]
        left_over = inputs["prev_chunk_left_over"][0]
        n_left = int(np.count_nonzero(left_over[:, 0]))
        assert 0 < n_left < 10
        assert int(inputs["execution_horizon"]) <= n_left
        assert int(inputs["inference_delay"]) <= int(inputs["execution_horizon"])


class _ClockedModel(_RTCModel):
    """A model whose calls take ``latencies[i]`` seconds on a virtual clock, with no real waiting.

    With ``gated``, each call after the first also waits for ``release`` (and sets ``entered``).
    """

    def __init__(self, latencies: list[float], *, gated: bool = False, tracker: Any = None) -> None:
        super().__init__()
        self.now = 0.0
        self._latencies = latencies
        self._gated = gated
        self.tracker = tracker  # called as a model callback, like InferenceModel does

    def clock(self) -> float:
        return self.now

    def _gate(self) -> None:
        self.now += self._latencies[min(self.calls, len(self._latencies) - 1)]
        self.calls += 1
        if self._gated and self.calls > 1:
            self.entered.set()
            assert self.release.wait(timeout=5.0)
            self.release.clear()

    def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        if self.tracker is not None:
            self.tracker.on_predict_start(inputs)
        out = super().__call__(inputs)
        if self.tracker is not None:
            self.tracker.on_predict_end(out)
        return out

    def reset(self) -> None:
        if self.tracker is not None:
            self.tracker.on_reset()


class _VirtualTime:
    """Stands in for the ``time`` module: ``perf_counter`` reads ``clock``, the rest is the real module."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self.perf_counter = clock

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


def _use_clock(monkeypatch: pytest.MonkeyPatch, module: str, clock: Callable[[], float]) -> None:
    monkeypatch.setattr(f"{module}.time", _VirtualTime(clock))


_RTC_MODULE = "physicalai.runtime.execution.rtc"
_TRACKER_MODULE = "physicalai.inference.callbacks.rtc_latency"


def _tick_until(ex: RTCExecution, queue: RTCActionQueue, n_inferences: int, tick: int = 0) -> int:
    """Tick (publish, wait for a taken request, pop) until ``n_inferences`` are done; return the next tick."""
    deadline = time.monotonic() + 10.0
    while ex.inference_count < n_inferences:
        assert time.monotonic() < deadline, f"stuck at {ex.inference_count} of {n_inferences} inferences"
        before = ex.inference_count
        ex.maybe_request(_obs(tick))
        if queue.below_threshold(ex.queue_threshold):
            assert _wait_for(lambda before=before: ex.inference_count > before)
        queue.pop()
        tick += 1
    return tick


def _replan_inputs(
    monkeypatch: pytest.MonkeyPatch, model: _ClockedModel, n_replans: int, **kwargs: Any
) -> list[dict[str, Any]]:
    """Run warmup plus ``n_replans`` re-plans and return the RTC inputs of each model call."""
    _use_clock(monkeypatch, _RTC_MODULE, model.clock)
    queue = RTCActionQueue()
    ex = RTCExecution(chunk_size=CHUNK, fps=30.0, **kwargs)
    ex.start(cast("InferenceModel", model), queue)
    try:
        ex.warmup(_obs(0))
        _tick_until(ex, queue, 1 + n_replans)
    finally:
        ex.stop()
    return model.rtc_inputs


def _threshold_after_warmup(monkeypatch: pytest.MonkeyPatch, latency_s: float, **kwargs: Any) -> int:
    model, queue = _ClockedModel([latency_s]), RTCActionQueue()
    _use_clock(monkeypatch, _RTC_MODULE, model.clock)
    ex = RTCExecution(chunk_size=CHUNK, fps=30.0, execution_horizon=10, **kwargs)
    ex.start(cast("InferenceModel", model), queue)
    try:
        ex.warmup(_obs(0))
        return ex.queue_threshold
    finally:
        ex.stop()


class TestRTCLatency:
    """RTC must tell the model how many actions go out during inference."""

    def test_delay_is_measured_without_a_tracker_from_the_first_replan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        inputs = _replan_inputs(monkeypatch, _ClockedModel([0.29]), 3, execution_horizon=10)
        assert [int(i["inference_delay"]) for i in inputs[1:4]] == [9, 9, 9]

    def test_cold_start_latency_is_an_estimate_then_forgotten(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Warmup takes 5 s (compilation). The first re-plan may only overestimate
        # (clamped to the horizon); once a steady call is measured the 5 s is gone.
        inputs = _replan_inputs(monkeypatch, _ClockedModel([5.0, 0.29]), 3, execution_horizon=10, warmup_inferences=1)
        delays = [int(i["inference_delay"]) for i in inputs[1:4]]
        assert delays[0] == 10
        assert delays[1:] == [9, 9]

    def test_initial_latency_overrides_the_cold_start(self, monkeypatch: pytest.MonkeyPatch) -> None:
        inputs = _replan_inputs(
            monkeypatch, _ClockedModel([5.0, 2.0, 0.29]), 3, execution_horizon=10, initial_latency_s=0.29
        )
        assert [int(i["inference_delay"]) for i in inputs[1:4]] == [9, 9, 9]

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1])
    def test_initial_latency_must_be_finite_and_not_negative(self, value: float) -> None:
        with pytest.raises(ValueError, match="initial_latency_s"):
            RTCExecution(initial_latency_s=value)

    def test_auto_threshold_tracks_the_delay(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _threshold_after_warmup(monkeypatch, 0.29) == 10 + 9

    def test_threshold_shorter_than_the_delay_is_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # delay 9 plus one tick for a result that lands just after the pop it was due for
        assert _threshold_after_warmup(monkeypatch, 0.29, queue_threshold=3) == 11

    def test_reading_the_threshold_logs_nothing_and_the_worker_warns_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        model, queue = _ClockedModel([0.29]), RTCActionQueue()
        _use_clock(monkeypatch, _RTC_MODULE, model.clock)
        ex = RTCExecution(chunk_size=CHUNK, fps=30.0, execution_horizon=10, queue_threshold=3)
        ex.start(cast("InferenceModel", model), queue)
        try:
            with caplog.at_level(logging.WARNING):
                ex.warmup(_obs(0))
                assert _wait_for(lambda: len(caplog.records) == 1)  # the worker warns after its merge
                for _ in range(5):
                    assert ex.queue_threshold == 11
                assert len(caplog.records) == 1  # the getter is on the control path: it never logs
                _tick_until(ex, queue, 3)
        finally:
            ex.stop()
        raised = [r for r in caplog.records if "queue_threshold 3" in r.getMessage()]
        assert len(raised) == 1
        assert raised[0].threadName != threading.current_thread().name

    @pytest.mark.parametrize("pop_before_delivery", [False, True], ids=["before-due-pop", "after-due-pop"])
    def test_hand_set_threshold_never_runs_dry_around_the_due_tick(
        self, monkeypatch: pytest.MonkeyPatch, *, pop_before_delivery: bool
    ) -> None:
        """The chunk measured at 9 ticks lands before, or just after, the pop of the ninth tick after the request."""
        model, queue = _ClockedModel([0.3], gated=True), RTCActionQueue()
        _use_clock(monkeypatch, _RTC_MODULE, model.clock)
        ex = RTCExecution(chunk_size=CHUNK, fps=30.0, execution_horizon=10, queue_threshold=3, warmup_inferences=1)
        ex.start(cast("InferenceModel", model), queue)
        sent: list[np.ndarray | None] = []
        try:
            ex.warmup(_obs(0))
            deadline = time.monotonic() + 5.0
            tick = 0
            while not model.entered.is_set():
                assert time.monotonic() < deadline, "no re-plan was requested"
                ex.maybe_request(_obs(tick))
                if queue.below_threshold(ex.queue_threshold):
                    assert model.entered.wait(timeout=2.0)
                sent.append(queue.pop())
                tick += 1
            for _ in range(8):  # ticks request+1 .. request+8
                sent.append(queue.pop())
                tick += 1
            if pop_before_delivery:  # the result lands just after the pop of tick request+9
                sent.append(queue.pop())
                tick += 1
            model.release.set()
            assert _wait_for(lambda: ex.inference_count == 2)
            for _ in range(3):
                sent.append(queue.pop())
                tick += 1
        finally:
            model.release.set()
            ex.stop()
        assert all(a is not None for a in sent), "the queue ran dry: the robot held its last action"
        assert [float(a[0]) for a in sent] == [float(t) for t in range(tick)]

    def test_the_old_wiring_records_each_inference_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A tracker registered as a model callback and passed to RTCExecution (the older example)."""
        from physicalai.inference.callbacks import RTCLatencyTracker

        tracker = RTCLatencyTracker()
        model = _ClockedModel([0.29], tracker=tracker)
        _use_clock(monkeypatch, _TRACKER_MODULE, model.clock)
        _replan_inputs(monkeypatch, model, 4, execution_horizon=10, latency_tracker=tracker, warmup_inferences=1)
        assert len(tracker._window) == 4  # noqa: SLF001  one per re-plan, the warmup only seeds

    def test_reset_keeps_the_old_worst_case_only_as_a_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from physicalai.inference.callbacks import RTCLatencyTracker

        tracker = RTCLatencyTracker()
        model, queue = _ClockedModel([0.8, 0.8, 0.8, 0.05]), RTCActionQueue()
        _use_clock(monkeypatch, _RTC_MODULE, model.clock)
        ex = RTCExecution(chunk_size=CHUNK * 3, fps=30.0, execution_horizon=10, latency_tracker=tracker, warmup_inferences=1)
        model.chunk_size = CHUNK
        ex.start(cast("InferenceModel", model), queue)
        try:
            ex.warmup(_obs(0))
            tick = _tick_until(ex, queue, 3)
            assert tracker.max_latency_s == pytest.approx(0.8)
            ex.reset()  # episode boundary
            assert tracker.max_latency_s == pytest.approx(0.8)  # still the estimate until a new measurement
            _tick_until(ex, queue, 4, tick)
        finally:
            ex.stop()
        assert tracker.max_latency_s == pytest.approx(0.05)

    @pytest.mark.parametrize(("prep_s", "merge_s", "expected_s"), [(0.1, 0.0, 0.39), (0.0, 0.2, 0.49)], ids=["preparation", "merge"])
    def test_delay_includes_the_worker_preparation_and_the_merge(
        self, monkeypatch: pytest.MonkeyPatch, prep_s: float, merge_s: float, expected_s: float
    ) -> None:
        """Model time is 0.29 s; reading the queue or merging the chunk takes extra (virtual) time."""
        model = _ClockedModel([0.29])

        class _SlowQueue(RTCActionQueue):
            def snapshot_left_over(self, from_index: int | None = None) -> tuple[int, np.ndarray | None]:
                model.now += prep_s
                return super().snapshot_left_over(from_index)

            def merge(self, *args: Any, **kwargs: Any) -> None:
                model.now += merge_s
                super().merge(*args, **kwargs)

        _use_clock(monkeypatch, _RTC_MODULE, model.clock)
        queue = _SlowQueue()
        ex = RTCExecution(chunk_size=CHUNK * 3, fps=30.0, execution_horizon=10, warmup_inferences=1)
        model.chunk_size = CHUNK
        ex.start(cast("InferenceModel", model), queue)
        try:
            ex.warmup(_obs(0))
            _tick_until(ex, queue, 3)
        finally:
            ex.stop()
        window = list(ex._latency_tracker._window)  # noqa: SLF001
        assert window == pytest.approx([expected_s, expected_s])

    def test_a_shared_tracker_still_samples_other_predictions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only the execution's own model calls skip the callback hooks, not the tracker's other uses."""
        from physicalai.inference.callbacks import RTCLatencyTracker

        tracker = RTCLatencyTracker()
        model = _ClockedModel([0.29], tracker=tracker)
        _use_clock(monkeypatch, _TRACKER_MODULE, model.clock)

        def predict_elsewhere(seconds: float) -> None:
            tracker.on_predict_start({})
            model.now += seconds
            tracker.on_predict_end({})

        _use_clock(monkeypatch, _RTC_MODULE, model.clock)
        ex = RTCExecution(chunk_size=CHUNK, fps=30.0, execution_horizon=10, latency_tracker=tracker, warmup_inferences=1)
        predict_elsewhere(0.05)  # before start()
        assert list(tracker._window) == pytest.approx([0.05])  # noqa: SLF001
        queue = RTCActionQueue()
        ex.start(cast("InferenceModel", model), queue)
        try:
            ex.warmup(_obs(0))
            _tick_until(ex, queue, 3)
        finally:
            ex.stop()
        assert list(tracker._window) == pytest.approx([0.29, 0.29])  # noqa: SLF001  warmup seeded, two re-plans recorded once
        predict_elsewhere(0.07)  # after stop()
        assert list(tracker._window)[-1] == pytest.approx(0.07)  # noqa: SLF001

    @pytest.mark.parametrize("fps", [float("nan"), float("inf"), 0.0, -30.0])
    def test_fps_must_be_finite_and_positive(self, fps: float) -> None:
        with pytest.raises(ValueError, match="fps"):
            RTCExecution(fps=fps)

    def test_a_huge_latency_saturates_instead_of_overflowing(self) -> None:
        ex = RTCExecution(chunk_size=20, fps=30.0, execution_horizon=10, initial_latency_s=1e308)
        assert ex._latency_tracker.compute_delay(30.0) == 1_000_000  # noqa: SLF001
        assert ex.queue_threshold == 10 + 1_000_000  # no cap: as on main, a huge estimate means "always re-plan"

    @pytest.mark.parametrize("queue_threshold", [None, 5], ids=["automatic", "hand-set"])
    def test_warmup_is_served_even_when_the_queue_is_full(self, queue_threshold: int | None) -> None:
        model, queue = _RTCModel(), RTCActionQueue()
        queue.push_chunk(np.zeros((CHUNK, 2), dtype=np.float32))  # a full queue: not below any threshold
        ex = RTCExecution(chunk_size=CHUNK, execution_horizon=5, queue_threshold=queue_threshold, initial_latency_s=0.0)
        ex.start(cast("InferenceModel", model), queue)
        try:
            done = threading.Event()
            threading.Thread(target=lambda: (ex.warmup(_obs(0)), done.set()), daemon=True).start()
            assert done.wait(timeout=2.0), "warmup waited for the queue to drain"
        finally:
            ex.stop()

    def test_another_model_on_restart_warms_up_again(self, monkeypatch: pytest.MonkeyPatch) -> None:
        first, second = _ClockedModel([0.05]), _ClockedModel([5.0, 0.05])  # the new model compiles on its first call
        clock = {"model": first}
        _use_clock(monkeypatch, _RTC_MODULE, lambda: clock["model"].now)
        queue = RTCActionQueue()
        ex = RTCExecution(chunk_size=CHUNK, fps=30.0, execution_horizon=5, initial_latency_s=0.05, warmup_inferences=1)
        ex.start(cast("InferenceModel", first), queue)
        try:
            ex.warmup(_obs(0))
            _tick_until(ex, queue, 2)
        finally:
            ex.stop()
        clock["model"] = second
        queue.clear()
        ex.start(cast("InferenceModel", second), queue)
        try:
            ex.warmup(_obs(0))
            _tick_until(ex, queue, 2)
        finally:
            ex.stop()
        assert ex._latency_tracker.max_latency_s == pytest.approx(0.05)  # noqa: SLF001  the 5 s compile was warmup
        assert [int(i["inference_delay"]) for i in second.rtc_inputs[1:]] == [2]

    def test_the_same_model_on_restart_does_not_warm_up_again(self) -> None:
        model, queue = _RTCModel(), RTCActionQueue()
        model.release.set()
        ex = RTCExecution(chunk_size=CHUNK, execution_horizon=5, warmup_inferences=1)
        for _ in range(2):
            ex.start(cast("InferenceModel", model), queue)
            try:
                queue.clear()
                ex.warmup(_obs(0))
            finally:
                ex.stop()
        assert ex._lifetime_inferences == 2  # noqa: SLF001  counted on, not reset

    @pytest.mark.parametrize(("chunk", "warns"), [(30, False), (20, True)])
    def test_two_deliveries_in_a_row_just_after_the_due_pop(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, chunk: int, *, warns: bool
    ) -> None:
        """Latency 0.33 s = delay 10. A 30-action chunk feeds two back-to-back late deliveries; a 20-action one cannot, and says so."""
        model, queue = _ClockedModel([0.33], gated=True), RTCActionQueue()
        model.chunk_size = chunk
        _use_clock(monkeypatch, _RTC_MODULE, model.clock)
        ex = RTCExecution(chunk_size=chunk, fps=30.0, execution_horizon=5, queue_threshold=3, warmup_inferences=1)
        ex.start(cast("InferenceModel", model), queue)
        sent: list[np.ndarray | None] = []
        try:
            with caplog.at_level(logging.WARNING):
                ex.warmup(_obs(0))
                deadline = time.monotonic() + 10.0
                tick, request_tick, delivered = 0, -1, 1
                while delivered < 3 and time.monotonic() < deadline:
                    ex.maybe_request(_obs(tick))
                    if request_tick < 0 and queue.below_threshold(ex.queue_threshold):
                        assert model.entered.wait(timeout=2.0)
                        model.entered.clear()
                        request_tick = tick
                    sent.append(queue.pop())
                    if request_tick >= 0 and tick == request_tick + 10:  # just after the pop it was due for
                        model.release.set()
                        assert _wait_for(lambda d=delivered: ex.inference_count == d + 1)
                        delivered += 1
                        request_tick = -1
                    tick += 1
                assert delivered == 3
        finally:
            model.release.set()
            ex.stop()
        chunk_warnings = [r for r in caplog.records if "too short" in r.getMessage()]
        assert bool(chunk_warnings) is warns
        holds = [i for i, a in enumerate(sent) if a is None]
        assert bool(holds) is warns  # the warning is for a real problem: the 20-action chunk runs dry

    def test_a_zero_window_is_rejected(self) -> None:
        from physicalai.inference.callbacks import RTCLatencyTracker

        with pytest.raises(ValueError, match="window_size"):
            RTCLatencyTracker(window_size=0)

    def test_the_tracker_can_be_copied_and_pickled(self) -> None:
        import copy
        import pickle

        from physicalai.inference.callbacks import RTCLatencyTracker

        tracker = RTCLatencyTracker()
        tracker.record(0.3)
        for clone in (copy.deepcopy(tracker), pickle.loads(pickle.dumps(tracker))):  # noqa: S301
            assert clone.max_latency_s == pytest.approx(0.3)
            with clone.skip_callbacks():
                clone.on_predict_start({})
                clone.on_predict_end({})
            assert len(clone._window) == 1  # noqa: SLF001

    def test_nested_skip_scopes_keep_the_outer_one(self) -> None:
        from physicalai.inference.callbacks import RTCLatencyTracker

        tracker = RTCLatencyTracker()
        with tracker.skip_callbacks():
            with tracker.skip_callbacks():
                pass
            tracker.on_predict_start({})
            tracker.on_predict_end({})
            assert len(tracker._window) == 0  # noqa: SLF001
        tracker.on_predict_start({})
        tracker.on_predict_end({})
        assert len(tracker._window) == 1  # noqa: SLF001

    def test_an_execution_reset_does_not_cut_a_shared_trackers_timed_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from physicalai.inference.callbacks import RTCLatencyTracker

        now = [1000.0]
        _use_clock(monkeypatch, _TRACKER_MODULE, lambda: now[0])
        tracker = RTCLatencyTracker()
        ex = RTCExecution(chunk_size=CHUNK, execution_horizon=5, latency_tracker=tracker)
        ex.start(cast("InferenceModel", _RTCModel()), RTCActionQueue())
        try:
            tracker.on_predict_start({})  # another caller's prediction, timed through the hooks
            ex.reset(reset_model=False)
            now[0] += 0.2
            tracker.on_predict_end({})
        finally:
            ex.stop()
        assert tracker.max_latency_s == pytest.approx(0.2)

    def test_the_warmup_log_does_not_hold_up_the_control_thread(self) -> None:
        """A log handler that blocks on the warmup message must not block maybe_request()."""
        logger = logging.getLogger("physicalai.runtime.execution.rtc")
        blocked, release = threading.Event(), threading.Event()

        class _BlockingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                if "Warmup inference 2/" in record.getMessage():
                    blocked.set()
                    release.wait(timeout=5.0)

        handler = _BlockingHandler(level=logging.INFO)
        old_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        model, queue = _RTCModel(), RTCActionQueue()
        model.release.set()
        # A huge first estimate caps the threshold at the chunk, so every tick publishes under the lock.
        ex = RTCExecution(chunk_size=CHUNK, execution_horizon=5, initial_latency_s=10.0)
        ex.start(cast("InferenceModel", model), queue)
        try:
            ex.warmup(_obs(0))
            deadline = time.monotonic() + 5.0
            tick = 0
            while not blocked.is_set():
                assert time.monotonic() < deadline, "the second inference never logged"
                ex.maybe_request(_obs(tick))
                queue.pop()
                tick += 1
                time.sleep(0.001)
            publisher = threading.Thread(target=ex.maybe_request, args=(_obs(tick),), daemon=True)
            publisher.start()
            publisher.join(timeout=1.0)
            assert not publisher.is_alive(), "maybe_request() waited for the log handler"
        finally:
            release.set()
            ex.stop()
            logger.removeHandler(handler)
            logger.setLevel(old_level)

    def test_tracker_keeps_its_estimate_across_a_reset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from physicalai.inference.callbacks import RTCLatencyTracker

        now = [0.1]
        _use_clock(monkeypatch, _TRACKER_MODULE, lambda: now[0])
        tracker = RTCLatencyTracker()
        tracker.on_predict_start({})
        now[0] += 0.3  # 0.30000000000000004 s: nine ticks plus float error
        tracker.on_predict_end({})
        assert tracker.compute_delay(30.0) == 9  # exactly nine ticks, not ten
        tracker.on_reset()
        assert tracker.compute_delay(30.0) == 9
        tracker.seed(0.1)
        assert tracker.compute_delay(30.0) == 3
