# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Timing of action chunks against the tick their observation was read on.

The fake model here returns chunks whose action ``k`` holds ``obs_tick + k``:
the tick that action is meant for. A test then pops actions tick by tick and
checks each one against the tick it is sent on.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from physicalai.runtime import AsyncExecution, ChunkedActionQueue

if TYPE_CHECKING:
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
