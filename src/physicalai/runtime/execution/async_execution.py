# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Asynchronous (background-thread) inference execution strategy."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from physicalai.config import export_config
from physicalai.runtime.execution.base import NOT_STARTED, Execution, WorkerDiedError

if TYPE_CHECKING:
    from physicalai.inference.model import InferenceModel
    from physicalai.runtime._callback_bus import _CallbackBus
    from physicalai.runtime.execution.queue import ActionQueue, ChunkedActionQueue

logger = logging.getLogger(__name__)

_JOIN_TIMEOUT_S: float = 10.0
_STRAGGLER_GRACE_S: float = 2.0


@export_config(class_path="physicalai.runtime.AsyncExecution")
class AsyncExecution(Execution):
    """Async inference in a background thread with health monitoring."""

    def __init__(
        self,
        request_threshold: float = 0.5,
        watchdog_timeout_s: float = 30.0,
    ) -> None:
        """Configure the async execution strategy.

        Args:
            request_threshold: Queue fraction at which to request new inference.
                When the action queue drops below this fraction of chunk_size,
                a new inference is scheduled. E.g. 0.25 means "request when
                only 25% of the chunk remains in the queue."
            watchdog_timeout_s: If inference is stuck longer than this, force-reset.
        """
        self._threshold_frac = request_threshold
        self._watchdog_timeout_s = watchdog_timeout_s

        self._model: InferenceModel | None = None
        self._queue: ChunkedActionQueue | None = None
        self._chunk_size: int = 0
        self._threshold_count: int = 0

        self._lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._obs_slot: tuple[dict[str, Any], int] | None = None
        self._obs_ready = threading.Event()
        self._running_inference = False
        self._request_time: float = 0.0
        self._pops_at_request: int = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._death_cause: BaseException | None = None
        self._inference_count: int = 0
        # Incarnation counter bumped by start()/reset(). Each observation handed
        # to the worker carries the current value; if reset() lands while
        # inference is in flight, the worker's incarnation no longer matches
        # and the result is discarded instead of reaching the queue.
        self._incarnation: int = 0
        self._bus: _CallbackBus | None = None
        self._session_id: str = ""

    def start(self, model: InferenceModel, action_queue: ActionQueue) -> None:
        """Bind model/queue and spawn a worker owned by this run.

        Each run gets fresh stop and wake events, passed straight to its worker.
        A straggler from a previous run therefore keeps its own, already-set stop
        event: this ``start()`` cannot revive it, it cannot steal the new
        worker's wake-up, and it discards its in-flight result instead of pushing
        it into this run's queue.

        Raises:
            RuntimeError: If the previous worker is still inside the model after
                a short grace period. Running anyway would put two threads
                through one ``InferenceModel``, which is not synchronised.
        """  # noqa: DOC502 — raised by the delegated _await_previous_worker(), but callers see it here.
        self._await_previous_worker()

        self._model = model
        self._queue = cast("ChunkedActionQueue", action_queue)
        # New objects, not clear(): a straggler keeps its own set stop event.
        self._stop_event = threading.Event()
        self._obs_ready = threading.Event()
        self._inference_count = 0
        with self._lock:
            self._incarnation += 1
            # A stale observation would be inferred on as if it were current.
            self._obs_slot = None
            self._running_inference = False
            self._request_time = 0.0
            self._pops_at_request = 0
        # A death from the previous run must not fail this one.
        self._death_cause = None
        self._thread = threading.Thread(
            target=self._run,
            args=(self._stop_event, self._obs_ready),
            name="InferenceThread",
            daemon=True,
        )
        self._thread.start()

    def _await_previous_worker(self) -> None:
        """Wait briefly for a worker that outlived ``stop()``, then refuse.

        ``InferenceModel`` is not synchronised, so letting a new run start while
        a straggler is still inside the model would put two threads through one
        model instance. Failing loudly here beats corrupting both.

        Raises:
            RuntimeError: If the straggler is still running after the grace period.
        """
        previous = self._thread
        if previous is None or not previous.is_alive():
            return

        logger.warning(
            "Previous inference worker is still running — waiting up to %.1fs for it to exit.",
            _STRAGGLER_GRACE_S,
        )
        previous.join(timeout=_STRAGGLER_GRACE_S)
        if previous.is_alive():
            msg = (
                "Previous inference worker is still inside the model after "
                f"{_JOIN_TIMEOUT_S + _STRAGGLER_GRACE_S:.1f}s. Refusing to start a second "
                "worker: InferenceModel is not safe for concurrent use. Wait for the "
                "inference to finish, or build a new execution with its own model."
            )
            raise RuntimeError(msg)

    def warmup(self, sample_observation: dict[str, Any]) -> None:
        """Run one inference in main thread, seed queue, discover chunk_size.

        Raises:
            RuntimeError: If start() has not been called.
        """
        if self._model is None or self._queue is None:
            raise RuntimeError(NOT_STARTED)
        with self._lock:
            incarnation = self._incarnation
        with self._model_lock:
            actions = self._model.predict_action_chunk(sample_observation)
        with self._lock:
            if incarnation != self._incarnation:
                msg = "AsyncExecution warmup cancelled by reset"
                raise RuntimeError(msg)
            self._chunk_size = actions.shape[0]
            self._threshold_count = int(self._chunk_size * self._threshold_frac)
            self._queue.push_chunk(actions, offset=0)

    def maybe_request(self, observation: dict[str, Any]) -> None:
        """Submit observation for background inference if queue is low and worker idle.

        Raises:
            RuntimeError: If start() has not been called.
            WorkerDiedError: If the inference thread has died.
        """
        if self._queue is None:
            raise RuntimeError(NOT_STARTED)
        if self._thread is not None and not self._thread.is_alive() and self._death_cause is not None:
            msg = f"Inference thread died: {self._death_cause}"
            raise WorkerDiedError(msg) from self._death_cause

        if self._busy_duration > self._watchdog_timeout_s:
            logger.warning("Inference stuck for %.0fs — force resetting", self._busy_duration)
            self._force_reset()

        if self._queue.below_threshold(self._threshold_count) and not self._busy:
            snapshot = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in observation.items()}
            with self._lock:
                self._obs_slot = (snapshot, self._incarnation)
                self._request_time = time.perf_counter()
                self._pops_at_request = self._queue.total_pops
            self._obs_ready.set()

    def reset(self, *, reset_model: bool = True) -> None:
        """Invalidate queued requests and optionally wait for active inference.

        The worker remains alive. A result whose inference began before this
        method is discarded even if it completes after the reset. With
        ``reset_model=True``, waits for inference already inside the model.

        Raises:
            RuntimeError: If :meth:`start` has not been called.
        """
        if self._model is None:
            raise RuntimeError(NOT_STARTED)
        with self._lock:
            self._incarnation += 1
            self._obs_slot = None
            self._request_time = time.perf_counter()
        if reset_model:
            with self._model_lock:
                self._model.reset()

    def stop(self) -> None:
        """Signal the worker and join it, with a timeout.

        Best-effort by nature: a worker inside a blocking inference cannot be
        preempted. It is left to finish and discard its result, so this never
        raises — teardown must not fail. A worker that outlives the join is
        logged, and the next :meth:`start` refuses to run alongside it.
        """
        if self._thread is not None:
            self._stop_event.set()
            self._obs_ready.set()
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                logger.warning(
                    "Inference worker did not exit within %.1fs — still inside the model. "
                    "It will discard its result and exit on its own.",
                    _JOIN_TIMEOUT_S,
                )

    @property
    def chunk_size(self) -> int:
        """Return discovered chunk size."""
        return self._chunk_size

    @property
    def alive(self) -> bool:
        """Whether the inference thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def inference_count(self) -> int:
        """Number of completed inference calls."""
        return self._inference_count

    @property
    def _busy(self) -> bool:
        with self._lock:
            return self._obs_slot is not None or self._running_inference

    @property
    def _busy_duration(self) -> float:
        with self._lock:
            if not (self._obs_slot is not None or self._running_inference):
                return 0.0
            return time.perf_counter() - self._request_time

    def _force_reset(self) -> None:
        with self._lock:
            self._obs_slot = None
            self._running_inference = False
        logger.warning("Force reset — cleared stuck inference state")

    def _run(self, stop_event: threading.Event, obs_ready: threading.Event) -> None:
        """Worker loop for one run.

        Takes its own events rather than reading ``self``, so a worker that
        outlives ``stop()`` keeps observing its own set stop event even after a
        later ``start()`` has installed fresh ones.

        Nothing propagates out of this thread.

        Raises:
            RuntimeError: If the model or queue is unset. Captured into
                ``_death_cause`` rather than leaving this thread; the control
                thread surfaces it as ``WorkerDiedError`` on the next
                ``maybe_request()``.
        """
        try:
            while not stop_event.is_set():
                obs_ready.wait()
                obs_ready.clear()

                if stop_event.is_set():
                    return

                with self._lock:
                    request = self._obs_slot
                    self._obs_slot = None
                    if request is None:
                        continue
                    obs, incarnation = request
                    self._running_inference = True

                if self._model is None or self._queue is None:
                    raise RuntimeError(NOT_STARTED)  # noqa: TRY301
                t0 = time.perf_counter()
                with self._model_lock:
                    with self._lock:
                        if incarnation != self._incarnation:
                            self._running_inference = False
                            continue
                    actions = self._model.predict_action_chunk(obs)
                    latency = time.perf_counter() - t0

                if stop_event.is_set():
                    # This run ended while the inference was in flight. The
                    # actions describe an observation from a finished session,
                    # so they must not reach a later run's queue.
                    return

                # Offset = actions actually sent since the observation was
                # captured. This is exact (no fps estimation error).
                with self._lock:
                    if incarnation != self._incarnation:
                        self._running_inference = False
                        continue
                    pops_since = self._queue.total_pops - self._pops_at_request
                    offset = min(max(pops_since, 0), len(actions) - 1)
                    self._queue.push_chunk(actions, offset=offset)
                self._inference_count += 1

                if self._bus:
                    from physicalai.runtime.events import InferenceEvent  # noqa: PLC0415

                    self._bus.emit_inference(
                        InferenceEvent(
                            session_id=self._session_id,
                            timestamp=time.time(),
                            latency_s=latency,
                            offset=offset,
                            chunk=actions,
                        )
                    )

                with self._lock:
                    self._running_inference = False

        except Exception as e:
            self._death_cause = e
            logger.exception("Inference thread died")
