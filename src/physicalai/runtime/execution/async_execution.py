# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Asynchronous (background-thread) inference execution strategy."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, cast

import numpy as np

from physicalai.config import export_config
from physicalai.runtime.execution.base import NOT_STARTED, Execution, WorkerDiedError

if TYPE_CHECKING:
    from physicalai.inference.model import InferenceModel
    from physicalai.runtime._callback_bus import _CallbackBus
    from physicalai.runtime.execution.queue import ActionQueue, ChunkedActionQueue

logger = logging.getLogger(__name__)


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
        self._obs_slot: dict[str, np.ndarray] | None = None
        self._obs_ready = threading.Event()
        self._running_inference = False
        self._request_time: float = 0.0
        self._pops_at_request: int = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._death_cause: BaseException | None = None
        self._inference_count: int = 0
        self._bus: _CallbackBus | None = None
        self._session_id: str = ""

    def start(self, model: InferenceModel, action_queue: ActionQueue) -> None:
        """Bind model/queue and spawn inference thread."""
        self._model = model
        self._queue = cast("ChunkedActionQueue", action_queue)
        self._thread = threading.Thread(target=self._run, name="InferenceThread", daemon=True)
        self._thread.start()

    def warmup(self, sample_observation: dict[str, np.ndarray]) -> None:
        """Run one inference in main thread, seed queue, discover chunk_size.

        Raises:
            RuntimeError: If start() has not been called.
        """
        if self._model is None or self._queue is None:
            raise RuntimeError(NOT_STARTED)
        actions = self._model.predict_action_chunk(sample_observation)
        self._chunk_size = actions.shape[0]
        self._threshold_count = int(self._chunk_size * self._threshold_frac)
        self._queue.push_chunk(actions, offset=0)

    def maybe_request(self, observation: dict[str, np.ndarray]) -> None:
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
                self._obs_slot = snapshot
                self._request_time = time.perf_counter()
                self._pops_at_request = self._queue.total_pops
            self._obs_ready.set()

    def stop(self) -> None:
        """Signal thread and join with timeout."""
        if self._thread is not None:
            self._stop_event.set()
            self._obs_ready.set()
            self._thread.join(timeout=10.0)

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

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._obs_ready.wait()
                self._obs_ready.clear()

                if self._stop_event.is_set():
                    return

                with self._lock:
                    obs = self._obs_slot
                    self._obs_slot = None
                    if obs is None:
                        continue
                    self._running_inference = True

                if self._model is None or self._queue is None:
                    raise RuntimeError(NOT_STARTED)  # noqa: TRY301
                t0 = time.perf_counter()
                actions = self._model.predict_action_chunk(obs)
                latency = time.perf_counter() - t0

                # Offset = actions actually sent since the observation was
                # captured. This is exact (no fps estimation error).
                with self._lock:
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
