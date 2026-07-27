# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Synchronous inference execution strategy."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

from physicalai.config import export_config
from physicalai.runtime.execution.base import NOT_STARTED, Execution

if TYPE_CHECKING:
    import numpy as np

    from physicalai.inference.model import InferenceModel
    from physicalai.runtime._callback_bus import _CallbackBus
    from physicalai.runtime.execution.queue import ActionQueue, ChunkedActionQueue


@export_config(class_path="physicalai.runtime.SyncExecution")
class SyncExecution(Execution):
    """Synchronous inference in the control thread."""

    def __init__(
        self,
        *,
        request_threshold: float = 0.5,
    ) -> None:
        """Configure synchronous execution.

        Args:
            request_threshold: Re-infer when queue drops below this fraction
                of chunk_size. E.g. 0.5 means re-infer after consuming half
                the chunk (discards the stale tail). Set to 0.0 to drain
                the entire chunk before re-inferring.
        """
        self._model: InferenceModel | None = None
        self._queue: ChunkedActionQueue | None = None
        self._chunk_size: int = 0
        self._threshold_frac = request_threshold
        self._threshold_count: int = 0
        self._inference_count: int = 0
        self._bus: _CallbackBus | None = None
        self._session_id: str = ""

    def start(self, model: InferenceModel, action_queue: ActionQueue) -> None:
        """Bind model and queue."""
        self._model = model
        self._queue = cast("ChunkedActionQueue", action_queue)

    def warmup(self, sample_observation: dict[str, np.ndarray]) -> None:
        """Run one inference, seed queue, discover chunk_size.

        Raises:
            RuntimeError: If start() has not been called.
        """
        if self._model is None or self._queue is None:
            raise RuntimeError(NOT_STARTED)
        actions = self._model.predict_action_chunk(sample_observation)
        self._chunk_size = actions.shape[0]
        self._threshold_count = max(1, int(self._chunk_size * self._threshold_frac))
        self._queue.push_chunk(actions, offset=0)

    def maybe_request(self, observation: dict[str, np.ndarray]) -> None:
        """Refill queue synchronously when below threshold.

        Raises:
            RuntimeError: If start() has not been called.
        """
        if self._model is None or self._queue is None:
            raise RuntimeError(NOT_STARTED)
        if self._queue.below_threshold(self._threshold_count):
            t0 = time.perf_counter()
            actions = self._model.predict_action_chunk(observation)
            latency = time.perf_counter() - t0
            self._queue.push_chunk(actions, offset=0)
            self._inference_count += 1
            if self._bus:
                from physicalai.runtime.events import InferenceEvent  # noqa: PLC0415

                self._bus.emit_inference(
                    InferenceEvent(
                        session_id=self._session_id,
                        timestamp=time.time(),
                        latency_s=latency,
                        offset=0,
                        chunk=actions,
                    )
                )

    def stop(self) -> None:
        """No-op for synchronous execution."""

    @property
    def chunk_size(self) -> int:
        """Return discovered chunk size."""
        return self._chunk_size

    @property
    def inference_count(self) -> int:
        """Number of completed inference calls."""
        return self._inference_count
