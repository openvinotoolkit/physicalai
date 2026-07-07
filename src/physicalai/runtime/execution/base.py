# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Execution ABC and the worker-death error type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from physicalai.inference.model import InferenceModel
    from physicalai.runtime._callback_bus import _CallbackBus
    from physicalai.runtime.execution.queue import ActionQueue

NOT_STARTED = "start() must be called before this method"


class WorkerDiedError(RuntimeError):
    """Raised when the inference worker thread dies unexpectedly."""


class Execution(ABC):
    """Decides when and where inference runs. Pushes results into ActionQueue."""

    _bus: _CallbackBus | None
    _session_id: str

    def set_bus(self, bus: _CallbackBus, session_id: str) -> None:
        """Inject callback bus and session ID before the control loop starts."""
        self._bus = bus
        self._session_id = session_id

    @abstractmethod
    def start(self, model: InferenceModel, action_queue: ActionQueue) -> None:
        """Bind to model and queue. Called once before the loop."""
        ...

    @abstractmethod
    def maybe_request(self, observation: dict[str, np.ndarray]) -> None:
        """Check if new inference is needed given the (already-read) observation. If so, run or schedule it."""
        ...

    @abstractmethod
    def warmup(self, sample_observation: dict[str, np.ndarray]) -> None:
        """Run one inference to discover chunk_size and seed the queue."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop scheduling."""
        ...

    @property
    @abstractmethod
    def chunk_size(self) -> int:
        """Discovered after warmup()."""
        ...
