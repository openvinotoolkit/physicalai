# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from physicalai.runtime.events import InferenceEvent, LifecycleEvent, TickEvent

logger = logging.getLogger(__name__)

_INFERENCE_QUEUE_MAXLEN = 64


class _CallbackBus:
    """Internal dispatch bus for runtime callbacks.

    Two dispatch modes:
    - Fire-and-forget (emit_*): telemetry hooks, exceptions isolated.
    - Request-response (invoke_*): action hooks, chained return values.

    Thread safety: ``emit_inference`` may be called from either the control
    thread (SyncExecution) or the inference thread (AsyncExecution). All other
    methods run on the control thread only.
    """

    def __init__(self, callbacks: Sequence[Any]) -> None:
        self._callbacks = list(callbacks)
        self._inference_queue: deque[InferenceEvent] = deque(maxlen=_INFERENCE_QUEUE_MAXLEN)

    def emit_tick(self, event: TickEvent) -> None:
        self._drain_inference()
        for cb in self._callbacks:
            fn = getattr(cb, "on_tick", None)
            if fn is None:
                continue
            try:
                fn(event)
            except Exception:
                logger.exception("Callback %r failed in on_tick", cb)

    def emit_inference(self, event: InferenceEvent) -> None:
        """Enqueue inference event from background thread for control-thread delivery."""
        self._inference_queue.append(event)

    def emit_lifecycle(self, event: LifecycleEvent) -> None:
        for cb in self._callbacks:
            fn = getattr(cb, "on_lifecycle", None)
            if fn is None:
                continue
            try:
                fn(event)
            except Exception:
                logger.exception("Callback %r failed in on_lifecycle", cb)

    def invoke_before_send_action(self, *, action: np.ndarray, step: int) -> np.ndarray:
        result = action
        for cb in self._callbacks:
            fn = getattr(cb, "before_send_action", None)
            if fn is None:
                continue
            try:
                modified = fn(action=result, step=step)
                if modified is not None:
                    result = modified
            except Exception:
                logger.exception("Callback %r failed in before_send_action", cb)
        return result

    def invoke_on_action_sent(self, *, action: np.ndarray, step: int) -> None:
        for cb in self._callbacks:
            fn = getattr(cb, "on_action_sent", None)
            if fn is None:
                continue
            try:
                fn(action=action, step=step)
            except Exception:
                logger.exception("Callback %r failed in on_action_sent", cb)

    def invoke_on_hold(self, *, step: int, holds: int) -> None:
        for cb in self._callbacks:
            fn = getattr(cb, "on_hold", None)
            if fn is None:
                continue
            try:
                fn(step=step, holds=holds)
            except Exception:
                logger.exception("Callback %r failed in on_hold", cb)

    def close(self) -> None:
        for cb in self._callbacks:
            close_fn = getattr(cb, "close", None)
            if close_fn is not None:
                try:
                    close_fn()
                except Exception:
                    logger.exception("Callback %r failed in close", cb)

    def _drain_inference(self) -> None:
        while self._inference_queue:
            event = self._inference_queue.popleft()
            for cb in self._callbacks:
                fn = getattr(cb, "on_inference", None)
                if fn is None:
                    continue
                try:
                    fn(event)
                except Exception:
                    logger.exception("Callback %r failed in on_inference", cb)
