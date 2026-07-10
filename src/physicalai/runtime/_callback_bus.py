# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from physicalai.runtime.events import InferenceEvent, LifecycleEvent, MetricsEvent, TickEvent

logger = logging.getLogger(__name__)

_INFERENCE_QUEUE_MAXLEN = 64


class _CallbackBus:
    """Internal dispatch bus for runtime callbacks.

    Two dispatch modes:
    - Fire-and-forget (emit_*, ``invoke_on_action_sent``): notification
      hooks, exceptions isolated and logged — a broken callback can't stop
      the run.
    - Request-response (``invoke_on_action_ready``): chains each callback's
      transform of the outgoing action. Exceptions propagate instead of
      being isolated — a callback that fails partway through the chain
      means the action can no longer be trusted, so it must not be sent to
      the robot silently un-transformed.

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

    def invoke_on_action_ready(self, *, action: np.ndarray, step: int) -> np.ndarray:
        """Chain each callback's ``on_action_ready``, threading the return value.

        Every callback must return a valid action (no ``None`` sentinel) — a
        callback that doesn't want to change anything returns its input
        unchanged.

        Unlike the ``emit_*`` hooks, exceptions here are not isolated: a
        callback (e.g. a safety filter) that fails partway through the chain
        means the action can no longer be trusted, so the failure is logged
        (identifying which callback raised) and then re-raised instead of
        silently sending a partially-transformed action to the robot.

        Returns:
            The action after every callback has had a chance to transform it.
        """
        result = action
        for cb in self._callbacks:
            fn = getattr(cb, "on_action_ready", None)
            if fn is None:
                continue
            try:
                modified_action = fn(action=result, step=step)
                if modified_action is None:
                    logger.warning("Callback %r returned None from on_action_ready, ignoring", cb)
                else:
                    result = modified_action
            except Exception:
                logger.exception("Callback %r failed in on_action_ready", cb)
                raise
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

    def emit_metrics(self, event: MetricsEvent) -> None:
        """Fire-and-forget dispatch to ``on_metrics``, exceptions isolated.

        Called synchronously on the control thread (e.g. from within
        ``PolicySource.update()``), so unlike ``emit_inference`` no queue is
        needed here.
        """
        for cb in self._callbacks:
            fn = getattr(cb, "on_metrics", None)
            if fn is None:
                continue
            try:
                fn(event)
            except Exception:
                logger.exception("Callback %r failed in on_metrics", cb)

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
