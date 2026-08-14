# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Real-Time Chunking (RTC) execution strategy.

Runs inference in a background daemon thread, injecting RTC-specific
inputs (noise, prev_chunk_left_over, inference_delay, etc.) and
managing a dual-track action queue for continuous robot control.
"""

from __future__ import annotations

import logging
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from physicalai.config import export_config
from physicalai.runtime.execution.base import Execution, WorkerDiedError

if TYPE_CHECKING:
    from physicalai.inference.callbacks.rtc_latency import RTCLatencyTracker
    from physicalai.inference.model import InferenceModel
    from physicalai.inference.postprocessors.base import Postprocessor
    from physicalai.runtime._callback_bus import _CallbackBus
    from physicalai.runtime.execution.rtc_queue import RTCActionQueue

logger = logging.getLogger(__name__)

_NOT_STARTED = "start() must be called before this method"
_IDLE_SLEEP_S: float = 0.005
_ERROR_RETRY_DELAY_S: float = 0.5
_MAX_CONSECUTIVE_ERRORS: int = 10
_JOIN_TIMEOUT_S: float = 5.0
_STRAGGLER_GRACE_S: float = 2.0


@dataclass
class _WarmupSignal:
    """Completion state for one blocking warmup request."""

    event: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


@export_config(class_path="physicalai.runtime.RTCExecution")
class RTCExecution(Execution):
    """Async RTC execution strategy with background inference thread.

    The background thread continuously predicts action chunks and
    merges them into an :class:`RTCActionQueue`. The main thread pops
    one action per tick — never blocking on inference.

    Tuning (defaults validated on π0.5 at 30 fps):

    * ``execution_horizon`` (15) — fresh actions used per chunk before
      re-planning. Raise for smoother, more open-loop motion; lower for
      more reactive motion at the cost of more inferences per second.
    * ``max_guidance_weight`` (5) — how tightly each new chunk is pulled
      toward the previous chunk's tail. Lower it if you see jitter or
      oscillation between chunks; raise it if chunk seams look
      discontinuous.

    RTC-specific inputs injected before each inference call:
    - ``noise``: random noise for denoising (shape: 1 x chunk x action_dim)
    - ``prev_chunk_left_over``: unconsumed tail (shape: 1 x chunk x action_dim)
    - ``inference_delay``: integer derived from measured latency
    - ``max_guidance_weight``: classifier-free guidance weight
    - ``execution_horizon``: number of fresh actions per chunk

    Args:
        chunk_size: Number of actions per model output chunk. If None, is
            automatically inferred from the model's manifest or model metadata.
        execution_horizon: Number of fresh actions to execute from each
            chunk before re-inferring (default 15). Larger = smoother
            and more open-loop (re-plans less often); smaller = more
            reactive (re-plans more often, more model calls per second).
        fps: Robot control frequency in Hz.
        max_action_dim: Model's internal action dimension (for noise/padding).
            If None, is automatically inferred from the model's manifest or
            defaulted to 32.
        max_guidance_weight: Strength of the RTC inpainting guidance
            (paper's β, default 5). Higher pulls each new chunk more
            tightly toward the previous chunk's tail (smoother seams)
            but can oscillate if pushed too high with few denoising
            steps; lower it if you see jitter between chunks.
        queue_threshold: Re-infer when queue drops below this level. If None,
            is dynamically computed as ``execution_horizon + latency_delay_actions``
            derived from worst-case inference latency and robot control rate (fps).
        latency_tracker: Callback that measures inference latency.
            If None, delay defaults to 0.
        warmup_inferences: Number of initial inferences treated as warmup.
            The latency tracker is reset after these to discard
            compilation/kernel-build overhead (e.g. OpenVINO first-run).
        postprocessors: Denormalization pipeline applied to raw actions.
            These run in the background thread to produce the processed
            track stored in the queue. If None, is automatically populated from
            the model's postprocessors.
    """

    def __init__(  # noqa: D107
        self,
        chunk_size: int | None = None,
        execution_horizon: int = 15,
        fps: float = 30.0,
        max_action_dim: int | None = None,
        max_guidance_weight: float = 5.0,
        queue_threshold: int | None = None,
        latency_tracker: RTCLatencyTracker | None = None,
        warmup_inferences: int = 2,
        postprocessors: list[Postprocessor] | None = None,
    ) -> None:
        self._chunk_size_param = chunk_size
        self._execution_horizon = execution_horizon
        self._fps = fps
        self._max_action_dim_param = max_action_dim
        self._max_guidance_weight = max_guidance_weight
        self._queue_threshold_param = queue_threshold
        self._latency_tracker = latency_tracker
        self._warmup_inferences = max(1, warmup_inferences)
        self._postprocessors: list[Postprocessor] = postprocessors or []

        self._rtc_queue: RTCActionQueue | None = None
        self._model: InferenceModel | None = None

        # Discovered/inferred state
        self._chunk_size: int = 50
        self._max_action_dim: int = 32
        self._chunk_size_discovered: int = 0

        # Thread state
        self._obs_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._obs_slot: tuple[dict[str, Any], int, _WarmupSignal | None] | None = None
        self._warmup_signal: _WarmupSignal | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._death_cause: BaseException | None = None
        self._inference_count: int = 0
        # Incarnation counter bumped by start()/reset(). Each observation handed
        # to the worker carries the current value; if reset() lands while
        # inference is in flight, the worker's incarnation no longer matches
        # and the result is discarded instead of reaching the queue.
        self._incarnation: int = 0
        # Separate from _inference_count, which restarts each run. This gate
        # prevents RTC's own cold-start discard from re-arming. A caller may
        # still reset the model's latency callback at an incarnation boundary.
        self._lifetime_inferences: int = 0
        self._bus: _CallbackBus | None = None
        self._session_id: str = ""

    @property
    def chunk_size(self) -> int:
        """Discovered chunk size (from warmup or config)."""
        return self._chunk_size_discovered or self._chunk_size

    @property
    def queue_threshold(self) -> int:
        """Threshold below which a new chunk inference is requested.

        If not explicitly passed during initialization, is dynamically computed
        as ``execution_horizon + latency_delay_actions``, where
        ``latency_delay_actions`` is derived from the measured worst-case
        inference latency and robot control rate of the loop.
        """
        if self._queue_threshold_param is not None:
            return self._queue_threshold_param
        delay = self._latency_tracker.compute_delay(self._fps) if self._latency_tracker is not None else 0
        return self._execution_horizon + delay

    @property
    def inference_count(self) -> int:
        """Number of completed inference calls."""
        return self._inference_count

    def start(self, model: InferenceModel, action_queue: RTCActionQueue) -> None:  # type: ignore[override]
        """Bind model and queue, spawn background thread.

        Args:
            model: The inference model.
            action_queue: The RTC dual-track action queue.

        Raises:
            RuntimeError: If the previous worker is still inside the model after
                a short grace period. Running anyway would put two threads
                through one ``InferenceModel``, which is not synchronised.
        """  # noqa: DOC502 — raised by the delegated _await_previous_worker(), but callers see it here.
        # Wake direct warmup callers before waiting on or refusing an active
        # worker. Otherwise they can remain blocked until the 120 s timeout.
        with self._obs_lock:
            self._incarnation += 1
            self._obs_slot = None
            self._cancel_warmup_locked("RTCExecution warmup cancelled by restart")

        # Before model/queue bindings are changed, refuse to run two workers
        # through the same unsynchronised model.
        self._await_previous_worker()

        self._model = model
        self._rtc_queue = action_queue

        # 1. Infer chunk_size
        if self._chunk_size_param is not None:
            self._chunk_size = self._chunk_size_param
        else:
            rtc_config = model.manifest.model_extra.get("rtc", {}) if hasattr(model, "manifest") else {}
            if isinstance(rtc_config, dict) and "chunk_size" in rtc_config:
                self._chunk_size = int(rtc_config["chunk_size"])
            elif model.chunk_size > 1:
                self._chunk_size = model.chunk_size
            else:
                self._chunk_size = 50  # fallback default to Pi05 chunk size

        # 2. Infer max_action_dim
        if self._max_action_dim_param is not None:
            self._max_action_dim = self._max_action_dim_param
        else:
            rtc_config = model.manifest.model_extra.get("rtc", {}) if hasattr(model, "manifest") else {}
            if isinstance(rtc_config, dict) and "max_action_dim" in rtc_config:
                self._max_action_dim = int(rtc_config["max_action_dim"])
            elif (
                hasattr(model, "manifest")
                and model.manifest.hardware.robots
                and model.manifest.hardware.robots[0].action is not None
                and model.manifest.hardware.robots[0].action.shape
            ):
                self._max_action_dim = model.manifest.hardware.robots[0].action.shape[-1]
            else:
                self._max_action_dim = 32

        # 3. Automatically discover postprocessors from model if empty/not provided
        if not self._postprocessors and hasattr(model, "postprocessors") and model.postprocessors:
            logger.info("Moving postprocessors from InferenceModel to RTCExecution for async background execution")
            self._postprocessors = model.postprocessors
            model.postprocessors = []  # Clear from model so they aren't run twice

        # Fresh events rather than clearing the shared ones: a worker that
        # outlived stop() keeps its own set stop event, so it cannot be revived
        # by this start(), and its in-flight chunk is discarded.
        self._stop_event = threading.Event()
        self._inference_count = 0
        # A death from the previous run must not fail this one, and a stale
        # observation must not be inferred on as if it were current.
        self._death_cause = None
        with self._obs_lock:
            self._obs_slot = None
        self._thread = threading.Thread(
            target=self._rtc_loop,
            args=(self._stop_event,),
            name="rtc-inference",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "RTCExecution started (fps=%.1f, chunk=%d, horizon=%d, threshold=%d)",
            self._fps,
            self._chunk_size,
            self._execution_horizon,
            self.queue_threshold,
        )

    def warmup(self, sample_observation: dict[str, Any]) -> None:
        """Run one inference to seed the queue and discover chunk size.

        Blocks until the first chunk is produced by the background
        thread (or timeout).

        Raises:
            RuntimeError: If start() not called or thread dies during warmup.
            WorkerDiedError: If the RTC thread dies during warmup.
        """
        if self._model is None or self._rtc_queue is None:
            raise RuntimeError(_NOT_STARTED)

        signal = _WarmupSignal()
        with self._obs_lock:
            if self._warmup_signal is not None:
                msg = "RTCExecution warmup is already in progress"
                raise RuntimeError(msg)
            incarnation = self._incarnation
            self._warmup_signal = signal
            self._obs_slot = (deepcopy(sample_observation), incarnation, signal)

        # Wait for the first chunk with a generous timeout
        if not signal.event.wait(timeout=120.0):
            with self._obs_lock:
                if self._warmup_signal is signal:
                    self._warmup_signal = None
            if self._death_cause is not None:
                msg = f"RTC thread died during warmup: {self._death_cause}"
                raise WorkerDiedError(msg) from self._death_cause
            msg = "RTCExecution warmup timed out waiting for first chunk"
            raise RuntimeError(msg)

        with self._obs_lock:
            if self._warmup_signal is signal:
                self._warmup_signal = None
            error = signal.error
        if error is not None:
            raise error

        self._chunk_size_discovered = self._chunk_size
        logger.info("RTCExecution warmup complete — chunk_size=%d", self._chunk_size_discovered)

    def maybe_request(self, observation: dict[str, Any]) -> None:
        """Publish the given observation for the background thread.

        The background thread decides when to re-infer based on
        queue threshold. This just updates the observation slot.

        Raises:
            WorkerDiedError: If the inference thread has died.
        """
        if self._thread is not None and not self._thread.is_alive() and self._death_cause is not None:
            msg = f"RTC inference thread died: {self._death_cause}"
            raise WorkerDiedError(msg) from self._death_cause

        if self._rtc_queue is None:
            return
        if not self._rtc_queue.below_threshold(self.queue_threshold):
            return

        with self._obs_lock:
            snapshot = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in observation.items()}
            self._obs_slot = (snapshot, self._incarnation, None)

    def reset(self, *, reset_model: bool = True) -> None:
        """Invalidate pending RTC work and optionally wait for active inference.

        With ``reset_model=True``, waits for inference already inside the model.

        Raises:
            RuntimeError: If :meth:`start` has not been called.
        """
        if self._model is None:
            raise RuntimeError(_NOT_STARTED)
        with self._obs_lock:
            self._incarnation += 1
            self._obs_slot = None
            self._cancel_warmup_locked("RTCExecution warmup cancelled by reset")
        if reset_model:
            with self._model_lock:
                self._model.reset()

    def stop(self) -> None:
        """Signal the worker and join it, with a timeout.

        Best-effort by nature: a worker inside a blocking inference cannot be
        preempted. It is left to finish and discard its chunk, so this never
        raises — teardown must not fail. A worker that outlives the join keeps
        its reference here so the next :meth:`start` can refuse to run
        alongside it.
        """
        with self._obs_lock:
            self._incarnation += 1
            self._obs_slot = None
            self._cancel_warmup_locked("RTCExecution warmup cancelled because execution stopped")
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                logger.warning(
                    "RTC worker did not exit within %.1fs — still inside the model. "
                    "It will discard its chunk and exit on its own.",
                    _JOIN_TIMEOUT_S,
                )
            else:
                self._thread = None
            logger.info("RTCExecution stopped (%d inferences)", self._inference_count)

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
            "Previous RTC worker is still running — waiting up to %.1fs for it to exit.",
            _STRAGGLER_GRACE_S,
        )
        previous.join(timeout=_STRAGGLER_GRACE_S)
        if previous.is_alive():
            msg = (
                "Previous RTC worker is still inside the model after "
                f"{_JOIN_TIMEOUT_S + _STRAGGLER_GRACE_S:.1f}s. Refusing to start a second "
                "worker: InferenceModel is not safe for concurrent use. Wait for the "
                "inference to finish, or build a new execution with its own model."
            )
            raise RuntimeError(msg)

    def _rtc_loop(self, stop_event: threading.Event) -> None:
        """Background loop for one run: infer chunks and merge into queue.

        Takes its own events rather than reading ``self``, so a worker that
        outlives ``stop()`` keeps observing its own set stop event even after a
        later ``start()`` has installed fresh ones.
        """
        assert self._model is not None  # noqa: S101
        assert self._rtc_queue is not None  # noqa: S101
        consecutive_errors = 0

        while not stop_event.is_set():
            # Only re-infer when queue is running low
            if not self._rtc_queue.below_threshold(self.queue_threshold):
                time.sleep(_IDLE_SLEEP_S)
                continue

            # Snapshot observation and consume the slot so a stale sample
            # (e.g. the warmup observation) is never reused for a later refill.
            with self._obs_lock:
                request = self._obs_slot
                self._obs_slot = None
            if request is None:
                time.sleep(_IDLE_SLEEP_S)
                continue
            inputs, incarnation, signal = request
            inputs = deepcopy(inputs)

            # Build RTC-specific inputs
            inputs = self._inject_rtc_inputs(inputs)

            # Snapshot cursor before inference
            action_index_before = self._rtc_queue.get_action_index()

            # Run inference (callbacks fire inside model.__call__)
            try:
                t0 = time.perf_counter()
                with self._model_lock:
                    with self._obs_lock:
                        if incarnation != self._incarnation:
                            self._cancel_signal_locked(signal, "RTCExecution warmup cancelled by reset")
                            continue
                    outputs = self._model(inputs)
                    elapsed = time.perf_counter() - t0
                consecutive_errors = 0
            except Exception:
                with self._obs_lock:
                    self._cancel_signal_locked(signal, "RTCExecution warmup inference failed")
                consecutive_errors += 1
                logger.exception(
                    "RTC inference error (%d/%d)",
                    consecutive_errors,
                    _MAX_CONSECUTIVE_ERRORS,
                )
                if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    self._death_cause = RuntimeError("Too many consecutive RTC errors")
                    logger.exception("RTC thread shutting down after %d consecutive errors", consecutive_errors)
                    return
                time.sleep(_ERROR_RETRY_DELAY_S)
                continue

            if stop_event.is_set():
                # This run ended while the inference was in flight. The chunk
                # describes an observation from a finished session, so it must
                # not reach a later run's queue.
                with self._obs_lock:
                    self._cancel_signal_locked(signal, "RTCExecution warmup cancelled because execution stopped")
                return

            processed_actions = self._accept_result(
                outputs,
                incarnation=incarnation,
                signal=signal,
                elapsed=elapsed,
                action_index_before=action_index_before,
            )
            if processed_actions is None:
                continue

            # Emit inference event so callbacks (e.g. RerunCallback) can
            # plot predicted future actions.
            if self._bus:
                from physicalai.runtime.events import InferenceEvent  # noqa: PLC0415

                self._bus.emit_inference(
                    InferenceEvent(
                        session_id=self._session_id,
                        timestamp=time.time(),
                        latency_s=elapsed,
                        offset=0,
                        chunk=processed_actions,
                    )
                )

            logger.debug(
                "RTC chunk: latency=%.3fs remaining=%d",
                elapsed,
                self._rtc_queue.remaining,
            )

    def _accept_result(
        self,
        outputs: dict[str, np.ndarray],
        *,
        incarnation: int,
        signal: _WarmupSignal | None,
        elapsed: float,
        action_index_before: int,
    ) -> np.ndarray | None:
        """Merge a result only if it belongs to the current incarnation.

        Returns:
            Processed actions, or ``None`` when the incarnation has changed.
        """
        assert self._rtc_queue is not None  # noqa: S101
        raw_actions = outputs["action"]
        if raw_actions.ndim == 3:  # noqa: PLR2004
            raw_actions = raw_actions[0]
        processed_actions = self._postprocess(raw_actions)

        with self._obs_lock:
            if incarnation != self._incarnation:
                self._cancel_signal_locked(signal, "RTCExecution warmup cancelled by reset")
                return None

            self._inference_count += 1
            self._lifetime_inferences += 1
            if self._lifetime_inferences <= self._warmup_inferences and self._latency_tracker is not None:
                self._latency_tracker.on_reset()
                logger.info(
                    "Warmup inference %d/%d complete (%.2fs) — latency tracker reset",
                    self._lifetime_inferences,
                    self._warmup_inferences,
                    elapsed,
                )

            self._rtc_queue.merge(
                raw_actions,
                processed_actions,
                action_index_before_inference=action_index_before,
            )
            if signal is not None:
                signal.event.set()
            return processed_actions

    def _cancel_warmup_locked(self, message: str) -> None:
        signal = self._warmup_signal
        if signal is None:
            return
        self._cancel_signal_locked(signal, message)
        self._warmup_signal = None

    @staticmethod
    def _cancel_signal_locked(signal: _WarmupSignal | None, message: str) -> None:
        if signal is None or signal.event.is_set():
            return
        signal.error = RuntimeError(message)
        signal.event.set()

    def _inject_rtc_inputs(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Add RTC-specific model inputs.

        Returns:
            Updated inputs dict with RTC keys added.
        """
        assert self._rtc_queue is not None  # noqa: S101

        # prev_chunk_left_over from queue
        prev_chunk = self._rtc_queue.get_left_over()
        if prev_chunk is None:
            prev_chunk_padded = np.zeros(
                (1, self._chunk_size, self._max_action_dim),
                dtype=np.float32,
            )
            # Suppress correction on the first step since there's no real previous trajectory
            max_guidance_weight = 0.0
            execution_horizon = 0
        else:
            remaining = prev_chunk.shape[0]
            out_dim = prev_chunk.shape[-1]

            # Pad action dim to model's max_action_dim if needed
            if out_dim < self._max_action_dim:
                prev_chunk = np.pad(
                    prev_chunk,
                    ((0, 0), (0, self._max_action_dim - out_dim)),
                )

            # Reshape to (1, remaining, max_action_dim) and pad time to chunk_size
            prev_chunk_padded = prev_chunk.reshape(1, remaining, self._max_action_dim)
            pad_len = self._chunk_size - remaining
            if pad_len > 0:
                prev_chunk_padded = np.pad(prev_chunk_padded, ((0, 0), (0, pad_len), (0, 0)))

            max_guidance_weight = self._max_guidance_weight
            execution_horizon = self._execution_horizon

        # Compute delay from latency tracker
        delay = self._latency_tracker.compute_delay(self._fps) if self._latency_tracker is not None else 0

        inputs["prev_chunk_left_over"] = prev_chunk_padded
        inputs["inference_delay"] = np.int64(delay)
        inputs["max_guidance_weight"] = np.float32(max_guidance_weight)
        inputs["execution_horizon"] = np.int64(execution_horizon)

        return inputs

    def _postprocess(self, actions: np.ndarray) -> np.ndarray:
        """Apply postprocessors (denormalization) to raw actions.

        Args:
            actions: Shape ``(chunk_size, action_dim)``.

        Returns:
            Postprocessed actions, same shape.
        """
        if not self._postprocessors:
            return actions.copy()

        outputs: dict[str, Any] = {"action": actions}
        for pp in self._postprocessors:
            outputs = pp(outputs)
        return outputs["action"]
