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
from typing import TYPE_CHECKING, Any

import numpy as np

from physicalai.runtime.execution import Execution, WorkerDiedError

if TYPE_CHECKING:
    from physicalai.inference.callbacks.rtc_latency import RTCLatencyTracker
    from physicalai.inference.model import InferenceModel
    from physicalai.inference.postprocessors.base import Postprocessor
    from physicalai.runtime._rtc_action_queue import RTCActionQueue

logger = logging.getLogger(__name__)

_NOT_STARTED = "start() must be called before this method"
_IDLE_SLEEP_S: float = 0.005
_ERROR_RETRY_DELAY_S: float = 0.5
_MAX_CONSECUTIVE_ERRORS: int = 10
_JOIN_TIMEOUT_S: float = 5.0


class RTCExecution(Execution):
    """Async RTC execution strategy with background inference thread.

    The background thread continuously predicts action chunks and
    merges them into an :class:`RTCActionQueue`. The main thread pops
    one action per tick — never blocking on inference.

    RTC-specific inputs injected before each inference call:
    - ``noise``: random noise for denoising (shape: 1 x chunk x action_dim)
    - ``prev_chunk_left_over``: unconsumed tail (shape: 1 x chunk x action_dim)
    - ``inference_delay``: integer derived from measured latency
    - ``max_guidance_weight``: classifier-free guidance weight
    - ``execution_horizon``: number of fresh actions per chunk

    Args:
        chunk_size: Number of actions per model output chunk. If None, is
            automatically inferred from the model's manifest or model metadata.
        execution_horizon: Fresh actions to execute per chunk (range: typically
            5 to 15, default 10). Low values speed up response but consume more
            CPU, high values tolerate higher latency spikes.
        fps: Robot control frequency in Hz.
        max_action_dim: Model's internal action dimension (for noise/padding).
            If None, is automatically inferred from the model's manifest or
            defaulted to 32.
        max_guidance_weight: Classifier-free guidance scale weight (range:
            typically 1.0 to 15.0, default 10.0) injected into diffusion models.
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
        execution_horizon: int = 10,
        fps: float = 30.0,
        max_action_dim: int | None = None,
        max_guidance_weight: float = 10.0,
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
        self._obs_slot: dict[str, np.ndarray] | None = None
        self._stop_event = threading.Event()
        self._first_chunk_ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._death_cause: BaseException | None = None
        self._inference_count: int = 0

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
        """
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

        self._stop_event.clear()
        self._first_chunk_ready.clear()
        self._thread = threading.Thread(
            target=self._rtc_loop,
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

    def warmup(self, sample_observation: dict[str, np.ndarray]) -> None:
        """Run one inference to seed the queue and discover chunk size.

        Blocks until the first chunk is produced by the background
        thread (or timeout).

        Raises:
            RuntimeError: If start() not called or thread dies during warmup.
            WorkerDiedError: If the RTC thread dies during warmup.
        """
        if self._model is None or self._rtc_queue is None:
            raise RuntimeError(_NOT_STARTED)

        # Publish the sample observation for the background thread
        with self._obs_lock:
            self._obs_slot = deepcopy(sample_observation)

        # Wait for the first chunk with a generous timeout
        if not self._first_chunk_ready.wait(timeout=120.0):
            if self._death_cause is not None:
                msg = f"RTC thread died during warmup: {self._death_cause}"
                raise WorkerDiedError(msg) from self._death_cause
            msg = "RTCExecution warmup timed out waiting for first chunk"
            raise RuntimeError(msg)

        self._chunk_size_discovered = self._chunk_size
        logger.info("RTCExecution warmup complete — chunk_size=%d", self._chunk_size_discovered)

    def maybe_request(self, observation: dict[str, np.ndarray]) -> None:
        """Publish latest observation for the background thread.

        The background thread decides when to re-infer based on
        queue threshold. This just updates the observation slot.

        Raises:
            WorkerDiedError: If the inference thread has died.
        """
        if self._thread is not None and not self._thread.is_alive() and self._death_cause is not None:
            msg = f"RTC inference thread died: {self._death_cause}"
            raise WorkerDiedError(msg) from self._death_cause

        with self._obs_lock:
            self._obs_slot = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in observation.items()}

    def stop(self) -> None:
        """Signal shutdown and join the background thread."""
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                logger.warning("RTC thread did not join within %.1fs", _JOIN_TIMEOUT_S)
            self._thread = None
            logger.info("RTCExecution stopped (%d inferences)", self._inference_count)

    def _rtc_loop(self) -> None:
        """Background loop: infer chunks and merge into queue."""
        assert self._model is not None  # noqa: S101
        assert self._rtc_queue is not None  # noqa: S101
        consecutive_errors = 0

        while not self._stop_event.is_set():
            # Only re-infer when queue is running low
            if not self._rtc_queue.below_threshold(self.queue_threshold):
                time.sleep(_IDLE_SLEEP_S)
                continue

            # Snapshot observation
            with self._obs_lock:
                if self._obs_slot is None:
                    time.sleep(_IDLE_SLEEP_S)
                    continue
                inputs = deepcopy(self._obs_slot)

            # Build RTC-specific inputs
            inputs = self._inject_rtc_inputs(inputs)

            # Snapshot cursor before inference
            action_index_before = self._rtc_queue.get_action_index()

            # Run inference (callbacks fire inside model.__call__)
            try:
                t0 = time.perf_counter()
                outputs = self._model(inputs)
                elapsed = time.perf_counter() - t0
                consecutive_errors = 0
            except Exception:
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

            self._inference_count += 1

            # Reset latency tracker after warmup inferences to discard
            # compilation overhead (e.g. OpenVINO first-run latency).
            if self._inference_count <= self._warmup_inferences and self._latency_tracker is not None:
                self._latency_tracker.on_reset()
                logger.info(
                    "Warmup inference %d/%d complete (%.2fs) — latency tracker reset",
                    self._inference_count,
                    self._warmup_inferences,
                    elapsed,
                )

            # Extract raw actions: (1, chunk_size, action_dim) → (chunk_size, action_dim)
            raw_actions = outputs["action"]
            if raw_actions.ndim == 3:  # noqa: PLR2004
                raw_actions = raw_actions[0]

            # Postprocess (denormalize) for robot
            processed_actions = self._postprocess(raw_actions)

            # Merge into dual-track queue — trim is based on actual
            # actions consumed (cursor movement) during inference, NOT
            # wall-clock time.  For the first chunk nothing was consumed
            # so trim=0 and the full chunk is kept.
            self._rtc_queue.merge(
                raw_actions,
                processed_actions,
                action_index_before_inference=action_index_before,
            )
            self._first_chunk_ready.set()

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
