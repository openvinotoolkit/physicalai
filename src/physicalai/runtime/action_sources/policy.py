# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Policy-backed action source: model + execution + action queue."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from physicalai.config import export_config
from physicalai.inference.constants import IMAGES, STATE, TASK
from physicalai.runtime.action_sources.base import ActionSource
from physicalai.runtime.events import MetricsEvent
from physicalai.runtime.execution.base import Execution
from physicalai.runtime.execution.queue import ChunkedActionQueue
from physicalai.runtime.execution.sync import SyncExecution
from physicalai.runtime.smoothers import LerpSmoother

if TYPE_CHECKING:
    from collections.abc import Mapping

    from physicalai.capture.frame import Frame
    from physicalai.inference.model import InferenceModel
    from physicalai.robot.interface import RobotObservation
    from physicalai.runtime._callback_bus import _CallbackBus
    from physicalai.runtime.execution.queue import ActionQueue

_DEFAULT_LERP_FRAMES = 5


@export_config(class_path="physicalai.runtime.PolicySource")
class PolicySource(ActionSource):
    """Action source adapting a model + execution + action-queue policy pipeline."""

    def __init__(
        self,
        model: InferenceModel,
        execution: Execution | None = None,
        action_queue: ActionQueue | None = None,
        *,
        task: str | None = None,
    ) -> None:
        """Initialize a policy-backed action source."""
        self._model = model
        self._execution = execution or SyncExecution()
        self._action_queue = action_queue or ChunkedActionQueue(
            smoother=LerpSmoother(duration_frames=_DEFAULT_LERP_FRAMES)
        )
        self._task = task
        self._last: np.ndarray | None = None
        self._warmed_up = False
        self._bus: _CallbackBus | None = None
        self._session_id: str = ""
        self._connected = False
        self._state_lock = threading.RLock()

    def set_task(self, task: str | None) -> None:
        """Update the task string used for the *next* inference request."""
        with self._state_lock:
            self._task = task

    @property
    def action_queue(self) -> ActionQueue:
        """Underlying action queue, exposed for end-of-run stats access.

        Returns:
            Action queue used by this action source.
        """
        return self._action_queue

    def connect(self, *, bus: _CallbackBus, session_id: str) -> None:
        """Inject bus/session into execution and start it."""
        with self._state_lock:
            if self._connected:
                return
            self._bus = bus
            self._session_id = session_id
            self._execution.set_bus(bus, session_id)
            self._execution.start(self._model, self._action_queue)
            try:
                if self._supports_reset:
                    self._execution.reset(reset_model=True)
                self._action_queue.reset()
                self._last = None
                self._warmed_up = False
            except BaseException:
                self._execution.stop()
                raise
            self._connected = True

    def reset(self, *, reset_model: bool = True) -> None:
        """Discard queued actions and per-episode state; the next update re-seeds.

        The execution worker remains running. Pending and in-flight results from
        the previous episode are invalidated before the queue is cleared. This
        may wait for inference already inside the model to finish.

        Args:
            reset_model: Also reset state held by the inference model.

        Raises:
            RuntimeError: If the source is not connected or its execution
                strategy does not support safe episode reset.
        """
        with self._state_lock:
            if not self._connected:
                msg = "PolicySource.reset() requires connect() first"
                raise RuntimeError(msg)
            if not self._supports_reset:
                msg = f"{type(self._execution).__name__} does not support safe episode reset"
                raise RuntimeError(msg)
            try:
                self._execution.reset(reset_model=reset_model)
            finally:
                self._action_queue.reset()
                self._last = None
                self._warmed_up = False

    def warmup(self, observation: dict[str, Any]) -> None:
        """Seed the action queue so the next :meth:`update` does not block.

        Args:
            observation: Model-ready input, such as the result of
                :meth:`to_model_input`.

        Raises:
            RuntimeError: If the source is not connected or warmup is cancelled.
        """
        with self._state_lock:
            if not self._connected:
                msg = "PolicySource.warmup() requires connect() first"
                raise RuntimeError(msg)
            if self._warmed_up:
                return
            self._execution.warmup(observation)
            self._warmed_up = True

    @property
    def _supports_reset(self) -> bool:
        reset = getattr(type(self._execution), "reset", None)
        return reset is not None and reset is not Execution.reset

    def update(self, robot_state: RobotObservation, camera_frames: Mapping[str, Frame], step: int) -> np.ndarray:
        """Maybe request inference and return the next action.

        On the first call, seeds the queue via ``execution.warmup()`` behind a
        private not-yet-seeded flag — there is no runtime-level warmup step.
        When the queue has nothing new, returns the last action sent (this
        action source's own hold decision); raises if none was ever produced.

        Returns:
            Action to send this tick.

        Raises:
            RuntimeError: If the queue is empty and no action has ever been produced.
        """
        with self._state_lock:
            model_input = self._to_model_input(robot_state, camera_frames)

            if not self._warmed_up:
                self.warmup(model_input)

            self._execution.maybe_request(model_input)

            action = self._action_queue.pop()
            if action is None:
                if self._last is None:
                    msg = "No action available and none produced yet (warmup may have failed)"
                    raise RuntimeError(msg)
                action = self._last
            else:
                self._last = action

            if self._bus is not None:
                self._bus.emit_metrics(
                    MetricsEvent(
                        session_id=self._session_id,
                        step=step,
                        timestamp=time.time(),
                        values={"queue_remaining": float(self._action_queue.remaining)},
                    )
                )

            return action

    def disconnect(self) -> None:
        """Stop execution — no drain, queued actions are discarded."""
        with self._state_lock:
            if not self._connected:
                return
            try:
                self._execution.stop()
            finally:
                self._connected = False

    def _to_model_input(self, robot_obs: RobotObservation, camera_frames: Mapping[str, Frame]) -> dict[str, Any]:
        """Assemble model input dict from observation and camera frames.

        Returns:
            Dictionary ready for model inference.
        """
        model_input: dict[str, Any] = {STATE: np.array([robot_obs.state], dtype=np.float32)}
        image_inputs: dict[str, np.ndarray] = {}
        # Merge robot-embedded images and external cameras
        if robot_obs.images:
            for name, frame in robot_obs.images.items():
                image_inputs[name] = frame.data[np.newaxis]
        for name, frame in camera_frames.items():
            image_inputs[name] = frame.data[np.newaxis]

        if len(image_inputs) > 1:
            for name, data in image_inputs.items():
                model_input[f"{IMAGES}.{name}"] = data
        elif len(image_inputs) == 1:
            model_input[IMAGES] = next(iter(image_inputs.values()))

        if self._task is not None:
            model_input[TASK] = [self._task]
        return model_input

    def to_model_input(self, robot_obs: RobotObservation, camera_frames: Mapping[str, Frame]) -> dict[str, Any]:
        """Public wrapper for model-input conversion (used by callers/tests).

        Returns:
            Dictionary ready for model inference.
        """
        with self._state_lock:
            return self._to_model_input(robot_obs, camera_frames)
