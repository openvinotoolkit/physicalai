# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action source abstraction and policy/teleop implementations."""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from physicalai.inference.constants import IMAGES, STATE, TASK
from physicalai.runtime._action_queue import ChunkedActionQueue  # noqa: PLC2701
from physicalai.runtime.events import MetricsEvent
from physicalai.runtime.smoothers import LerpSmoother

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from physicalai.capture.frame import Frame
    from physicalai.inference.model import InferenceModel
    from physicalai.robot.interface import Robot, RobotObservation
    from physicalai.runtime._callback_bus import _CallbackBus
    from physicalai.runtime.execution import Execution
    from physicalai.runtime.runtime import ActionQueue

_DEFAULT_LERP_FRAMES = 5


class ActionSource(Protocol):
    """The minimum a developer must implement to plug an action source into RobotRuntime.

    Three required methods, nothing optional — no capability protocols, no
    ``isinstance`` anywhere in the runtime.
    """

    def connect(self, *, bus: _CallbackBus, session_id: str) -> None:
        """Set up resources (spawn threads, connect a leader device, etc.).

        Called fresh every ``run()``, which is exactly when the runtime
        generates a new ``session_id`` — construction-time injection would
        miss that.
        """
        ...

    def update(self, robot_state: RobotObservation, camera_frames: Mapping[str, Frame], step: int) -> np.ndarray:
        """Return the action to send this tick.

        Always returns a sendable action — no ``None`` sentinel. What to do
        when there is nothing new to decide (repeat the last action, go to a
        safe pose, whatever) is entirely this action source's own call, made
        internally. If it truly cannot produce anything, it raises.

        Returns:
            Action vector to send to the robot this tick.
        """
        ...

    def disconnect(self) -> None:
        """Tear down only (stop threads, release a leader device).

        Returns nothing — any queued-but-unsent actions are discarded, not
        flushed. The action source never receives a robot reference.
        """
        ...


class PolicySource:
    """Action source adapting a model + execution + action-queue policy pipeline."""

    def __init__(
        self,
        model: InferenceModel,
        execution: Execution,
        action_queue: ActionQueue | None = None,
        *,
        task: str | None = None,
    ) -> None:
        """Initialize a policy-backed action source."""
        self._model = model
        self._execution = execution
        self._action_queue = action_queue or ChunkedActionQueue(
            smoother=LerpSmoother(duration_frames=_DEFAULT_LERP_FRAMES)
        )
        self._task = task
        self._last: np.ndarray | None = None
        self._warmed_up = False
        self._bus: _CallbackBus | None = None
        self._session_id: str = ""

    @property
    def action_queue(self) -> ActionQueue:
        """Underlying action queue, exposed for end-of-run stats access.

        Returns:
            Action queue used by this action source.
        """
        return self._action_queue

    def connect(self, *, bus: _CallbackBus, session_id: str) -> None:
        """Inject bus/session into execution and start it."""
        self._bus = bus
        self._session_id = session_id
        self._execution.set_bus(bus, session_id)
        self._execution.start(self._model, self._action_queue)

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
        model_input = self._to_model_input(robot_state, camera_frames)

        if not self._warmed_up:
            self._execution.warmup(model_input)
            self._warmed_up = True

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
        self._execution.stop()

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
        return self._to_model_input(robot_obs, camera_frames)


class TeleopSource:
    """Action source that reads a leader arm and writes to the follower.

    The action source is the leader device, not the follower's observation or
    any inference model. Both ``robot_state``/``camera_frames`` are ignored —
    a teleop tick with no recording attached performs zero extra reads beyond
    what the runtime already does for telemetry.

    Args:
        leader: The leader robot (same ``Robot`` protocol; must support
            ``get_observation()``).
        to_action: Optional callable mapping a ``RobotObservation`` from the
            leader to an action array for the follower. Defaults to
            ``obs.joint_positions`` (identity for same-morphology leader/follower).
    """

    def __init__(  # noqa: D107
        self,
        leader: Robot,
        *,
        to_action: Callable[[RobotObservation], np.ndarray] | None = None,
    ) -> None:
        self._leader = leader
        self._to_action = to_action or (lambda obs: obs.joint_positions)
        self._leader_owned = False

    def connect(self, *, bus: _CallbackBus, session_id: str) -> None:  # noqa: ARG002
        """Connect leader if not already connected."""
        if not self._leader.is_connected():
            self._leader.connect()
            self._leader_owned = True

    def update(self, robot_state: RobotObservation, camera_frames: Mapping[str, Frame], step: int) -> np.ndarray:  # noqa: ARG002
        """Read the leader arm and return the action for the follower.

        Returns:
            Action array for the follower robot.
        """
        return self._to_action(self._leader.get_observation())

    def disconnect(self) -> None:
        """Disconnect leader if we connected it."""
        if self._leader_owned:
            with contextlib.suppress(Exception):
                self._leader.disconnect()
