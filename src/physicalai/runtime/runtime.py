# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PolicyRuntime — runs a trained policy on robot hardware."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from physicalai.runtime._action_queue import ActionQueue  # noqa: PLC2701
from physicalai.runtime.execution import Execution, WorkerDiedError
from physicalai.runtime.smoothers import LerpSmoother

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from physicalai.capture.camera import Camera
    from physicalai.capture.frame import Frame
    from physicalai.inference.model import InferenceModel
    from physicalai.robot.interface import Robot, RobotObservation

logger = logging.getLogger(__name__)

_DEFAULT_LERP_FRAMES = 5


def default_observation_to_input(
    robot_obs: RobotObservation,
    camera_frames: dict[str, Frame],
) -> dict[str, Any]:
    """Convert robot observation and camera frames to model input dict.

    Maps:
        - ``robot_obs.joint_positions`` → ``"state"`` (as batch dim)
        - ``frame.data`` per camera → ``"images.{name}"``

    Returns:
        Model input dictionary.
    """
    model_input: dict[str, Any] = {}

    if robot_obs.joint_positions is not None:
        model_input["state"] = np.array([robot_obs.joint_positions], dtype=np.float32)

    for name, frame in camera_frames.items():
        model_input[f"images.{name}"] = frame.data

    return model_input


class RuntimeCallback(Protocol):
    """Optional hook points in the PolicyRuntime control loop."""

    def before_send_action(self, *, action: np.ndarray, step: int) -> np.ndarray | None:
        """Called before sending action. Return modified action or None."""
        ...

    def on_action_sent(self, *, action: np.ndarray, step: int) -> None:
        """Called after action is sent to robot."""
        ...

    def on_hold(self, *, step: int, holds: int) -> None:
        """Called when action queue is empty and robot holds last position."""
        ...


@dataclass(frozen=True)
class RunStats:
    """Statistics from a PolicyRuntime.run() session."""

    steps: int
    total_pops: int
    total_holds: int
    inference_count: int
    transient_errors: int = 0
    stale_obs_ticks: int = 0


class PolicyRuntime:
    """Runs a policy on robot hardware.

    Loop: observe → maybe_request → pop → send → sleep.
    Robot and cameras must be connected before run(). Caller owns lifecycle.
    """

    def __init__(  # noqa: D107
        self,
        robot: Robot,
        model: InferenceModel,
        execution: Execution,
        fps: float,
        cameras: Mapping[str, Camera] | None = None,
        action_queue: ActionQueue | None = None,
        obs_to_input: Callable[[RobotObservation, dict[str, Frame]], dict[str, Any]] | None = None,
        callbacks: Sequence[RuntimeCallback] = (),
    ) -> None:
        if fps <= 0:
            msg = f"fps must be positive, got {fps}"
            raise ValueError(msg)
        self._robot = robot
        self._model = model
        self._execution = execution
        self._fps = fps
        self._cameras: Mapping[str, Camera] = cameras or {}
        self._action_queue = action_queue or ActionQueue(smoother=LerpSmoother(duration_frames=_DEFAULT_LERP_FRAMES))
        self._obs_to_input = obs_to_input or default_observation_to_input
        self._callbacks = list(callbacks)

    def run(self, *, duration_s: float | None = None) -> RunStats:
        """Run the control loop.

        Args:
            duration_s: Maximum duration in seconds. None runs indefinitely.

        Returns:
            Statistics from the run session.

        Raises:
            WorkerDiedError: If the inference worker thread dies.
        """
        self._execution.start(self._model, self._action_queue)
        sample_obs = self._build_model_input()
        self._execution.warmup(sample_obs)

        goal_time = 1.0 / self._fps
        step = 0
        last_action: np.ndarray | None = None

        try:
            while True:
                if duration_s is not None and step * goal_time >= duration_s:
                    break

                loop_start = time.perf_counter()

                obs = self._build_model_input()
                self._execution.maybe_request(obs)

                action = self._action_queue.pop()
                if action is not None:
                    last_action = action
                else:
                    action = last_action
                    holds = self._action_queue.consecutive_holds
                    if holds == 1:
                        logger.warning("Queue empty — holding position")
                    elif self._fps > 0 and holds % int(self._fps) == 0:
                        logger.warning(
                            "Queue starvation: %d consecutive holds (%.1fs)",
                            holds,
                            holds / self._fps,
                        )
                    self._invoke_callback("on_hold", step=step, holds=holds)

                if action is None:
                    logger.error("No action available (warmup may have failed)")
                    step += 1
                    continue

                modified = self._invoke_callback("before_send_action", action=action, step=step)
                if modified is not None:
                    action = modified

                self._robot.send_action(action)
                self._invoke_callback("on_action_sent", action=action, step=step)

                elapsed = time.perf_counter() - loop_start
                sleep_time = goal_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                step += 1

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except WorkerDiedError:
            logger.exception("Worker died during runtime")
            raise
        finally:
            self._shutdown(step)

        return RunStats(
            steps=step,
            total_pops=self._action_queue.total_pops,
            total_holds=self._action_queue.total_holds,
            inference_count=getattr(self._execution, "inference_count", 0),
        )

    def _build_model_input(self) -> dict[str, Any]:
        robot_obs = self._robot.get_observation()
        camera_frames = {name: cam.read_latest() for name, cam in self._cameras.items()}
        return self._obs_to_input(robot_obs, camera_frames)

    def _shutdown(self, step: int) -> None:
        self._execution.stop()

        remaining = self._action_queue.remaining
        drain_limit = min(remaining, int(self._fps))
        for _ in range(drain_limit):
            action = self._action_queue.pop()
            if action is not None:
                self._robot.send_action(action)
                time.sleep(1.0 / self._fps)

        logger.info(
            "Shutdown complete — %d steps, %d pops, %d holds",
            step,
            self._action_queue.total_pops,
            self._action_queue.total_holds,
        )

    def _invoke_callback(self, method: str, **kwargs: Any) -> Any:  # noqa: ANN401
        result = None
        for cb in self._callbacks:
            fn = getattr(cb, method, None)
            if fn is not None:
                try:
                    result = fn(**kwargs)
                except Exception:
                    logger.exception("Callback %s.%s raised", type(cb).__name__, method)
        return result
