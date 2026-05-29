# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PolicyRuntime — runs a trained policy on robot hardware."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

import numpy as np

from physicalai.capture.errors import CaptureError
from physicalai.runtime._action_queue import ChunkedActionQueue  # noqa: PLC2701
from physicalai.runtime._callback_bus import _CallbackBus  # noqa: PLC2701
from physicalai.runtime.events import LifecycleEvent, TickEvent
from physicalai.runtime.execution import Execution, WorkerDiedError
from physicalai.runtime.smoothers import LerpSmoother

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from physicalai.capture.camera import Camera
    from physicalai.capture.frame import Frame
    from physicalai.inference.model import InferenceModel
    from physicalai.robot.interface import Robot, RobotObservation

logger = logging.getLogger(__name__)

_DEFAULT_LERP_FRAMES = 5
_MAX_OBS_RETRIES = 3
_MAX_SEND_RETRIES = 2
_RETRY_BACKOFF_S = 0.001
_WARMUP_RETRIES = 5
_WARMUP_BACKOFF_S = 1.0
_GOAL_TIME_TICKS = 3


@runtime_checkable
class ActionQueue(Protocol):
    """Protocol for a thread-safe action queue."""

    def pop(self) -> np.ndarray | None:
        """Pop the next action.

        Returns:
            Single action vector, or None if empty.
        """
        ...

    @property
    def remaining(self) -> int:
        """Number of unconsumed actions in the queue."""
        ...

    @property
    def consecutive_holds(self) -> int:
        """Number of consecutive holds (resets on successful pop)."""
        ...

    @property
    def total_holds(self) -> int:
        """Total number of hold events (pop on empty queue)."""
        ...

    @property
    def total_pops(self) -> int:
        """Total number of actions popped."""
        ...

    def below_threshold(self, threshold: int) -> bool:
        """Check if remaining actions are below threshold."""
        ...

    def clear(self) -> None:
        """Clear all state from the queue."""
        ...

    def push_chunk(self, chunk: np.ndarray, offset: int = 0) -> None:
        """Push an action chunk into the queue."""
        ...

    def reset(self) -> None:
        """Clear queue and reset all counters for a fresh session."""
        ...


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


class LowPassFilterCallback:
    """Stateful low-pass filter (Exponential Moving Average) callback for smooth actions.

    Filters outgoing multidimensional joint positions/actions using a simple
    discrete one-pole IIR filter (exponential moving average):
        y_t = alpha * x_t + (1 - alpha) * y_{t-1}

    Args:
        alpha: Smoothing factor in range (0, 1]. A lower value introduces
            more smoothing (heavy low-pass filter), whereas 1.0 is a no-op.
    """

    def __init__(self, alpha: float = 0.5) -> None:  # noqa: D107
        if not (0.0 < alpha <= 1.0):
            msg = f"alpha must be in (0, 1], got {alpha}"
            raise ValueError(msg)
        self.alpha = alpha
        self._last_action: np.ndarray | None = None

    def before_send_action(self, *, action: np.ndarray, step: int) -> np.ndarray:  # noqa: ARG002
        """Filter target action vector using previous action state.

        Args:
            action: The target raw/unfiltered joint configuration.
            step: The iteration step index in the control loop.

        Returns:
            The smoothed/filtered action target configuration.
        """
        if self._last_action is None or self._last_action.shape != action.shape:
            # First tick or shape mismatch: initialize filter state to current action
            self._last_action = action.copy()
            return action

        # Apply low-pass recursive formula
        filtered_action = self.alpha * action + (1.0 - self.alpha) * self._last_action
        self._last_action = filtered_action.copy()
        return filtered_action

    def on_action_sent(self, *, action: np.ndarray, step: int) -> None:
        """No-op."""

    def on_hold(self, *, step: int, holds: int) -> None:
        """No-op."""


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

    Supports context manager for safe lifecycle management::

        with PolicyRuntime(robot=robot, model=model, ...) as runtime:
            stats = runtime.run(duration_s=60.0)
    """

    def __init__(  # noqa: D107
        self,
        robot: Robot,
        model: InferenceModel,
        execution: Execution,
        fps: float,
        cameras: Mapping[str, Camera] | None = None,
        action_queue: ActionQueue | None = None,
        callbacks: Sequence[Any] = (),
        task: str | None = None,
    ) -> None:
        if fps <= 0:
            msg = f"fps must be positive, got {fps}"
            raise ValueError(msg)
        self._robot = robot
        self._model = model
        self._execution = execution
        self._fps = fps
        self._cameras: Mapping[str, Camera] = cameras or {}
        self._action_queue = action_queue or ChunkedActionQueue(
            smoother=LerpSmoother(duration_frames=_DEFAULT_LERP_FRAMES)
        )
        self._bus = _CallbackBus(callbacks)
        self._goal_time = (1.0 / fps) * _GOAL_TIME_TICKS
        self._task = task
        self._connected = False
        self._last_robot_obs: RobotObservation | None = None
        self._last_camera_frames: dict[str, Frame] = {}
        self._consecutive_error_ticks: int = 0
        self._max_consecutive_error_ticks: int = int(3 * fps)
        self._stale_obs_ticks: int = 0
        self._transient_errors: int = 0
        self._session_id: str = ""

    @property
    def robot(self) -> Robot:
        """The robot instance managed by this runtime."""
        return self._robot

    @property
    def cameras(self) -> Mapping[str, Camera]:
        """Camera instances managed by this runtime, keyed by name."""
        return self._cameras

    def connect(self) -> None:
        """Connect robot and cameras.

        Connects robot first, then cameras in dict order. On failure,
        disconnects everything already connected and re-raises.

        Idempotent — calling on an already-connected runtime is a no-op.
        """
        if self._connected:
            logger.debug("connect() called but already connected — no-op")
            return

        self._robot.connect()
        connected_cameras: list[str] = []
        try:
            for name, cam in self._cameras.items():
                cam.connect()
                connected_cameras.append(name)
        except Exception:
            for cam_name in connected_cameras:
                try:
                    self._cameras[cam_name].disconnect()
                except Exception:
                    logger.warning("Failed to disconnect camera '%s' during rollback", cam_name, exc_info=True)
            try:
                self._robot.disconnect()
            except Exception:
                logger.warning("Failed to disconnect robot during rollback", exc_info=True)
            raise

        self._connected = True

    def disconnect(self) -> None:
        """Disconnect cameras then robot. Never raises.

        Idempotent — calling on an already-disconnected runtime is a no-op.
        """
        if not self._connected:
            return

        for name, cam in self._cameras.items():
            try:
                cam.disconnect()
            except Exception:
                logger.warning("Failed to disconnect camera '%s'", name, exc_info=True)
        try:
            self._robot.disconnect()
        except Exception:
            logger.warning("Failed to disconnect robot", exc_info=True)

        self._connected = False

    def __enter__(self) -> Self:  # noqa: D105
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:  # noqa: D105
        self.disconnect()

    def run(self, *, duration_s: float | None = None) -> RunStats:  # noqa: PLR0915
        """Run the control loop.

        Args:
            duration_s: Maximum duration in seconds. None runs indefinitely.

        Returns:
            Statistics from the run session.

        Raises:
            RuntimeError: If called before connect().
            WorkerDiedError: If the inference worker thread dies.
        """
        if not self._connected:
            msg = "PolicyRuntime.run() called before connect(). Use 'with runtime:' or call runtime.connect() first."
            raise RuntimeError(msg)

        self._reset_session()

        self._execution.set_bus(self._bus, self._session_id)
        self._execution.start(self._model, self._action_queue)  # type: ignore[arg-type]
        self._bus.emit_lifecycle(
            LifecycleEvent(
                session_id=self._session_id,
                timestamp=time.time(),
                event="start",
                metadata={
                    "fps": self._fps,
                    "duration_s": duration_s,
                    "cameras": list(self._cameras.keys()),
                    "joint_names": self._robot.joint_names,
                },
            )
        )
        self._warmup_with_retry()

        goal_time = 1.0 / self._fps
        step = 0
        last_action: np.ndarray | None = None
        stale_this_tick = False

        try:
            while True:
                if duration_s is not None and step * goal_time >= duration_s:
                    break

                loop_start = time.perf_counter()
                stale_this_tick = False

                robot_obs, camera_frames = self._resilient_observe()
                if self._consecutive_error_ticks > 0:
                    stale_this_tick = True
                self._execution.maybe_request(self._build_model_input_from(robot_obs, camera_frames))

                action = self._action_queue.pop()
                if action is not None:
                    last_action = action
                else:
                    action = last_action
                    self._handle_hold(step=step)

                if action is None:
                    logger.error("No action available (warmup may have failed)")
                    self._tick_sleep(loop_start, goal_time)
                    step += 1
                    continue

                action = self._bus.invoke_before_send_action(action=action, step=step)

                self._resilient_send(action)
                self._bus.invoke_on_action_sent(action=action, step=step)

                elapsed = time.perf_counter() - loop_start
                sleep_time = goal_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                self._bus.emit_tick(
                    TickEvent(
                        session_id=self._session_id,
                        step=step,
                        timestamp=time.time(),
                        robot_observation=robot_obs,
                        camera_frames=camera_frames,
                        action_sent=action,
                        queue_remaining=self._action_queue.remaining,
                        loop_duration_s=elapsed,
                        sleep_time_s=max(sleep_time, 0.0),
                        stale_obs=stale_this_tick,
                    )
                )
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
            transient_errors=self._transient_errors,
            stale_obs_ticks=self._stale_obs_ticks,
        )

    def _handle_hold(self, *, step: int) -> None:
        holds = self._action_queue.consecutive_holds
        if holds == 1:
            logger.warning("Queue empty — holding position")
        elif self._fps > 0:
            warning_interval = max(int(self._fps), 1)
            if holds % warning_interval == 0:
                logger.warning(
                    "Queue starvation: %d consecutive holds (%.1fs)",
                    holds,
                    holds / self._fps,
                )
        self._bus.invoke_on_hold(step=step, holds=holds)

    def _reset_session(self) -> None:
        """Reset all session-scoped state for a fresh run."""
        self._session_id = uuid.uuid4().hex[:8]
        self._last_robot_obs = None
        self._last_camera_frames = {}
        self._consecutive_error_ticks = 0
        self._stale_obs_ticks = 0
        self._transient_errors = 0
        self._action_queue.reset()

    @staticmethod
    def _tick_sleep(loop_start: float, goal_time: float) -> tuple[float, float]:
        elapsed = time.perf_counter() - loop_start
        sleep_time = goal_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        return elapsed, sleep_time

    def _build_model_input(self) -> dict[str, Any]:
        robot_obs = self._robot.get_observation()
        model_input: dict[str, Any] = {"state": np.array([robot_obs.state], dtype=np.float32)}

        # Merge robot-embedded images and external cameras
        if robot_obs.images:
            for name, frame in robot_obs.images.items():
                model_input[f"images.{name}"] = frame.data[np.newaxis]
        for name, cam in self._cameras.items():
            model_input[f"images.{name}"] = cam.read_latest().data[np.newaxis]

        if self._task is not None:
            model_input["task"] = [self._task]

        return model_input

    def _retry_robot_obs(self) -> tuple[RobotObservation | None, ConnectionError | OSError | None]:
        robot_obs: RobotObservation | None = None
        last_error: ConnectionError | OSError | None = None
        for attempt in range(_MAX_OBS_RETRIES):
            try:
                robot_obs = self._robot.get_observation()
            except (ConnectionError, OSError) as exc:
                last_error = exc
                if attempt + 1 < _MAX_OBS_RETRIES:
                    time.sleep(_RETRY_BACKOFF_S)
            else:
                break
        return robot_obs, last_error

    def _resilient_observe(self) -> tuple[RobotObservation, dict[str, Frame]]:
        """Read robot observation and camera frames with retry and stale fallback.

        Returns:
            Tuple of (robot observation, camera frames keyed by name).

        Raises:
            ConnectionError: If robot observation fails with no stale fallback or
                max consecutive errors exceeded.
            CaptureError: If a camera read fails and no stale frame is available.
        """
        robot_obs, last_robot_error = self._retry_robot_obs()

        if robot_obs is None:
            if self._last_robot_obs is None:
                self._bus.emit_lifecycle(
                    LifecycleEvent(
                        session_id=self._session_id,
                        timestamp=time.time(),
                        event="connection_lost",
                        metadata={"error": str(last_robot_error)},
                    )
                )
                msg = "Robot observation failed and no stale observation available"
                raise ConnectionError(msg) from last_robot_error

            self._consecutive_error_ticks += 1
            self._stale_obs_ticks += 1
            if self._consecutive_error_ticks >= self._max_consecutive_error_ticks:
                self._bus.emit_lifecycle(
                    LifecycleEvent(
                        session_id=self._session_id,
                        timestamp=time.time(),
                        event="connection_lost",
                        metadata={"error": str(last_robot_error)},
                    )
                )
                msg = "Exceeded max consecutive robot observation failures"
                raise ConnectionError(msg) from last_robot_error

            self._bus.emit_lifecycle(
                LifecycleEvent(
                    session_id=self._session_id,
                    timestamp=time.time(),
                    event="obs_error",
                    metadata={"error": str(last_robot_error), "stale": True},
                )
            )
            robot_obs = self._last_robot_obs
        else:
            self._consecutive_error_ticks = 0
            self._last_robot_obs = robot_obs

        camera_frames: dict[str, Frame] = {}
        for name, camera in self._cameras.items():
            try:
                frame = camera.read_latest()
                camera_frames[name] = frame
                self._last_camera_frames[name] = frame
            except CaptureError as exc:
                stale_frame = self._last_camera_frames.get(name)
                if stale_frame is None:
                    raise
                logger.warning(
                    "Camera %s read failed — using stale frame: %s",
                    name,
                    exc,
                )
                camera_frames[name] = stale_frame

        return robot_obs, camera_frames

    def _build_model_input_from(self, robot_obs: RobotObservation, camera_frames: dict[str, Frame]) -> dict[str, Any]:
        """Assemble model input dict from observation and camera frames.

        Returns:
            Dictionary ready to pass to the inference model.
        """
        model_input: dict[str, Any] = {"state": np.array([robot_obs.state], dtype=np.float32)}
        if robot_obs.images:
            for name, frame in robot_obs.images.items():
                model_input[f"images.{name}"] = frame.data[np.newaxis]
        for name, frame in camera_frames.items():
            model_input[f"images.{name}"] = frame.data[np.newaxis]
        if self._task is not None:
            model_input["task"] = [self._task]
        return model_input

    def _resilient_send(self, action: np.ndarray) -> None:
        last_error: ConnectionError | OSError | None = None

        for attempt in range(_MAX_SEND_RETRIES):
            try:
                self._robot.send_action(action, goal_time=self._goal_time)
            except (ConnectionError, OSError) as exc:
                last_error = exc
                if attempt + 1 < _MAX_SEND_RETRIES:
                    time.sleep(_RETRY_BACKOFF_S)
            else:
                self._consecutive_error_ticks = 0
                return

        self._transient_errors += 1
        self._consecutive_error_ticks += 1
        if self._consecutive_error_ticks >= self._max_consecutive_error_ticks:
            self._bus.emit_lifecycle(
                LifecycleEvent(
                    session_id=self._session_id,
                    timestamp=time.time(),
                    event="connection_lost",
                    metadata={"error": str(last_error), "source": "send"},
                )
            )
            msg = "Exceeded max consecutive send failures"
            raise ConnectionError(msg) from last_error
        self._bus.emit_lifecycle(
            LifecycleEvent(
                session_id=self._session_id,
                timestamp=time.time(),
                event="send_error",
                metadata={"error": str(last_error)},
            )
        )
        logger.error(
            "Failed to send action after %d attempts; skipping tick: %s",
            _MAX_SEND_RETRIES,
            last_error,
        )

    def _warmup_with_retry(self) -> None:
        last_error: ConnectionError | OSError | None = None

        for attempt in range(_WARMUP_RETRIES):
            try:
                sample_obs = self._build_model_input()
                self._execution.warmup(sample_obs)
            except (ConnectionError, OSError) as exc:
                last_error = exc
                if attempt + 1 < _WARMUP_RETRIES:
                    time.sleep(_WARMUP_BACKOFF_S)
            else:
                return

        msg = f"Warmup failed after {_WARMUP_RETRIES} attempts"
        self._bus.emit_lifecycle(
            LifecycleEvent(
                session_id=self._session_id,
                timestamp=time.time(),
                event="warmup_failed",
                metadata={"error": str(last_error), "attempts": _WARMUP_RETRIES},
            )
        )
        raise ConnectionError(msg) from last_error

    def _shutdown(self, step: int) -> None:
        self._execution.stop()

        remaining = self._action_queue.remaining
        drain_limit = min(remaining, int(self._fps))
        for _ in range(drain_limit):
            action = self._action_queue.pop()
            if action is not None:
                try:
                    self._resilient_send(action)
                except ConnectionError:
                    logger.warning("Send failed during drain; skipping remaining actions")
                    break
                time.sleep(1.0 / self._fps)

        self._bus.emit_lifecycle(
            LifecycleEvent(
                session_id=self._session_id,
                timestamp=time.time(),
                event="shutdown",
                metadata={
                    "steps": step,
                    "transient_errors": self._transient_errors,
                    "stale_obs_ticks": self._stale_obs_ticks,
                },
            )
        )
        self._bus.close()

        logger.info(
            "Shutdown complete — %d steps, %d pops, %d holds",
            step,
            self._action_queue.total_pops,
            self._action_queue.total_holds,
        )
