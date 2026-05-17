# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PolicyRuntime — runs a trained policy on robot hardware."""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Self

import numpy as np

from physicalai.capture.errors import CaptureError
from physicalai.runtime._action_queue import ActionQueue  # noqa: PLC2701
from physicalai.runtime.execution import Execution, WorkerDiedError
from physicalai.runtime.smoothers import LerpSmoother

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from physicalai.capture.camera import Camera
    from physicalai.capture.frame import Frame
    from physicalai.cli._config import RuntimeConfig
    from physicalai.inference.model import InferenceModel
    from physicalai.robot.interface import Robot, RobotObservation
    from physicalai.runtime._telemetry import TelemetryEmitter

logger = logging.getLogger(__name__)

_DEFAULT_LERP_FRAMES = 5
_MAX_OBS_RETRIES = 3
_MAX_SEND_RETRIES = 2
_RETRY_BACKOFF_S = 0.001
_WARMUP_RETRIES = 5
_WARMUP_BACKOFF_S = 1.0


def _import_class(class_path: str) -> type:
    """Import a class from a dotted path like ``package.module.ClassName``.

    Returns:
        The imported class object.

    Raises:
        ValueError: If *class_path* has no module component.
        ImportError: If the class is not found in the module.
    """
    module_path, _, class_name = class_path.rpartition(".")
    if not module_path:
        msg = f"Invalid class_path: {class_path!r} — must be 'module.ClassName'"
        raise ValueError(msg)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        msg = f"{class_name!r} not found in {module_path!r}"
        raise ImportError(msg)
    return cls


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
        callbacks: Sequence[RuntimeCallback] = (),
        telemetry: TelemetryEmitter | None = None,
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
        self._callbacks = list(callbacks)
        self._goal_time = (1.0 / fps) * 3
        self._connected = False
        self._telemetry = telemetry
        self._last_robot_obs: RobotObservation | None = None
        self._last_camera_frames: dict[str, Frame] = {}
        self._consecutive_error_ticks: int = 0
        self._max_consecutive_error_ticks: int = int(3 * fps)
        self._stale_obs_ticks: int = 0
        self._transient_errors: int = 0

    @property
    def robot(self) -> Robot:
        """The robot instance managed by this runtime."""
        return self._robot

    @property
    def cameras(self) -> Mapping[str, Camera]:
        """Camera instances managed by this runtime, keyed by name."""
        return self._cameras

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> PolicyRuntime:
        """Construct a PolicyRuntime from a RuntimeConfig.

        Lazily imports InferenceModel, Camera, and robot classes.
        Robot config must include a ``class_path`` key (dotted import path).

        Args:
            config: Validated runtime configuration.

        Returns:
            Configured PolicyRuntime instance.

        Raises:
            ValueError: If robot config is missing ``class_path``.
        """
        from physicalai.capture.camera import Camera  # noqa: PLC0415
        from physicalai.inference.model import InferenceModel  # noqa: PLC0415

        model = InferenceModel(config.model.path, backend=config.model.backend, device=config.model.device)

        robot_cfg = dict(config.robot)
        class_path = robot_cfg.pop("class_path", None)
        if class_path is None:
            msg = "robot config must include 'class_path' (e.g. 'physicalai.robot.so101.SO101Follower')"
            raise ValueError(msg)
        robot_cls = _import_class(class_path)
        robot = robot_cls(**robot_cfg)

        cameras: dict[str, Camera] = {}
        for name, cam_cfg in config.cameras.items():
            cam_dict = cam_cfg.model_dump()
            cameras[name] = Camera.from_config(cam_dict)

        if config.execution.mode == "sync":
            from physicalai.runtime.execution import SyncExecution  # noqa: PLC0415

            execution: Execution = SyncExecution()
        else:
            from physicalai.runtime.execution import AsyncExecution  # noqa: PLC0415

            execution = AsyncExecution(
                threshold=config.execution.threshold,
                fps=int(config.fps),
                watchdog_timeout_s=config.execution.watchdog_timeout_s,
            )

        from physicalai.runtime.smoothers import ReplaceSmoother  # noqa: PLC0415

        if config.smoother.type == "lerp":
            smoother = LerpSmoother(duration_frames=config.smoother.duration_frames)
        else:
            smoother = ReplaceSmoother()
        action_queue = ActionQueue(smoother=smoother)

        return cls(
            robot=robot,
            model=model,
            execution=execution,
            action_queue=action_queue,
            cameras=cameras,
            fps=config.fps,
        )

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

    def run(self, *, duration_s: float | None = None) -> RunStats:
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
        self._execution.start(self._model, self._action_queue)
        self._warmup_with_retry()
        if self._telemetry:
            self._telemetry.emit_lifecycle("start", fps=self._fps, duration_s=duration_s)

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

                obs = self._resilient_observe()
                if self._consecutive_error_ticks > 0:
                    stale_this_tick = True
                self._execution.maybe_request(obs)

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

                modified = self._invoke_callback("before_send_action", action=action, step=step)
                if modified is not None:
                    action = modified

                self._resilient_send(action)
                self._invoke_callback("on_action_sent", action=action, step=step)
                elapsed, sleep_time = self._tick_sleep(loop_start, goal_time)

                self._emit_tick_telemetry(step, action, elapsed, sleep_time, stale_obs=stale_this_tick)
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

    def _emit_tick_telemetry(
        self,
        step: int,
        action: np.ndarray,
        elapsed: float,
        sleep_time: float,
        *,
        stale_obs: bool,
    ) -> None:
        if not self._telemetry:
            return
        robot_obs = self._last_robot_obs
        self._telemetry.emit_tick(
            step=step,
            timestamp=time.perf_counter(),
            joint_positions=robot_obs.joint_positions if robot_obs else None,
            action_sent=action,
            queue_remaining=self._action_queue.remaining,
            loop_duration_s=elapsed,
            sleep_time_s=max(sleep_time, 0.0),
            stale_obs=stale_obs,
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
        self._invoke_callback("on_hold", step=step, holds=holds)

    @staticmethod
    def _tick_sleep(loop_start: float, goal_time: float) -> tuple[float, float]:
        elapsed = time.perf_counter() - loop_start
        sleep_time = goal_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        return elapsed, sleep_time

    def _build_model_input(self) -> dict[str, Any]:
        robot_obs = self._robot.get_observation()
        model_input: dict[str, Any] = {}

        if robot_obs.joint_positions is not None:
            model_input["state"] = np.array([robot_obs.joint_positions], dtype=np.float32)

        # Merge robot-embedded images and external cameras
        if robot_obs.images:
            for name, frame in robot_obs.images.items():
                model_input[f"images.{name}"] = frame.data[np.newaxis]
        for name, cam in self._cameras.items():
            model_input[f"images.{name}"] = cam.read_latest().data[np.newaxis]

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

    def _resilient_observe(self) -> dict[str, Any]:
        robot_obs, last_robot_error = self._retry_robot_obs()

        if robot_obs is None:
            if self._last_robot_obs is None:
                if self._telemetry:
                    self._telemetry.emit_lifecycle("connection_lost", error=str(last_robot_error))
                msg = "Robot observation failed and no stale observation available"
                raise ConnectionError(msg) from last_robot_error

            self._consecutive_error_ticks += 1
            self._stale_obs_ticks += 1
            if self._consecutive_error_ticks >= self._max_consecutive_error_ticks:
                if self._telemetry:
                    self._telemetry.emit_lifecycle("connection_lost", error=str(last_robot_error))
                msg = "Exceeded max consecutive robot observation failures"
                raise ConnectionError(msg) from last_robot_error

            if self._telemetry:
                self._telemetry.emit_lifecycle("obs_error", error=str(last_robot_error), stale=True)
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

        model_input: dict[str, Any] = {}
        if robot_obs is not None and robot_obs.joint_positions is not None:
            model_input["state"] = np.array([robot_obs.joint_positions], dtype=np.float32)
        if robot_obs is not None and robot_obs.images:
            for name, frame in robot_obs.images.items():
                model_input[f"images.{name}"] = frame.data[np.newaxis]
        for name, frame in camera_frames.items():
            model_input[f"images.{name}"] = frame.data[np.newaxis]
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
        if self._telemetry:
            self._telemetry.emit_lifecycle("send_error", error=str(last_error))
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
        raise ConnectionError(msg) from last_error

    def _shutdown(self, step: int) -> None:
        self._execution.stop()

        remaining = self._action_queue.remaining
        drain_limit = min(remaining, int(self._fps))
        for _ in range(drain_limit):
            action = self._action_queue.pop()
            if action is not None:
                self._resilient_send(action)
                time.sleep(1.0 / self._fps)

        if self._telemetry:
            self._telemetry.emit_lifecycle(
                "shutdown",
                steps=step,
                transient_errors=self._transient_errors,
                stale_obs_ticks=self._stale_obs_ticks,
            )
            self._telemetry.close()

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
                    callback_result = fn(**kwargs)
                    if callback_result is not None:
                        result = callback_result
                        if method == "before_send_action":
                            kwargs["action"] = callback_result
                except Exception:
                    logger.exception("Callback %s.%s raised", type(cb).__name__, method)
        return result
