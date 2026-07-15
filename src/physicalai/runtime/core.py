# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime loop implementations for robot control."""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Protocol, Self

from physicalai.capture.errors import CaptureError
from physicalai.runtime._callback_bus import _CallbackBus  # noqa: PLC2701
from physicalai.runtime.events import LifecycleEvent, TickEvent
from physicalai.runtime.execution.base import WorkerDiedError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from typing import Any

    import numpy as np

    from physicalai.capture.camera import Camera
    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import Robot, RobotObservation
    from physicalai.runtime.action_sources.base import ActionSource

logger = logging.getLogger(__name__)


_MAX_OBS_RETRIES = 3
_MAX_SEND_RETRIES = 2
_RETRY_BACKOFF_S = 0.001
_GOAL_TIME_TICKS = 3


class RuntimeCallback(Protocol):
    """Optional action-transform hooks a callback may implement.

    Only ``on_action_ready``/``on_action_sent`` are part of this protocol — the
    fire-and-forget telemetry hooks (``on_tick``, ``on_inference``,
    ``on_lifecycle``, ``on_metrics``) are duck-typed via ``getattr`` in the
    callback bus and intentionally not part of this protocol.
    """

    def on_action_ready(self, *, action: np.ndarray, step: int) -> np.ndarray:
        """Called with the chosen action before it's sent; may transform it.

        Every callback must return a valid action (no ``None`` sentinel) — a
        callback that doesn't want to change anything returns its input
        unchanged. Exceptions raised here are not isolated by the callback
        bus — they propagate and end the run, since a failed transform (e.g.
        a safety filter) means the action can no longer be trusted.

        Returns:
            The action after this callback's transform.
        """
        ...

    def on_action_sent(self, *, action: np.ndarray, step: int) -> None:
        """Called after action is sent to robot. Notification only."""
        ...


class RobotRuntime:
    """Generic robot runtime loop with a required, pluggable action source."""

    def __init__(  # noqa: D107
        self,
        robot: Robot,
        action_source: ActionSource,
        fps: float,
        cameras: Mapping[str, Camera] | None = None,
        callbacks: Sequence[Any] = (),
    ) -> None:
        if fps <= 0:
            msg = f"fps must be positive, got {fps}"
            raise ValueError(msg)
        self._robot = robot
        self._action_source = action_source
        self._fps = fps
        self._cameras: Mapping[str, Camera] = cameras or {}
        self._bus = _CallbackBus(callbacks)
        self._goal_time = (1.0 / fps) * _GOAL_TIME_TICKS
        self._connected = False
        self._last_robot_obs: RobotObservation | None = None
        self._last_camera_frames: dict[str, Frame] = {}
        self._consecutive_error_ticks: int = 0
        self._max_consecutive_error_ticks: int = int(3 * fps)
        self._stale_obs_ticks: int = 0
        self._transient_errors: int = 0
        self._session_id: str = ""
        self._last_tick_stale: bool = False

    @property
    def robot(self) -> Robot:
        """The robot instance managed by this runtime."""
        return self._robot

    @property
    def cameras(self) -> Mapping[str, Camera]:
        """Camera instances managed by this runtime, keyed by name."""
        return self._cameras

    @property
    def action_source(self) -> ActionSource:
        """The action source driving this runtime's control loop.

        Public so config-built runs can still reach action-source-owned
        stats (e.g. ``runtime.action_source.action_queue.total_pops``)
        after ``run()`` returns.
        """
        return self._action_source

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

    @classmethod
    def from_config(cls, config: str | Path) -> Self:
        """Build runtime from YAML/JSON config file.

        ``action_source:`` is always required and explicit — one schema, no
        flat/legacy shorthand.

        Returns:
            Instantiated runtime object.
        """
        from jsonargparse import ActionConfigFile, ArgumentParser  # noqa: PLC0415

        parser = ArgumentParser()
        parser.add_argument("--config", action=ActionConfigFile)
        parser.add_class_arguments(cls, "runtime")
        parser.add_method_arguments(cls, "run", "run")
        ns = parser.parse_args(["--config", str(config)])
        return parser.instantiate(ns).runtime

    def run(self, *, duration_s: float | None = None) -> int:
        """Run the control loop.

        Returns:
            Number of steps completed this run.
            A step is one iteration of the loop at ``fps``: read an observation,
            get one action from ``action_source``, and send it to the robot.

        Raises:
            RuntimeError: If called before ``connect()``.
            WorkerDiedError: If the action source's execution worker dies.
        """
        if not self._connected:
            msg = "RobotRuntime.run() called before connect(). Use 'with runtime:' or call runtime.connect() first."
            raise RuntimeError(msg)

        self._reset_session()
        self._action_source.connect(bus=self._bus, session_id=self._session_id)
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

        goal_time = 1.0 / self._fps
        step = 0

        try:
            while True:
                if duration_s is not None and step * goal_time >= duration_s:
                    break

                loop_start = time.perf_counter()
                robot_state, camera_frames = self._read_observation()

                action = self._action_source.update(robot_state, camera_frames, step)
                action = self._bus.invoke_on_action_ready(action=action, step=step)

                self._resilient_send(action)
                self._bus.invoke_on_action_sent(action=action, step=step)

                elapsed, sleep_time = self._tick_sleep(loop_start, goal_time)
                self._bus.emit_tick(
                    TickEvent(
                        session_id=self._session_id,
                        step=step,
                        timestamp=time.time(),
                        robot_state=robot_state,
                        camera_frames=camera_frames,
                        action_sent=action,
                        loop_duration_s=elapsed,
                        sleep_time_s=max(sleep_time, 0.0),
                        stale_obs=self._last_tick_stale,
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

        return step

    def _reset_session(self) -> None:
        """Reset all session-scoped state for a fresh run."""
        # Telemetry/log correlation id only (ties together events from one run()
        # call), not a security token or capability.
        self._session_id = uuid.uuid4().hex[:8]
        self._last_robot_obs = None
        self._last_camera_frames = {}
        self._consecutive_error_ticks = 0
        self._stale_obs_ticks = 0
        self._transient_errors = 0
        self._last_tick_stale = False

    @staticmethod
    def _tick_sleep(loop_start: float, goal_time: float) -> tuple[float, float]:
        elapsed = time.perf_counter() - loop_start
        sleep_time = goal_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        return elapsed, sleep_time

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

    def _read_robot_resilient(self) -> tuple[RobotObservation, bool]:
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
            return self._last_robot_obs, True

        self._consecutive_error_ticks = 0
        self._last_robot_obs = robot_obs
        return robot_obs, False

    def _read_cameras_resilient(self) -> dict[str, Frame]:
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
        return camera_frames

    def _read_observation(self) -> tuple[RobotObservation, dict[str, Frame]]:
        """Read robot state + camera frames once for this tick (retry + stale fallback).

        The single read for this tick — the same values are passed to the
        action source's ``update()`` and to telemetry via ``TickEvent``.
        Staleness is stashed on the instance (``_last_tick_stale``) for the
        caller to attach to the tick's ``TickEvent``.

        Returns:
            Tuple ``(robot_state, camera_frames)``.
        """
        robot_state, stale = self._read_robot_resilient()
        self._last_tick_stale = stale
        camera_frames = self._read_cameras_resilient()
        return robot_state, camera_frames

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

    def _shutdown(self, step: int) -> None:
        try:
            self._action_source.disconnect()
        except Exception:
            logger.exception("Action source disconnect failed")

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

        logger.info("Shutdown complete — %d steps", step)
