# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime loop implementations for robot control."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, Self

from physicalai.capture.errors import CaptureError
from physicalai.config import export_config
from physicalai.runtime._callback_bus import _CallbackBus  # noqa: PLC2701
from physicalai.runtime.events import LifecycleEvent, TickEvent
from physicalai.runtime.execution.base import WorkerDiedError

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
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

RunReason = Literal["stop_requested", "duration_elapsed", "interrupted", "error"]
"""Why a :meth:`RobotRuntime.run` call ended."""


def _unwrap_runtime_document(document: dict[str, Any], *, target: type) -> dict[str, Any]:
    """Rewrite a bare exported Config document to the CLI ``runtime:`` shape.

    A document without a top-level ``class_path`` is returned unchanged (it is
    already a CLI document). Otherwise it must be a valid
    :class:`~physicalai.config.Config` whose ``class_path`` resolves
    to *target* (or a subclass), and its ``init_args`` become the ``runtime:``
    section — making ``Config.from_instance(runtime)`` → YAML → CLI round-trip.

    Returns:
        A CLI-shaped document with constructor args under ``runtime:``.

    Raises:
        ConfigError: If ``class_path`` does not resolve to *target*.
    """
    if "class_path" not in document:
        return document

    from physicalai.config import (  # noqa: PLC0415
        ConfigError,
        import_dotted_path,
        validate_config,
    )

    config = validate_config(document)
    resolved = import_dotted_path(config["class_path"])
    if not (isinstance(resolved, type) and issubclass(resolved, target)):
        msg = (
            f"config class_path {config['class_path']!r} does not resolve to "
            f"{target.__module__}.{target.__qualname__} (or a subclass); "
            "expected a runtime config exported via Config.from_instance(runtime)"
        )
        raise ConfigError(msg)
    return {"runtime": dict(config["init_args"])}


class StopSignal(Protocol):
    """A stop flag the control loop polls once per tick.

    Anything exposing a thread- or process-safe ``is_set()`` satisfies it, so
    ``threading.Event`` and ``multiprocessing.Event`` both work unchanged. Duck
    typing is the point: a parent process can end a session process directly,
    with no mailbox round-trip, and the loop keeps its no-``isinstance`` rule.
    """

    def is_set(self) -> bool:
        """Report whether a stop has been requested.

        Returns:
            ``True`` once a stop has been requested.
        """
        ...


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


@export_config(class_path="physicalai.runtime.RobotRuntime")
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
        self._closed = False
        self._lifecycle_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._last_robot_obs: RobotObservation | None = None
        self._last_camera_frames: dict[str, Frame] = {}
        self._consecutive_error_ticks: int = 0
        self._max_consecutive_error_ticks: int = int(3 * fps)
        self._stale_obs_ticks: int = 0
        self._transient_errors: int = 0
        self._session_id: str = ""
        self._last_tick_stale: bool = False
        # Cleared in _shutdown(), not _reset_session(): _reset_session() runs at
        # the top of run(), so clearing there would drop a stop() that landed
        # between connect() and run() and the session would never end.
        self._stop = threading.Event()
        self._last_run_reason: RunReason | None = None

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

    @property
    def last_run_reason(self) -> RunReason | None:
        """Why the most recent :meth:`run` ended.

        Also emitted as ``reason`` in the ``shutdown`` lifecycle event, so
        telemetry sinks receive it without the caller wiring up a callback.

        Reports ``None`` before any run has started and while a run is in
        flight. A :meth:`run` rejected before it starts — the not-connected
        ``RuntimeError`` — leaves the previous run's value untouched.

        Returns:
            The reason the last started run ended, or ``None``.
        """
        return self._last_run_reason

    def connect(self) -> None:
        """Connect robot and cameras.

        Connects robot first, then cameras in dict order. On failure,
        disconnects everything already connected and re-raises.

        Idempotent — calling on an already-connected runtime is a no-op.

        Raises:
            RuntimeError: If a run is active or this runtime was disconnected.
        """
        with self._lifecycle_lock:
            if self._run_lock.locked():
                msg = "Cannot connect RobotRuntime while run() is active"
                raise RuntimeError(msg)
            if self._closed:
                msg = "RobotRuntime has been disconnected and cannot be connected again; create a new runtime"
                raise RuntimeError(msg)
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
        """Permanently close callbacks and disconnect cameras then robot.

        Idempotent. A disconnected runtime cannot be connected again; construct
        a new runtime with fresh callbacks instead.

        Raises:
            RuntimeError: If a run is still active.
        """
        with self._lifecycle_lock:
            if self._closed:
                return
            if self._run_lock.locked():
                msg = "Cannot disconnect RobotRuntime while run() is active; stop it and wait for run() to return"
                raise RuntimeError(msg)
            try:
                if self._connected:
                    for name, cam in self._cameras.items():
                        try:
                            cam.disconnect()
                        except Exception:
                            logger.warning("Failed to disconnect camera '%s'", name, exc_info=True)
                    try:
                        self._robot.disconnect()
                    except Exception:
                        logger.warning("Failed to disconnect robot", exc_info=True)
            finally:
                self._connected = False
                self._closed = True
                self._bus.close()

    def __enter__(self) -> Self:  # noqa: D105
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:  # noqa: D105
        self.disconnect()

    @classmethod
    def from_config(cls, config: str | Path) -> Self:
        """Build runtime from YAML/JSON config file.

        Accepts two document shapes: the CLI document (constructor args under
        ``runtime:``, optional ``run:``) and a bare exported
        :class:`~physicalai.config.Config` as produced by
        :func:`~physicalai.config.to_config` / :func:`~physicalai.config.save_yaml`
        (top-level ``class_path`` resolving to this class).
        ``action_source:`` is always required and explicit — one schema, no
        flat/legacy shorthand.

        Returns:
            Instantiated runtime object.

        Raises:
            TypeError: If the file root is not a mapping.
        """
        import yaml  # noqa: PLC0415
        from jsonargparse import ArgumentParser  # noqa: PLC0415

        document = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        if document is None:
            document = {}
        if not isinstance(document, dict):
            msg = f"runtime config must be a mapping, got {type(document).__name__}"
            raise TypeError(msg)
        document = _unwrap_runtime_document(document, target=cls)
        parser = ArgumentParser()
        parser.add_class_arguments(cls, "runtime")
        # stop_event is a live in-process object with no serializable form, so
        # it stays out of the config schema entirely.
        parser.add_method_arguments(cls, "run", "run", skip={"stop_event"})
        ns = parser.parse_object(document)
        return parser.instantiate(ns).runtime

    def stop(self) -> None:
        """Request the loop to exit after the current tick. Thread-safe.

        The tick already in flight finishes and its action is still sent, so the
        robot is never left mid-command. Safe to call before :meth:`run` (that
        run then exits after zero steps), from another thread while :meth:`run`
        is in flight, or on an idle runtime. The flag clears once the run ends,
        so the runtime stays reusable.

        Distinct from ``Execution.stop()`` (tear down the inference worker) and
        ``ActionSource.disconnect()`` (release source-owned resources): this
        only asks the control loop to finish.
        """
        self._stop.set()

    def run(self, *, duration_s: float | None = None, stop_event: StopSignal | None = None) -> int:
        """Run the control loop.

        Exits on the first of: a stop request (:meth:`stop` or *stop_event*),
        *duration_s* elapsing, ``KeyboardInterrupt``, or a propagating
        exception. :attr:`last_run_reason` records which one it was.

        Args:
            duration_s: Maximum duration in seconds. ``None`` runs until stopped.
            stop_event: External stop flag polled once per tick, honoured in
                addition to :meth:`stop`. Any object with ``is_set()`` works,
                ``multiprocessing.Event`` included — see :class:`StopSignal`.

        Returns:
            Number of steps completed this run.
            A step is one iteration of the loop at ``fps``: read an observation,
            get one action from ``action_source``, and send it to the robot.

        Raises:
            RuntimeError: If called before ``connect()``.
            WorkerDiedError: If the action source's execution worker dies.
        """  # noqa: DOC502
        goal_time = 1.0 / self._fps
        step = 0
        # Every normal exit overwrites this, so only a propagating exception
        # leaves "error" in place — the shutdown event then reports the truth
        # instead of inheriting whichever normal reason happened to be set.
        reason: RunReason = "error"

        with self._active_run():
            # Source connect and the start event sit inside the try so a failure
            # there still runs _shutdown(): otherwise the stop flag would survive
            # into the next run() and silently zero-step every later session.
            try:
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

                while True:
                    if self._stop.is_set() or (stop_event is not None and stop_event.is_set()):
                        reason = "stop_requested"
                        # Fires after the last tick. Callbacks may take control of the robot from this point.
                        self._bus.emit_lifecycle(
                            LifecycleEvent(
                                session_id=self._session_id,
                                timestamp=time.time(),
                                event="stop_requested",
                                metadata={"step": step},
                            )
                        )
                        break
                    if duration_s is not None and step * goal_time >= duration_s:
                        reason = "duration_elapsed"
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
                reason = "interrupted"
                logger.info("Interrupted by user")
            except WorkerDiedError:
                logger.exception("Worker died during runtime")
                raise
            finally:
                self._shutdown(step, reason=reason)

        return step

    @contextmanager
    def _active_run(self) -> Iterator[None]:
        """Acquire the run slot and release it after the caller's block.

        Raises:
            RuntimeError: If another run is active or the runtime is not connected.
        """
        acquired = False
        try:
            with self._lifecycle_lock:
                acquired = self._run_lock.acquire(blocking=False)
                if not acquired:
                    msg = "RobotRuntime.run() is already active"
                    raise RuntimeError(msg)
                if not self._connected:
                    msg = (
                        "RobotRuntime.run() called before connect(). "
                        "Use 'with runtime:' or call runtime.connect() first."
                    )
                    raise RuntimeError(msg)
            yield
        finally:
            if acquired:
                self._run_lock.release()

    def _reset_session(self) -> None:
        """Reset all session-scoped state for a fresh run.

        Deliberately leaves ``self._stop`` alone: this runs at the top of
        ``run()``, so clearing the flag here would discard a ``stop()`` that
        arrived between ``connect()`` and ``run()``. ``_shutdown()`` clears it.
        """
        # Telemetry/log correlation id only (ties together events from one run()
        # call), not a security token or capability.
        self._session_id = uuid.uuid4().hex[:8]
        self._last_robot_obs = None
        self._last_camera_frames = {}
        self._consecutive_error_ticks = 0
        self._stale_obs_ticks = 0
        self._transient_errors = 0
        self._last_tick_stale = False
        self._last_run_reason = None

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

    def _disconnect_and_log_errors(self) -> None:
        """Release action-source resources, logging rather than raising on failure.

        Only ``Exception`` is swallowed. A ``BaseException`` — a Ctrl+C landing
        mid-teardown — still propagates; the caller's ``finally`` chain keeps the
        rest of shutdown intact.
        """
        try:
            self._action_source.disconnect()
        except Exception:
            logger.exception("Action source disconnect failed")

    def _emit_shutdown(self, step: int, *, reason: RunReason) -> None:
        """Emit the shutdown lifecycle event, then flush every callback."""
        try:
            self._bus.emit_lifecycle(
                LifecycleEvent(
                    session_id=self._session_id,
                    timestamp=time.time(),
                    event="shutdown",
                    metadata={
                        "steps": step,
                        "reason": reason,
                        "transient_errors": self._transient_errors,
                        "stale_obs_ticks": self._stale_obs_ticks,
                    },
                )
            )
        finally:
            # A Ctrl+C in on_lifecycle must not lose buffered telemetry.
            self._bus.flush()

    def _shutdown(self, step: int, *, reason: RunReason) -> None:
        self._last_run_reason = reason
        try:
            self._disconnect_and_log_errors()
        finally:
            try:
                self._emit_shutdown(step, reason=reason)
            finally:
                # Outermost guarantee: whatever a callback throws — including a
                # Ctrl+C the bus does not swallow — the stop flag must not
                # survive into the next run() and silently zero-step it.
                self._stop.clear()
                logger.info("Shutdown complete — %d steps (%s)", step, reason)
