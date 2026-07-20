# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Standalone robot owner worker (``python -m physicalai.robot.transport._owner_worker``).

One owner process holds the exclusive hardware connection and runs a single
write-first control loop: apply the newest action, read state, publish it.
No background threads touch the driver, so there is no serial-bus contention.

Startup order (name lock -> device locks -> hardware connect) matters: the
driver is constructed once to read its identity, then every lock is held
*before* any hardware access starts, so a losing race or a device already
owned under another name is caught before ever touching the bus.

Startup protocol (mirrors the capture publisher worker): the parent sends a
JSON config on stdin; the worker answers a single ``READY`` or
``ERROR:{json}`` line on stdout. The ``ERROR`` payload's ``phase`` field
lets the parent (:class:`~physicalai.robot.transport._owner.RobotOwner`)
distinguish failure kinds without string-matching the message.
"""

from __future__ import annotations

import contextlib
import json
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from physicalai.robot.interface import Robot
from physicalai.robot.transport._codec import (  # noqa: PLC2701
    ROBOT_TRANSPORT_PROTOCOL_VERSION,
    decode_action,
    encode_metadata,
    encode_state,
)
from physicalai.robot.transport._ids import (  # noqa: PLC2701
    action_key,
    default_host,
    derive_endpoint_port,
    metadata_key,
    state_key,
)
from physicalai.robot.transport._lock import NAME_KIND, LockContention, OwnedLocks, acquire_locks  # noqa: PLC2701
from physicalai.robot.transport._owner_config import RobotOwnerConfig  # noqa: PLC2701
from physicalai.robot.transport._session import open_session  # noqa: PLC2701

if TYPE_CHECKING:
    from types import FrameType

_MAX_CONSECUTIVE_FAILURES = 5

shutdown = threading.Event()


def sigterm_handler(_signum: int, _frame: FrameType | None) -> None:
    shutdown.set()


def signal_ready() -> None:
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    sys.stdout.close()


def signal_error(
    msg: str,
    tb: str | None = None,
    *,
    phase: str | None = None,
    device_ids: tuple[str, ...] | None = None,
) -> None:
    payload: dict[str, Any] = {"msg": msg}
    if tb:
        payload["traceback"] = tb
    if phase:
        payload["phase"] = phase
    if device_ids:
        payload["device_ids"] = list(device_ids)
    sys.stdout.write(f"ERROR:{json.dumps(payload)}\n")
    sys.stdout.flush()
    sys.stdout.close()


def suppress_stdout() -> int:
    """Redirect fd 1 to /dev/null during startup.

    Operates at the OS file-descriptor level so native libraries writing
    directly to the C stdout fd are silenced too, keeping the single-line
    ``READY``/``ERROR`` IPC protocol on stdout uncorrupted.

    Returns:
        The saved original fd 1, to be passed to :func:`restore_stdout`.
    """
    import os  # noqa: PLC0415

    saved_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)
    return saved_fd


def restore_stdout(saved_fd: int) -> None:
    """Restore fd 1 from *saved_fd* and rewrap ``sys.stdout``.

    Args:
        saved_fd: The fd returned by :func:`suppress_stdout`.
    """
    import os  # noqa: PLC0415

    os.dup2(saved_fd, 1)
    os.close(saved_fd)
    sys.stdout = os.fdopen(1, "w")


class _StartupError(Exception):
    """Internal signal carrying the failure phase for the ``ERROR:{json}`` payload."""

    def __init__(self, message: str, *, phase: str, device_ids: tuple[str, ...] | None = None) -> None:
        """Store the phase and (if relevant) this worker's own device ids."""
        super().__init__(message)
        self.phase = phase
        self.device_ids = device_ids


def _build_metadata(
    config: RobotOwnerConfig,
    driver: Robot,
    device_ids: tuple[str, ...],
    state_dim: int,
) -> dict[str, Any]:
    """Assemble the ``/metadata`` record advertised by the queryable.

    Args:
        config: Worker config (for name / robot_class).
        driver: Connected driver (for joint names).
        device_ids: This owner's sorted, deduplicated device ids. Omitted
            from advertised metadata when remote transport is enabled.
        state_dim: Length of the owner-computed state vector.

    Returns:
        The metadata dict shipped to discovering/validating subscribers.
        Deliberately excludes constructor kwargs, calibration paths, and
        any other construction secret.
    """
    joint_names = list(driver.joint_names)
    metadata: dict[str, Any] = {
        "protocol_version": ROBOT_TRANSPORT_PROTOCOL_VERSION,
        "name": config.name,
        "robot_class": config.robot_class,
        "host": default_host(),
        "joint_names": joint_names,
        "num_joints": len(joint_names),
        "state_dim": state_dim,
    }
    if not config.allow_remote:
        metadata["device_ids"] = list(device_ids)
    return metadata


def _run_loop(
    driver: Robot,
    state_pub: Any,  # noqa: ANN401
    action_sub: Any,  # noqa: ANN401
    *,
    rate_hz: float,
    idle_timeout: float,
    name: str,
) -> None:
    """Single-threaded write-first owner loop.

    Ordering per tick: apply newest action (minimizes action latency), read
    measured state, publish. When no action is pending the servos hold their
    last commanded position (freezing is safer than silent motion). Exits
    when no ``/state`` subscriber has been matched for *idle_timeout*
    seconds.

    Args:
        driver: Connected robot driver (exclusively owned by this thread).
        state_pub: Zenoh publisher on the ``/state`` key.
        action_sub: Zenoh subscriber (Ring(1)) on the ``/action`` key.
        rate_hz: Fixed loop rate.
        idle_timeout: Seconds with zero subscribers before self-exit.
        name: For logging.
    """
    period = 1.0 / rate_hz
    idle_since: float | None = None
    consecutive_failures = 0
    next_tick = time.monotonic()

    while not shutdown.is_set():
        sample = action_sub.try_recv()
        if sample is not None:
            try:
                action, goal_time, _send_ts = decode_action(sample.payload.to_bytes())
                driver.send_action(action, goal_time=goal_time)
            except Exception:  # noqa: BLE001
                # A malformed or out-of-range action from one subscriber must
                # not kill the owner shared by everyone else.
                logger.warning(f"Failed to apply action for {name}", exc_info=True)

        try:
            obs = driver.get_observation()
        except Exception:  # noqa: BLE001
            consecutive_failures += 1
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                logger.error(f"{consecutive_failures} consecutive read failures -- shutting down owner {name}")
                break
            continue
        consecutive_failures = 0

        state_pub.put(
            encode_state(
                joint_positions=obs.joint_positions,
                state=obs.state,
                timestamp=obs.timestamp,
                sensor_data=obs.sensor_data,
            ),
        )

        now = time.monotonic()
        # Runtime returns a MatchingStatus object (always truthy); the bool
        # lives on its .matching attribute — the type stub says `-> bool`.
        if state_pub.matching_status.matching:
            idle_since = None
        elif idle_since is None:
            idle_since = now
        elif now - idle_since > idle_timeout:
            logger.info(f"No subscribers for {idle_timeout}s -- shutting down owner {name}")
            break

        next_tick += period
        sleep_time = next_tick - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            # Fell behind (slow bus / blocking read); don't accumulate debt.
            next_tick = time.monotonic()


@dataclass
class _Endpoints:
    """Resources produced by startup and consumed by the owner loop."""

    driver: Robot
    locks: OwnedLocks
    session: Any
    state_pub: Any
    action_sub: Any
    metadata_queryable: Any


def _connect_and_build_metadata(
    config: RobotOwnerConfig,
    driver: Robot,
    device_ids: tuple[str, ...],
) -> bytes:
    """Connect the driver and assemble the encoded ``/metadata`` payload.

    Returns:
        The msgpack-encoded ``/metadata`` payload.

    Raises:
        _StartupError: ``phase="connection_failed"`` if ``driver.connect()``
            raises.
    """
    try:
        driver.connect()
    except Exception as exc:
        msg = f"driver.connect() failed: {exc}"
        raise _StartupError(msg, phase="connection_failed") from exc

    first_obs = driver.get_observation()
    metadata = _build_metadata(config, driver, device_ids, state_dim=int(first_obs.state.shape[0]))
    return encode_metadata(metadata)


@dataclass
class _ZenohEndpoints:
    """Zenoh resources declared for one owner."""

    session: Any
    state_pub: Any
    action_sub: Any
    metadata_queryable: Any


def _declare_zenoh_endpoints(config: RobotOwnerConfig, metadata_bytes: bytes) -> _ZenohEndpoints:
    """Open the session and declare ``/state``, ``/action``, ``/metadata``.

    Returns:
        The declared Zenoh session and endpoints.

    Raises:
        _StartupError: ``phase="endpoint_collision"`` if opening the session
            or declaring any endpoint fails (most likely a derived-port
            bind collision).
    """
    import zenoh  # noqa: PLC0415

    metadata_key_expr = metadata_key(config.name)

    def _answer_metadata(query: Any) -> None:  # noqa: ANN401
        # Static bytes computed once at startup — safe to serve from
        # zenoh's callback thread without touching the driver.
        with contextlib.suppress(Exception):
            query.reply(metadata_key_expr, metadata_bytes)

    session: Any = None
    try:
        session = open_session(config.name, listen=True, allow_remote=config.allow_remote)

        # QoS (D20): defaults tune for throughput, not latency. Small
        # messages at 100-200 Hz sit in Zenoh's batching danger zone, so
        # `express` sends each sample immediately; best-effort/drop match
        # the fire-and-forget, latest-wins semantics.
        state_pub = session.declare_publisher(
            state_key(config.name),
            reliability=zenoh.Reliability.BEST_EFFORT,
            congestion_control=zenoh.CongestionControl.DROP,
            express=True,
        )
        action_sub = session.declare_subscriber(action_key(config.name), zenoh.handlers.RingChannel(1))
        metadata_queryable = session.declare_queryable(metadata_key_expr, _answer_metadata)
    except Exception as exc:
        if session is not None:
            with contextlib.suppress(Exception):
                session.close()
        bind_host = "0.0.0.0" if config.allow_remote else "127.0.0.1"  # noqa: S104  # nosec B104: explicit remote opt-in
        endpoint = f"tcp/{bind_host}:{derive_endpoint_port(config.name)}"
        msg = (
            f"failed to declare Zenoh endpoints at derived endpoint {endpoint}: {exc}. "
            "Choose a different robot name or configure a local Zenoh router."
        )
        raise _StartupError(msg, phase="endpoint_collision") from exc

    return _ZenohEndpoints(
        session=session,
        state_pub=state_pub,
        action_sub=action_sub,
        metadata_queryable=metadata_queryable,
    )


def _startup(config: RobotOwnerConfig) -> _Endpoints:
    """Construct the driver, acquire locks, connect, and declare Zenoh endpoints.

    Order matters (see module docstring): construct -> read device_ids ->
    acquire name lock -> acquire device locks -> connect -> publish.
    Cleans up any partially-created resources on failure.

    Args:
        config: Worker config from stdin.

    Returns:
        The connected endpoints.

    Raises:
        _StartupError: Naming the failure phase, for the caller to report
            via :func:`signal_error`.
    """
    try:
        driver = config.build()
    except Exception as exc:
        msg = f"failed to construct {config.robot_class!r}: {exc}"
        raise _StartupError(msg, phase="construction_failed") from exc

    if not isinstance(driver, Robot):
        msg = f"{config.robot_class!r} does not satisfy the Robot protocol"
        raise _StartupError(msg, phase="construction_failed")

    device_ids = tuple(sorted(set(driver.device_ids)))

    try:
        locks = acquire_locks(config.name, device_ids)
    except LockContention as exc:
        if exc.kind == NAME_KIND:
            msg = f"name {config.name!r} is already owned by another process"
            raise _StartupError(msg, phase="name_lock_contention", device_ids=device_ids) from exc
        msg = f"device {exc.identity!r} is already locked under another name"
        raise _StartupError(msg, phase="device_lock_contention", device_ids=(exc.identity,)) from exc

    try:
        metadata_bytes = _connect_and_build_metadata(config, driver, device_ids)
        zenoh_endpoints = _declare_zenoh_endpoints(config, metadata_bytes)
    except Exception as exc:
        with contextlib.suppress(Exception):
            driver.disconnect()
        locks.release_all()
        if isinstance(exc, _StartupError):
            raise
        msg = f"unexpected startup failure after acquiring locks: {type(exc).__name__}: {exc}"
        raise _StartupError(msg, phase="unexpected_startup_failure") from exc

    return _Endpoints(
        driver=driver,
        locks=locks,
        session=zenoh_endpoints.session,
        state_pub=zenoh_endpoints.state_pub,
        action_sub=zenoh_endpoints.action_sub,
        metadata_queryable=zenoh_endpoints.metadata_queryable,
    )


def main() -> int:
    """Entry point for the owner worker process.

    Returns:
        Exit code: 0 on success, 1 on startup failure.
    """
    signal.signal(signal.SIGTERM, sigterm_handler)

    raw = sys.stdin.read()
    sys.stdin.close()
    try:
        config = RobotOwnerConfig.from_json_dict(json.loads(raw))
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        signal_error(f"invalid worker config: {exc}", phase="invalid_config")
        return 1

    saved_stdout_fd = suppress_stdout()
    try:
        endpoints = _startup(config)
    except _StartupError as exc:
        restore_stdout(saved_stdout_fd)
        signal_error(str(exc), tb=traceback.format_exc(), phase=exc.phase, device_ids=exc.device_ids)
        return 1
    except Exception as exc:  # noqa: BLE001
        restore_stdout(saved_stdout_fd)
        signal_error(f"{type(exc).__name__}: {exc}", tb=traceback.format_exc(), phase="unexpected_startup_failure")
        return 1
    restore_stdout(saved_stdout_fd)

    signal_ready()

    try:
        _run_loop(
            endpoints.driver,
            endpoints.state_pub,
            endpoints.action_sub,
            rate_hz=config.rate_hz,
            idle_timeout=config.idle_timeout,
            name=config.name,
        )
    except Exception:  # noqa: BLE001
        logger.exception(f"owner loop failed for {config.name}")
    finally:
        shutdown.set()
        # Safe-state contract: the owner (not subscribers) stops/homes the
        # robot on exit, whether idle-timeout, SIGTERM, or loop failure.
        try:
            endpoints.driver.disconnect()
        except Exception:  # noqa: BLE001
            logger.exception(f"driver disconnect failed for {config.name}")
        with contextlib.suppress(Exception):
            endpoints.metadata_queryable.undeclare()
        with contextlib.suppress(Exception):
            endpoints.session.close()
        endpoints.locks.release_all()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
