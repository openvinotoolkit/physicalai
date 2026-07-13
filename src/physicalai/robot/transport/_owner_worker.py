# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Standalone robot owner worker (``python -m physicalai.robot.transport._owner_worker``).

One owner process holds the exclusive hardware connection and runs a single
write-first control loop: apply the newest action, read state, publish it.
No background threads touch the driver, so there is no serial-bus contention.

Startup protocol (mirrors the capture publisher worker): the parent sends a
JSON config on stdin; the worker answers a single ``READY`` or
``ERROR:{json}`` line on stdout.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from physicalai.robot.transport._codec import (  # noqa: PLC2701
    decode_action,
    encode_meta,
    encode_state,
)
from physicalai.robot.transport._ids import action_key, meta_key, state_key  # noqa: PLC2701
from physicalai.robot.transport._lock import RobotLock  # noqa: PLC2701
from physicalai.robot.transport._session import open_session  # noqa: PLC2701
from physicalai.robot.transport._spec import RobotSpec, default_rate_hz  # noqa: PLC2701

if TYPE_CHECKING:
    from types import FrameType

    from physicalai.robot.interface import Robot

_MAX_CONSECUTIVE_FAILURES = 5

shutdown = threading.Event()


def sigterm_handler(_signum: int, _frame: FrameType | None) -> None:
    shutdown.set()


def signal_ready() -> None:
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    sys.stdout.close()


def signal_error(msg: str, tb: str | None = None) -> None:
    payload = {"msg": msg, "traceback": tb} if tb else {"msg": msg}
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
    os.dup2(saved_fd, 1)
    os.close(saved_fd)
    sys.stdout = os.fdopen(1, "w")


def build_robot(config: dict) -> Robot:
    """Instantiate a robot driver from a JSON config dict.

    Args:
        config: Configuration dict with ``robot_type``, ``robot_kwargs``,
            and optional ``_factory_override`` (tests only).

    Returns:
        A not-yet-connected driver instance.
    """
    factory_override = config.get("_factory_override")
    if factory_override:
        module_path, _, attr = factory_override.rpartition(":")
        mod = importlib.import_module(module_path)
        factory = getattr(mod, attr)
        return factory(**config.get("robot_kwargs", {}))

    spec = RobotSpec.from_json_dict(config)
    return spec.build()


def _build_meta(config: dict, driver: Robot, state_dim: int) -> dict[str, Any]:
    """Assemble the ``/meta`` record advertised by the queryable.

    Args:
        config: Worker config (for robot_type / device_id).
        driver: Connected driver (for joint names).
        state_dim: Length of the owner-computed state vector.

    Returns:
        The meta dict shipped to discovering/validating subscribers.
    """
    import socket  # noqa: PLC0415

    joint_names = list(driver.joint_names)
    return {
        "robot_type": config["robot_type"],
        "joint_names": joint_names,
        "host": socket.gethostname(),
        "connection": config["device_id"],
        "state_dim": state_dim,
        "num_joints": len(joint_names),
    }


def _run_loop(
    driver: Robot,
    state_pub: Any,  # noqa: ANN401
    action_sub: Any,  # noqa: ANN401
    *,
    rate_hz: float,
    idle_timeout: float,
    robot_id: str,
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
        robot_id: For logging.
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
                logger.warning(f"Failed to apply action for {robot_id}", exc_info=True)

        try:
            obs = driver.get_observation()
        except Exception:  # noqa: BLE001
            consecutive_failures += 1
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                logger.error(f"{consecutive_failures} consecutive read failures -- shutting down owner {robot_id}")
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
            logger.info(f"No subscribers for {idle_timeout}s -- shutting down owner {robot_id}")
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
    session: Any
    state_pub: Any
    action_sub: Any
    meta_queryable: Any


def _startup(config: dict, lock: RobotLock) -> _Endpoints:
    """Acquire the lock, connect the driver, and declare Zenoh endpoints.

    Cleans up any partially-created resources on failure.

    Args:
        config: Worker config from stdin.
        lock: Single-owner lock (not yet acquired).

    Returns:
        The connected endpoints.

    Raises:
        RuntimeError: If another owner already holds the device lock.
    """
    import zenoh  # noqa: PLC0415

    robot_id: str = config["robot_id"]
    device_id: str = config["device_id"]

    if not lock.acquire():
        msg = f"robot lock already held for device {device_id!r} (another owner is starting or running)"
        raise RuntimeError(msg)

    driver = build_robot(config)
    driver.connect()
    session: Any = None
    try:
        first_obs = driver.get_observation()
        meta = _build_meta(config, driver, state_dim=int(first_obs.state.shape[0]))
        meta_bytes = encode_meta(meta)

        session = open_session(robot_id, listen=True)

        # QoS (D20): defaults tune for throughput, not latency. Small
        # messages at 100-200 Hz sit in Zenoh's batching danger zone, so
        # `express` sends each sample immediately; best-effort/drop match
        # the fire-and-forget, latest-wins semantics.
        state_pub = session.declare_publisher(
            state_key(robot_id),
            reliability=zenoh.Reliability.BEST_EFFORT,
            congestion_control=zenoh.CongestionControl.DROP,
            express=True,
        )
        action_sub = session.declare_subscriber(
            action_key(robot_id),
            zenoh.handlers.RingChannel(1),
        )

        meta_key_expr = meta_key(robot_id)

        def _answer_meta(query: Any) -> None:  # noqa: ANN401
            # Static bytes computed once at startup — safe to serve from
            # zenoh's callback thread without touching the driver.
            with contextlib.suppress(Exception):
                query.reply(meta_key_expr, meta_bytes)

        meta_queryable = session.declare_queryable(meta_key_expr, _answer_meta)
    except Exception:
        try:
            driver.disconnect()
        except Exception:  # noqa: BLE001
            logger.exception("driver disconnect failed during error cleanup")
        if session is not None:
            with contextlib.suppress(Exception):
                session.close()
        raise

    return _Endpoints(
        driver=driver,
        session=session,
        state_pub=state_pub,
        action_sub=action_sub,
        meta_queryable=meta_queryable,
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
        config = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        signal_error(f"invalid JSON config: {exc}")
        return 1

    robot_id: str = config["robot_id"]
    lock = RobotLock(config["device_id"])

    saved_stdout_fd = suppress_stdout()
    try:
        endpoints = _startup(config, lock)
    except Exception as exc:  # noqa: BLE001
        restore_stdout(saved_stdout_fd)
        signal_error(f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())
        lock.release()
        return 1
    restore_stdout(saved_stdout_fd)

    signal_ready()

    try:
        _run_loop(
            endpoints.driver,
            endpoints.state_pub,
            endpoints.action_sub,
            rate_hz=config.get("rate_hz") or default_rate_hz(config.get("robot_type", "")),
            idle_timeout=config.get("idle_timeout", 10.0),
            robot_id=robot_id,
        )
    except Exception:  # noqa: BLE001
        logger.exception(f"owner loop failed for {robot_id}")
    finally:
        shutdown.set()
        # Safe-state contract: the owner (not subscribers) stops/homes the
        # robot on exit, whether idle-timeout, SIGTERM, or loop failure.
        try:
            endpoints.driver.disconnect()
        except Exception:  # noqa: BLE001
            logger.exception(f"driver disconnect failed for {robot_id}")
        with contextlib.suppress(Exception):
            endpoints.meta_queryable.undeclare()
        with contextlib.suppress(Exception):
            endpoints.session.close()
        lock.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
