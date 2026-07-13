# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared robot subscriber over Zenoh.

``SharedRobot`` structurally satisfies the :class:`~physicalai.robot.Robot`
protocol: it pulls the latest owner-published state on demand and publishes
actions fire-and-forget. The first instance that finds no existing owner
spawns one; later instances attach.

Reads never use a background callback thread: the ``/state`` subscriber is
declared with a native ``RingChannel(1)`` whose buffering is GIL-independent,
and ``get_observation()`` retrieves the newest sample with a non-blocking
``try_recv()``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from physicalai.robot.errors import RobotIdConflict, RobotNotConnectedError, RobotTransportError

from ._codec import TransportObservation, decode_meta, decode_state, encode_action
from ._ids import (
    META_WILDCARD,
    action_key,
    derive_device_id,
    derive_robot_id,
    meta_key,
    state_key,
)
from ._session import open_session

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np

    from physicalai.robot import RobotObservation

_PROBE_TIMEOUT = 1.0
_RACE_RETRY_TIMEOUT = 5.0
_RETRY_INTERVAL = 0.2
_FIRST_STATE_TIMEOUT = 5.0


def _query_meta(session: Any, key: str, timeout: float) -> dict[str, Any] | None:  # noqa: ANN401
    """Query a ``/meta`` key and decode the first successful reply.

    Args:
        session: Open Zenoh session.
        key: Concrete ``.../meta`` key expression.
        timeout: Zenoh query timeout in seconds.

    Returns:
        The decoded meta dict, or ``None`` when no owner answered.
    """
    try:
        replies = session.get(key, timeout=timeout)
        for reply in replies:
            sample = reply.ok
            if sample is not None:
                return decode_meta(sample.payload.to_bytes())
    except Exception:  # noqa: BLE001
        logger.debug(f"meta query failed for {key}", exc_info=True)
    return None


def _query_meta_with_retry(session: Any, key: str, timeout: float) -> dict[str, Any] | None:  # noqa: ANN401
    """Poll :func:`_query_meta` until an owner answers or *timeout* elapses.

    Returns:
        The decoded meta dict, or ``None`` on timeout.
    """
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        meta = _query_meta(session, key, timeout=min(_PROBE_TIMEOUT, remaining))
        if meta is not None:
            return meta
        time.sleep(_RETRY_INTERVAL)


def discover_robots(timeout: float = 2.0, *, session: Any = None) -> list[dict[str, Any]]:  # noqa: ANN401
    """Enumerate all reachable shared robots via the ``/meta`` wildcard.

    Args:
        timeout: Zenoh query timeout in seconds.
        session: Optional existing Zenoh session to query through (kept
            open); by default a scouting-only session is opened and closed.

    Returns:
        One decoded meta dict per answering owner, each including its
        ``robot_id`` (derived from the answering key expression).
    """
    own_session = session is None
    if own_session:
        session = open_session()
    robots: list[dict[str, Any]] = []
    try:
        replies = session.get(META_WILDCARD, timeout=timeout)
        for reply in replies:
            sample = reply.ok
            if sample is None:
                continue
            try:
                meta = decode_meta(sample.payload.to_bytes())
            except Exception:  # noqa: BLE001
                logger.debug("skipping malformed meta reply", exc_info=True)
                continue
            meta["robot_id"] = str(sample.key_expr).removesuffix("/meta")
            robots.append(meta)
    finally:
        if own_session:
            session.close()
    return robots


class SharedRobot:
    """Robot subscriber that attaches to (or spawns) a shared owner process.

    The owner process holds the exclusive hardware connection; any number
    of ``SharedRobot`` instances read its state and send actions over
    Zenoh. Satisfies the :class:`~physicalai.robot.Robot` protocol, so it
    is a drop-in replacement for a direct driver.

    Args:
        robot_type: Logical robot type (``"so101"`` / ``"widowxai"``) for
            auto-spawn mode, or ``None`` to attach to an existing owner
            only (requires ``robot_id``).
        robot_id: Explicit robot id override. Defaults to one derived from
            ``robot_type`` and the connection kwargs, which is what lets a
            second same-machine instance attach instead of spawning a
            competing owner.
        rate_hz: Owner loop rate when this instance spawns the owner;
            ``None`` selects the per-robot default.
        idle_timeout: Seconds with zero subscribers before a spawned owner
            self-exits (and homes/holds the robot).
        connect_timeout: Overall budget for :meth:`connect`; also caps the
            owner-spawn handshake (hardware connect may legitimately block
            for seconds).
        robot_kwargs: JSON-serializable driver constructor kwargs
            (e.g. ``port``, ``calibration`` as a path, ``role`` as a str).
        **extra_robot_kwargs: Convenience merge into ``robot_kwargs``.
    """

    def __init__(
        self,
        robot_type: str | None,
        *,
        robot_id: str | None = None,
        rate_hz: float | None = None,
        idle_timeout: float = 10.0,
        connect_timeout: float = 10.0,
        robot_kwargs: Mapping[str, object] | None = None,
        _factory_override: str | None = None,
        **extra_robot_kwargs: object,
    ) -> None:
        if robot_type is None and robot_id is None:
            msg = "must provide robot_type or robot_id"
            raise ValueError(msg)

        self._robot_type = robot_type
        self._robot_kwargs: dict[str, object] = {**(robot_kwargs or {}), **extra_robot_kwargs}
        self._rate_hz = rate_hz
        self._idle_timeout = idle_timeout
        self._connect_timeout = connect_timeout
        self._factory_override = _factory_override

        try:
            self._device_id: str | None = derive_device_id(self._robot_kwargs)
        except ValueError:
            self._device_id = None

        if robot_id is None and self._device_id is None:
            msg = "cannot derive robot_id: provide 'port'/'ip' in robot kwargs or an explicit robot_id"
            raise ValueError(msg)

        self._robot_id = derive_robot_id(robot_type or "", self._robot_kwargs, robot_id=robot_id)

        self._session: Any = None
        self._owner: Any = None
        self._state_sub: Any = None
        self._action_pub: Any = None
        self._meta: dict[str, Any] | None = None
        self._latest: TransportObservation | None = None
        self._connected = False

    @classmethod
    def from_owner(cls, robot_id: str) -> SharedRobot:
        """Attach-only construction: subscribe to an existing owner by id.

        Args:
            robot_id: The owner's robot id (as advertised on ``/meta``).

        Returns:
            A ``SharedRobot`` that never spawns an owner.
        """
        return cls(None, robot_id=robot_id)

    @property
    def robot_id(self) -> str:
        """The full Zenoh robot id keying this robot's topics."""
        return self._robot_id

    @property
    def joint_names(self) -> list[str]:
        """Ordered joint names, from the owner's ``/meta`` record.

        Raises:
            RobotNotConnectedError: If called before :meth:`connect`.
        """
        if self._meta is None:
            msg = "SharedRobot is not connected. Call connect() first."
            raise RobotNotConnectedError(msg)
        return list(self._meta["joint_names"])

    @property
    def meta(self) -> dict[str, Any] | None:
        """The owner's ``/meta`` record (None before connect)."""
        return self._meta

    def connect(self) -> None:
        """Attach to an existing owner, spawning one first if needed.

        Idempotent: calling ``connect()`` when already connected is a no-op.
        Raises :class:`RobotIdConflict` when an existing owner's advertised
        identity does not match this instance's construction kwargs, and
        :class:`RobotTransportError` when no owner could be found or spawned.

        Uses the ``connect_timeout`` passed to the constructor as its
        overall budget (also caps the owner-spawn handshake — hardware
        connect may legitimately block for seconds).
        """
        if self._connected:
            return

        self._session = open_session(self._robot_id)
        try:
            meta = _query_meta(self._session, meta_key(self._robot_id), timeout=_PROBE_TIMEOUT)
            if meta is None:
                meta = self._spawn_or_reprobe(self._connect_timeout)
            self._validate_meta(meta)
            self._meta = meta
            self._attach()
        except Exception:
            self._teardown()
            raise
        self._connected = True

    def _spawn_or_reprobe(self, timeout: float) -> dict[str, Any]:
        """Spawn an owner; on failure fall back to a bounded ``/meta`` re-probe.

        The re-probe covers the lost-spawn-race case: the competing owner
        holds the lock file and this worker's spawn reports an error, but
        the robot is (or is about to be) served.

        Returns:
            The owner's meta record.

        Raises:
            RobotTransportError: If no owner is reachable after the retry
                budget.
        """
        if self._robot_type is None:
            msg = f"no owner found for {self._robot_id} (attach-only mode: robot_type not provided)"
            raise RobotTransportError(msg)

        from ._owner import RobotOwner  # noqa: PLC0415
        from ._spec import RobotSpec  # noqa: PLC0415

        spec = RobotSpec(self._robot_type, dict(self._robot_kwargs))
        owner = RobotOwner(
            spec,
            self._robot_id,
            self._device_id or self._robot_id.replace("/", "_"),
            rate_hz=self._rate_hz,
            idle_timeout=self._idle_timeout,
            _factory_override=self._factory_override,
        )
        try:
            owner.start(timeout=timeout)
        except RobotTransportError as exc:
            meta = _query_meta_with_retry(self._session, meta_key(self._robot_id), timeout=_RACE_RETRY_TIMEOUT)
            if meta is not None:
                logger.debug(f"Lost owner race for {self._robot_id} — attaching to existing owner")
                return meta
            msg = f"failed to start robot owner for {self._robot_id}"
            raise RobotTransportError(msg) from exc
        else:
            # Keep the handle alive; the detached owner self-terminates via
            # idle timeout, so disconnect() never stops it explicitly.
            self._owner = owner

        meta = _query_meta_with_retry(self._session, meta_key(self._robot_id), timeout=_RACE_RETRY_TIMEOUT)
        if meta is None:
            msg = f"owner for {self._robot_id} reported READY but its /meta queryable is unreachable"
            raise RobotTransportError(msg)
        return meta

    def _validate_meta(self, meta: dict[str, Any]) -> None:
        """Compare the owner's advertised identity against our kwargs.

        A mismatch means the id was reused for different hardware — fail
        loudly instead of silently binding to the wrong robot.

        Raises:
            RobotIdConflict: On robot-type or connection mismatch.
        """
        if self._robot_type is not None and meta.get("robot_type") != self._robot_type:
            msg = (
                f"robot id {self._robot_id} is served by robot_type={meta.get('robot_type')!r}, "
                f"but this instance was constructed for {self._robot_type!r}"
            )
            raise RobotIdConflict(msg)
        if self._device_id is not None and meta.get("connection") != self._device_id:
            msg = (
                f"robot id {self._robot_id} is served by connection={meta.get('connection')!r}, "
                f"but this instance was constructed for {self._device_id!r}"
            )
            raise RobotIdConflict(msg)

    def _attach(self) -> None:
        """Declare the ``/state`` subscriber and ``/action`` publisher.

        Blocks until the first state sample arrives (the ring is empty
        until the first publish after attach; a continuously-publishing
        owner fills it within one period).

        Raises:
            RobotTransportError: If no state arrives within the timeout.
        """
        import zenoh  # noqa: PLC0415

        self._state_sub = self._session.declare_subscriber(
            state_key(self._robot_id),
            zenoh.handlers.RingChannel(1),
        )
        # QoS (D20): express bypasses batching; best-effort/drop match the
        # fire-and-forget, latest-wins action semantics.
        self._action_pub = self._session.declare_publisher(
            action_key(self._robot_id),
            reliability=zenoh.Reliability.BEST_EFFORT,
            congestion_control=zenoh.CongestionControl.DROP,
            express=True,
        )

        deadline = time.monotonic() + _FIRST_STATE_TIMEOUT
        while time.monotonic() < deadline:
            sample = self._state_sub.try_recv()
            if sample is not None:
                self._latest = decode_state(sample.payload.to_bytes())
                return
            time.sleep(0.005)

        msg = f"no state received from owner of {self._robot_id} within {_FIRST_STATE_TIMEOUT:.1f}s"
        raise RobotTransportError(msg)

    def get_observation(self) -> RobotObservation:
        """Pull the newest owner-published state (non-blocking).

        Returns:
            The newest observation, or the cached last-known one when no
            fresher sample has arrived (staleness is visible via
            ``timestamp``). ``images`` is always ``None`` — frames go
            through the capture transport.

        Raises:
            RobotNotConnectedError: If called before :meth:`connect`.
        """
        if not self._connected or self._state_sub is None:
            msg = "SharedRobot is not connected. Call connect() first."
            raise RobotNotConnectedError(msg)

        # Ring(1) holds only the newest sample; a stalled subscriber
        # resumes on current state, never a backlog of stale samples.
        sample = self._state_sub.try_recv()
        if sample is not None:
            self._latest = decode_state(sample.payload.to_bytes())

        assert self._latest is not None  # noqa: S101  # guaranteed by _attach()
        return self._latest

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Publish an absolute joint target, fire-and-forget.

        Latest-wins on the owner's Ring(1) is safe because actions are
        absolute targets — dropping an intermediate one just skips to the
        newest.

        Args:
            action: Absolute joint targets matching :attr:`joint_names`.
            goal_time: Minimum time (seconds) to reach the target.

        Raises:
            RobotNotConnectedError: If called before :meth:`connect`.
        """
        if not self._connected or self._action_pub is None:
            msg = "SharedRobot is not connected. Call connect() first."
            raise RobotNotConnectedError(msg)
        self._action_pub.put(encode_action(action, goal_time))

    def disconnect(self) -> None:
        """Close this subscriber's own Zenoh session only.

        Deliberately does **not** signal the owner to stop motors — the
        owner owns safe-state and detects departed subscribers via Zenoh
        matching status, whether they exit cleanly or crash.
        """
        self._teardown()

    def is_connected(self) -> bool:
        """Whether this subscriber is attached to an owner.

        Returns:
            True when connected.
        """
        return self._connected

    def _teardown(self) -> None:
        self._owner = None
        self._state_sub = None
        self._action_pub = None
        if self._session is not None:
            try:
                self._session.close()
            except Exception:  # noqa: BLE001
                logger.debug("Error closing zenoh session", exc_info=True)
            self._session = None
        self._connected = False
        self._latest = None
