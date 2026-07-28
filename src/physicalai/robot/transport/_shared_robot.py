# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared robot subscriber over Zenoh.

``SharedRobot`` structurally satisfies the :class:`~physicalai.robot.Robot`
protocol: it pulls the latest owner-published state on demand and publishes
actions fire-and-forget. The first instance constructed for a given *name*
that finds no existing owner spawns one; later instances (for the same
*name*, anywhere reachable) attach.

Construction is :class:`~physicalai.config.ComponentConfig`-only via
``robot=`` or :meth:`from_config`. ``robot`` is declared as an
``@export_config(config_args=...)`` argument, so a nested
:func:`~physicalai.config.instantiate` hands over the recipe instead of
building a driver this process would immediately discard. Pass ``robot=None``
(or use :meth:`attach`) for attach-only.

Unlike the superseded connection-derived ``robot_id``, *name* is a required,
caller-chosen logical identifier — routing never needs a live driver
instance to resolve. Physical device identity
(:attr:`~physicalai.robot.interface.Robot.device_ids`) only matters to the
*owner*, for host-local exclusivity locking; a ``SharedRobot`` never
constructs a driver itself.

Reads never use a background callback thread: the ``/state`` subscriber is
declared with a native ``RingChannel(1)`` whose buffering is GIL-independent,
and ``get_observation()`` retrieves the newest sample with a non-blocking
``try_recv()``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from loguru import logger

from physicalai.config import export_config
from physicalai.robot.errors import (
    RobotDeviceAlreadyOwned,
    RobotNameConflict,
    RobotNotConnectedError,
    RobotProtocolMismatch,
    RobotTransportError,
)

from ._codec import ROBOT_TRANSPORT_PROTOCOL_VERSION, TransportObservation, decode_metadata, decode_state, encode_action
from ._ids import KEY_PREFIX, METADATA_WILDCARD, action_key, metadata_key, state_key, validate_name
from ._lock import active_owner_device_ids, registered_owner_names
from ._owner_config import DEFAULT_RATE_HZ, normalize_robot_config
from ._session import open_session

if TYPE_CHECKING:
    import numpy as np

    from physicalai.config import ComponentConfig
    from physicalai.robot import RobotObservation


_PROBE_TIMEOUT = 1.0
_RACE_RETRY_TIMEOUT = 5.0
_RETRY_INTERVAL = 0.2
_FIRST_STATE_TIMEOUT = 5.0


def _query_metadata(session: Any, key: str, timeout: float) -> dict[str, Any] | None:  # noqa: ANN401
    """Query a ``/metadata`` key and decode the first successful reply.

    Args:
        session: Open Zenoh session.
        key: Concrete ``.../metadata`` key expression.
        timeout: Zenoh query timeout in seconds.

    Returns:
        The decoded metadata dict, or ``None`` when no owner answered.
    """
    try:
        replies = session.get(key, timeout=timeout)
        for reply in replies:
            sample = reply.ok
            if sample is not None:
                return decode_metadata(sample.payload.to_bytes())
    except Exception:  # noqa: BLE001
        logger.debug(f"metadata query failed for {key}", exc_info=True)
    return None


def _query_metadata_with_retry(session: Any, key: str, timeout: float) -> dict[str, Any] | None:  # noqa: ANN401
    """Poll :func:`_query_metadata` until an owner answers or *timeout* elapses.

    Returns:
        The decoded metadata dict, or ``None`` on timeout.
    """
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        metadata = _query_metadata(session, key, timeout=min(_PROBE_TIMEOUT, remaining))
        if metadata is not None:
            return metadata
        time.sleep(min(_RETRY_INTERVAL, remaining))


def discover_robots(
    timeout: float = 2.0,
    *,
    session: Any = None,  # noqa: ANN401
    allow_remote: bool = False,
) -> list[dict[str, Any]]:
    """Enumerate all reachable shared robots via the ``/metadata`` wildcard.

    Args:
        timeout: Zenoh query timeout in seconds.
        session: Optional existing Zenoh session to query through (kept
            open). By default, host-local owners are enumerated through
            the name-lock registry and queried over their deterministic
            loopback endpoints.
        allow_remote: Whether the (default, own) session may discover
            robots beyond localhost. Ignored when *session* is given —
            that session's own scope applies.

    Returns:
        One decoded metadata dict per answering owner, each including its
        ``name``.
    """
    robots: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout

    def _append_replies(query_session: Any, query_timeout: float) -> None:  # noqa: ANN401
        replies = query_session.get(METADATA_WILDCARD, timeout=query_timeout)
        for reply in replies:
            sample = reply.ok
            if sample is None:
                continue
            try:
                metadata = decode_metadata(sample.payload.to_bytes())
            except Exception:  # noqa: BLE001
                logger.debug("skipping malformed metadata reply", exc_info=True)
                continue
            metadata.setdefault(
                "name",
                str(sample.key_expr).removeprefix(f"{KEY_PREFIX}/").removesuffix("/metadata"),
            )
            robots.append(metadata)

    if session is not None:
        _append_replies(session, timeout)
    else:
        for name in registered_owner_names():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                local_session = open_session(name, allow_remote=False)
            except ValueError:
                logger.debug(f"skipping invalid robot name from local registry: {name!r}")
                continue
            try:
                metadata = _query_metadata(local_session, metadata_key(name), timeout=remaining)
                if metadata is not None:
                    metadata.setdefault("name", name)
                    robots.append(metadata)
            finally:
                local_session.close()

        if allow_remote and (remaining := deadline - time.monotonic()) > 0:
            remote_session = open_session(allow_remote=True)
            try:
                while (remaining := deadline - time.monotonic()) > 0:
                    _append_replies(remote_session, min(_PROBE_TIMEOUT, remaining))
                    time.sleep(min(_RETRY_INTERVAL, remaining))
            finally:
                remote_session.close()

    return list({metadata.get("name"): metadata for metadata in robots}.values())


@export_config(class_path="physicalai.robot.SharedRobot", config_args=("robot",))
class SharedRobot:
    """Robot subscriber that attaches to (or spawns) a shared owner process.

    The owner process holds the exclusive hardware connection; any number
    of ``SharedRobot`` instances read its state and send actions over
    Zenoh. Satisfies the :class:`~physicalai.robot.Robot` protocol, so it
    is a drop-in replacement for a direct driver.

    Prefer :meth:`from_config`. The constructor takes
    ``robot: ComponentConfig`` to spawn, or ``robot=None`` (attach-only;
    :meth:`attach` is the explicit form).

    Opted into :func:`~physicalai.config.export_config` as a **construction
    recipe** only (name, nested ``robot`` ComponentConfig, transport knobs).
    Connection / Zenoh session / publisher state is never part of
    :func:`~physicalai.config.to_config`.

    Args:
        name: Required logical name — keys the Zenoh topics directly. Two
            instances constructed with the same *name* (anywhere reachable
            under the chosen transport scope) share one owner.
        robot: Driver :class:`~physicalai.config.ComponentConfig` from local
            config input (same boundary as CLI/app args), used
            to spawn if no owner exists yet for *name*. ``None`` means
            attach-only — use :meth:`attach` for that case. Declared as an
            ``@export_config`` config arg, so nested
            :func:`~physicalai.config.instantiate` passes the recipe through
            without constructing the driver here.
        allow_remote: Whether this instance's own session — and, if it
            spawns the owner, the owner's session for its whole lifetime —
            is reachable beyond localhost. Defaults to the secure,
            localhost-only scope; a later attacher's value never widens or
            narrows an already-running owner's reachability.
        rate_hz: Owner loop rate when this instance spawns the owner.
        idle_timeout: Seconds with zero subscribers before a spawned owner
            self-exits (and homes/holds the robot). ``None`` disables idle
            exit (owner stays up until stopped).
        connect_timeout: Default overall budget for :meth:`connect`.
    """

    def __init__(
        self,
        name: str,
        *,
        robot: ComponentConfig | Mapping[str, object] | None = None,
        allow_remote: bool = False,
        rate_hz: float = DEFAULT_RATE_HZ,
        idle_timeout: float | None = 10.0,
        connect_timeout: float = 10.0,
        _session: object | None = None,
    ) -> None:
        self._name = validate_name(name)
        self._robot = None if robot is None else normalize_robot_config(robot)
        self._allow_remote = allow_remote
        self._rate_hz = rate_hz
        self._idle_timeout = idle_timeout
        self._connect_timeout = connect_timeout

        self._provided_session = _session
        self._session: Any = None
        self._owns_session = _session is None
        self._owner: Any = None
        self._state_sub: Any = None
        self._action_pub: Any = None
        self._metadata: dict[str, Any] | None = None
        self._latest: TransportObservation | None = None
        self._connected = False

    @classmethod
    def _physicalai_normalize_captured_init_args(cls, supplied: dict[str, object]) -> None:
        robot = supplied.get("robot")
        if isinstance(robot, Mapping):
            supplied["robot"] = normalize_robot_config(robot)

    @classmethod
    def from_config(
        cls,
        robot_config: ComponentConfig | Mapping[str, object],
        *,
        name: str,
        allow_remote: bool = False,
        rate_hz: float = DEFAULT_RATE_HZ,
        idle_timeout: float | None = 10.0,
        connect_timeout: float = 10.0,
        _session: object | None = None,
    ) -> SharedRobot:
        """Primary API: spawn/attach from a local robot ComponentConfig.

        Args:
            robot_config: Local ``class_path`` + ``init_args`` for the driver.
            name: Logical owner name (Zenoh topic key).
            allow_remote: Whether this session / spawned owner may leave localhost.
            rate_hz: Owner loop rate when this instance spawns the owner.
            idle_timeout: Idle self-exit timeout for a spawned owner. ``None``
                disables idle exit.
            connect_timeout: Overall budget for :meth:`connect`.

        Returns:
            A ``SharedRobot`` that stores the normalized ComponentConfig and
            writes only the new owner stdin shape on spawn.
        """
        # Omit default None so @export_config does not capture "_session": null.
        if _session is None:
            return cls(
                name,
                robot=robot_config,
                allow_remote=allow_remote,
                rate_hz=rate_hz,
                idle_timeout=idle_timeout,
                connect_timeout=connect_timeout,
            )
        return cls(
            name,
            robot=robot_config,
            allow_remote=allow_remote,
            rate_hz=rate_hz,
            idle_timeout=idle_timeout,
            connect_timeout=connect_timeout,
            _session=_session,
        )

    @classmethod
    def attach(
        cls,
        name: str,
        *,
        allow_remote: bool = False,
        connect_timeout: float = 10.0,
        _session: object | None = None,
    ) -> SharedRobot:
        """Attach-only construction: subscribe to an existing owner by name.

        Never spawns an owner — :meth:`connect` raises
        :class:`~physicalai.robot.errors.RobotTransportError` if none is
        reachable.

        Args:
            name: The owner's logical name (as advertised on ``/metadata``).
            allow_remote: Whether this session may discover an owner beyond
                localhost.
            connect_timeout: Overall budget for :meth:`connect`.

        Returns:
            A ``SharedRobot`` that never spawns an owner.
        """
        # Omit default None so @export_config does not capture "_session": null.
        if _session is None:
            return cls(name, allow_remote=allow_remote, connect_timeout=connect_timeout)
        return cls(name, allow_remote=allow_remote, connect_timeout=connect_timeout, _session=_session)

    @property
    def _robot_class(self) -> str | None:
        """Public ``class_path`` used for metadata conflict diagnostics."""
        return None if self._robot is None else self._robot["class_path"]

    @property
    def name(self) -> str:
        """This robot's logical name, keying its Zenoh topics."""
        return self._name

    @property
    def device_ids(self) -> tuple[str, ...]:
        """Always empty — a subscriber owns no physical hardware itself."""
        return ()

    @property
    def joint_names(self) -> list[str]:
        """Ordered joint names, from the owner's ``/metadata`` record.

        Raises:
            RobotNotConnectedError: If called before :meth:`connect`.
        """
        if self._metadata is None:
            msg = "SharedRobot is not connected. Call connect() first."
            raise RobotNotConnectedError(msg)
        return list(self._metadata["joint_names"])

    @property
    def metadata(self) -> dict[str, Any] | None:
        """The owner's ``/metadata`` record (None before connect)."""
        return self._metadata

    def connect(self) -> None:
        """Attach to an existing owner, spawning one first if needed.

        Idempotent: calling ``connect()`` when already connected is a no-op.
        Raises :class:`~physicalai.robot.errors.RobotProtocolMismatch`,
        :class:`~physicalai.robot.errors.RobotNameConflict`,
        :class:`~physicalai.robot.errors.RobotDeviceAlreadyOwned`, or
        :class:`~physicalai.robot.errors.RobotTransportError` depending on
        why no owner could be attached to or spawned — see
        :meth:`_spawn_or_reprobe` and :meth:`_validate_metadata`.

        """
        if self._connected:
            return

        budget = self._connect_timeout
        self._session = self._provided_session or open_session(self._name, allow_remote=self._allow_remote)
        try:
            metadata = self._resolve_metadata(budget)
            self._validate_metadata(metadata)
            self._metadata = metadata
            self._attach()
        except Exception:
            self._teardown()
            raise
        self._connected = True

    def _resolve_metadata(self, timeout: float) -> dict[str, Any]:
        """Probe for an existing owner's ``/metadata``, spawning one if none answers.

        Returns:
            The owner's metadata record — either from an existing owner or
            a freshly spawned one.
        """
        metadata = _query_metadata(self._session, metadata_key(self._name), timeout=_PROBE_TIMEOUT)
        if metadata is None and self._allow_remote:
            # A fresh remote peer needs time to learn routes through Zenoh
            # scouting before the owner's queryable is reachable.
            metadata = _query_metadata_with_retry(
                self._session,
                metadata_key(self._name),
                timeout=min(_RACE_RETRY_TIMEOUT, timeout),
            )
        if metadata is None:
            return self._spawn_or_reprobe(timeout)
        return metadata

    def _spawn_or_reprobe(self, timeout: float) -> dict[str, Any]:
        """Spawn an owner; on a benign name-lock race, attach to the winner instead.

        Returns:
            The owner's metadata record.

        Raises:
            RobotTransportError: If no robot config was given (attach-only)
                or spawning failed for a reason other than a benign
                same-device race.
            RobotNameConflict: If the race was against a *different*-device
                owner under the same name.
            RobotDeviceAlreadyOwned: If a requested device is already
                locked under another name — propagated as-is, no re-probe
                needed.
        """
        if self._robot is None:
            msg = f"no owner found for {self._name!r} (attach-only mode: robot config not provided)"
            raise RobotTransportError(msg)

        from ._owner import RobotOwner  # noqa: PLC0415
        from ._owner_config import RobotOwnerConfig  # noqa: PLC0415

        config = RobotOwnerConfig(
            name=self._name,
            robot=self._robot,
            allow_remote=self._allow_remote,
            rate_hz=self._rate_hz,
            idle_timeout=self._idle_timeout,
        )
        owner = RobotOwner(config)
        try:
            owner.start(timeout=timeout)
        except RobotDeviceAlreadyOwned:
            raise
        except RobotTransportError as exc:
            if exc.phase == "name_lock_contention":
                metadata = _query_metadata_with_retry(
                    self._session,
                    metadata_key(self._name),
                    timeout=_RACE_RETRY_TIMEOUT,
                )
                if metadata is not None:
                    # A name-lock contention is always host-local, so the winner's
                    # device ids come from its private name-lock diagnostic
                    winner_ids = active_owner_device_ids(self._name)
                    if winner_ids is None:
                        msg = f"could not determine device identities for local owner {self._name!r}"
                        raise RobotTransportError(msg, phase=exc.phase) from exc
                    mine_ids = tuple(sorted(exc.device_ids or ()))
                    if winner_ids == mine_ids:
                        logger.debug(f"Lost owner race for {self._name!r} — attaching to existing owner")
                        return metadata
                    msg = (
                        f"name {self._name!r} is claimed by different devices: "
                        f"this instance={mine_ids}, existing owner={winner_ids}"
                    )
                    raise RobotNameConflict(msg, phase=exc.phase) from exc
            msg = f"failed to start robot owner for {self._name!r}"
            raise RobotTransportError(msg) from exc
        else:
            # Keep the handle alive; the detached owner self-terminates via
            # idle timeout, so disconnect() never stops it explicitly.
            self._owner = owner

        metadata = _query_metadata_with_retry(self._session, metadata_key(self._name), timeout=_RACE_RETRY_TIMEOUT)
        if metadata is None:
            msg = f"owner for {self._name!r} reported READY but its /metadata queryable is unreachable"
            raise RobotTransportError(msg)
        return metadata

    def _validate_metadata(self, metadata: dict[str, Any]) -> None:
        """Validate protocol compatibility and internal metadata consistency.

        Never imports the owner-advertised ``robot_class`` — network
        metadata is untrusted (only compared as a string). A mismatch is
        logged, not raised: public re-exports, wrappers, and subclasses can
        all preserve the wire contract, so an exact string match is useful
        evidence of a wrong name, not proof of incompatibility.

        Raises:
            RobotProtocolMismatch: If the owner's transport protocol
                version is unsupported.
            RobotTransportError: If the owner's metadata is internally
                inconsistent (malformed joint/dimension fields).
        """
        protocol_version = metadata.get("protocol_version")
        if protocol_version != ROBOT_TRANSPORT_PROTOCOL_VERSION:
            msg = (
                f"owner of {self._name!r} speaks protocol_version={protocol_version!r}, "
                f"this SharedRobot supports {ROBOT_TRANSPORT_PROTOCOL_VERSION!r}"
            )
            raise RobotProtocolMismatch(msg)

        joint_names = metadata.get("joint_names")
        num_joints = metadata.get("num_joints")
        state_dim = metadata.get("state_dim")
        malformed = (
            not isinstance(joint_names, list)
            or not joint_names
            or len(set(joint_names)) != len(joint_names)
            or num_joints != len(joint_names)
            or not isinstance(state_dim, int)
            or state_dim <= 0
        )
        if malformed:
            msg = f"owner of {self._name!r} published malformed /metadata: {metadata!r}"
            raise RobotTransportError(msg)

        if self._robot is not None:
            expected = self._robot["class_path"]
            advertised = metadata.get("robot_class")
            if advertised != expected:
                logger.warning(
                    f"SharedRobot(name={self._name!r}) was constructed with "
                    f"robot.class_path={expected!r} but the existing owner advertises "
                    f"robot_class={advertised!r}. Not fatal — public re-exports, wrappers, or "
                    "subclasses can preserve the wire contract — but double-check this is the "
                    "robot you expect.",
                )

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
            state_key(self._name),
            zenoh.handlers.RingChannel(1),
        )
        # QoS (D20): express bypasses batching; best-effort/drop match the
        # fire-and-forget, latest-wins action semantics.
        self._action_pub = self._session.declare_publisher(
            action_key(self._name),
            reliability=zenoh.Reliability.BEST_EFFORT,
            congestion_control=zenoh.CongestionControl.DROP,
            express=True,
        )

        deadline = time.monotonic() + _FIRST_STATE_TIMEOUT
        while time.monotonic() < deadline:
            sample = self._state_sub.try_recv()
            if sample is not None:
                try:
                    self._latest = decode_state(sample.payload.to_bytes())
                except Exception:  # noqa: BLE001
                    # A malformed sample from a corrupted or incompatible
                    # owner must not abort attach -- keep waiting for a
                    # good one until the deadline.
                    logger.warning(f"Failed to decode state for {self._name!r}", exc_info=True)
                else:
                    return
            time.sleep(0.005)

        msg = f"no state received from owner of {self._name!r} within {_FIRST_STATE_TIMEOUT:.1f}s"
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
            RobotTransportError: If no owner state has been cached.
        """
        if not self._connected or self._state_sub is None:
            msg = "SharedRobot is not connected. Call connect() first."
            raise RobotNotConnectedError(msg)

        # Ring(1) holds only the newest sample; a stalled subscriber
        # resumes on current state, never a backlog of stale samples.
        sample = self._state_sub.try_recv()
        if sample is not None:
            try:
                self._latest = decode_state(sample.payload.to_bytes())
            except Exception:  # noqa: BLE001
                # A malformed sample must not crash the caller (e.g. a
                # PolicyRuntime control loop) -- fall back to the last
                # known-good state instead.
                logger.warning(f"Failed to decode state for {self._name!r}", exc_info=True)

        if self._latest is None:
            msg = "SharedRobot has no cached state. Call connect() first."
            raise RobotTransportError(msg)
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
        if self._session is not None and self._owns_session:
            try:
                self._session.close()
            except Exception:  # noqa: BLE001
                logger.debug("Error closing zenoh session", exc_info=True)
        self._session = None
        self._connected = False
        self._latest = None
        self._metadata = None
