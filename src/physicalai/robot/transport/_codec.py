# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Wire format for the robot Zenoh transport.

Zenoh moves opaque bytes; this module defines the msgpack record schemas
for the ``/state``, ``/action``, and ``/meta`` keys. Numpy arrays are
encoded via :mod:`physicalai._serialization` so dtype and shape round-trip
exactly. Images are intentionally excluded from ``/state`` — frames go
through the capture transport (``SharedCamera``), not this one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from physicalai._serialization import encode_numpy, pack_payload, unpack_payload  # noqa: PLC2701


@dataclass
class TransportObservation:
    """Robot observation reconstructed from a ``/state`` sample.

    Satisfies the :class:`~physicalai.robot.interface.RobotObservation`
    protocol. ``state`` is the owner-computed vector shipped on the wire —
    the robot-specific concat logic (e.g. WidowXAI positions+velocities)
    lives only on the owner and is never re-implemented subscriber-side.
    """

    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Any] | None = None
    _state: np.ndarray | None = None

    @property
    def state(self) -> np.ndarray:
        """Owner-computed state vector shipped on the wire.

        Returns:
            The shipped ``state`` vector, falling back to
            ``joint_positions`` if the owner did not ship one.
        """
        if self._state is not None:
            return self._state
        return self.joint_positions


def encode_state(
    *,
    joint_positions: np.ndarray,
    state: np.ndarray,
    timestamp: float,
    sensor_data: dict[str, np.ndarray] | None,
) -> bytes:
    """Encode a ``/state`` record.

    Args:
        joint_positions: Measured joint positions.
        state: Owner-computed robot-specific state vector.
        timestamp: ``time.monotonic()`` at capture (staleness signal).
        sensor_data: Optional auxiliary arrays (velocities, efforts, ...).

    Returns:
        msgpack-encoded bytes.
    """
    payload: dict[str, Any] = {
        "joint_positions": encode_numpy(np.ascontiguousarray(joint_positions)),
        "state": encode_numpy(np.ascontiguousarray(state)),
        "timestamp": timestamp,
        "sensor_data": (
            {k: encode_numpy(np.ascontiguousarray(v)) for k, v in sensor_data.items()}
            if sensor_data is not None
            else None
        ),
    }
    return pack_payload(payload)


def decode_state(data: bytes) -> TransportObservation:
    """Decode a ``/state`` record into an observation.

    Args:
        data: msgpack-encoded bytes from :func:`encode_state`.

    Returns:
        A :class:`TransportObservation` with exact dtypes/shapes.
    """
    record = unpack_payload(data)
    return TransportObservation(
        joint_positions=record["joint_positions"],
        timestamp=record["timestamp"],
        sensor_data=record.get("sensor_data"),
        images=None,
        _state=record.get("state"),
    )


def encode_action(action: np.ndarray, goal_time: float) -> bytes:
    """Encode an ``/action`` record.

    Actions are absolute joint targets, so latest-wins delivery is safe:
    dropping an intermediate target just skips to the newest.

    Args:
        action: Absolute joint targets.
        goal_time: Minimum time (seconds) to reach the target.

    Returns:
        msgpack-encoded bytes.
    """
    payload: dict[str, Any] = {
        "action": encode_numpy(np.ascontiguousarray(action)),
        "goal_time": goal_time,
        "ts": time.monotonic(),
    }
    return pack_payload(payload)


def decode_action(data: bytes) -> tuple[np.ndarray, float, float]:
    """Decode an ``/action`` record.

    Args:
        data: msgpack-encoded bytes from :func:`encode_action`.

    Returns:
        Tuple of ``(action, goal_time, send_ts)``.
    """
    record = unpack_payload(data)
    return record["action"], record["goal_time"], record["ts"]


def encode_meta(meta: dict[str, Any]) -> bytes:
    """Encode a ``/meta`` record (discovery + attach validation).

    Args:
        meta: Informational dict — ``robot_type``, ``joint_names``,
            ``host``, ``connection``, ``state_dim``, ``num_joints``.

    Returns:
        msgpack-encoded bytes.
    """
    return pack_payload(meta)


def decode_meta(data: bytes) -> dict[str, Any]:
    """Decode a ``/meta`` record.

    Args:
        data: msgpack-encoded bytes from :func:`encode_meta`.

    Returns:
        The meta dict.
    """
    return unpack_payload(data)
