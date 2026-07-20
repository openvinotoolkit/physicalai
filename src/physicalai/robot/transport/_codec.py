# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Wire format for the robot Zenoh transport.

Zenoh moves opaque bytes; this module defines the msgpack record schemas
for the ``/state``, ``/action``, and ``/metadata`` keys. Numpy arrays are
encoded with dtype and shape so they round-trip exactly. Images are
intentionally excluded from ``/state`` — frames go through the capture
transport (``SharedCamera``), not this one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

ROBOT_TRANSPORT_PROTOCOL_VERSION = 1
"""Version of the robot transport wire contract (not the robot class or
package release). Subscribers reject an owner advertising an unsupported
version before declaring the action publisher, so an incompatible owner
never receives commands it might misinterpret.

Bump this constant in the same change that introduces a backward-
incompatible ``/state``, ``/action``, or ``/metadata`` payload, required-
field, or semantic change. Do **not** bump it for additive optional
fields, internal refactors, robot-driver changes, or package releases that
preserve wire compatibility.
"""
_MAX_PAYLOAD_BYTES = 1024 * 1024
"""Upper bound on a single encoded record.

The richest realistic record (bimanual state: 14-dim ``joint_positions`` +
28-dim ``state`` + a few ``sensor_data`` arrays) is on the order of a few
KB. 1 MiB leaves generous headroom for future growth while still rejecting
a corrupted or hostile payload before any unpacking work is spent on it.
"""


def _encode_numpy(array: np.ndarray) -> dict[str, Any]:
    return {
        "__np__": True,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data": array.tobytes(),
    }


def _decode_payload(value: object) -> object:
    if isinstance(value, dict):
        if value.get("__np__"):
            return np.frombuffer(value["data"], dtype=np.dtype(value["dtype"])).reshape(value["shape"])
        return {key: _decode_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_payload(item) for item in value]
    return value


def _msgpack_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _encode_numpy(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    msg = f"Unsupported robot transport type for msgpack serialization: {type(value).__name__}"
    raise TypeError(msg)


def _pack_payload(payload: dict[str, Any]) -> bytes:
    import msgpack  # noqa: PLC0415

    return msgpack.packb(payload, default=_msgpack_default, use_bin_type=True)  # type: ignore[return-value]


def _unpack_payload(data: bytes) -> dict[str, Any]:
    import msgpack  # noqa: PLC0415

    if len(data) > _MAX_PAYLOAD_BYTES:
        msg = f"Robot transport payload of {len(data)} bytes exceeds the {_MAX_PAYLOAD_BYTES}-byte limit"
        raise ValueError(msg)

    payload = _decode_payload(msgpack.unpackb(data, raw=False))
    if not isinstance(payload, dict):
        msg = f"Expected a dict payload, got {type(payload).__name__}"
        raise TypeError(msg)
    return payload


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
        "joint_positions": _encode_numpy(np.ascontiguousarray(joint_positions)),
        "state": _encode_numpy(np.ascontiguousarray(state)),
        "timestamp": timestamp,
        "sensor_data": (
            {k: _encode_numpy(np.ascontiguousarray(v)) for k, v in sensor_data.items()}
            if sensor_data is not None
            else None
        ),
    }
    return _pack_payload(payload)


def decode_state(data: bytes) -> TransportObservation:
    """Decode a ``/state`` record into an observation.

    Args:
        data: msgpack-encoded bytes from :func:`encode_state`.

    Returns:
        A :class:`TransportObservation` with exact dtypes/shapes.
    """
    record = _unpack_payload(data)
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
        "action": _encode_numpy(np.ascontiguousarray(action)),
        "goal_time": goal_time,
        "ts": time.monotonic(),
    }
    return _pack_payload(payload)


def decode_action(data: bytes) -> tuple[np.ndarray, float, float]:
    """Decode an ``/action`` record.

    Args:
        data: msgpack-encoded bytes from :func:`encode_action`.

    Returns:
        Tuple of ``(action, goal_time, send_ts)``.
    """
    record = _unpack_payload(data)
    return record["action"], record["goal_time"], record["ts"]


def encode_metadata(metadata: dict[str, Any]) -> bytes:
    """Encode a ``/metadata`` record (discovery + protocol validation).

    Args:
        metadata: Informational dict — ``protocol_version``, ``name``,
            ``robot_class``, optional ``device_ids``, ``host``,
            ``joint_names``, ``num_joints``, ``state_dim``. Remote-capable
            owners omit ``device_ids``. Must never include constructor
            kwargs, calibration paths/contents, credentials, or tokens.

    Returns:
        msgpack-encoded bytes.
    """
    return _pack_payload(metadata)


def decode_metadata(data: bytes) -> dict[str, Any]:
    """Decode a ``/metadata`` record.

    Args:
        data: msgpack-encoded bytes from :func:`encode_metadata`.

    Returns:
        The metadata dict.
    """
    return _unpack_payload(data)
