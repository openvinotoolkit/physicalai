# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Msgpack and NumPy encoding for runtime observer telemetry."""

from __future__ import annotations

from typing import Any

import numpy as np

_MAX_PAYLOAD_DEPTH = 32
"""Nesting depth limit for :func:`decode_payload`'s recursive walk.
"""


def encode_numpy(array: np.ndarray) -> dict[str, Any]:
    """Encode an array with its exact dtype and shape.

    Returns:
        A msgpack-compatible array record.
    """
    return {
        "__np__": True,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data": array.tobytes(),
    }


def decode_numpy(record: dict[str, Any]) -> np.ndarray:
    """Decode an array record produced by :func:`encode_numpy`.

    Returns:
        The reconstructed array.
    """
    return np.frombuffer(record["data"], dtype=np.dtype(record["dtype"])).reshape(record["shape"])


def decode_payload(value: object, _depth: int = 0) -> object:
    """Recursively decode array records in a telemetry payload.

    Returns:
        The payload with array records replaced by NumPy arrays.

    Raises:
        ValueError: If nesting exceeds ``_MAX_PAYLOAD_DEPTH``.
    """
    if isinstance(value, (dict, list)) and _depth > _MAX_PAYLOAD_DEPTH:
        msg = f"Telemetry payload nesting exceeds the {_MAX_PAYLOAD_DEPTH}-level limit"
        raise ValueError(msg)
    if isinstance(value, dict):
        if value.get("__np__"):
            return decode_numpy(value)
        return {key: decode_payload(item, _depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_payload(item, _depth + 1) for item in value]
    return value


def _msgpack_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return encode_numpy(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    msg = f"Unsupported telemetry type for msgpack serialization: {type(value).__name__}"
    raise TypeError(msg)


def pack_payload(payload: dict[str, Any]) -> bytes:
    """Serialize a telemetry record with nested NumPy values.

    Returns:
        The msgpack-encoded payload.
    """
    import msgpack  # noqa: PLC0415

    return msgpack.packb(payload, default=_msgpack_default, use_bin_type=True)  # type: ignore[return-value]
