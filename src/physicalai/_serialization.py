# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared msgpack + numpy serialization helpers.

Used by runtime telemetry and the robot Zenoh transport. Numpy arrays are
encoded as ``{"__np__": True, "dtype": ..., "shape": ..., "data": ...}``
records so that dtype and shape round-trip exactly — no float upcasting
(as a JSON list would do) and no precision loss (as ``str(array)`` would).

``msgpack`` is imported lazily so that importing this module does not
require the optional ``transport`` extra.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def encode_numpy(arr: np.ndarray) -> dict[str, Any]:
    """Encode a numpy array as a msgpack-friendly record.

    Args:
        arr: Array to encode.

    Returns:
        A self-describing dict carrying dtype, shape, and raw bytes.
    """
    return {
        "__np__": True,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": arr.tobytes(),
    }


def decode_numpy(obj: dict[str, Any]) -> np.ndarray:
    """Decode a record produced by :func:`encode_numpy`.

    Args:
        obj: Record dict with ``dtype``, ``shape``, and ``data`` keys.

    Returns:
        The reconstructed array with exact dtype and shape.
    """
    return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])


def decode_numpy_recursive(obj: object) -> object:
    """Recursively decode :func:`encode_numpy` records nested in containers.

    Args:
        obj: Arbitrary msgpack-decoded object (dict, list, or scalar).

    Returns:
        The same structure with every numpy record replaced by an array.
    """
    if isinstance(obj, dict):
        if obj.get("__np__"):
            return decode_numpy(obj)
        return {k: decode_numpy_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_numpy_recursive(v) for v in obj]
    return obj


def _msgpack_default(obj: object) -> object:
    """Convert numpy types msgpack cannot serialize natively.

    Args:
        obj: Object rejected by msgpack's native packer.

    Returns:
        A msgpack-serializable representation.

    Raises:
        TypeError: If the object is not a supported numpy type.
    """
    if isinstance(obj, np.ndarray):
        return encode_numpy(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    msg = f"Unsupported payload type for msgpack serialization: {type(obj).__name__}"
    raise TypeError(msg)


def pack_payload(payload: dict[str, Any]) -> bytes:
    """Serialize a heterogeneous record dict with msgpack.

    Numpy arrays (at any nesting depth) are encoded via
    :func:`encode_numpy`; numpy scalars are converted to Python scalars.

    Args:
        payload: Record dict to serialize.

    Returns:
        The msgpack-encoded bytes.
    """
    import msgpack  # noqa: PLC0415

    # packb only returns None in streaming mode (unused here).
    return msgpack.packb(payload, default=_msgpack_default, use_bin_type=True)  # type: ignore[return-value]


def unpack_payload(data: bytes) -> dict[str, Any]:
    """Deserialize bytes produced by :func:`pack_payload`.

    Args:
        data: msgpack-encoded bytes.

    Returns:
        The record dict with nested numpy records decoded to arrays.

    Raises:
        TypeError: If the decoded payload is not a dict.
    """
    import msgpack  # noqa: PLC0415

    obj = decode_numpy_recursive(msgpack.unpackb(data, raw=False))
    if not isinstance(obj, dict):
        msg = f"Expected a dict payload, got {type(obj).__name__}"
        raise TypeError(msg)
    return obj
