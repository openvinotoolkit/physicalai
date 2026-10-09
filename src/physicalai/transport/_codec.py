# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Safe shared MessagePack helpers for NumPy-bearing transport payloads."""

from __future__ import annotations

import math
from typing import Any

import msgpack
import numpy as np

DEFAULT_MAX_PAYLOAD_BYTES = 32 * 2**20
DEFAULT_MAX_PAYLOAD_DEPTH = 64


def _ensure_contiguous(array: np.ndarray) -> np.ndarray:
    """Make non-scalar arrays contiguous without changing scalar shape."""
    if array.ndim == 0:
        return array
    return np.ascontiguousarray(array)


def encode_numpy(array: np.ndarray) -> dict[str, Any]:
    """Encode a NumPy array using the shared ``__np__`` MessagePack shape."""
    if array.dtype.kind not in "biufc":
        raise ValueError("NumPy array dtype must be numeric or bool")
    contiguous = _ensure_contiguous(array)
    return {
        "__np__": True,
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "data": contiguous.tobytes(),
    }


def msgpack_default(value: object) -> object:
    """Encode supported NumPy values for ``msgpack.packb``."""
    if isinstance(value, np.ndarray):
        return encode_numpy(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    msg = f"Unsupported MessagePack transport type: {type(value).__name__}"
    raise TypeError(msg)


def pack_msgpack(payload: object) -> bytes:
    """Pack a value using binary-safe MessagePack and the shared NumPy encoder."""
    return msgpack.packb(payload, default=msgpack_default, use_bin_type=True)


def _decode_value(
    value: object,
    *,
    max_bytes: int,
    array_bytes: list[int],
    depth: int,
    max_depth: int,
) -> object:
    if isinstance(value, dict):
        if depth > max_depth:
            raise ValueError(f"MessagePack payload nesting exceeds the {max_depth}-level limit")
        if value.get("__np__") is True:
            dtype_value = value.get("dtype")
            shape_value = value.get("shape")
            data = value.get("data")
            if not isinstance(dtype_value, str) or not isinstance(shape_value, list) or not isinstance(data, bytes):
                raise ValueError("Malformed __np__ array payload")
            try:
                dtype = np.dtype(dtype_value)
            except (TypeError, ValueError) as error:
                raise ValueError("Invalid __np__ array dtype") from error
            if dtype.kind not in "biufc":
                raise ValueError("__np__ array dtype must be numeric or bool")
            if any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape_value):
                raise ValueError("Invalid __np__ array shape")
            expected_size = math.prod(shape_value) * dtype.itemsize
            if expected_size != len(data):
                raise ValueError("__np__ array shape and dtype do not match its data length")
            if expected_size > max_bytes - array_bytes[0]:
                raise ValueError(f"Decoded NumPy arrays exceed the {max_bytes}-byte limit")
            array_bytes[0] += expected_size
            return np.frombuffer(data, dtype=dtype).reshape(tuple(shape_value))
        return {
            key: _decode_value(
                item,
                max_bytes=max_bytes,
                array_bytes=array_bytes,
                depth=depth + 1,
                max_depth=max_depth,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        if depth > max_depth:
            raise ValueError(f"MessagePack payload nesting exceeds the {max_depth}-level limit")
        return [
            _decode_value(
                item,
                max_bytes=max_bytes,
                array_bytes=array_bytes,
                depth=depth + 1,
                max_depth=max_depth,
            )
            for item in value
        ]
    return value


def decode_payload(
    value: object,
    *,
    max_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    max_depth: int = DEFAULT_MAX_PAYLOAD_DEPTH,
) -> object:
    """Recursively decode tagged arrays with dtype, shape, and aggregate-size validation."""
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0:
        raise ValueError("max_bytes must be a non-negative integer")
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0:
        raise ValueError("max_depth must be a non-negative integer")
    return _decode_value(value, max_bytes=max_bytes, array_bytes=[0], depth=0, max_depth=max_depth)


def unpack_msgpack(
    data: bytes,
    *,
    max_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    max_depth: int = DEFAULT_MAX_PAYLOAD_DEPTH,
) -> object:
    """Unpack and safely decode a MessagePack payload before NumPy allocation."""
    if len(data) > max_bytes:
        raise ValueError(f"MessagePack payload of {len(data)} bytes exceeds the {max_bytes}-byte limit")
    try:
        unpacked = msgpack.unpackb(data, raw=False, strict_map_key=False)
    except (msgpack.UnpackException, TypeError, ValueError, OverflowError) as error:
        raise ValueError("Invalid MessagePack payload") from error
    return decode_payload(unpacked, max_bytes=max_bytes, max_depth=max_depth)
