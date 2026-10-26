# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Safe shared MessagePack helpers for NumPy-bearing transport payloads."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

DEFAULT_MAX_PAYLOAD_BYTES = 32 * 2**20
DEFAULT_MAX_PAYLOAD_DEPTH = 64


def _ensure_contiguous(array: np.ndarray) -> np.ndarray:
    """Make non-scalar arrays contiguous without changing scalar shape.

    Returns:
        A contiguous array retaining the original shape.
    """
    if array.ndim == 0:
        return array
    return np.ascontiguousarray(array)


def encode_numpy(array: np.ndarray) -> dict[str, Any]:
    """Encode a NumPy array using the shared ``__np__`` MessagePack shape.

    Returns:
        A tagged map of array dtype, shape, and raw bytes.

    Raises:
        ValueError: If the array is not numeric or boolean.
    """
    if array.dtype.kind not in "biufc":
        msg = "NumPy array dtype must be numeric or bool"
        raise ValueError(msg)
    contiguous = _ensure_contiguous(array)
    return {
        "__np__": True,
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "data": contiguous.tobytes(),
    }


def msgpack_default(value: object) -> object:
    """Encode supported NumPy values for ``msgpack.packb``.

    Returns:
        A tagged array map or native numeric scalar.

    Raises:
        TypeError: If the value has no supported MessagePack representation.
    """
    if isinstance(value, np.ndarray):
        return encode_numpy(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    msg = f"Unsupported MessagePack transport type: {type(value).__name__}"
    raise TypeError(msg)


def pack_msgpack(payload: object) -> bytes:
    """Pack a value using binary-safe MessagePack and the shared NumPy encoder.

    Returns:
        Encoded MessagePack bytes.

    Raises:
        TypeError: If the encoder cannot represent a value or return bytes.
    """
    import msgpack  # noqa: PLC0415

    packed = msgpack.packb(payload, default=msgpack_default, use_bin_type=True)
    if not isinstance(packed, bytes):
        msg = "MessagePack encoder must return bytes"
        raise TypeError(msg)
    return packed


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
            msg = f"MessagePack payload nesting exceeds the {max_depth}-level limit"
            raise ValueError(msg)
        if value.get("__np__") is True:
            dtype_value = value.get("dtype")
            shape_value = value.get("shape")
            data = value.get("data")
            if not isinstance(dtype_value, str) or not isinstance(shape_value, list) or not isinstance(data, bytes):
                msg = "Malformed __np__ array payload"
                raise ValueError(msg)
            try:
                dtype = np.dtype(dtype_value)
            except (TypeError, ValueError) as error:
                msg = "Invalid __np__ array dtype"
                raise ValueError(msg) from error
            if dtype.kind not in "biufc":
                msg = "__np__ array dtype must be numeric or bool"
                raise ValueError(msg)
            if any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape_value):
                msg = "Invalid __np__ array shape"
                raise ValueError(msg)
            expected_size = math.prod(shape_value) * dtype.itemsize
            if expected_size != len(data):
                msg = "__np__ array shape and dtype do not match its data length"
                raise ValueError(msg)
            if expected_size > max_bytes - array_bytes[0]:
                msg = f"Decoded NumPy arrays exceed the {max_bytes}-byte limit"
                raise ValueError(msg)
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
            msg = f"MessagePack payload nesting exceeds the {max_depth}-level limit"
            raise ValueError(msg)
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
    """Recursively decode tagged arrays with dtype, shape, and aggregate-size validation.

    Returns:
        The decoded object with tagged arrays replaced by NumPy arrays.

    Raises:
        ValueError: If limits or array metadata are invalid.
    """
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0:
        msg = "max_bytes must be a non-negative integer"
        raise ValueError(msg)
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0:
        msg = "max_depth must be a non-negative integer"
        raise ValueError(msg)
    return _decode_value(value, max_bytes=max_bytes, array_bytes=[0], depth=0, max_depth=max_depth)


def unpack_msgpack(
    data: bytes,
    *,
    max_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    max_depth: int = DEFAULT_MAX_PAYLOAD_DEPTH,
) -> object:
    """Unpack and safely decode a MessagePack payload before NumPy allocation.

    Returns:
        The decoded object, including validated NumPy arrays.

    Raises:
        ValueError: If the payload is malformed or exceeds a decoding limit.
    """
    import msgpack  # noqa: PLC0415

    if len(data) > max_bytes:
        msg = f"MessagePack payload of {len(data)} bytes exceeds the {max_bytes}-byte limit"
        raise ValueError(msg)
    try:
        unpacked = msgpack.unpackb(data, raw=False, strict_map_key=False)
    except (msgpack.UnpackException, TypeError, ValueError, OverflowError) as error:
        msg = "Invalid MessagePack payload"
        raise ValueError(msg) from error
    return decode_payload(unpacked, max_bytes=max_bytes, max_depth=max_depth)
