# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fuzz the shared transport msgpack codec (``_codec.py``).

Raw-bytes mode feeds arbitrary bytes into ``unpack_msgpack`` with fuzz-derived
size and depth limits. Structure-aware mode round-trips genuinely valid tagged
arrays and NumPy scalars through ``pack_msgpack``/``unpack_msgpack``. Malformed
mode sends corrupted ``__np__`` markers, nesting at the depth gate, huge shape
integers, and non-string map keys through ``decode_payload`` so mutation reaches
dtype/shape/data and limit validation instead of stopping at msgpack syntax.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import atheris
import numpy as np

with atheris.instrument_imports():
    import msgpack

    from physicalai.transport._codec import (
        DEFAULT_MAX_PAYLOAD_DEPTH,
        decode_payload,
        encode_numpy,
        pack_msgpack,
        unpack_msgpack,
    )

# The shared codec documents ValueError for every unpack/decode rejection: the
# payload-size gate, the nesting-depth gate, malformed ``__np__`` metadata, and
# msgpack syntax errors. MemoryError, RecursionError, a hang, or any other
# undocumented exception is a finding -- do not widen this tuple to `Exception`.
_EXPECTED_DECODE_EXCEPTIONS = (ValueError,)

_REAL_DTYPES = ["float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "bool", "complex64"]
_BAD_DTYPE_STRINGS = ["", "not-a-dtype", "O", "V0", "U10"]
_BAD_ARRAY_DTYPES = ["O", "U4", "V4", "M8[ns]", "m8[ns]"]


def _float_eq(first: float, second: float) -> bool:
    """Equality that treats NaN as equal to NaN, like a literal wire round-trip should."""
    if first == second:
        return True
    return isinstance(first, float) and isinstance(second, float) and np.isnan(first) and np.isnan(second)


def _well_formed_array(fdp: atheris.FuzzedDataProvider) -> np.ndarray:
    dtype = fdp.PickValueInList(_REAL_DTYPES)
    # ndim=0 (scalar) is included: encode_numpy() must preserve it exactly.
    ndim = fdp.ConsumeIntInRange(0, 3)
    shape = tuple(fdp.ConsumeIntInRange(0, 6) for _ in range(ndim))
    count = 1
    for dim in shape:
        count *= dim
    n_bytes = count * np.dtype(dtype).itemsize
    raw = fdp.ConsumeBytes(n_bytes)
    if len(raw) < n_bytes:
        raw += b"\x00" * (n_bytes - len(raw))
    return np.frombuffer(raw[:n_bytes], dtype=dtype).copy().reshape(shape)


def _malformed_marker(fdp: atheris.FuzzedDataProvider) -> dict:
    """Build a ``{"__np__": ...}`` marker with fuzz-derived corruption."""
    marker: dict = {"__np__": True}
    if fdp.ConsumeBool():
        marker["dtype"] = fdp.PickValueInList([*_REAL_DTYPES, *_BAD_DTYPE_STRINGS, 1, None])
    if fdp.ConsumeBool():
        shape_kind = fdp.ConsumeIntInRange(0, 3)
        if shape_kind == 0:
            marker["shape"] = [fdp.ConsumeIntInRange(-8, 16) for _ in range(fdp.ConsumeIntInRange(0, 4))]
        elif shape_kind == 1:
            marker["shape"] = fdp.ConsumeUnicodeNoSurrogates(8)  # wrong type: str instead of list
        elif shape_kind == 2:
            marker["shape"] = fdp.ConsumeIntInRange(0, 2**40)  # wrong type: int instead of list
        else:
            marker["shape"] = [fdp.ConsumeIntInRange(0, 2**40) for _ in range(fdp.ConsumeIntInRange(0, 3))]
    if fdp.ConsumeBool():
        data_kind = fdp.ConsumeIntInRange(0, 2)
        if data_kind == 0:
            marker["data"] = fdp.ConsumeBytes(fdp.ConsumeIntInRange(0, 128))
        elif data_kind == 1:
            marker["data"] = fdp.ConsumeUnicodeNoSurrogates(16)  # wrong type: str instead of bytes
        else:
            marker["data"] = fdp.ConsumeIntInRange(0, 2**31)  # wrong type: int instead of bytes
    return marker


def _fuzz_max_bytes(fdp: atheris.FuzzedDataProvider) -> int:
    """Pick a size limit, including zero and values near a typical payload size."""
    return fdp.PickValueInList([0, 1, 2, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])


def _fuzz_max_depth(fdp: atheris.FuzzedDataProvider) -> int:
    """Pick a depth limit, including zero and small values near the default."""
    return fdp.PickValueInList([0, 1, 2, 3, 4, 5, 8, DEFAULT_MAX_PAYLOAD_DEPTH - 1, DEFAULT_MAX_PAYLOAD_DEPTH])


@atheris.instrument_func
def _raw_bytes_mode(fdp: atheris.FuzzedDataProvider) -> None:
    """Feed arbitrary bytes straight into the shared unpack entry point."""
    max_bytes = _fuzz_max_bytes(fdp)
    max_depth = _fuzz_max_depth(fdp)
    data = bytes(fdp.ConsumeBytes(fdp.remaining_bytes()) or b"")
    try:
        unpack_msgpack(data, max_bytes=max_bytes, max_depth=max_depth)
    except _EXPECTED_DECODE_EXCEPTIONS:
        pass


@atheris.instrument_func
def _structured_valid_mode(fdp: atheris.FuzzedDataProvider) -> None:
    """Round-trip genuinely valid arrays and NumPy scalars through encode/decode."""
    array = _well_formed_array(fdp)
    scalar_int = np.int64(fdp.ConsumeIntInRange(-(2**31), 2**31))
    scalar_float = np.float64(fdp.ConsumeFloat())

    packed = pack_msgpack({"array": array, "scalar_int": scalar_int, "scalar_float": scalar_float})
    decoded = unpack_msgpack(packed)
    assert isinstance(decoded, dict), "pack_msgpack round-trip did not return a map"

    decoded_array = decoded["array"]
    assert isinstance(decoded_array, np.ndarray), "array round-trip did not return an ndarray"
    assert decoded_array.dtype == array.dtype, f"array round-trip changed dtype: {decoded_array.dtype} != {array.dtype}"
    assert decoded_array.shape == array.shape, f"array round-trip changed shape: {decoded_array.shape} != {array.shape}"
    np.testing.assert_array_equal(decoded_array, array, err_msg="array round-trip changed values")
    assert decoded["scalar_int"] == int(scalar_int), "integer scalar round-trip changed value"
    assert _float_eq(decoded["scalar_float"], float(scalar_float)), "float scalar round-trip changed value"

    marker = encode_numpy(array)
    direct = decode_payload(marker)
    assert isinstance(direct, np.ndarray), "decode_payload did not return an ndarray"
    assert direct.dtype == array.dtype, f"decode_payload changed dtype: {direct.dtype} != {array.dtype}"
    assert direct.shape == array.shape, f"decode_payload changed shape: {direct.shape} != {array.shape}"
    np.testing.assert_array_equal(direct, array, err_msg="decode_payload changed values")


@atheris.instrument_func
def _structured_malformed_mode(fdp: atheris.FuzzedDataProvider) -> None:
    """Pack a deliberately malformed value; only documented exceptions may reach the caller."""
    target = fdp.ConsumeIntInRange(0, 4)
    max_bytes = _fuzz_max_bytes(fdp)
    max_depth = _fuzz_max_depth(fdp)

    if target == 0:
        marker = _malformed_marker(fdp)
        try:
            decode_payload(marker, max_bytes=max_bytes, max_depth=max_depth)
        except _EXPECTED_DECODE_EXCEPTIONS:
            pass
    elif target == 1:
        marker = _malformed_marker(fdp)
        try:
            data = msgpack.packb(marker, use_bin_type=True)
        except (TypeError, ValueError, OverflowError):
            return  # the corrupted field value was not msgpack-packable in the first place
        if not isinstance(data, bytes):
            return
        try:
            unpack_msgpack(data, max_bytes=max_bytes, max_depth=max_depth)
        except _EXPECTED_DECODE_EXCEPTIONS:
            pass
    elif target == 2:
        depth = fdp.ConsumeIntInRange(0, max_depth + 2)
        nested: object = fdp.ConsumeIntInRange(0, 10)
        for _ in range(depth):
            nested = [nested] if fdp.ConsumeBool() else {"k": nested}
        try:
            decode_payload(nested, max_bytes=max_bytes, max_depth=max_depth)
        except _EXPECTED_DECODE_EXCEPTIONS:
            pass
    elif target == 3:
        dtype = fdp.PickValueInList(_BAD_ARRAY_DTYPES)
        try:
            encode_numpy(np.zeros((fdp.ConsumeIntInRange(1, 4),), dtype=np.dtype(dtype)))
        except _EXPECTED_DECODE_EXCEPTIONS:
            pass
    else:
        marker = {
            "__np__": True,
            "dtype": fdp.PickValueInList([*_REAL_DTYPES, *_BAD_DTYPE_STRINGS]),
            "shape": [fdp.ConsumeIntInRange(0, 2**40) for _ in range(fdp.ConsumeIntInRange(0, 3))],
            "data": b"",
        }
        record = {fdp.ConsumeIntInRange(-4, 4): marker, "ok": fdp.ConsumeUnicodeNoSurrogates(8)}
        try:
            decode_payload(record, max_bytes=max_bytes, max_depth=max_depth)
        except _EXPECTED_DECODE_EXCEPTIONS:
            pass


def test_one_input(data: bytes) -> None:
    if len(data) < 4:
        return
    fdp = atheris.FuzzedDataProvider(data)
    mode = fdp.ConsumeIntInRange(0, 2)
    if mode == 0:
        _raw_bytes_mode(fdp)
    elif mode == 1:
        _structured_valid_mode(fdp)
    else:
        _structured_malformed_mode(fdp)


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
