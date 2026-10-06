# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fuzz the robot transport msgpack codec (``_codec.py``).

Raw-bytes mode feeds arbitrary bytes straight into the four decode entry
points for msgpack/numpy parser robustness. Structure-aware mode builds an
action/state/metadata record from fuzz-derived field values including
malformed ``__np__`` markers, missing keys, and wrong-typed fields
``msgpack.packb()``s it, and decodes it so mutation reaches dtype/shape/key
validation instead of stopping at msgpack syntax.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import atheris
import numpy as np

with atheris.instrument_imports():
    import msgpack

    from physicalai.robot.transport._codec import (
        _unpack_payload,
        decode_action,
        decode_metadata,
        decode_state,
        encode_action,
        encode_metadata,
        encode_state,
    )

from _helpers import make_float_array

# Exceptions the codec's own contract documents as expected parser rejections: the 1 MiB
# size gate, the nesting-depth gate, the dict-root type gate, msgpack syntax errors, numpy
# dtype/shape errors, and missing required record keys. MemoryError, RecursionError, a hang,
# or any other undocumented exception is a finding -- do not widen this tuple to `Exception`.
_EXPECTED_DECODE_EXCEPTIONS = (
    ValueError,
    TypeError,
    KeyError,
    UnicodeDecodeError,
    OverflowError,
    msgpack.exceptions.UnpackException,
    msgpack.exceptions.ExtraData,
)

_REAL_DTYPES = ["float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "bool", "complex64"]
_BAD_DTYPE_STRINGS = ["", "not-a-dtype", "O", "V0", "U10"]
# Metadata is a flat dict of up to this many fields -- not a nesting depth.
# Deep nesting is exercised separately, by raw-bytes mode mutating real msgpack bytes.
_MAX_METADATA_FIELDS = 6


def _float_eq(first: float, second: float) -> bool:
    """Equality that treats NaN as equal to NaN, like a literal wire round-trip should."""
    if first == second:
        return True
    return isinstance(first, float) and isinstance(second, float) and np.isnan(first) and np.isnan(second)


def _well_formed_array(fdp: atheris.FuzzedDataProvider) -> np.ndarray:
    dtype = fdp.PickValueInList(_REAL_DTYPES)
    # ndim=0 (scalar) is included: encode_action()/encode_state() must preserve it exactly.
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


def _malformed_np_marker(fdp: atheris.FuzzedDataProvider) -> dict:
    """Build a ``{"__np__": ...}`` marker with fuzz-derived corruption."""
    marker: dict = {"__np__": True}
    if fdp.ConsumeBool():
        marker["dtype"] = fdp.PickValueInList(_REAL_DTYPES + _BAD_DTYPE_STRINGS)
    if fdp.ConsumeBool():
        shape_kind = fdp.ConsumeIntInRange(0, 2)
        if shape_kind == 0:
            marker["shape"] = [fdp.ConsumeIntInRange(-8, 16) for _ in range(fdp.ConsumeIntInRange(0, 4))]
        elif shape_kind == 1:
            marker["shape"] = fdp.ConsumeUnicodeNoSurrogates(8)  # wrong type: str instead of list
        else:
            marker["shape"] = fdp.ConsumeIntInRange(0, 2**40)  # wrong type: int instead of list
    if fdp.ConsumeBool():
        marker["data"] = fdp.ConsumeBytes(fdp.ConsumeIntInRange(0, 128))
    return marker


def _fuzz_field_value(fdp: atheris.FuzzedDataProvider, *, prefer_np: bool) -> object:
    """One fuzz-derived field value: a numpy marker (well- or malformed), a scalar, or text."""
    kind = fdp.ConsumeIntInRange(0, 3)
    if kind == 0 and prefer_np:
        return {
            "__np__": True,
            "dtype": (array := _well_formed_array(fdp)).dtype.name,
            "shape": list(array.shape),
            "data": array.tobytes(),
        }
    if kind == 1:
        return _malformed_np_marker(fdp)
    if kind == 2:
        return fdp.ConsumeFloat()
    return fdp.ConsumeUnicodeNoSurrogates(16)


def _fuzz_metadata(fdp: atheris.FuzzedDataProvider, *, allow_np_markers: bool) -> dict:
    metadata: dict = {}
    for _ in range(fdp.ConsumeIntInRange(0, _MAX_METADATA_FIELDS)):
        key = fdp.ConsumeUnicodeNoSurrogates(16) or "k"
        if allow_np_markers and fdp.ConsumeBool():
            metadata[key] = _fuzz_field_value(fdp, prefer_np=True)
            continue
        kind = fdp.ConsumeIntInRange(0, 3)
        if kind == 0:
            metadata[key] = fdp.ConsumeIntInRange(-(2**31), 2**31)
        elif kind == 1:
            metadata[key] = fdp.ConsumeUnicodeNoSurrogates(32)
        elif kind == 2:
            metadata[key] = fdp.ConsumeBool()
        else:
            metadata[key] = [fdp.ConsumeUnicodeNoSurrogates(8) for _ in range(fdp.ConsumeIntInRange(0, 3))]
    return metadata


@atheris.instrument_func
def _raw_bytes_mode(fdp: atheris.FuzzedDataProvider) -> None:
    """Feed arbitrary bytes straight into the decode entry points."""
    target = fdp.ConsumeIntInRange(0, 3)
    data = fdp.ConsumeBytes(fdp.remaining_bytes())
    try:
        if target == 0:
            _unpack_payload(data)
        elif target == 1:
            decode_action(data)
        elif target == 2:
            decode_state(data)
        else:
            decode_metadata(data)
    except _EXPECTED_DECODE_EXCEPTIONS:
        pass


@atheris.instrument_func
def _structured_valid_mode(fdp: atheris.FuzzedDataProvider) -> None:
    """Round-trip a genuinely valid record through encode/decode."""
    target = fdp.ConsumeIntInRange(0, 2)
    if target == 0:
        action = _well_formed_array(fdp)
        goal_time = fdp.ConsumeFloat()

        decoded, decoded_goal_time, decoded_ts = decode_action(encode_action(action, goal_time))

        assert decoded.shape == action.shape, f"action round-trip changed shape: {decoded.shape} != {action.shape}"
        np.testing.assert_array_equal(decoded, action, err_msg="action round-trip changed values")
        assert decoded.dtype == action.dtype, f"action round-trip changed dtype: {decoded.dtype} != {action.dtype}"
        assert _float_eq(decoded_goal_time, goal_time), "goal_time round-trip changed value"
        assert decoded_ts > 0, "encode_action must stamp a positive monotonic ts"
    elif target == 1:
        joint_positions = make_float_array(fdp, max_ndim=1, max_dim=32)
        state = make_float_array(fdp, max_ndim=1, max_dim=32)
        timestamp = fdp.ConsumeFloat()
        sensor_data = None
        if fdp.ConsumeBool():
            sensor_data = {fdp.ConsumeUnicodeNoSurrogates(8) or "s": make_float_array(fdp, max_ndim=1, max_dim=16)}

        obs = decode_state(
            encode_state(joint_positions=joint_positions, state=state, timestamp=timestamp, sensor_data=sensor_data)
        )

        assert obs.joint_positions.shape == joint_positions.shape, (
            f"joint_positions round-trip changed shape: {obs.joint_positions.shape} != {joint_positions.shape}"
        )
        assert obs.joint_positions.dtype == joint_positions.dtype, (
            f"joint_positions round-trip changed dtype: {obs.joint_positions.dtype} != {joint_positions.dtype}"
        )
        np.testing.assert_array_equal(
            obs.joint_positions, joint_positions, err_msg="joint_positions round-trip changed values"
        )
        assert obs.state.shape == state.shape, f"state round-trip changed shape: {obs.state.shape} != {state.shape}"
        assert obs.state.dtype == state.dtype, f"state round-trip changed dtype: {obs.state.dtype} != {state.dtype}"
        np.testing.assert_array_equal(obs.state, state, err_msg="state round-trip changed values")
        assert _float_eq(obs.timestamp, timestamp), "timestamp round-trip changed value"
        if sensor_data is None:
            assert obs.sensor_data is None, "sensor_data round-trip changed None to a value"
        else:
            assert obs.sensor_data is not None, "sensor_data round-trip dropped a value"
            assert obs.sensor_data.keys() == sensor_data.keys(), (
                f"sensor_data round-trip changed keys: {obs.sensor_data.keys()} != {sensor_data.keys()}"
            )
            for key, array in sensor_data.items():
                decoded_array = obs.sensor_data[key]
                assert decoded_array.shape == array.shape, (
                    f"sensor_data[{key!r}] round-trip changed shape: {decoded_array.shape} != {array.shape}"
                )
                assert decoded_array.dtype == array.dtype, (
                    f"sensor_data[{key!r}] round-trip changed dtype: {decoded_array.dtype} != {array.dtype}"
                )
                np.testing.assert_array_equal(
                    decoded_array, array, err_msg=f"sensor_data[{key!r}] round-trip changed values"
                )
    else:
        metadata = _fuzz_metadata(fdp, allow_np_markers=False)
        assert decode_metadata(encode_metadata(metadata)) == metadata, "metadata round-trip changed value"


@atheris.instrument_func
def _structured_malformed_mode(fdp: atheris.FuzzedDataProvider) -> None:
    """Pack a deliberately malformed record; only documented exceptions may reach the caller."""
    target = fdp.ConsumeIntInRange(0, 2)
    fields_by_target = {
        0: ("action", "goal_time", "ts"),
        1: ("joint_positions", "state", "timestamp", "sensor_data"),
    }

    if target == 2:
        record = _fuzz_metadata(fdp, allow_np_markers=True)
    else:
        record = {}
        for name in fields_by_target[target]:
            if fdp.ConsumeBool():
                continue  # omit this key -- exercises KeyError at decode_*
            record[name] = _fuzz_field_value(fdp, prefer_np=name not in ("goal_time", "ts", "timestamp"))

    try:
        data = msgpack.packb(record, use_bin_type=True)
    except (TypeError, ValueError, OverflowError):
        return  # the corrupted field value was not msgpack-packable in the first place

    try:
        if target == 0:
            decode_action(data)
        elif target == 1:
            decode_state(data)
        else:
            decode_metadata(data)
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
