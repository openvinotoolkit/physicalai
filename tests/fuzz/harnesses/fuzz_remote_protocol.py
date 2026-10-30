# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fuzz the remote inference protocol (``_protocol.py``).

Raw-bytes mode feeds arbitrary bytes into the message, request, and reply
decoders. Structure-aware mode round-trips genuinely valid predict requests and
replies. Malformed mode packs wrong-typed versions, sequences, budgets, and
timings plus missing keys and bad input/output containers, then verifies that
only ``RemoteInferenceProtocolError`` reaches the caller. Encode-only argument
validation is exercised separately and may raise ``ValueError`` or
``RemoteInferenceProtocolError``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import atheris
import numpy as np

with atheris.instrument_imports():
    from physicalai.inference.remote._protocol import (
        MAX_REQUEST_BUDGET_MS,
        PROTOCOL_VERSION,
        RemoteInferenceProtocolError,
        decode_message,
        decode_predict_reply,
        decode_predict_request,
        encode_message,
        encode_predict_reply,
        encode_predict_request,
    )

from _helpers import make_float_array

# Every protocol decoder documents RemoteInferenceProtocolError for malformed
# payloads; lower-level msgpack/type errors are wrapped before they escape.
# MemoryError, RecursionError, a hang, or any other undocumented exception is a
# finding -- do not widen this tuple to `Exception`.
_EXPECTED_DECODE_EXCEPTIONS = (RemoteInferenceProtocolError,)
# Encoders validate caller arguments directly (ValueError) and wrap codec
# failures (RemoteInferenceProtocolError).
_EXPECTED_ENCODE_EXCEPTIONS = (ValueError, RemoteInferenceProtocolError)

_BAD_VERSIONS = [True, False, 1.0, "1", None, 2, 0, -1]
_BAD_INTEGERS = [True, False, -1, 0, 1.0, "1", None, 2**40, 2**63 - 1]
_BAD_TIMINGS = [0.0, 0.5, -1.0, float("nan"), float("inf"), float("-inf"), True, False, "0.1", None]
_BAD_IMAGES = [{}, [], "images", 1, None, {1: {"codec": "raw"}}]
_BAD_OUTPUTS = [{}, [], "outputs", 1, None, {"action": [1, 2]}, {"action": "not-an-array"}]
_BAD_INPUTS = [{}, [], "inputs", 1, None, {"state": None}, {"state": [1, 2]}, {"images.front": "not-an-image"}]


def _float_eq(first: float, second: float) -> bool:
    """Equality that treats NaN as equal to NaN, like a literal wire round-trip should."""
    if first == second:
        return True
    return isinstance(first, float) and isinstance(second, float) and np.isnan(first) and np.isnan(second)


def _fuzz_max_bytes(fdp: atheris.FuzzedDataProvider) -> int:
    """Pick a request size limit, including zero and values near real payloads."""
    return fdp.PickValueInList([0, 1, 2, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])


def _fuzz_name(fdp: atheris.FuzzedDataProvider, prefix: str) -> str:
    return f"{prefix}_{fdp.ConsumeUnicodeNoSurrogates(8) or 'k'}"


def _valid_inputs(fdp: atheris.FuzzedDataProvider) -> dict[str, object]:
    inputs: dict[str, object] = {}
    for _ in range(fdp.ConsumeIntInRange(0, 3)):
        name = _fuzz_name(fdp, "in")
        kind = fdp.ConsumeIntInRange(0, 2)
        if kind == 0:
            inputs[name] = make_float_array(fdp, max_ndim=3, max_dim=8)
        elif kind == 1:
            inputs[name] = [fdp.ConsumeUnicodeNoSurrogates(8) for _ in range(fdp.ConsumeIntInRange(0, 3))]
        else:
            inputs[name] = fdp.ConsumeIntInRange(-(2**31), 2**31)
    return inputs


def _valid_outputs(fdp: atheris.FuzzedDataProvider) -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    for _ in range(fdp.ConsumeIntInRange(0, 3)):
        outputs[_fuzz_name(fdp, "out")] = make_float_array(fdp, max_ndim=3, max_dim=6)
    return outputs


def _valid_timing(fdp: atheris.FuzzedDataProvider) -> float:
    return fdp.ConsumeIntInRange(0, 1000) / 1000.0


def _valid_request_fields(fdp: atheris.FuzzedDataProvider) -> tuple[int, int, dict[str, object]]:
    return (
        fdp.ConsumeIntInRange(0, 2**31 - 1),
        fdp.ConsumeIntInRange(1, MAX_REQUEST_BUDGET_MS),
        _valid_inputs(fdp),
    )


@atheris.instrument_func
def _raw_bytes_mode(fdp: atheris.FuzzedDataProvider) -> None:
    """Feed arbitrary bytes into the protocol decoders."""
    target = fdp.ConsumeIntInRange(0, 2)
    max_bytes = _fuzz_max_bytes(fdp)
    data = fdp.ConsumeBytes(fdp.remaining_bytes())
    try:
        if target == 0:
            decode_message(data, max_bytes=max_bytes)
        elif target == 1:
            decode_predict_request(data, max_bytes=max_bytes)
        else:
            decode_predict_reply(data)
    except _EXPECTED_DECODE_EXCEPTIONS:
        pass


@atheris.instrument_func
def _structured_valid_mode(fdp: atheris.FuzzedDataProvider) -> None:
    """Round-trip genuinely valid protocol messages through encode/decode."""
    target = fdp.ConsumeIntInRange(0, 2)
    if target == 0:
        seq, budget_ms, inputs = _valid_request_fields(fdp)
        payload = encode_predict_request(
            inputs,
            seq=seq,
            budget_ms=budget_ms,
            image_codec="raw",
            jpeg_quality=fdp.ConsumeIntInRange(0, 100),
            max_image_side=None,
        )
        metadata, decoded = decode_predict_request(payload)

        assert metadata["protocol_version"] == PROTOCOL_VERSION, "request round-trip changed protocol_version"
        assert metadata["seq"] == seq, "request round-trip changed seq"
        assert metadata["budget_ms"] == budget_ms, "request round-trip changed budget_ms"
        assert metadata["images"] == {}, "request round-trip invented image entries"
        for name, value in inputs.items():
            if isinstance(value, np.ndarray):
                assert decoded[name].dtype == value.dtype, (
                    f"input {name!r} round-trip changed dtype: {decoded[name].dtype} != {value.dtype}"
                )
                assert decoded[name].shape == value.shape, (
                    f"input {name!r} round-trip changed shape: {decoded[name].shape} != {value.shape}"
                )
                np.testing.assert_array_equal(decoded[name], value, err_msg=f"input {name!r} round-trip changed values")
            elif isinstance(value, list):
                assert decoded[name] == value, f"string list {name!r} round-trip changed value"
            else:
                np.testing.assert_array_equal(
                    decoded[name], np.asarray(value), err_msg=f"scalar {name!r} round-trip changed value"
                )
    elif target == 1:
        outputs = _valid_outputs(fdp)
        seq = fdp.ConsumeIntInRange(0, 2**31 - 1)
        server_id = _fuzz_name(fdp, "server")
        queue_s = _valid_timing(fdp)
        compute_s = _valid_timing(fdp)
        decode_s = _valid_timing(fdp)
        payload = encode_predict_reply(
            outputs,
            seq=seq,
            server_id=server_id,
            server_queue_s=queue_s,
            server_compute_s=compute_s,
            server_decode_s=decode_s,
        )
        metadata = decode_predict_reply(payload)

        assert metadata["protocol_version"] == PROTOCOL_VERSION, "reply round-trip changed protocol_version"
        assert metadata["seq"] == seq, "reply round-trip changed seq"
        assert metadata["server_id"] == server_id, "reply round-trip changed server_id"
        assert _float_eq(metadata["server_queue_s"], queue_s), "reply round-trip changed server_queue_s"
        assert _float_eq(metadata["server_compute_s"], compute_s), "reply round-trip changed server_compute_s"
        assert _float_eq(metadata["server_decode_s"], decode_s), "reply round-trip changed server_decode_s"
        for name, value in outputs.items():
            decoded = metadata["outputs"][name]
            assert decoded.dtype == value.dtype, (
                f"output {name!r} round-trip changed dtype: {decoded.dtype} != {value.dtype}"
            )
            assert decoded.shape == value.shape, (
                f"output {name!r} round-trip changed shape: {decoded.shape} != {value.shape}"
            )
            np.testing.assert_array_equal(decoded, value, err_msg=f"output {name!r} round-trip changed values")
    else:
        message = {"protocol_version": PROTOCOL_VERSION}
        assert decode_message(encode_message(message)) == message, "metadata message round-trip changed value"


@atheris.instrument_func
def _malformed_request(fdp: atheris.FuzzedDataProvider) -> None:
    record: dict[str, object] = {
        "protocol_version": fdp.PickValueInList([PROTOCOL_VERSION, *_BAD_VERSIONS]),
        "seq": fdp.PickValueInList([1, *_BAD_INTEGERS]),
        "budget_ms": fdp.PickValueInList([1000, *_BAD_INTEGERS]),
        "inputs": fdp.PickValueInList(_BAD_INPUTS),
        "images": fdp.PickValueInList(_BAD_IMAGES),
    }
    for name in ("protocol_version", "seq", "budget_ms", "inputs", "images"):
        if fdp.ConsumeBool():
            record.pop(name, None)
    if fdp.ConsumeBool():
        record[_fuzz_name(fdp, "extra")] = fdp.ConsumeIntInRange(0, 10)
    try:
        payload = encode_message(record)
    except (TypeError, ValueError, OverflowError, RemoteInferenceProtocolError):
        return
    try:
        decode_predict_request(payload)
    except _EXPECTED_DECODE_EXCEPTIONS:
        pass


@atheris.instrument_func
def _malformed_reply(fdp: atheris.FuzzedDataProvider) -> None:
    record: dict[str, object] = {
        "protocol_version": fdp.PickValueInList([PROTOCOL_VERSION, *_BAD_VERSIONS]),
        "seq": fdp.PickValueInList([1, *_BAD_INTEGERS]),
        "server_id": fdp.PickValueInList(["server", "", 1, None, True]),
        "outputs": fdp.PickValueInList(_BAD_OUTPUTS),
        "server_queue_s": fdp.PickValueInList(_BAD_TIMINGS),
        "server_compute_s": fdp.PickValueInList(_BAD_TIMINGS),
        "server_decode_s": fdp.PickValueInList(_BAD_TIMINGS),
    }
    for name in ("protocol_version", "seq", "server_id", "outputs", "server_queue_s", "server_compute_s"):
        if fdp.ConsumeBool():
            record.pop(name, None)
    if fdp.ConsumeBool():
        record[_fuzz_name(fdp, "extra")] = fdp.ConsumeIntInRange(0, 10)
    try:
        payload = encode_message(record)
    except (TypeError, ValueError, OverflowError, RemoteInferenceProtocolError):
        return
    try:
        decode_predict_reply(payload)
    except _EXPECTED_DECODE_EXCEPTIONS:
        pass


@atheris.instrument_func
def _malformed_encode_args(fdp: atheris.FuzzedDataProvider) -> None:
    if fdp.ConsumeBool():
        inputs = {
            _fuzz_name(fdp, "in"): fdp.PickValueInList([None, [1, 2], {"nested": 1}, "text", object])
        }
        try:
            encode_predict_request(
                inputs,
                seq=fdp.PickValueInList(_BAD_INTEGERS),
                budget_ms=fdp.PickValueInList(_BAD_INTEGERS),
                image_codec=fdp.PickValueInList([None, 1, "png", True, "raw"]),
                jpeg_quality=fdp.PickValueInList([None, -1, 101, True, "90", 90]),
                max_image_side=fdp.PickValueInList([None, 0, -1, True, "32", 32]),
            )
        except _EXPECTED_ENCODE_EXCEPTIONS:
            pass
    else:
        outputs = {_fuzz_name(fdp, "out"): fdp.PickValueInList([None, [1, 2], "text", {1}])}
        try:
            encode_predict_reply(
                outputs,
                seq=fdp.PickValueInList(_BAD_INTEGERS),
                server_id=fdp.PickValueInList(["server", "", 1, None, True]),
                server_queue_s=fdp.PickValueInList(_BAD_TIMINGS),
                server_compute_s=fdp.PickValueInList(_BAD_TIMINGS),
                server_decode_s=fdp.PickValueInList(_BAD_TIMINGS),
            )
        except _EXPECTED_ENCODE_EXCEPTIONS:
            pass


@atheris.instrument_func
def _structured_malformed_mode(fdp: atheris.FuzzedDataProvider) -> None:
    """Pack a deliberately malformed message; only documented exceptions may reach the caller."""
    target = fdp.ConsumeIntInRange(0, 2)
    if target == 0:
        _malformed_request(fdp)
    elif target == 1:
        _malformed_reply(fdp)
    else:
        _malformed_encode_args(fdp)


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
