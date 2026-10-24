# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fuzz the remote inference image path (``decode_images`` and predict requests).

Raw-bytes mode feeds arbitrary image entries (JPEG and raw) into
``decode_images`` with fuzz-derived size limits. Structure-aware mode round-trips
small uint8 RGB images through a predict request and asserts only shape/dtype
(JPEG is lossy). Malformed mode covers truncated/random JPEG bytes with and
without the SOI prefix, shape/dtype mismatches, oversized declared shapes, bad
codecs, and invalid encode-side image options.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import atheris
import numpy as np

with atheris.instrument_imports():
    from physicalai.inference.constants import IMAGES
    from physicalai.inference.remote._protocol import (
        PROTOCOL_VERSION,
        RemoteInferenceProtocolError,
        decode_images,
        decode_predict_request,
        encode_images,
        encode_message,
        encode_predict_request,
    )

# Image decoding documents RemoteInferenceProtocolError for every rejection:
# malformed metadata, wrong namespaces, size gates, and JPEG/raw decode failures.
# MemoryError, RecursionError, a hang, or any other undocumented exception is a
# finding -- do not widen this tuple to `Exception`.
_EXPECTED_DECODE_EXCEPTIONS = (RemoteInferenceProtocolError,)
# Encoders validate options with ValueError and image values with
# RemoteInferenceProtocolError.
_EXPECTED_ENCODE_EXCEPTIONS = (ValueError, RemoteInferenceProtocolError)

_MAX_SIDE = 32
_MAX_RANDOM_BYTES = 64
_MAX_DECODED_BYTES = 2048


def _fuzz_name(fdp: atheris.FuzzedDataProvider) -> str:
    return f"{IMAGES}.{fdp.ConsumeUnicodeNoSurrogates(8) or 'front'}"


def _well_formed_image(fdp: atheris.FuzzedDataProvider) -> np.ndarray:
    """Return a small uint8 HxWx3 or 1xHxWx3 image (at most a few KB)."""
    batched = fdp.ConsumeBool()
    height = fdp.ConsumeIntInRange(1, _MAX_SIDE)
    width = fdp.ConsumeIntInRange(1, _MAX_SIDE)
    shape = (1, height, width, 3) if batched else (height, width, 3)
    count = int(np.prod(shape))
    raw = fdp.ConsumeBytes(count)
    if len(raw) < count:
        raw += b"\x00" * (count - len(raw))
    return np.frombuffer(raw[:count], dtype=np.uint8).copy().reshape(shape)


def _fuzz_max_bytes(fdp: atheris.FuzzedDataProvider) -> int:
    return fdp.PickValueInList([0, 1, 16, 64, 128, 256, 512, 1024, _MAX_DECODED_BYTES])


def _fuzz_shape(fdp: atheris.FuzzedDataProvider) -> object:
    kind = fdp.ConsumeIntInRange(0, 6)
    height = fdp.ConsumeIntInRange(1, 12)
    width = fdp.ConsumeIntInRange(1, 12)
    if kind == 0:
        return [height, width, 3]
    if kind == 1:
        return [1, height, width, 3]
    if kind == 2:
        return [height, width, 4]  # wrong channel count
    if kind == 3:
        return [height, width]  # wrong ndim
    if kind == 4:
        return fdp.ConsumeUnicodeNoSurrogates(8)  # wrong type: str instead of list
    if kind == 5:
        return [1, 2, 3, 4, 5]  # wrong ndim
    return [4096, 4096, 3]  # oversized declared shape, rejected before allocation


def _fuzz_dtype(fdp: atheris.FuzzedDataProvider) -> object:
    return fdp.PickValueInList(["uint8", "uint8", "float32", None, 1, True])


def _fuzz_image_data(fdp: atheris.FuzzedDataProvider, *, jpeg: bool) -> bytes:
    data = fdp.ConsumeBytes(fdp.ConsumeIntInRange(0, _MAX_RANDOM_BYTES))
    if jpeg and fdp.ConsumeBool():
        data = b"\xff\xd8" + data
    return data


def _malformed_entry(fdp: atheris.FuzzedDataProvider) -> dict[str, object]:
    return {
        "codec": fdp.PickValueInList(["jpeg", "raw", "png", None, 1, True]),
        "shape": _fuzz_shape(fdp),
        "dtype": _fuzz_dtype(fdp),
        "data": _fuzz_image_data(fdp, jpeg=True),
    }


def _truncated_real_jpeg_entry(fdp: atheris.FuzzedDataProvider) -> tuple[str, dict[str, object]]:
    image = _well_formed_image(fdp)
    name = _fuzz_name(fdp)
    entry = encode_images({name: image}, image_codec="jpeg", jpeg_quality=fdp.ConsumeIntInRange(0, 100), max_image_side=None)
    corrupted = dict(entry[name])
    data = corrupted["data"]
    assert isinstance(data, bytes)
    cut = fdp.ConsumeIntInRange(0, len(data))
    data = data[:cut]
    if data and fdp.ConsumeBool():
        index = fdp.ConsumeIntInRange(0, len(data) - 1)
        data = data[:index] + bytes([fdp.ConsumeIntInRange(0, 255)]) + data[index + 1 :]
    corrupted["data"] = data
    return name, corrupted


@atheris.instrument_func
def _raw_bytes_mode(fdp: atheris.FuzzedDataProvider) -> None:
    """Feed arbitrary image entries straight into decode_images."""
    name = _fuzz_name(fdp)
    max_bytes = _fuzz_max_bytes(fdp)
    entry = _malformed_entry(fdp)
    try:
        decode_images(
            {name: entry},
            max_bytes=max_bytes,
            already_decoded_bytes=fdp.ConsumeIntInRange(0, max_bytes),
        )
    except _EXPECTED_DECODE_EXCEPTIONS:
        pass


@atheris.instrument_func
def _structured_valid_mode(fdp: atheris.FuzzedDataProvider) -> None:
    """Round-trip a small uint8 RGB image through a predict request."""
    image = _well_formed_image(fdp)
    name = _fuzz_name(fdp)
    codec = fdp.PickValueInList(["raw", "jpeg"])
    payload = encode_predict_request(
        {name: image},
        seq=fdp.ConsumeIntInRange(0, 2**31 - 1),
        budget_ms=fdp.ConsumeIntInRange(1, 60_000),
        image_codec=codec,
        jpeg_quality=fdp.ConsumeIntInRange(0, 100),
        max_image_side=None,
    )
    _, decoded = decode_predict_request(payload)

    output = decoded[name]
    assert isinstance(output, np.ndarray), f"image {name!r} round-trip did not return an ndarray"
    assert output.dtype == image.dtype, f"image {name!r} round-trip changed dtype: {output.dtype} != {image.dtype}"
    # JPEG is lossy: assert layout only, never exact pixels.
    assert output.shape == image.shape, f"image {name!r} round-trip changed shape: {output.shape} != {image.shape}"


@atheris.instrument_func
def _malformed_decode_images(fdp: atheris.FuzzedDataProvider) -> None:
    max_bytes = _fuzz_max_bytes(fdp)
    if fdp.ConsumeBool():
        name, entry = _truncated_real_jpeg_entry(fdp)
    else:
        name = _fuzz_name(fdp)
        entry = _malformed_entry(fdp)
    try:
        decode_images(
            {name: entry},
            max_bytes=max_bytes,
            already_decoded_bytes=fdp.ConsumeIntInRange(0, max_bytes),
        )
    except _EXPECTED_DECODE_EXCEPTIONS:
        pass


@atheris.instrument_func
def _malformed_predict_request(fdp: atheris.FuzzedDataProvider) -> None:
    name = _fuzz_name(fdp) if fdp.ConsumeBool() else fdp.ConsumeUnicodeNoSurrogates(8)
    images = {name: _malformed_entry(fdp)}
    record = {
        "protocol_version": PROTOCOL_VERSION,
        "seq": fdp.ConsumeIntInRange(0, 2**31 - 1),
        "budget_ms": fdp.ConsumeIntInRange(1, 60_000),
        "inputs": {},
        "images": images,
    }
    try:
        payload = encode_message(record)
    except (TypeError, ValueError, OverflowError, RemoteInferenceProtocolError):
        return
    try:
        decode_predict_request(payload, max_bytes=_fuzz_max_bytes(fdp))
    except _EXPECTED_DECODE_EXCEPTIONS:
        pass


@atheris.instrument_func
def _malformed_encode_args(fdp: atheris.FuzzedDataProvider) -> None:
    name = _fuzz_name(fdp)
    image = _well_formed_image(fdp)
    kind = fdp.ConsumeIntInRange(0, 4)
    if kind == 0:
        value: object = image.astype(np.float32)  # wrong dtype
    elif kind == 1:
        value = image[..., :2]  # wrong channel count
    elif kind == 2:
        value = image[..., 0]  # wrong ndim
    elif kind == 3:
        value = np.zeros((2, *image.shape[-3:]), dtype=np.uint8)  # batch > 1 for JPEG
    else:
        value = image
    try:
        encode_predict_request(
            {name: value},
            seq=fdp.ConsumeIntInRange(0, 2**31 - 1),
            budget_ms=fdp.ConsumeIntInRange(1, 60_000),
            image_codec=fdp.PickValueInList(["raw", "jpeg", "png", None, 1]),
            jpeg_quality=fdp.PickValueInList([0, 90, 100, -1, 101, None, True]),
            max_image_side=fdp.PickValueInList([None, 1, 16, 0, -1, True, "32"]),
        )
    except _EXPECTED_ENCODE_EXCEPTIONS:
        pass


@atheris.instrument_func
def _structured_malformed_mode(fdp: atheris.FuzzedDataProvider) -> None:
    """Build a deliberately malformed image path; only documented exceptions may escape."""
    target = fdp.ConsumeIntInRange(0, 2)
    if target == 0:
        _malformed_decode_images(fdp)
    elif target == 1:
        _malformed_predict_request(fdp)
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
