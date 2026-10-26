# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Version-one MessagePack protocol and RGB image codecs for remote inference."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import msgpack
import numpy as np

from physicalai.inference.constants import IMAGES
from physicalai.transport import DEFAULT_MAX_PAYLOAD_BYTES, pack_msgpack, unpack_msgpack

PROTOCOL_VERSION = 1
DEFAULT_IMAGE_JPEG_QUALITY = 90
DEFAULT_MAX_REQUEST_BYTES = DEFAULT_MAX_PAYLOAD_BYTES
MAX_REQUEST_BUDGET_MS = 60_000
MAX_IMAGE_JPEG_QUALITY = 100
_RGB_CHANNELS = 3
_RGB_IMAGE_NDIM = 3
_BATCHED_RGB_IMAGE_NDIM = 4
_JPEG_MIN_HEADER_BYTES = 4
_JPEG_MARKER_PREFIX = 0xFF
_JPEG_SEGMENT_LENGTH_BYTES = 2
_JPEG_FRAME_MIN_BYTES = 7
ImageCodec = Literal["jpeg", "raw"]


class RemoteInferenceError(RuntimeError):
    """Base class for remote inference failures."""


class RemoteInferenceUnavailableError(RemoteInferenceError):
    """The configured remote inference server is unavailable."""


class RemoteInferenceTimeoutError(RemoteInferenceError):
    """A remote inference request exceeded its deadline."""


class RemoteInferenceProtocolError(RemoteInferenceError):
    """The peer sent a malformed or incompatible protocol message."""


class RemoteInferenceModelMismatchError(RemoteInferenceError):
    """The peer serves a different or replaced model."""


@dataclass(frozen=True, slots=True)
class RemoteTiming:
    """Request timings with server decoding, post-decode queue wait, and compute split.

    ``encode_s`` and ``decode_s`` measure client-side request encoding and reply
    decoding; ``server_decode_s`` measures server-side request decoding.
    """

    round_trip_s: float
    server_queue_s: float
    server_compute_s: float
    encode_s: float
    decode_s: float
    request_bytes: int
    reply_bytes: int
    server_decode_s: float


def encode_message(message: dict[str, Any]) -> bytes:
    """Encode one protocol map using the shared tagged-array MessagePack codec.

    Returns:
        Binary MessagePack representation of the map.

    Raises:
        TypeError: If the message is not a map.
        RemoteInferenceProtocolError: If a value cannot be encoded.
    """
    if not isinstance(message, dict):
        msg = "Protocol messages must be maps"
        raise TypeError(msg)
    try:
        return pack_msgpack(message)
    except (TypeError, ValueError, OverflowError) as error:
        msg = f"Could not encode protocol message: {error}"
        raise RemoteInferenceProtocolError(msg) from error


def decode_message(payload: bytes, *, max_bytes: int = DEFAULT_MAX_REQUEST_BYTES) -> dict[str, Any]:
    """Decode a protocol map, validating every tagged ndarray before allocation.

    Returns:
        The decoded string-keyed protocol map.

    Raises:
        RemoteInferenceProtocolError: If the payload or its arrays are invalid.
    """
    try:
        message = unpack_msgpack(payload, max_bytes=max_bytes)
    except (msgpack.UnpackException, TypeError, ValueError, OverflowError) as error:
        msg = f"Invalid MessagePack protocol map: {error}"
        raise RemoteInferenceProtocolError(msg) from error
    if not isinstance(message, dict) or any(not isinstance(key, str) for key in message):
        msg = "Protocol message must be a string-keyed MessagePack map"
        raise RemoteInferenceProtocolError(msg)
    return message


def decode_error(payload: bytes) -> dict[str, Any]:
    """Decode the protocol's compact ``query.reply_err`` map.

    Returns:
        The validated error map.

    Raises:
        RemoteInferenceProtocolError: If the error map is malformed.
    """
    message = decode_message(payload)
    version = message.get("protocol_version")
    if isinstance(version, bool) or not isinstance(version, int) or not isinstance(message.get("code"), str):
        msg = "Malformed remote error map"
        raise RemoteInferenceProtocolError(msg)
    return message


def _is_image_input(name: str) -> bool:
    return name == IMAGES or name.startswith(f"{IMAGES}.")


def _resize_image(image: np.ndarray, max_image_side: int | None) -> np.ndarray:
    if (
        image.dtype != np.uint8
        or image.ndim not in {_RGB_IMAGE_NDIM, _BATCHED_RGB_IMAGE_NDIM}
        or image.shape[-1] != _RGB_CHANNELS
    ):
        msg = "Images must be uint8 HxWx3 or BxHxWx3 arrays"
        raise RemoteInferenceProtocolError(msg)
    height, width = image.shape[-3:-1]
    if height < 1 or width < 1 or (image.ndim == _BATCHED_RGB_IMAGE_NDIM and image.shape[0] < 1):
        msg = "Image dimensions must be positive"
        raise RemoteInferenceProtocolError(msg)
    if max_image_side is None or max(height, width) <= max_image_side:
        return np.ascontiguousarray(image)
    scale = max_image_side / max(height, width)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if image.ndim == _RGB_IMAGE_NDIM:
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return np.stack([cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA) for frame in image])


def validate_image_options(image_codec: ImageCodec, jpeg_quality: int, max_image_side: int | None) -> None:
    """Validate the shared image encoding settings.

    Raises:
        ValueError: If the codec, JPEG quality, or resize limit is invalid.
    """
    if image_codec not in {"jpeg", "raw"}:
        msg = "image_codec must be 'jpeg' or 'raw'"
        raise ValueError(msg)
    if (
        not isinstance(jpeg_quality, int)
        or isinstance(jpeg_quality, bool)
        or not 0 <= jpeg_quality <= MAX_IMAGE_JPEG_QUALITY
    ):
        msg = "jpeg_quality must be between 0 and 100"
        raise ValueError(msg)
    if max_image_side is not None and (
        not isinstance(max_image_side, int) or isinstance(max_image_side, bool) or max_image_side < 1
    ):
        msg = "max_image_side must be positive"
        raise ValueError(msg)


def encode_images(
    images: dict[str, Any],
    *,
    image_codec: ImageCodec,
    jpeg_quality: int,
    max_image_side: int | None,
) -> dict[str, dict[str, Any]]:
    """Encode each named RGB image at native resolution unless downscaling is requested.

    Returns:
        Image descriptors containing encoded bytes, shapes, and codec names.

    Raises:
        RemoteInferenceProtocolError: If an image cannot be encoded.
    """
    validate_image_options(image_codec, jpeg_quality, max_image_side)
    encoded: dict[str, dict[str, Any]] = {}
    for name, value in images.items():
        if not isinstance(name, str):
            msg = "Image keys must be strings"
            raise RemoteInferenceProtocolError(msg)
        image = _resize_image(np.asarray(value), max_image_side)
        data = image.tobytes()
        if image_codec == "jpeg":
            if image.ndim == _BATCHED_RGB_IMAGE_NDIM:
                if image.shape[0] != 1:
                    msg = "JPEG supports a leading batch dimension of one"
                    raise RemoteInferenceProtocolError(msg)
                frame = image[0]
            else:
                frame = image
            # Model-facing pixels are RGB; convert to and from OpenCV's BGR convention symmetrically.
            success, jpeg = cv2.imencode(
                ".jpg",
                np.ascontiguousarray(frame[..., ::-1]),
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            if not success:
                msg = "OpenCV failed to encode JPEG image"
                raise RemoteInferenceProtocolError(msg)
            data = jpeg.tobytes()
        encoded[name] = {
            "codec": image_codec,
            "shape": list(image.shape),
            "dtype": "uint8",
            "data": data,
        }
    return encoded


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < _JPEG_MIN_HEADER_BYTES or data[:2] != b"\xff\xd8":
        msg = "Malformed JPEG image data"
        raise RemoteInferenceProtocolError(msg)
    start_of_frame = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    offset = 2
    while offset + 4 <= len(data):
        if data[offset] != _JPEG_MARKER_PREFIX:
            msg = "Malformed JPEG marker"
            raise RemoteInferenceProtocolError(msg)
        while offset < len(data) and data[offset] == _JPEG_MARKER_PREFIX:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in {0xD8, 0xD9, 0x01, *range(0xD0, 0xD8)}:
            continue
        if offset + 2 > len(data):
            break
        segment_length = struct.unpack_from(">H", data, offset)[0]
        if segment_length < _JPEG_SEGMENT_LENGTH_BYTES or offset + segment_length > len(data):
            msg = "Malformed JPEG segment"
            raise RemoteInferenceProtocolError(msg)
        if marker in start_of_frame:
            if segment_length < _JPEG_FRAME_MIN_BYTES:
                msg = "Malformed JPEG frame header"
                raise RemoteInferenceProtocolError(msg)
            height, width = struct.unpack_from(">HH", data, offset + 3)
            if width < 1 or height < 1:
                msg = "Invalid JPEG dimensions"
                raise RemoteInferenceProtocolError(msg)
            return height, width
        offset += segment_length
    msg = "JPEG image has no dimensions"
    raise RemoteInferenceProtocolError(msg)


def _decode_image_frame(codec: str, data: bytes, shape: tuple[int, ...]) -> np.ndarray:
    if codec == "raw":
        if len(data) != math.prod(shape):
            msg = "Raw image data length does not match its shape"
            raise RemoteInferenceProtocolError(msg)
        return np.frombuffer(data, dtype=np.uint8).reshape(shape).copy()
    if _jpeg_dimensions(data) != shape[-3:-1]:
        msg = "JPEG dimensions do not match the declared shape"
        raise RemoteInferenceProtocolError(msg)
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        msg = "OpenCV failed to decode JPEG image"
        raise RemoteInferenceProtocolError(msg)
    image = np.ascontiguousarray(bgr[..., ::-1])
    if len(shape) == _BATCHED_RGB_IMAGE_NDIM:
        image = image[np.newaxis]
    if image.shape != shape:
        msg = "Decoded JPEG image does not match its declared shape"
        raise RemoteInferenceProtocolError(msg)
    return image


def decode_images(
    images: object,
    *,
    max_bytes: int,
    already_decoded_bytes: int,
) -> dict[str, np.ndarray]:
    """Decode image entries after validating shape and total allocation size.

    Returns:
        Writable RGB image arrays keyed by observation name.

    Raises:
        RemoteInferenceProtocolError: If image metadata, bytes, or sizes are invalid.
    """
    if not isinstance(images, dict) or any(not isinstance(key, str) for key in images):
        msg = "Request images must be a string-keyed map"
        raise RemoteInferenceProtocolError(msg)
    decoded: dict[str, np.ndarray] = {}
    decoded_bytes = already_decoded_bytes
    for name, entry in images.items():
        if not _is_image_input(name):
            msg = f"Image key {name!r} must use the '{IMAGES}' namespace"
            raise RemoteInferenceProtocolError(msg)
        if not isinstance(entry, dict):
            msg = "Image entry must be a map"
            raise RemoteInferenceProtocolError(msg)
        codec = entry.get("codec")
        shape_value = entry.get("shape")
        dtype = entry.get("dtype")
        data = entry.get("data")
        if codec not in {"jpeg", "raw"} or dtype != "uint8" or not isinstance(data, bytes):
            msg = "Malformed image codec, dtype, or data"
            raise RemoteInferenceProtocolError(msg)
        if not isinstance(shape_value, list) or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 1 for dim in shape_value
        ):
            msg = "Image shape dimensions must be positive integers"
            raise RemoteInferenceProtocolError(msg)
        shape = tuple(shape_value)
        if len(shape) not in {_RGB_IMAGE_NDIM, _BATCHED_RGB_IMAGE_NDIM} or shape[-1] != _RGB_CHANNELS:
            msg = "Images must declare HxWx3 or BxHxWx3 shapes"
            raise RemoteInferenceProtocolError(msg)
        if len(shape) == _BATCHED_RGB_IMAGE_NDIM and shape[0] != 1:
            msg = "Remote inference image batch dimension must be one"
            raise RemoteInferenceProtocolError(msg)
        expected_size = math.prod(shape)
        if expected_size > max_bytes - decoded_bytes:
            msg = "Decoded image arrays exceed max_request_bytes"
            raise RemoteInferenceProtocolError(msg)
        image = _decode_image_frame(codec, data, shape)
        decoded_bytes += expected_size
        decoded[name] = image
    return decoded


def encode_predict_request(
    inputs: dict[str, Any],
    *,
    seq: int,
    budget_ms: int,
    image_codec: ImageCodec,
    jpeg_quality: int,
    max_image_side: int | None,
) -> bytes:
    """Encode the protocol-v1 predict map, keeping string lists in MessagePack.

    Returns:
        Encoded request bytes with separate image entries.

    Raises:
        ValueError: If sequence, budget, or image options are invalid.
        RemoteInferenceProtocolError: If an input cannot be encoded.
    """
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        msg = "seq must be a non-negative integer"
        raise ValueError(msg)
    if not isinstance(budget_ms, int) or isinstance(budget_ms, bool) or budget_ms < 1:
        msg = "budget_ms must be a positive integer"
        raise ValueError(msg)
    if budget_ms > MAX_REQUEST_BUDGET_MS:
        msg = f"budget_ms must not exceed {MAX_REQUEST_BUDGET_MS}"
        raise ValueError(msg)
    model_inputs: dict[str, Any] = {}
    image_inputs: dict[str, Any] = {}
    for name, value in inputs.items():
        if not isinstance(name, str):
            msg = "Input keys must be strings"
            raise RemoteInferenceProtocolError(msg)
        if _is_image_input(name):
            image_inputs[name] = value
        elif (isinstance(value, list) and all(isinstance(item, str) for item in value)) or (
            isinstance(value, np.ndarray) and value.dtype.kind in "biufc"
        ):
            model_inputs[name] = value
        elif isinstance(value, (bool, int, float, np.number)):
            scalar = np.asarray(value)
            if scalar.dtype.kind not in "biufc":
                msg = f"Input {name!r} must be numeric or bool"
                raise RemoteInferenceProtocolError(msg)
            model_inputs[name] = scalar
        else:
            msg = f"Input {name!r} must be a numeric ndarray or list[str]"
            raise RemoteInferenceProtocolError(msg)
    image_entries = encode_images(
        image_inputs,
        image_codec=image_codec,
        jpeg_quality=jpeg_quality,
        max_image_side=max_image_side,
    )
    return encode_message({
        "protocol_version": PROTOCOL_VERSION,
        "seq": seq,
        "budget_ms": budget_ms,
        "inputs": model_inputs,
        "images": image_entries,
    })


def _validate_predict_inputs(inputs: object) -> tuple[dict[str, Any], int]:
    if not isinstance(inputs, dict) or any(not isinstance(name, str) for name in inputs):
        msg = "Predict inputs must be a string-keyed map"
        raise RemoteInferenceProtocolError(msg)
    decoded_bytes = 0
    for name, value in inputs.items():
        if _is_image_input(name):
            msg = "Image inputs must use the separate images map"
            raise RemoteInferenceProtocolError(msg)
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "biufc":
                msg = "Input array dtype must be numeric or bool"
                raise RemoteInferenceProtocolError(msg)
            decoded_bytes += value.nbytes
        elif not (isinstance(value, list) and all(isinstance(item, str) for item in value)):
            msg = "Predict inputs must contain numeric arrays or list[str]"
            raise RemoteInferenceProtocolError(msg)
    return inputs, decoded_bytes


def decode_predict_request(
    payload: bytes,
    *,
    max_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a v1 predict request and reconstruct all arrays/images.

    Returns:
        Request metadata and writable model inputs.

    Raises:
        RemoteInferenceProtocolError: If the request is malformed or exceeds its size limit.
    """
    message = decode_message(payload, max_bytes=max_bytes)
    if set(message) != {"protocol_version", "seq", "budget_ms", "inputs", "images"}:
        msg = "Predict request has unexpected or missing fields"
        raise RemoteInferenceProtocolError(msg)
    version = message.get("protocol_version")
    if not isinstance(version, int) or isinstance(version, bool) or version != PROTOCOL_VERSION:
        msg = "Unsupported predict protocol version"
        raise RemoteInferenceProtocolError(msg)
    seq = message.get("seq")
    budget = message.get("budget_ms")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        msg = "Invalid prediction sequence"
        raise RemoteInferenceProtocolError(msg)
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        msg = "Invalid prediction budget"
        raise RemoteInferenceProtocolError(msg)
    if budget > MAX_REQUEST_BUDGET_MS:
        msg = "Prediction budget exceeds the server maximum"
        raise RemoteInferenceProtocolError(msg)
    inputs, decoded_bytes = _validate_predict_inputs(message.get("inputs"))
    if decoded_bytes > max_bytes:
        msg = "Decoded input arrays exceed max_request_bytes"
        raise RemoteInferenceProtocolError(msg)
    image_inputs = decode_images(message.get("images"), max_bytes=max_bytes, already_decoded_bytes=decoded_bytes)
    overlap = inputs.keys() & image_inputs.keys()
    if overlap:
        msg = f"Duplicate input and image keys: {sorted(overlap)!r}"
        raise RemoteInferenceProtocolError(msg)
    for name, value in inputs.items():
        if isinstance(value, np.ndarray):
            inputs[name] = value.copy()
    inputs.update(image_inputs)
    return message, inputs


def encode_predict_reply(
    outputs: dict[str, Any],
    *,
    seq: int,
    server_id: str,
    server_queue_s: float,
    server_compute_s: float,
    server_decode_s: float,
) -> bytes:
    """Encode the complete model output map and server timing in a v1 reply.

    Returns:
        Encoded prediction reply bytes.
    """
    return encode_message({
        "protocol_version": PROTOCOL_VERSION,
        "seq": seq,
        "server_id": server_id,
        "outputs": outputs,
        "server_queue_s": server_queue_s,
        "server_compute_s": server_compute_s,
        "server_decode_s": server_decode_s,
    })


def decode_predict_reply(payload: bytes) -> dict[str, Any]:
    """Decode and validate a v1 predict reply with recursively decoded arrays.

    Returns:
        Validated reply metadata with writable output arrays.

    Raises:
        RemoteInferenceProtocolError: If the reply schema or timing is invalid.
    """
    message = decode_message(payload)
    required = {
        "protocol_version",
        "seq",
        "server_id",
        "outputs",
        "server_decode_s",
        "server_queue_s",
        "server_compute_s",
    }
    version = message.get("protocol_version")
    if (
        set(message) != required
        or not isinstance(version, int)
        or isinstance(version, bool)
        or version != PROTOCOL_VERSION
    ):
        msg = "Malformed prediction reply"
        raise RemoteInferenceProtocolError(msg)
    seq = message.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        msg = "Prediction reply has an invalid sequence"
        raise RemoteInferenceProtocolError(msg)
    if not isinstance(message.get("server_id"), str) or not message["server_id"]:
        msg = "Prediction reply has an invalid server_id"
        raise RemoteInferenceProtocolError(msg)
    outputs = message.get("outputs")
    if not isinstance(outputs, dict) or any(not isinstance(key, str) for key in outputs):
        msg = "Prediction outputs must be a string-keyed map"
        raise RemoteInferenceProtocolError(msg)
    if any(not isinstance(value, np.ndarray) for value in outputs.values()):
        msg = "Prediction outputs must be NumPy arrays"
        raise RemoteInferenceProtocolError(msg)
    for field in ("server_decode_s", "server_queue_s", "server_compute_s"):
        value = message.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            msg = f"Prediction reply has invalid {field}"
            raise RemoteInferenceProtocolError(msg)
    for name, value in outputs.items():
        outputs[name] = value.copy()
    return message
