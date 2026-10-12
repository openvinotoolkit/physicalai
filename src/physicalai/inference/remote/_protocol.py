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
from physicalai.transport._codec import DEFAULT_MAX_PAYLOAD_BYTES, pack_msgpack, unpack_msgpack

PROTOCOL_VERSION = 1
DEFAULT_IMAGE_JPEG_QUALITY = 90
DEFAULT_MAX_REQUEST_BYTES = DEFAULT_MAX_PAYLOAD_BYTES
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
    """Timing information for the last completed remote request."""

    round_trip_s: float
    server_queue_s: float
    server_compute_s: float
    encode_s: float
    decode_s: float
    request_bytes: int
    reply_bytes: int


def encode_message(message: dict[str, Any]) -> bytes:
    """Encode one protocol map using the shared tagged-array MessagePack codec."""
    if not isinstance(message, dict):
        raise TypeError("Protocol messages must be maps")
    try:
        return pack_msgpack(message)
    except (TypeError, ValueError, OverflowError) as error:
        raise RemoteInferenceProtocolError(f"Could not encode protocol message: {error}") from error


def decode_message(payload: bytes, *, max_bytes: int = DEFAULT_MAX_REQUEST_BYTES) -> dict[str, Any]:
    """Decode a protocol map, validating every tagged ndarray before allocation."""
    try:
        message = unpack_msgpack(payload, max_bytes=max_bytes)
    except (msgpack.UnpackException, TypeError, ValueError, OverflowError) as error:
        raise RemoteInferenceProtocolError(f"Invalid MessagePack protocol map: {error}") from error
    if not isinstance(message, dict) or any(not isinstance(key, str) for key in message):
        raise RemoteInferenceProtocolError("Protocol message must be a string-keyed MessagePack map")
    return message


def decode_error(payload: bytes) -> dict[str, Any]:
    """Decode the protocol's compact ``query.reply_err`` map."""
    message = decode_message(payload)
    version = message.get("protocol_version")
    if isinstance(version, bool) or not isinstance(version, int) or not isinstance(message.get("code"), str):
        raise RemoteInferenceProtocolError("Malformed remote error map")
    return message


def _is_image_input(name: str) -> bool:
    return name == IMAGES or name.startswith(f"{IMAGES}.")


def _resize_image(image: np.ndarray, max_image_side: int | None) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim not in (3, 4) or image.shape[-1] != 3:
        raise RemoteInferenceProtocolError("Images must be uint8 HxWx3 or BxHxWx3 arrays")
    height, width = image.shape[-3:-1]
    if height < 1 or width < 1 or (image.ndim == 4 and image.shape[0] < 1):
        raise RemoteInferenceProtocolError("Image dimensions must be positive")
    if max_image_side is None or max(height, width) <= max_image_side:
        return np.ascontiguousarray(image)
    scale = max_image_side / max(height, width)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if image.ndim == 3:
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return np.stack([cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA) for frame in image])


def encode_images(
    images: dict[str, Any],
    *,
    image_codec: ImageCodec,
    jpeg_quality: int,
    max_image_side: int | None,
) -> dict[str, dict[str, Any]]:
    """Encode each named RGB image at native resolution unless downscaling is requested."""
    if image_codec not in ("jpeg", "raw"):
        raise ValueError("image_codec must be 'jpeg' or 'raw'")
    if not isinstance(jpeg_quality, int) or isinstance(jpeg_quality, bool) or not 0 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 0 and 100")
    if max_image_side is not None and (
        not isinstance(max_image_side, int) or isinstance(max_image_side, bool) or max_image_side < 1
    ):
        raise ValueError("max_image_side must be positive")
    encoded: dict[str, dict[str, Any]] = {}
    for name, value in images.items():
        if not isinstance(name, str):
            raise RemoteInferenceProtocolError("Image keys must be strings")
        image = _resize_image(np.asarray(value), max_image_side)
        data = image.tobytes()
        if image_codec == "jpeg":
            if image.ndim == 4:
                if image.shape[0] != 1:
                    raise RemoteInferenceProtocolError("JPEG supports a leading batch dimension of one")
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
                raise RemoteInferenceProtocolError("OpenCV failed to encode JPEG image")
            data = jpeg.tobytes()
        encoded[name] = {
            "codec": image_codec,
            "shape": list(image.shape),
            "dtype": "uint8",
            "data": data,
        }
    return encoded


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise RemoteInferenceProtocolError("Malformed JPEG image data")
    start_of_frame = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    offset = 2
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            raise RemoteInferenceProtocolError("Malformed JPEG marker")
        while offset < len(data) and data[offset] == 0xFF:
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
        if segment_length < 2 or offset + segment_length > len(data):
            raise RemoteInferenceProtocolError("Malformed JPEG segment")
        if marker in start_of_frame:
            if segment_length < 7:
                raise RemoteInferenceProtocolError("Malformed JPEG frame header")
            height, width = struct.unpack_from(">HH", data, offset + 3)
            if width < 1 or height < 1:
                raise RemoteInferenceProtocolError("Invalid JPEG dimensions")
            return height, width
        offset += segment_length
    raise RemoteInferenceProtocolError("JPEG image has no dimensions")


def decode_images(
    images: object,
    *,
    max_bytes: int,
    already_decoded_bytes: int,
) -> dict[str, np.ndarray]:
    """Decode image entries after validating shape and total allocation size."""
    if not isinstance(images, dict) or any(not isinstance(key, str) for key in images):
        raise RemoteInferenceProtocolError("Request images must be a string-keyed map")
    decoded: dict[str, np.ndarray] = {}
    decoded_bytes = already_decoded_bytes
    for name, entry in images.items():
        if not _is_image_input(name):
            raise RemoteInferenceProtocolError(f"Image key {name!r} must use the '{IMAGES}' namespace")
        if not isinstance(entry, dict):
            raise RemoteInferenceProtocolError("Image entry must be a map")
        codec = entry.get("codec")
        shape_value = entry.get("shape")
        dtype = entry.get("dtype")
        data = entry.get("data")
        if codec not in ("jpeg", "raw") or dtype != "uint8" or not isinstance(data, bytes):
            raise RemoteInferenceProtocolError("Malformed image codec, dtype, or data")
        if not isinstance(shape_value, list) or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 1 for dim in shape_value
        ):
            raise RemoteInferenceProtocolError("Image shape dimensions must be positive integers")
        shape = tuple(shape_value)
        if len(shape) not in (3, 4) or shape[-1] != 3:
            raise RemoteInferenceProtocolError("Images must declare HxWx3 or BxHxWx3 shapes")
        if len(shape) == 4 and shape[0] != 1:
            raise RemoteInferenceProtocolError("Remote inference image batch dimension must be one")
        expected_size = math.prod(shape)
        if expected_size > max_bytes - decoded_bytes:
            raise RemoteInferenceProtocolError("Decoded image arrays exceed max_request_bytes")
        if codec == "raw":
            if len(data) != expected_size:
                raise RemoteInferenceProtocolError("Raw image data length does not match its shape")
            image = np.frombuffer(data, dtype=np.uint8).reshape(shape).copy()
        else:
            if _jpeg_dimensions(data) != shape[-3:-1]:
                raise RemoteInferenceProtocolError("JPEG dimensions do not match the declared shape")
            bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RemoteInferenceProtocolError("OpenCV failed to decode JPEG image")
            image = np.ascontiguousarray(bgr[..., ::-1])
            if len(shape) == 4:
                image = image[np.newaxis]
            if image.shape != shape:
                raise RemoteInferenceProtocolError("Decoded JPEG image does not match its declared shape")
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
    """Encode the protocol-v1 predict map, keeping string lists in MessagePack."""
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        raise ValueError("seq must be a non-negative integer")
    if not isinstance(budget_ms, int) or isinstance(budget_ms, bool) or budget_ms < 1:
        raise ValueError("budget_ms must be a positive integer")
    model_inputs: dict[str, Any] = {}
    image_inputs: dict[str, Any] = {}
    for name, value in inputs.items():
        if not isinstance(name, str):
            raise RemoteInferenceProtocolError("Input keys must be strings")
        if _is_image_input(name):
            image_inputs[name] = value
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            model_inputs[name] = value
        elif isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
            model_inputs[name] = value
        elif isinstance(value, (bool, int, float, np.number)):
            scalar = np.asarray(value)
            if scalar.dtype.kind not in "biufc":
                raise RemoteInferenceProtocolError(f"Input {name!r} must be numeric or bool")
            model_inputs[name] = scalar
        else:
            raise RemoteInferenceProtocolError(f"Input {name!r} must be a numeric ndarray or list[str]")
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


def decode_predict_request(
    payload: bytes,
    *,
    max_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a v1 predict request and reconstruct all arrays/images."""
    message = decode_message(payload, max_bytes=max_bytes)
    if set(message) != {"protocol_version", "seq", "budget_ms", "inputs", "images"}:
        raise RemoteInferenceProtocolError("Predict request has unexpected or missing fields")
    version = message.get("protocol_version")
    if not isinstance(version, int) or isinstance(version, bool) or version != PROTOCOL_VERSION:
        raise RemoteInferenceProtocolError("Unsupported predict protocol version")
    seq = message.get("seq")
    budget = message.get("budget_ms")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        raise RemoteInferenceProtocolError("Invalid prediction sequence")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise RemoteInferenceProtocolError("Invalid prediction budget")
    inputs = message.get("inputs")
    if not isinstance(inputs, dict) or any(not isinstance(name, str) for name in inputs):
        raise RemoteInferenceProtocolError("Predict inputs must be a string-keyed map")
    decoded_bytes = 0
    for name, value in inputs.items():
        if _is_image_input(name):
            raise RemoteInferenceProtocolError("Image inputs must use the separate images map")
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "biufc":
                raise RemoteInferenceProtocolError("Input array dtype must be numeric or bool")
            decoded_bytes += value.nbytes
        elif not (isinstance(value, list) and all(isinstance(item, str) for item in value)):
            raise RemoteInferenceProtocolError("Predict inputs must contain numeric arrays or list[str]")
    if decoded_bytes > max_bytes:
        raise RemoteInferenceProtocolError("Decoded input arrays exceed max_request_bytes")
    image_inputs = decode_images(message.get("images"), max_bytes=max_bytes, already_decoded_bytes=decoded_bytes)
    overlap = inputs.keys() & image_inputs.keys()
    if overlap:
        raise RemoteInferenceProtocolError(f"Duplicate input and image keys: {sorted(overlap)!r}")
    inputs.update(image_inputs)
    return message, inputs


def encode_predict_reply(
    outputs: dict[str, Any],
    *,
    seq: int,
    server_id: str,
    server_queue_s: float,
    server_compute_s: float,
) -> bytes:
    """Encode the complete model output map and server timing in a v1 reply."""
    return encode_message({
        "protocol_version": PROTOCOL_VERSION,
        "seq": seq,
        "server_id": server_id,
        "outputs": outputs,
        "server_queue_s": server_queue_s,
        "server_compute_s": server_compute_s,
    })


def decode_predict_reply(payload: bytes) -> dict[str, Any]:
    """Decode and validate a v1 predict reply with recursively decoded arrays."""
    message = decode_message(payload)
    required = {
        "protocol_version",
        "seq",
        "server_id",
        "outputs",
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
        raise RemoteInferenceProtocolError("Malformed prediction reply")
    seq = message.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        raise RemoteInferenceProtocolError("Prediction reply has an invalid sequence")
    if not isinstance(message.get("server_id"), str) or not message["server_id"]:
        raise RemoteInferenceProtocolError("Prediction reply has an invalid server_id")
    outputs = message.get("outputs")
    if not isinstance(outputs, dict) or any(not isinstance(key, str) for key in outputs):
        raise RemoteInferenceProtocolError("Prediction outputs must be a string-keyed map")
    if any(not isinstance(value, np.ndarray) for value in outputs.values()):
        raise RemoteInferenceProtocolError("Prediction outputs must be NumPy arrays")
    for field in ("server_queue_s", "server_compute_s"):
        value = message.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise RemoteInferenceProtocolError(f"Prediction reply has invalid {field}")
    return message
