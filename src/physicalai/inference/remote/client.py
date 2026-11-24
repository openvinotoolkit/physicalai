# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lazy client proxy for a remotely served inference model."""

from __future__ import annotations

import contextlib
import copy
import logging
import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Self

import numpy as np
import zenoh

from physicalai.config import export_config
from physicalai.inference.constants import ACTION
from physicalai.inference.model import InferenceModel
from physicalai.transport import derive_port, endpoint_for_key, make_zenoh_config, model_key_prefix, open_zenoh_session

from ._protocol import (
    DEFAULT_IMAGE_JPEG_QUALITY,
    DEFAULT_MAX_REQUEST_BYTES,
    MAX_REQUEST_BUDGET_MS,
    PROTOCOL_VERSION,
    ImageCodec,
    RemoteInferenceError,
    RemoteInferenceModelMismatchError,
    RemoteInferenceProtocolError,
    RemoteInferenceTimeoutError,
    RemoteInferenceUnavailableError,
    RemoteTiming,
    decode_error,
    decode_message,
    decode_predict_reply,
    encode_message,
    encode_predict_request,
    validate_image_options,
)

logger = logging.getLogger(__name__)
_ACTION_TIME_AXIS_NDIM = 2


@export_config(class_path="physicalai.inference.RemoteInferenceModel")
class RemoteInferenceModel(InferenceModel):
    """Proxy an inference model using a lazy, persistent Zenoh client session."""

    def __init__(
        self,
        name: str,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        expected_policy: str | None = None,
        expected_manifest_sha256: str | None = None,
        request_timeout_s: float = 2.0,
        image_codec: ImageCodec = "jpeg",
        jpeg_quality: int = DEFAULT_IMAGE_JPEG_QUALITY,
        max_image_side: int | None = None,
        zenoh_config: str | Path | None = None,
    ) -> None:
        """Initialize a proxy without opening a network session.

        Raises:
            TypeError: If an expected policy identity has the wrong type.
            ValueError: If a name, address, timeout, or image option is invalid.
        """
        self._key_prefix = model_key_prefix(name)
        validate_image_options(image_codec, jpeg_quality, max_image_side)
        if (
            isinstance(request_timeout_s, bool)
            or not math.isfinite(request_timeout_s)
            or request_timeout_s <= 0
            or int(request_timeout_s * 1000) < 1
            or request_timeout_s > MAX_REQUEST_BUDGET_MS / 1000
        ):
            msg = "request_timeout_s must be between 0.001 and 60 seconds"
            raise ValueError(msg)
        if expected_policy is not None and not isinstance(expected_policy, str):
            msg = "expected_policy must be a string or None"
            raise TypeError(msg)
        if expected_manifest_sha256 is not None and not isinstance(expected_manifest_sha256, str):
            msg = "expected_manifest_sha256 must be a string or None"
            raise TypeError(msg)

        self._name = name
        self.policy_name = expected_policy or name
        self._host = host
        self._port = derive_port("inference", name) if port is None else port
        self._endpoint = endpoint_for_key(self._key_prefix, host, self._port)
        self._expected_policy = expected_policy
        self._expected_manifest_sha256 = expected_manifest_sha256
        self._request_timeout_s = float(request_timeout_s)
        self._image_codec: ImageCodec = image_codec
        self._jpeg_quality = jpeg_quality
        self._max_image_side = max_image_side
        self._zenoh_config_path = Path(zenoh_config) if zenoh_config is not None else None
        if self._zenoh_config_path is not None:
            logger.warning("user config %s replaces the secure Zenoh defaults", self._zenoh_config_path)
        self._builtin_config = None
        if self._zenoh_config_path is None:
            self._builtin_config = make_zenoh_config(
                "client",
                connect_endpoints=[self._endpoint],
                listen_endpoints=[],
                multicast_enabled=False,
                gossip_enabled=False,
            )
            self._builtin_config.insert_json5(
                "transport/link/rx/max_message_size",
                str(DEFAULT_MAX_REQUEST_BYTES),
            )

        self._lock = threading.RLock()
        self._session: zenoh.Session | None = None
        self._queriers: dict[str, Any] = {}
        self._seq = 0
        self._metadata: dict[str, Any] = {}
        self._server_id: str | None = None
        self._last_timing: RemoteTiming | None = None
        self._action_buffer: deque[np.ndarray] = deque()
        self._max_request_bytes = DEFAULT_MAX_REQUEST_BYTES

    def connect(self) -> None:
        """Open the session, declare queriers, and validate the server handshake.

        Raises:
            RemoteInferenceError: If the server fails the handshake.
            RemoteInferenceUnavailableError: If the server cannot be reached.
        """
        with self._lock:
            if self._session is not None and self._server_id is not None:
                return
            self._close_locked()
            try:
                if self._zenoh_config_path is None:
                    self._session = open_zenoh_session(
                        "client",
                        connect_endpoints=[self._endpoint],
                        listen_endpoints=[],
                        multicast_enabled=False,
                        gossip_enabled=False,
                        config=self._builtin_config,
                    )
                else:
                    custom_config = zenoh.Config.from_file(str(self._zenoh_config_path))
                    self._session = zenoh.open(custom_config)
                for key in (f"{self._key_prefix}/metadata", f"{self._key_prefix}/predict", f"{self._key_prefix}/reset"):
                    self._queriers[key.rsplit("/", maxsplit=1)[-1]] = self._session.declare_querier(
                        key,
                        target=zenoh.QueryTarget.BEST_MATCHING,
                        consolidation=zenoh.ConsolidationMode.NONE,
                        timeout=self._request_timeout_s,
                        congestion_control=zenoh.CongestionControl.BLOCK,
                        priority=zenoh.Priority.INTERACTIVE_HIGH,
                    )
                self._wait_for_server()
                request = encode_message({"protocol_version": PROTOCOL_VERSION})
                started = time.monotonic()
                metadata = self._query_metadata(request)
                self._validate_handshake(metadata)
                self._metadata = metadata
                self._server_id = metadata["server_id"]
                policy = metadata["policy"]
                self.policy_name = str(policy["name"])
                self._last_timing = None
                elapsed = time.monotonic() - started
                rtc_supported = metadata["rtc"]["supported"]
                rtc = metadata.get("rtc", {})
                chunk_size = (
                    rtc.get("chunk_size") if rtc_supported and isinstance(rtc, dict) else metadata.get("chunk_size")
                )
                logger.info(
                    "Connected to inference server %r: policy=%r, manifest_sha256=%r, chunk_size=%s "
                    "rtc_supported=%s (handshake %.3fs)",
                    self._name,
                    policy["name"],
                    policy["manifest_sha256"],
                    chunk_size,
                    rtc_supported,
                    elapsed,
                )
            except RemoteInferenceError:
                self._close_locked()
                raise
            except Exception as error:
                self._close_locked()
                msg = (
                    f"Inference server {self._name!r} at host {self._host!r}, port {self._port} is unavailable; "
                    f"is `physicalai inference serve --name {self._name}` running, and is the SSH tunnel up / "
                    f"address reachable? ({error})"
                )
                raise RemoteInferenceUnavailableError(msg) from error

    def _wait_for_server(self) -> None:
        deadline = time.monotonic() + self._request_timeout_s
        while not self._all_queries_match():
            if time.monotonic() >= deadline:
                msg = (
                    f"Inference server {self._name!r} at host {self._host!r}, port {self._port} is unavailable; "
                    f"is `physicalai inference serve --name {self._name}` running, and is the SSH tunnel up / "
                    "address reachable?"
                )
                raise RemoteInferenceUnavailableError(msg)
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))

    def __call__(self, inputs: dict[str, np.ndarray | list[str]]) -> dict[str, np.ndarray]:
        """Run the generic InferenceModel call interface remotely.

        Returns:
            The complete model output map.

        """
        with self._lock:
            self.connect()
            _, outputs = self._predict(inputs)
            self._validate_action_output(outputs)
            return outputs

    def _validate_action_output(self, outputs: dict[str, np.ndarray]) -> None:
        """Reject malformed action arrays before exposing them to callers.

        Raises:
            RemoteInferenceProtocolError: If action values, dtype, or chunk length are invalid.
        """
        actions = outputs.get(ACTION)
        if actions is None:
            return
        if actions.dtype.kind != "f":
            msg = f"Remote action output must use a floating-point dtype, got {actions.dtype}"
            raise RemoteInferenceProtocolError(msg)
        if not np.isfinite(actions).all():
            msg = "Remote action output must contain only finite values"
            raise RemoteInferenceProtocolError(msg)
        action_count = actions.shape[-_ACTION_TIME_AXIS_NDIM] if actions.ndim >= _ACTION_TIME_AXIS_NDIM else 1
        chunk_size = self.chunk_size
        # InferenceModel uses 1 when an export does not declare its chunk size.
        if chunk_size > 1 and action_count > chunk_size:
            msg = f"Remote action chunk length {action_count} exceeds advertised chunk_size {chunk_size}"
            raise RemoteInferenceProtocolError(msg)

    def reset(self) -> None:
        """Reset the remote policy on the model-owning server worker."""
        with self._lock:
            self.connect()
            request = encode_message({"protocol_version": PROTOCOL_VERSION})
            self._send("reset", request, None)
            self._action_buffer.clear()
            self._last_timing = None

    def close(self) -> None:
        """Close queriers and the persistent Zenoh session."""
        with self._lock:
            self._close_locked()

    def __enter__(self) -> Self:
        """Connect and return this model proxy.

        Returns:
            The connected proxy.
        """
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the proxy when leaving a context manager."""
        self.close()

    def __repr__(self) -> str:
        """Describe the configured address and connection state.

        Returns:
            A diagnostic representation of this proxy.
        """
        return (
            f"RemoteInferenceModel(name={self._name!r}, host={self._host!r}, port={self._port!r}, "
            f"connected={self._server_id is not None})"
        )

    @property
    def host(self) -> str:
        """The configured server hostname or IP address."""
        return self._host

    @property
    def port(self) -> int:
        """The explicit or name-derived TCP port."""
        return self._port

    @property
    def chunk_size(self) -> int:
        """Action chunk size from the server metadata, connecting if required."""
        with self._lock:
            self.connect()
            rtc = self._metadata.get("rtc", {})
            if rtc.get("supported") and rtc.get("chunk_size") is not None:
                return int(rtc["chunk_size"])
            return int(self._metadata["chunk_size"])

    @property
    def metadata(self) -> dict[str, Any]:
        """Copy of the last validated server handshake metadata."""
        with self._lock:
            return copy.deepcopy(self._metadata)

    @property
    def last_timing(self) -> RemoteTiming | None:
        """Timing information for the most recent completed request."""
        return self._last_timing

    def _all_queries_match(self) -> bool:
        for querier in self._queriers.values():
            status = querier.matching_status
            status = status() if callable(status) else status
            if not bool(getattr(status, "matching", False)):
                return False
        return bool(self._queriers)

    def _validate_handshake(self, metadata: dict[str, Any]) -> None:
        version = metadata.get("protocol_version")
        if not isinstance(version, int) or isinstance(version, bool) or version != PROTOCOL_VERSION:
            msg = f"Remote protocol version {metadata.get('protocol_version')!r} does not match {PROTOCOL_VERSION}"
            raise RemoteInferenceProtocolError(msg)
        if metadata.get("name") != self._name:
            msg = f"Server name {metadata.get('name')!r} does not match requested name {self._name!r}"
            raise RemoteInferenceProtocolError(msg)
        server_id = metadata.get("server_id")
        policy = metadata.get("policy")
        chunk_size = metadata.get("chunk_size")
        rtc = metadata.get("rtc", {})
        if not isinstance(server_id, str) or not server_id:
            msg = "Server metadata has no server_id"
            raise RemoteInferenceProtocolError(msg)
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            msg = "Server metadata has an invalid chunk_size"
            raise RemoteInferenceProtocolError(msg)
        if not isinstance(policy, dict) or any(
            not isinstance(policy.get(field), str) for field in ("name", "manifest_sha256", "backend", "device")
        ):
            msg = "Server metadata has invalid policy identity"
            raise RemoteInferenceProtocolError(msg)
        if not isinstance(rtc, dict) or not isinstance(rtc.get("supported"), bool):
            msg = "Server metadata has invalid RTC information"
            raise RemoteInferenceProtocolError(msg)
        rtc_chunk_size = rtc.get("chunk_size")
        if (
            rtc["supported"]
            and rtc_chunk_size is not None
            and (not isinstance(rtc_chunk_size, int) or isinstance(rtc_chunk_size, bool) or rtc_chunk_size < 1)
        ):
            msg = "Server metadata has an invalid RTC chunk_size"
            raise RemoteInferenceProtocolError(msg)
        image_codecs = metadata.get("image_codecs", ["jpeg", "raw"])
        if (
            not isinstance(image_codecs, list)
            or any(not isinstance(codec, str) for codec in image_codecs)
            or self._image_codec not in image_codecs
        ):
            msg = f"Server does not advertise the configured image codec {self._image_codec!r}"
            raise RemoteInferenceProtocolError(msg)
        max_request_bytes = metadata.get("max_request_bytes", DEFAULT_MAX_REQUEST_BYTES)
        if not isinstance(max_request_bytes, int) or isinstance(max_request_bytes, bool) or max_request_bytes < 1:
            msg = "Server metadata has invalid max_request_bytes"
            raise RemoteInferenceProtocolError(msg)
        if self._expected_policy is not None and policy["name"] != self._expected_policy:
            msg = f"Expected policy {self._expected_policy!r}, got {policy['name']!r}"
            raise RemoteInferenceModelMismatchError(msg)
        if self._expected_manifest_sha256 is not None and policy["manifest_sha256"] != self._expected_manifest_sha256:
            msg = "Remote manifest SHA-256 does not match the expected value"
            raise RemoteInferenceModelMismatchError(msg)
        self._max_request_bytes = max_request_bytes

    def _query_metadata(self, payload: bytes) -> dict[str, Any]:
        reply = self._receive_matching_reply("metadata", payload)
        if reply.err is not None:
            self._raise_remote_error(bytes(reply.err.payload), None)
        if reply.ok is None:
            msg = "Metadata query returned an empty reply"
            raise RemoteInferenceProtocolError(msg)
        return decode_message(bytes(reply.ok.payload))

    def _predict(self, inputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        self._last_timing = None
        round_trip_started = time.perf_counter()
        seq = self._next_seq()
        encode_started = time.perf_counter()
        payload = encode_predict_request(
            inputs,
            seq=seq,
            budget_ms=int(self._request_timeout_s * 1000),
            image_codec=self._image_codec,
            jpeg_quality=self._jpeg_quality,
            max_image_side=self._max_image_side,
        )
        if len(payload) > self._max_request_bytes:
            msg = f"Encoded request exceeds server max_request_bytes ({self._max_request_bytes})"
            raise RemoteInferenceProtocolError(msg)
        encode_s = time.perf_counter() - encode_started
        request_bytes = len(payload)
        reply = self._receive_matching_reply("predict", payload)
        if reply.err is not None:
            self._raise_remote_error(bytes(reply.err.payload), seq)
        if reply.ok is None:
            msg = "Prediction query returned an empty reply"
            raise RemoteInferenceProtocolError(msg)
        reply_payload = bytes(reply.ok.payload)
        decode_started = time.perf_counter()
        metadata = decode_predict_reply(reply_payload)
        decode_s = time.perf_counter() - decode_started
        self._validate_response(metadata, seq)
        self._last_timing = RemoteTiming(
            round_trip_s=time.perf_counter() - round_trip_started,
            server_queue_s=float(metadata["server_queue_s"]),
            server_compute_s=float(metadata["server_compute_s"]),
            server_decode_s=float(metadata["server_decode_s"]),
            encode_s=encode_s,
            decode_s=decode_s,
            request_bytes=request_bytes,
            reply_bytes=len(reply_payload),
        )
        return metadata, metadata["outputs"]

    def _receive_matching_reply(self, name: str, payload: bytes) -> zenoh.Reply:
        if self._session is None:
            msg = "Remote inference session is not connected"
            raise RemoteInferenceUnavailableError(msg)
        querier = self._queriers.get(name)
        if querier is None:
            msg = f"Remote inference querier {name!r} is not declared"
            raise RemoteInferenceUnavailableError(msg)
        receiver = querier.get(payload=payload)
        try:
            reply = receiver.recv()
        except zenoh.ZError as error:
            msg = f"Remote inference query failed: {error}"
            raise RemoteInferenceUnavailableError(msg) from error
        if reply is None:
            msg = f"Remote inference request {name} timed out after {self._request_timeout_s:.3f}s"
            raise RemoteInferenceTimeoutError(msg)
        if reply.err is not None and bytes(reply.err.payload) == b"Timeout":
            msg = f"Remote inference request {name} timed out after {self._request_timeout_s:.3f}s"
            raise RemoteInferenceTimeoutError(msg)
        return reply

    def _validate_response(self, metadata: dict[str, Any], seq: int) -> None:
        version = metadata.get("protocol_version")
        response_seq = metadata.get("seq")
        if not isinstance(version, int) or isinstance(version, bool) or version != PROTOCOL_VERSION:
            msg = "Prediction response protocol version mismatch"
            raise RemoteInferenceProtocolError(msg)
        if not isinstance(response_seq, int) or isinstance(response_seq, bool) or response_seq != seq:
            msg = "Prediction response sequence mismatch"
            raise RemoteInferenceProtocolError(msg)
        for field in ("server_queue_s", "server_compute_s"):
            value = metadata.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                msg = f"Prediction response has invalid {field}"
                raise RemoteInferenceProtocolError(msg)
        server_id = metadata.get("server_id")
        if not isinstance(server_id, str) or server_id != self._server_id:
            self._close_locked()
            msg = "Inference server restarted or was replaced"
            raise RemoteInferenceModelMismatchError(msg)

    def _send(self, name: str, payload: bytes, seq: int | None) -> dict[str, Any]:
        reply = self._receive_matching_reply(name, payload)
        if reply.err is not None:
            self._raise_remote_error(bytes(reply.err.payload), seq)
        if reply.ok is None:
            msg = "Remote request returned an empty reply"
            raise RemoteInferenceProtocolError(msg)
        response = decode_message(bytes(reply.ok.payload))
        if response != {"protocol_version": PROTOCOL_VERSION, "ok": True}:
            msg = "Invalid reset response"
            raise RemoteInferenceProtocolError(msg)
        return response

    @staticmethod
    def _raise_remote_error(payload: bytes, seq: int | None) -> None:
        metadata = decode_error(payload)
        if metadata.get("protocol_version") != PROTOCOL_VERSION:
            msg = "Remote error protocol version mismatch"
            raise RemoteInferenceProtocolError(msg)
        error_seq = metadata.get("seq")
        if not isinstance(error_seq, int) or isinstance(error_seq, bool):
            msg = "Remote error has an invalid sequence"
            raise RemoteInferenceProtocolError(msg)
        if seq is not None and error_seq != seq:
            msg = "Remote error sequence mismatch"
            raise RemoteInferenceProtocolError(msg)
        code = metadata.get("code")
        if code == "expired":
            msg = f"Remote inference request {seq} expired in the server queue"
            raise RemoteInferenceTimeoutError(msg)
        if code == "unsupported_version":
            msg = "Remote server does not support this protocol version"
            raise RemoteInferenceProtocolError(msg)
        if code == "invalid_request":
            msg = "Remote server rejected the request as invalid"
            raise RemoteInferenceProtocolError(msg)
        if code == "shutting_down":
            msg = "Remote inference server is shutting down"
            raise RemoteInferenceUnavailableError(msg)
        if code == "superseded":
            msg = f"Remote inference request {seq} was superseded by a newer request"
            raise RemoteInferenceError(msg)
        if code == "model_error":
            msg = "Remote model failed to process the request"
            raise RemoteInferenceError(msg)
        msg = f"Unknown remote error code: {code!r}"
        raise RemoteInferenceProtocolError(msg)

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _close_locked(self) -> None:
        for querier in self._queriers.values():
            with contextlib.suppress(Exception):
                querier.undeclare()
        self._queriers.clear()
        if self._session is not None:
            try:
                self._session.close()
            finally:
                self._session = None
        self._server_id = None
        self._metadata = {}
        self._action_buffer.clear()
        self._last_timing = None
        self._max_request_bytes = DEFAULT_MAX_REQUEST_BYTES
