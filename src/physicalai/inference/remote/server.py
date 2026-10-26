# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Single-worker Zenoh server for a remotely accessible inference model."""

from __future__ import annotations

import contextlib
import hashlib
import ipaddress
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgpack
import numpy as np
import zenoh

from physicalai.transport import derive_port, endpoint_for_key, make_zenoh_config, model_key_prefix

from ._protocol import (
    DEFAULT_MAX_REQUEST_BYTES,
    PROTOCOL_VERSION,
    RemoteInferenceProtocolError,
    decode_message,
    decode_predict_request,
    encode_message,
    encode_predict_reply,
)

if TYPE_CHECKING:
    from physicalai.inference.model import InferenceModel

logger = logging.getLogger(__name__)
_SUMMARY_INTERVAL_S = 30.0
_TIMING_SAMPLES = 2048


@dataclass(slots=True)
class _Request:
    query: Any
    command: str
    seq: int
    budget_ms: int
    received_at: float
    request_bytes: int
    inputs: dict[str, Any] | None = None
    decoded_at: float | None = None


class InferenceServer:
    """Serve one inference model using exactly one model-owning worker thread."""

    def __init__(
        self,
        model: InferenceModel,
        name: str,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        allow_all_interfaces: bool = False,
        zenoh_config: str | Path | None = None,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    ) -> None:
        """Configure the model worker and listener without opening a session.

        Raises:
            ValueError: If the name, address, request limit, or wildcard opt-in is invalid.
        """
        self._key_prefix = model_key_prefix(name)
        if not isinstance(max_request_bytes, int) or isinstance(max_request_bytes, bool) or max_request_bytes < 1:
            msg = "max_request_bytes must be a positive integer"
            raise ValueError(msg)
        self._name = name
        self._model = model
        self._host = host
        self._port = derive_port("inference", name) if port is None else port
        self._listen = endpoint_for_key(self._key_prefix, host, self._port)
        if self._is_wildcard_host(host) and not allow_all_interfaces:
            msg = "Wildcard hosts require allow_all_interfaces=True"
            raise ValueError(msg)
        self._allow_all_interfaces = allow_all_interfaces
        self._zenoh_config_path = Path(zenoh_config) if zenoh_config is not None else None
        if self._zenoh_config_path is not None:
            logger.warning("user config %s replaces the secure Zenoh defaults", self._zenoh_config_path)
        self._max_request_bytes = max_request_bytes
        self._server_id = ""
        self._metadata_key = f"{self._key_prefix}/metadata"
        self._predict_key = f"{self._key_prefix}/predict"
        self._reset_key = f"{self._key_prefix}/reset"

        self._lifecycle_lock = threading.RLock()
        self._condition = threading.Condition()
        self._queue: deque[_Request] = deque()
        self._session: zenoh.Session | None = None
        self._queryables: list[Any] = []
        self._worker: threading.Thread | None = None
        self._started = threading.Event()
        self._stopping = threading.Event()
        self._queue_samples: deque[float] = deque(maxlen=_TIMING_SAMPLES)
        self._decode_samples: deque[float] = deque(maxlen=_TIMING_SAMPLES)
        self._compute_samples: deque[float] = deque(maxlen=_TIMING_SAMPLES)
        self._counts = {"count": 0, "superseded": 0, "expired": 0}
        self._last_summary_counts = self._counts.copy()
        self._last_summary = 0.0
        self._metadata: dict[str, Any] = {}

    @property
    def host(self) -> str:
        """The configured bind hostname or IP address."""
        return self._host

    @property
    def port(self) -> int:
        """The explicit or name-derived TCP port."""
        return self._port

    def start(self) -> None:
        """Open the peer session and worker without blocking.

        Raises:
            RuntimeError: If this server was already stopped.
        """
        with self._lifecycle_lock:
            if self._started.is_set():
                return
            if self._stopping.is_set():
                msg = "InferenceServer cannot be restarted after stop()"
                raise RuntimeError(msg)
            self._server_id = uuid.uuid4().hex
            self._metadata = self._build_metadata()
            config = self._build_config()
            try:
                self._warn_non_loopback_listener()
                self._session = zenoh.open(config)
                self._last_summary = time.monotonic()
                self._worker = threading.Thread(target=self._worker_loop, name="PhysicalAIInference", daemon=True)
                self._worker.start()
                for key, command in (
                    (self._metadata_key, "metadata"),
                    (self._predict_key, "predict"),
                    (self._reset_key, "reset"),
                ):
                    queryable = self._session.declare_queryable(
                        key,
                        lambda query, request_command=command: self._on_query(query, request_command),
                        complete=True,
                    )
                    self._queryables.append(queryable)
            except Exception:
                self._stopping.set()
                with self._condition:
                    self._condition.notify_all()
                self._close_resources()
                raise
            self._started.set()
            rtc_supported = self._metadata["rtc"]["supported"]
            logger.info(
                "Inference server started: name=%s policy=%s manifest_sha256=%s chunk_size=%s "
                "rtc_supported=%s host=%s port=%s",
                self._name,
                self._metadata["policy"]["name"],
                self._metadata["policy"]["manifest_sha256"],
                self._metadata["chunk_size"],
                rtc_supported,
                self._host,
                self._port,
            )

    def serve_forever(self) -> None:
        """Start the server and block until :meth:`stop` is called."""
        self.start()
        try:
            self._stopping.wait()
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop accepting requests, fail pending work, and close resources."""
        with self._lifecycle_lock:
            if self._stopping.is_set():
                return
            self._stopping.set()
            with self._condition:
                while self._queue:
                    request = self._queue.popleft()
                    self._reply_error(request.query, request.seq, "shutting_down")
                self._condition.notify_all()
            self._undeclare_queryables()
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join()
        with self._lifecycle_lock:
            self._close_resources()
            self._started.clear()

    def _build_config(self) -> zenoh.Config:
        if self._zenoh_config_path is None:
            config = make_zenoh_config(
                "peer",
                connect_endpoints=[],
                listen_endpoints=[self._listen],
                multicast_enabled=False,
                gossip_enabled=False,
            )
        else:
            config = zenoh.Config.from_file(str(self._zenoh_config_path))
        config.insert_json5("transport/link/rx/max_message_size", str(self._max_request_bytes))
        config.insert_json5("transport/shared_memory/enabled", "false")
        return config

    def _build_metadata(self) -> dict[str, Any]:
        manifest = getattr(self._model, "manifest", None)
        model_extra = getattr(manifest, "model_extra", {})
        has_rtc = isinstance(model_extra, dict) and "rtc" in model_extra
        rtc_config = model_extra.get("rtc") if has_rtc else None
        rtc_chunk_size = rtc_config.get("chunk_size") if isinstance(rtc_config, dict) else None
        if not isinstance(rtc_chunk_size, int) or isinstance(rtc_chunk_size, bool) or rtc_chunk_size < 1:
            rtc_chunk_size = None
        chunk_size = getattr(self._model, "chunk_size", 1)
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            chunk_size = 1
        manifest_policy = getattr(getattr(manifest, "policy", None), "name", None)
        policy_name = manifest_policy or getattr(self._model, "policy_name", "") or ""
        manifest_sha256 = ""
        export_dir = getattr(self._model, "export_dir", None)
        if export_dir is not None:
            manifest_path = Path(export_dir) / "manifest.json"
            with contextlib.suppress(OSError):
                manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        return {
            "protocol_version": PROTOCOL_VERSION,
            "server_id": self._server_id,
            "name": self._name,
            "policy": {
                "name": policy_name,
                "manifest_sha256": manifest_sha256,
                "backend": str(getattr(self._model, "backend", "")),
                "device": str(getattr(self._model, "device", "")),
            },
            "chunk_size": chunk_size,
            "rtc": {"supported": has_rtc, "chunk_size": rtc_chunk_size},
            "image_codecs": ["jpeg", "raw"],
            "max_request_bytes": self._max_request_bytes,
        }

    @staticmethod
    def _is_wildcard_host(host: str) -> bool:
        try:
            return ipaddress.ip_address(host).is_unspecified
        except ValueError:
            return False

    @staticmethod
    def _is_loopback_host(host: str) -> bool:
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return host.lower() == "localhost"

    def _warn_non_loopback_listener(self) -> None:
        if not self._is_loopback_host(self._host):
            address = f"[{self._host}]" if ":" in self._host else self._host
            logger.warning(
                "inference server reachable at %s:%s without Zenoh-level authentication — "
                "use only on a trusted network or tailnet",
                address,
                self._port,
            )

    def _on_query(self, query: zenoh.Query, command: str) -> None:
        received_at = time.monotonic()
        payload = bytes(query.payload) if query.payload is not None else b""
        if self._stopping.is_set():
            self._reply_error(query, self._read_seq(payload), "shutting_down")
            return
        if len(payload) > self._max_request_bytes:
            self._reply_error(query, -1, "invalid_request")
            return
        try:
            self._handle_query(query, command, payload, received_at)
        except RemoteInferenceProtocolError:
            logger.debug("Rejected malformed remote inference request", exc_info=True)
            self._reply_error(query, self._read_seq(payload), "invalid_request")
        except Exception:
            logger.exception("Failed to decode remote inference request")
            self._reply_error(query, self._read_seq(payload), "invalid_request")

    def _handle_query(self, query: zenoh.Query, command: str, payload: bytes, received_at: float) -> None:
        if command == "metadata":
            metadata = decode_message(payload, max_bytes=self._max_request_bytes)
            version = metadata.get("protocol_version")
            if not isinstance(version, int) or isinstance(version, bool) or version != PROTOCOL_VERSION:
                self._reply_error(query, -1, "unsupported_version")
                return
            if metadata != {"protocol_version": PROTOCOL_VERSION}:
                msg = "Metadata request must contain only protocol_version"
                raise RemoteInferenceProtocolError(msg)
            self._reply_metadata(query)
            return
        if command == "predict":
            metadata, inputs = decode_predict_request(payload, max_bytes=self._max_request_bytes)
            request = _Request(
                query=query,
                command="predict",
                seq=metadata["seq"],
                budget_ms=metadata["budget_ms"],
                received_at=received_at,
                request_bytes=len(payload),
                inputs=inputs,
                decoded_at=time.monotonic(),
            )
            self._enqueue(request)
            return
        if command == "reset":
            metadata = decode_message(payload, max_bytes=self._max_request_bytes)
            version = metadata.get("protocol_version")
            if not isinstance(version, int) or isinstance(version, bool) or version != PROTOCOL_VERSION:
                self._reply_error(query, -1, "unsupported_version")
                return
            if metadata != {"protocol_version": PROTOCOL_VERSION}:
                msg = "Reset request must contain only protocol_version"
                raise RemoteInferenceProtocolError(msg)
            request = _Request(
                query=query,
                command="reset",
                seq=-1,
                budget_ms=2**31 - 1,
                received_at=received_at,
                request_bytes=len(payload),
                decoded_at=time.monotonic(),
            )
            self._enqueue(request)
            return
        msg = "Unsupported query key"
        raise RemoteInferenceProtocolError(msg)

    def _enqueue(self, request: _Request) -> None:
        with self._condition:
            if self._stopping.is_set():
                self._reply_error(request.query, request.seq, "shutting_down")
                return
            for pending in tuple(self._queue):
                if pending.command != request.command:
                    continue
                self._queue.remove(pending)
                self._reply_error(pending.query, pending.seq, "superseded")
                self._counts["superseded"] += 1
                break
            self._queue.append(request)
            self._condition.notify()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._stopping.is_set():
                    remaining = _SUMMARY_INTERVAL_S - (time.monotonic() - self._last_summary)
                    self._condition.wait(timeout=max(0.0, remaining))
                    if time.monotonic() - self._last_summary >= _SUMMARY_INTERVAL_S:
                        self._log_summary()
                if self._stopping.is_set() and not self._queue:
                    return
                request = self._queue.popleft()
            worker_started = time.monotonic()
            decoded_at = request.decoded_at if request.decoded_at is not None else request.received_at
            decode_ms = (decoded_at - request.received_at) * 1000
            queue_ms = (worker_started - decoded_at) * 1000
            if (worker_started - request.received_at) * 1000 > request.budget_ms:
                self._counts["expired"] += 1
                self._log_request(request, queue_ms, 0.0, "expired")
                self._reply_error(request.query, request.seq, "expired")
                self._maybe_log_summary()
                continue
            compute_started = time.monotonic()
            try:
                if request.command == "reset":
                    self._model.reset()
                    outputs: dict[str, Any] = {}
                else:
                    outputs = self._run_model(request.inputs or {})
                compute_ms = (time.monotonic() - compute_started) * 1000
                if request.command == "reset":
                    response = encode_message({"protocol_version": PROTOCOL_VERSION, "ok": True})
                else:
                    response = encode_predict_reply(
                        outputs,
                        seq=request.seq,
                        server_id=self._server_id,
                        server_queue_s=queue_ms / 1000,
                        server_compute_s=compute_ms / 1000,
                        server_decode_s=decode_ms / 1000,
                    )
                request.query.reply(self._key(request.command), response)
                self._counts["count"] += 1
                self._queue_samples.append(queue_ms)
                self._decode_samples.append(decode_ms)
                self._compute_samples.append(compute_ms)
                self._log_request(request, queue_ms, compute_ms, "ok")
            except Exception:
                compute_ms = (time.monotonic() - compute_started) * 1000
                logger.exception("Remote inference model request failed (seq=%s)", request.seq)
                self._counts["count"] += 1
                self._queue_samples.append(queue_ms)
                self._decode_samples.append(decode_ms)
                self._compute_samples.append(compute_ms)
                self._log_request(request, queue_ms, compute_ms, "model_error")
                self._reply_error(request.query, request.seq, "model_error")
            self._maybe_log_summary()

    def _run_model(self, inputs: dict[str, Any]) -> dict[str, Any]:
        outputs = self._model(inputs)
        if not isinstance(outputs, dict):
            msg = "model __call__ must return a mapping of output arrays"
            raise TypeError(msg)
        return outputs

    def _reply_metadata(self, query: zenoh.Query) -> None:
        query.reply(self._metadata_key, encode_message(self._metadata))

    @staticmethod
    def _reply_error(query: zenoh.Query, seq: int, code: str) -> None:
        payload = encode_message(
            {"protocol_version": PROTOCOL_VERSION, "code": code, "seq": seq},
        )
        try:
            query.reply_err(payload)
        except Exception:
            logger.debug("Unable to send remote inference error reply", exc_info=True)

    @staticmethod
    def _log_request(request: _Request, queue_ms: float, compute_ms: float, outcome: str) -> None:
        decoded_at = request.decoded_at if request.decoded_at is not None else request.received_at
        logger.debug(
            "Remote inference request seq=%s bytes=%s decode_ms=%.3f queue_ms=%.3f compute_ms=%.3f outcome=%s",
            request.seq,
            request.request_bytes,
            (decoded_at - request.received_at) * 1000,
            queue_ms,
            compute_ms,
            outcome,
        )

    def _log_summary(self) -> None:
        self._last_summary = time.monotonic()
        if self._counts == self._last_summary_counts:
            return
        self._last_summary_counts = self._counts.copy()
        queue = np.asarray(self._queue_samples, dtype=np.float64)
        decode = np.asarray(self._decode_samples, dtype=np.float64)
        compute = np.asarray(self._compute_samples, dtype=np.float64)
        logger.info(
            "Remote inference summary count=%d decode_ms_p50=%.3f decode_ms_p99=%.3f "
            "queue_ms_p50=%.3f queue_ms_p99=%.3f "
            "compute_ms_p50=%.3f compute_ms_p99=%.3f superseded=%d expired=%d",
            self._counts["count"],
            float(np.percentile(decode, 50)) if decode.size else 0.0,
            float(np.percentile(decode, 99)) if decode.size else 0.0,
            float(np.percentile(queue, 50)) if queue.size else 0.0,
            float(np.percentile(queue, 99)) if queue.size else 0.0,
            float(np.percentile(compute, 50)) if compute.size else 0.0,
            float(np.percentile(compute, 99)) if compute.size else 0.0,
            self._counts["superseded"],
            self._counts["expired"],
        )

    def _maybe_log_summary(self) -> None:
        with self._condition:
            if time.monotonic() - self._last_summary >= _SUMMARY_INTERVAL_S:
                self._log_summary()

    def _key(self, command: str) -> str:
        return self._reset_key if command == "reset" else self._predict_key

    @staticmethod
    def _read_seq(payload: bytes) -> int:
        try:
            unpacked = msgpack.unpackb(payload, raw=False, strict_map_key=False)
            seq = unpacked.get("seq", -1) if isinstance(unpacked, dict) else -1
            return seq if isinstance(seq, int) and not isinstance(seq, bool) else -1
        except (msgpack.UnpackException, TypeError, ValueError, OverflowError):
            return -1

    def _undeclare_queryables(self) -> None:
        for queryable in self._queryables:
            try:
                queryable.undeclare()
            except Exception:
                logger.debug("Could not undeclare inference queryable", exc_info=True)
        self._queryables.clear()

    def _close_resources(self) -> None:
        self._undeclare_queryables()
        if self._session is not None:
            try:
                self._session.close()
            finally:
                self._session = None
