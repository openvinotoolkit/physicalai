# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for model-namespaced Zenoh remote inference."""

from __future__ import annotations

import hashlib
import inspect
import logging
import socket
import threading
import time
from types import SimpleNamespace
from typing import Any

import msgpack
import numpy as np
import pytest

zenoh = pytest.importorskip("zenoh")

from physicalai.inference.constants import ACTION, PREV_CHUNK_LEFT_OVER, RTC_INFERENCE_DELAY  # noqa: E402
from physicalai.inference.remote import (  # noqa: E402
    InferenceServer,
    RemoteInferenceModel,
    RemoteInferenceModelMismatchError,
)
from physicalai.inference.remote import _protocol as protocol_module  # noqa: E402
from physicalai.inference.remote import client as client_module  # noqa: E402
from physicalai.inference.remote import server as server_module  # noqa: E402
from physicalai.inference.remote._protocol import (  # noqa: E402
    RemoteInferenceError,
    RemoteInferenceProtocolError,
    RemoteInferenceTimeoutError,
    decode_message,
    decode_predict_reply,
    decode_predict_request,
    encode_message,
    encode_predict_reply,
    encode_predict_request,
)
from physicalai.inference.remote.server import _Request  # noqa: E402
from physicalai.runtime.action_sources.policy import PolicySource  # noqa: E402
from physicalai.runtime.execution.async_execution import AsyncExecution  # noqa: E402
from physicalai.runtime.execution.queue import ChunkedActionQueue  # noqa: E402
from physicalai.runtime.execution.rtc import RTCExecution  # noqa: E402
from physicalai.runtime.execution.rtc_queue import RTCActionQueue  # noqa: E402
from physicalai.runtime.execution.sync import SyncExecution  # noqa: E402
from physicalai.transport._zenoh import endpoint_for_key, model_key_prefix  # noqa: E402


class _Model:
    def __init__(self, policy_name: str) -> None:
        self.policy_name = policy_name
        self.predict_count = 0
        self.call_count = 0
        self.reset_count = 0
        self.reset_thread_id: int | None = None
        self.chunk_size = 4
        self.manifest = type("Manifest", (), {"model_extra": {"rtc": {}}})()
        self.last_call_inputs: dict[str, Any] | None = None

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        self.predict_count += 1
        return np.ones((4, 3), dtype=np.float32)

    def reset(self) -> None:
        self.reset_count += 1
        self.reset_thread_id = threading.get_ident()

    def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        self.call_count += 1
        self.last_call_inputs = inputs
        return {ACTION: self.predict_action_chunk(inputs)}


def _free_endpoint() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"tcp/127.0.0.1:{port}"


def test_client_rejects_server_with_different_expected_policy() -> None:
    model_b = _Model("model-b")
    endpoint_b = _free_endpoint()
    server_b = InferenceServer(model_b, "model-b", listen=endpoint_b)
    server_b.start()
    assert len(server_b._queryables) == 3  # noqa: SLF001
    mismatched_client = RemoteInferenceModel(
        "model-b",
        endpoint=endpoint_b,
        expected_policy="another-policy",
        request_timeout_s=0.5,
    )

    with pytest.raises(RemoteInferenceModelMismatchError):
        mismatched_client.connect()
    mismatched_client.close()

    client_for_b = RemoteInferenceModel("model-b", endpoint=endpoint_b)
    try:
        actions = client_for_b.predict_action_chunk({"state": np.zeros(3, dtype=np.float32)})
        assert actions.shape == (4, 3)
        assert model_b.predict_count == 1
        assert client_for_b.last_timing is not None
        assert client_for_b.metadata["name"] == "model-b"
        assert client_for_b.metadata["server_id"] == server_b._server_id  # noqa: SLF001
        assert client_for_b.last_timing.request_bytes > 0
        assert client_for_b.last_timing.reply_bytes > 0
        assert client_for_b.last_timing.round_trip_s >= client_for_b.last_timing.encode_s
        assert client_for_b.last_timing.decode_s >= 0
    finally:
        client_for_b.close()
        server_b.stop()


def test_client_accepts_matching_policy_and_manifest_pins(tmp_path: Any) -> None:
    manifest_bytes = b'{"policy":{"name":"pi05"}}'
    (tmp_path / "manifest.json").write_bytes(manifest_bytes)
    model = _Model("pi05")
    model.export_dir = tmp_path
    model.manifest.policy = SimpleNamespace(name="pi05")
    endpoint = _free_endpoint()
    server = InferenceServer(model, "cell1-act", listen=endpoint)
    server.start()
    correct_hash = hashlib.sha256(manifest_bytes).hexdigest()
    matching = RemoteInferenceModel(
        "cell1-act",
        endpoint=endpoint,
        expected_policy="pi05",
        expected_manifest_sha256=correct_hash,
    )
    wrong_hash = RemoteInferenceModel(
        "cell1-act",
        endpoint=endpoint,
        expected_manifest_sha256="0" * 64,
    )
    try:
        matching.connect()
        assert matching.metadata["policy"]["manifest_sha256"] == correct_hash
        with pytest.raises(RemoteInferenceModelMismatchError, match="SHA-256"):
            wrong_hash.connect()
    finally:
        matching.close()
        wrong_hash.close()
        server.stop()


def test_server_error_reply_does_not_expose_exception_text() -> None:
    class _FailingModel(_Model):
        def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
            raise RuntimeError("private model failure details")

    name = "error-reply-test"
    server = InferenceServer(_FailingModel(name), name, listen=_free_endpoint())
    server.start()
    client = RemoteInferenceModel(name, endpoint=server.endpoint, request_timeout_s=0.5)
    try:
        with pytest.raises(RemoteInferenceError) as error:
            client.predict_action_chunk({"state": np.zeros(3, dtype=np.float32)})
        assert "private model failure details" not in str(error.value)
    finally:
        client.close()
        server.stop()


def test_task_remains_a_string_list_through_the_model_preprocessor() -> None:
    class _TaskCheckingModel(_Model):
        def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
            task = inputs["task"]
            if not isinstance(task, list) or not all(isinstance(item, str) for item in task):
                raise TypeError("task must be list[str]")
            self.seen_task = task
            return {ACTION: np.ones((1, 4, 3), dtype=np.float32)}

    name = "task-list-preprocessor"
    model = _TaskCheckingModel(name)
    server = InferenceServer(model, name, listen=_free_endpoint())
    server.start()
    client = RemoteInferenceModel(name, endpoint=server.endpoint)
    try:
        outputs = client({"task": ["pick red"]})
        assert outputs[ACTION].shape == (1, 4, 3)
        assert model.seen_task == ["pick red"]
    finally:
        client.close()
        server.stop()


def test_live_server_returns_invalid_request_without_exception_text() -> None:
    name = "invalid-request-live"
    server = InferenceServer(_Model(name), name, listen=_free_endpoint(), max_request_bytes=1024)
    server.start()
    client = RemoteInferenceModel(name, endpoint=server.endpoint)
    malformed_payloads = [
        encode_message({
            "protocol_version": 1,
            "seq": 41,
            "budget_ms": 1000,
            "inputs": {"state": {"__np__": True, "dtype": "|O", "shape": [1], "data": b"x"}},
            "images": {},
        }),
        encode_message({
            "protocol_version": 1,
            "seq": 42,
            "budget_ms": 1000,
            "inputs": {"state": {"__np__": True, "dtype": "float32", "shape": [2], "data": b"1234"}},
            "images": {},
        }),
    ]
    try:
        client.connect()
        for payload, seq in zip(malformed_payloads, (41, 42), strict=True):
            reply = client._queriers["predict"].get(payload=payload).recv()  # noqa: SLF001
            assert reply.err is not None
            error = msgpack.unpackb(bytes(reply.err.payload), raw=False)
            assert error == {"protocol_version": 1, "code": "invalid_request", "seq": seq}
    finally:
        client.close()
        server.stop()


def test_two_server_namespaces_are_isolated_after_one_server_stops() -> None:
    class _ValueModel(_Model):
        def __init__(self, value: float, chunk_size: int) -> None:
            super().__init__("pi05")
            self.value = value
            self.chunk_size = chunk_size

        def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
            return {ACTION: np.full((1, self.chunk_size, 2), self.value, dtype=np.float32)}

    endpoint_a, endpoint_b = _free_endpoint(), _free_endpoint()
    server_a = InferenceServer(_ValueModel(11.0, 3), "cell1-act", listen=endpoint_a)
    server_b = InferenceServer(_ValueModel(22.0, 5), "cell2-act", listen=endpoint_b)
    server_a.start()
    server_b.start()
    client_a = RemoteInferenceModel("cell1-act", endpoint=endpoint_a, request_timeout_s=0.2)
    client_b = RemoteInferenceModel("cell2-act", endpoint=endpoint_b, request_timeout_s=0.2)
    try:
        actions_a = client_a.predict_action_chunk({"state": np.zeros(2, dtype=np.float32)})
        actions_b = client_b.predict_action_chunk({"state": np.zeros(2, dtype=np.float32)})
        assert actions_a.shape == (3, 2)
        assert actions_b.shape == (5, 2)
        assert np.all(actions_a == 11.0)
        assert np.all(actions_b == 22.0)

        server_a.stop()
        with pytest.raises(RemoteInferenceError):
            client_a.predict_action_chunk({"state": np.zeros(2, dtype=np.float32)})
        actions_b_again = client_b.predict_action_chunk({"state": np.zeros(2, dtype=np.float32)})
        assert np.all(actions_b_again == 22.0)
    finally:
        client_a.close()
        client_b.close()
        server_a.stop()
        server_b.stop()


def test_two_servers_with_same_name_cannot_bind_the_same_port() -> None:
    name = "same-slot-collision"
    endpoint = _free_endpoint()
    first = InferenceServer(_Model("pi05"), name, listen=endpoint)
    second = InferenceServer(_Model("pi05"), name, listen=endpoint)
    first.start()
    try:
        with pytest.raises(Exception, match="Address already in use"):
            second.start()
        assert not second._started.is_set()  # noqa: SLF001
    finally:
        second.stop()
        first.stop()


def test_client_rejects_protocol_version_mismatch() -> None:
    client = RemoteInferenceModel("version-mismatch", endpoint="tcp/127.0.0.1:12346")

    with pytest.raises(RemoteInferenceProtocolError, match="protocol version"):
        client._validate_handshake({"protocol_version": 2})  # noqa: SLF001


def test_client_detects_server_restart_during_run() -> None:
    name = "restart-detection"
    endpoint = _free_endpoint()
    first_server = InferenceServer(_Model("pi05"), name, listen=endpoint)
    first_server.start()
    client = RemoteInferenceModel(name, endpoint=endpoint, request_timeout_s=1.0)
    second_server: InferenceServer | None = None
    try:
        client.predict_action_chunk({"state": np.zeros(3, dtype=np.float32)})
        first_id = client.metadata["server_id"]
        first_server.stop()

        second_server = InferenceServer(_Model("pi05"), name, listen=endpoint)
        second_server.start()
        assert second_server._server_id != first_id  # noqa: SLF001
        assert _wait_until(client._all_queries_match, 2.0)  # noqa: SLF001

        with pytest.raises(RemoteInferenceModelMismatchError, match="restarted or was replaced"):
            client.predict_action_chunk({"state": np.zeros(3, dtype=np.float32)})
    finally:
        client.close()
        first_server.stop()
        if second_server is not None:
            second_server.stop()


def test_server_serializes_predict_and_reset_on_its_model_worker() -> None:
    class _ConcurrentModel(_Model):
        def __init__(self) -> None:
            super().__init__("pi05")
            self.predict_started = threading.Event()
            self.release_predict = threading.Event()
            self._state_lock = threading.Lock()
            self._active_calls = 0
            self.max_active_calls = 0
            self.overlapped = False

        def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
            with self._state_lock:
                self._active_calls += 1
                self.max_active_calls = max(self.max_active_calls, self._active_calls)
                self.overlapped |= self._active_calls > 1
            try:
                self.predict_started.set()
                if not self.release_predict.wait(2.0):
                    raise TimeoutError("test did not release predict")
                return {ACTION: np.ones((1, 4, 3), dtype=np.float32)}
            finally:
                with self._state_lock:
                    self._active_calls -= 1

        def reset(self) -> None:
            with self._state_lock:
                self.overlapped |= self._active_calls > 0
                self._active_calls += 1
                self.max_active_calls = max(self.max_active_calls, self._active_calls)
            try:
                self.reset_count += 1
            finally:
                with self._state_lock:
                    self._active_calls -= 1

    name = "serialized-model"
    model = _ConcurrentModel()
    server = InferenceServer(model, name, listen=_free_endpoint())
    server.start()
    predict_client = RemoteInferenceModel(name, endpoint=server.endpoint, request_timeout_s=1.0)
    reset_client = RemoteInferenceModel(name, endpoint=server.endpoint, request_timeout_s=1.0)
    errors: list[BaseException] = []

    def predict() -> None:
        try:
            predict_client.predict_action_chunk({"state": np.zeros(3, dtype=np.float32)})
        except BaseException as error:
            errors.append(error)

    def reset() -> None:
        try:
            reset_client.reset()
        except BaseException as error:
            errors.append(error)

    predict_thread = threading.Thread(target=predict)
    reset_thread = threading.Thread(target=reset)
    try:
        predict_client.connect()
        reset_client.connect()
        predict_thread.start()
        assert model.predict_started.wait(1.0)
        reset_thread.start()
        assert _wait_until(lambda: any(request.command == "reset" for request in server._queue), 1.0)  # noqa: SLF001
        assert model.reset_count == 0
        assert not model.overlapped

        model.release_predict.set()
        predict_thread.join(timeout=2.0)
        reset_thread.join(timeout=2.0)

        assert not predict_thread.is_alive()
        assert not reset_thread.is_alive()
        assert errors == []
        assert model.reset_count == 1
        assert model.max_active_calls == 1
        assert not model.overlapped
    finally:
        model.release_predict.set()
        predict_thread.join(timeout=2.0)
        reset_thread.join(timeout=2.0)
        predict_client.close()
        reset_client.close()
        server.stop()


def test_remote_model_default_request_timeout_is_two_seconds() -> None:
    timeout_parameter = inspect.signature(RemoteInferenceModel.__init__).parameters["request_timeout_s"]

    assert timeout_parameter.default == 2.0


def test_predict_timeout_is_typed_and_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    class _TimeoutReceiver:
        def recv(self) -> None:
            raise RuntimeError("Timeout")

    class _Querier:
        get_count = 0

        def get(self, *, payload: bytes) -> _TimeoutReceiver:
            self.get_count += 1
            return _TimeoutReceiver()

    client = RemoteInferenceModel("no-retry", endpoint="tcp/127.0.0.1:14570", request_timeout_s=0.05)
    querier = _Querier()
    client._session = object()  # noqa: SLF001
    client._server_id = "test-server"  # noqa: SLF001
    client._queriers["predict"] = querier  # noqa: SLF001
    monkeypatch.setattr(client, "connect", lambda: None)

    with pytest.raises(RemoteInferenceTimeoutError):
        client.predict_action_chunk({"state": np.zeros(3, dtype=np.float32)})

    assert querier.get_count == 1


@pytest.mark.parametrize("name", ["", "..", "bad/name", "bad name", "bad*name", "$name"])
def test_server_name_rejects_invalid_key_segments(name: str) -> None:
    with pytest.raises(ValueError):
        model_key_prefix(name)


def test_server_name_follows_policy_pattern_without_an_arbitrary_length_limit() -> None:
    name = "a" + "b" * 128

    assert model_key_prefix(name) == f"physicalai/inference/{name}"


def test_server_name_mismatch_is_not_reported_as_model_identity_mismatch() -> None:
    client = RemoteInferenceModel("cell1-act", endpoint="tcp/127.0.0.1:12345")

    with pytest.raises(RemoteInferenceProtocolError, match="Server name"):
        client._validate_handshake(  # noqa: SLF001
            {"protocol_version": 1, "name": "cell2-act"}
        )


def test_server_name_and_policy_identity_remain_separate() -> None:
    server = InferenceServer(_Model("pi05"), "cell1-act")

    metadata = server._build_metadata()  # noqa: SLF001

    assert metadata["name"] == "cell1-act"
    assert metadata["policy"]["name"] == "pi05"
    assert metadata["policy"]["manifest_sha256"] == ""
    assert metadata["rtc"] == {"supported": True, "chunk_size": None}
    assert metadata["cameras"] == []
    assert metadata["image_codecs"] == ["jpeg", "raw"]
    assert metadata["max_request_bytes"] == 32 * 2**20
    assert server._metadata_key == "physicalai/inference/cell1-act/metadata"  # noqa: SLF001
    assert server._predict_key == "physicalai/inference/cell1-act/predict"  # noqa: SLF001
    assert server._reset_key == "physicalai/inference/cell1-act/reset"  # noqa: SLF001


def test_metadata_reply_schema_and_manifest_hash_are_v1(tmp_path: Any) -> None:
    model = _Model("pi05")
    model.backend = "openvino"
    model.device = "GPU"
    manifest_data = b'{"policy":{"name":"pi05"}}'
    (tmp_path / "manifest.json").write_bytes(manifest_data)
    model.export_dir = tmp_path
    model.manifest.policy = SimpleNamespace(name="pi05")
    model.manifest.hardware = SimpleNamespace(cameras=[SimpleNamespace(name="wrist", shape=[3, 480, 640])])
    model.manifest.model_extra["rtc"] = {"chunk_size": 4}
    server = InferenceServer(model, "cell1-act")

    metadata = server._build_metadata()  # noqa: SLF001

    assert set(metadata) == {
        "protocol_version",
        "server_id",
        "name",
        "policy",
        "chunk_size",
        "rtc",
        "cameras",
        "image_codecs",
        "max_request_bytes",
    }
    assert metadata["protocol_version"] == 1
    assert metadata["server_id"] == ""
    assert metadata["policy"] == {
        "name": "pi05",
        "manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
        "backend": "openvino",
        "device": "GPU",
    }
    assert metadata["rtc"] == {"supported": True, "chunk_size": 4}
    assert metadata["cameras"] == [{"name": "wrist", "shape": [3, 480, 640]}]
    assert metadata["image_codecs"] == ["jpeg", "raw"]


def test_protocol_v1_message_schemas_are_direct_maps() -> None:
    metadata_request = encode_message({"protocol_version": 1})
    reset_request = encode_message({"protocol_version": 1})
    predict_request = encode_predict_request(
        {"state": np.zeros(2, dtype=np.float32)},
        seq=7,
        budget_ms=2000,
        image_codec="raw",
        jpeg_quality=90,
        max_image_side=None,
    )
    predict_reply = encode_predict_reply(
        {"action": np.ones((1, 4, 2), dtype=np.float32)},
        seq=7,
        server_id="server-id",
        server_queue_s=0.01,
        server_compute_s=0.02,
    )

    assert decode_message(metadata_request) == {"protocol_version": 1}
    assert decode_message(reset_request) == {"protocol_version": 1}
    assert set(decode_message(predict_request)) == {
        "protocol_version",
        "seq",
        "budget_ms",
        "inputs",
        "images",
    }
    assert set(decode_message(predict_reply)) == {
        "protocol_version",
        "seq",
        "server_id",
        "outputs",
        "server_queue_s",
        "server_compute_s",
    }


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_predict_request_rejects_non_integer_protocol_versions(version: object) -> None:
    request = encode_message({
        "protocol_version": version,
        "seq": 1,
        "budget_ms": 1000,
        "inputs": {},
        "images": {},
    })

    with pytest.raises(RemoteInferenceProtocolError, match="protocol version"):
        decode_predict_request(request)


def test_predict_reply_preserves_the_full_model_output_map() -> None:
    outputs = {
        "action": np.ones((1, 4, 3), dtype=np.float32),
        "aux": np.asarray([2, 3], dtype=np.int16),
    }
    reply = encode_predict_reply(
        outputs,
        seq=5,
        server_id="test-server",
        server_queue_s=0.01,
        server_compute_s=0.02,
    )
    decoded = decode_predict_reply(reply)

    assert set(decoded["outputs"]) == {"action", "aux"}
    np.testing.assert_array_equal(decoded["outputs"]["action"], outputs["action"])
    np.testing.assert_array_equal(decoded["outputs"]["aux"], outputs["aux"])


@pytest.mark.parametrize("host", ["0.0.0.0", "[::]", "100.64.0.10", "192.168.1.20"])
def test_non_loopback_listeners_require_opt_in(host: str) -> None:
    with pytest.raises(ValueError, match="allow_all_interfaces"):
        InferenceServer(_Model("listener-security"), "listener-security", listen=f"tcp/{host}:5555")


@pytest.mark.parametrize("host", ["0.0.0.0", "[::]", "100.64.0.10", "192.168.1.20", ""])
def test_non_loopback_listeners_are_allowed_with_explicit_opt_in(host: str) -> None:
    server = InferenceServer(
        _Model("listener-opt-in"),
        "listener-opt-in",
        listen=f"tcp/{host}:5555",
        allow_all_interfaces=True,
    )

    assert server.endpoint == f"tcp/{host}:5555"


@pytest.mark.parametrize("host", ["0.0.0.0", "[::]", "100.64.0.10", "192.168.1.20"])
def test_non_loopback_listener_logs_unauthenticated_exposure_warning(
    host: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="physicalai.inference.remote.server"):
        InferenceServer._warn_non_loopback_listener(f"tcp/{host}:5555")  # noqa: SLF001

    assert "inference server reachable at :5555 without Zenoh-level authentication" in caplog.text
    assert "use only on a trusted network or tailnet" in caplog.text


def test_client_constructor_is_lazy_and_exports_its_public_config_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_opened(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("constructor must not open a Zenoh session")

    monkeypatch.setattr(client_module, "open_zenoh_session", fail_if_opened)
    model = RemoteInferenceModel("lazy-construction", endpoint="tcp/127.0.0.1:14567")

    assert not hasattr(model, "manifest")
    assert model.endpoint == "tcp/127.0.0.1:14567"
    assert model.metadata == {}

    from physicalai.config import Config
    from physicalai.inference import InferenceModel
    from physicalai.runtime.action_sources.policy import PolicySource

    recipe = Config.from_instance(model)
    assert recipe.class_path == "physicalai.inference.RemoteInferenceModel"
    restored = recipe.instantiate(expected_type=InferenceModel)
    assert isinstance(restored, InferenceModel)
    assert not hasattr(restored, "manifest")
    source_recipe = Config.from_instance(PolicySource(model=model))
    restored_source = source_recipe.instantiate()
    assert isinstance(restored_source, PolicySource)


def test_custom_zenoh_config_replaces_the_built_in_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_if_built(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("custom config must replace built-in config construction")

    monkeypatch.setattr(client_module, "make_zenoh_config", fail_if_built)
    config_path = tmp_path / "custom.json5"
    with caplog.at_level(logging.WARNING):
        model = RemoteInferenceModel("custom-config", zenoh_config=config_path)
        InferenceServer(_Model("custom-config"), "custom-config", zenoh_config=config_path)

    assert model._builtin_config is None  # noqa: SLF001
    assert caplog.text.count("user config") == 2
    assert "replaces the secure Zenoh defaults" in caplog.text


def test_default_client_config_sets_one_connect_and_no_listen_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    inserted: dict[str, str] = {}

    class _Config:
        def insert_json5(self, key: str, value: str) -> None:
            inserted[key] = value

    def capture_config(mode: str, **kwargs: Any) -> Any:
        captured.update(mode=mode, **kwargs)
        return _Config()

    monkeypatch.setattr(client_module, "make_zenoh_config", capture_config)
    RemoteInferenceModel("client-config", endpoint="tcp/192.0.2.1:45000")

    assert captured == {
        "mode": "client",
        "connect_endpoints": ["tcp/192.0.2.1:45000"],
        "listen_endpoints": [],
        "multicast_enabled": False,
        "gossip_enabled": False,
    }
    assert inserted["transport/link/rx/max_message_size"] == str(32 * 2**20)


def test_default_server_config_uses_peer_mode_and_one_listen_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    inserted: dict[str, str] = {}

    class _Config:
        def insert_json5(self, key: str, value: str) -> None:
            inserted[key] = value

    def capture_config(mode: str, **kwargs: Any) -> Any:
        captured.update(mode=mode, **kwargs)
        return _Config()

    monkeypatch.setattr(server_module, "make_zenoh_config", capture_config)
    server = InferenceServer(_Model("server-config"), "server-config")
    server._build_config()  # noqa: SLF001

    assert captured == {
        "mode": "peer",
        "connect_endpoints": [],
        "listen_endpoints": [server.endpoint],
        "multicast_enabled": False,
        "gossip_enabled": False,
    }
    assert inserted["transport/link/rx/max_message_size"] == str(32 * 2**20)


def test_client_fails_unavailable_when_no_queryable_matches() -> None:
    model = RemoteInferenceModel("unavailable-test", endpoint=_free_endpoint(), request_timeout_s=0.05)

    with pytest.raises(RemoteInferenceError, match="SSH tunnel up / address reachable"):
        model.connect()
    model.close()


def test_string_lists_remain_msgpack_strings_and_other_arrays_are_raw() -> None:
    payload = encode_predict_request(
        {"task": ["pick", "place"], "state": np.asarray([1, 2], dtype=np.int16)},
        seq=17,
        budget_ms=2000,
        image_codec="raw",
        jpeg_quality=90,
        max_image_side=None,
    )
    message = decode_message(payload)
    decoded_metadata, inputs = decode_predict_request(payload)

    assert decoded_metadata["seq"] == 17
    assert message["inputs"]["task"] == ["pick", "place"]
    assert isinstance(message["inputs"]["state"], np.ndarray)
    assert message["images"] == {}
    assert inputs["task"] == ["pick", "place"]
    np.testing.assert_array_equal(inputs["state"], [1, 2])


def test_request_decoder_rejects_oversized_and_non_numeric_arrays() -> None:
    payload = encode_predict_request(
        {"state": np.zeros(1024, dtype=np.float32)},
        seq=1,
        budget_ms=1000,
        image_codec="raw",
        jpeg_quality=90,
        max_image_side=None,
    )
    with pytest.raises(RemoteInferenceProtocolError, match="limit"):
        decode_predict_request(payload, max_bytes=256)

    invalid = encode_message(
        {
            "protocol_version": 1,
            "seq": 1,
            "budget_ms": 1000,
            "inputs": {
                "state": {"__np__": True, "dtype": "|O", "shape": [1], "data": b"x"},
            },
            "images": {},
        },
    )
    with pytest.raises(RemoteInferenceProtocolError, match="numeric or bool"):
        decode_predict_request(invalid)


def test_new_predict_supersedes_the_only_pending_prediction() -> None:
    class _Query:
        def __init__(self) -> None:
            self.error: bytes | None = None

        def reply_err(self, payload: bytes) -> None:
            self.error = payload

    server = InferenceServer(_Model("newest-wins"), "newest-wins")
    old_query = _Query()
    latest_query = _Query()
    now = time.monotonic()
    server._enqueue(_Request(old_query, "predict", 10, 1000, now, 12, {}))  # noqa: SLF001
    server._enqueue(_Request(latest_query, "predict", 11, 1000, now, 13, {}))  # noqa: SLF001

    assert old_query.error is not None
    assert msgpack.unpackb(old_query.error, raw=False)["code"] == "superseded"
    assert len(server._queue) == 1  # noqa: SLF001
    assert server._queue[0].seq == 11  # noqa: SLF001
    server._queue.clear()  # noqa: SLF001
    server.stop()


def test_live_newest_wins_supersedes_pending_request_while_worker_is_busy() -> None:
    class _BlockingModel(_Model):
        def __init__(self) -> None:
            super().__init__("pi05")
            self.first_started = threading.Event()
            self.release_first = threading.Event()

        def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
            self.call_count += 1
            call_number = self.call_count
            if call_number == 1:
                self.first_started.set()
                if not self.release_first.wait(3.0):
                    raise TimeoutError("test did not release running predict")
            return {ACTION: np.full((1, 4, 3), call_number, dtype=np.float32)}

    name = "newest-wins-live"
    model = _BlockingModel()
    server = InferenceServer(model, name, listen=_free_endpoint())
    server.start()
    clients = [RemoteInferenceModel(name, endpoint=server.endpoint, request_timeout_s=3.0) for _ in range(3)]
    results: dict[str, np.ndarray] = {}
    errors: dict[str, BaseException] = {}
    completed = {key: threading.Event() for key in ("running", "pending", "newest")}

    def request(label: str, client: RemoteInferenceModel) -> None:
        try:
            results[label] = client.predict_action_chunk({"state": np.zeros(3, dtype=np.float32)})
        except BaseException as error:
            errors[label] = error
        finally:
            completed[label].set()

    threads = [
        threading.Thread(target=request, args=(label, client))
        for label, client in zip(("running", "pending", "newest"), clients, strict=True)
    ]
    try:
        for client in clients:
            client.connect()

        threads[0].start()
        assert model.first_started.wait(1.0)
        threads[1].start()
        assert _wait_until(lambda: len(server._queue) == 1, 1.0)  # noqa: SLF001
        threads[2].start()

        assert completed["pending"].wait(1.0)
        assert isinstance(errors.get("pending"), RemoteInferenceError)
        assert "superseded" in str(errors["pending"]).lower()
        assert server._counts["superseded"] == 1  # noqa: SLF001
        assert len(server._queue) == 1  # noqa: SLF001

        model.release_first.set()
        assert completed["running"].wait(2.0)
        assert completed["newest"].wait(2.0)
        assert set(errors) == {"pending"}
        assert model.call_count == 2
        np.testing.assert_array_equal(results["running"], np.ones((4, 3), dtype=np.float32))
        np.testing.assert_array_equal(results["newest"], np.full((4, 3), 2, dtype=np.float32))
    finally:
        model.release_first.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=3.0)
        for client in clients:
            client.close()
        server.stop()


def test_worker_expires_a_request_before_calling_the_model() -> None:
    class _Query:
        def __init__(self) -> None:
            self.event = threading.Event()
            self.error: bytes | None = None

        def reply_err(self, payload: bytes) -> None:
            self.error = payload
            self.event.set()

    model = _Model("deadline-test")
    server = InferenceServer(model, "deadline-test")
    server._last_summary = time.monotonic()  # noqa: SLF001
    server._worker = threading.Thread(target=server._worker_loop, daemon=True)  # noqa: SLF001
    server._worker.start()  # noqa: SLF001
    query = _Query()
    server._enqueue(  # noqa: SLF001
        _Request(query, "predict", 15, 1, time.monotonic() - 1, 10, {"state": np.zeros(3)})
    )

    assert query.event.wait(1.0)
    assert query.error is not None
    assert msgpack.unpackb(query.error, raw=False)["code"] == "expired"
    assert model.predict_count == 0
    server.stop()


def test_server_and_client_derive_port_from_model_namespace() -> None:
    model_name = "pi05-port-test"
    key_prefix = f"physicalai/inference/{model_name}"
    server = InferenceServer(_Model(model_name), model_name)
    client = RemoteInferenceModel(model_name)

    assert server.endpoint == endpoint_for_key(key_prefix, "127.0.0.1")
    assert client.endpoint == endpoint_for_key(key_prefix, "127.0.0.1")


def test_reset_roundtrip_runs_on_the_server_worker() -> None:
    name = "reset-roundtrip"
    model = _Model(name)
    server = InferenceServer(model, name, listen=_free_endpoint())
    server.start()
    client = RemoteInferenceModel(name, endpoint=server.endpoint)
    try:
        client.predict_action_chunk({"state": np.zeros(3, dtype=np.float32)})
        client.reset()
        assert model.reset_count == 1
        assert model.reset_thread_id == server._worker.ident  # noqa: SLF001
    finally:
        client.close()
        server.stop()


def test_predict_querier_can_be_reused() -> None:
    name = "repeat-predict"
    server = InferenceServer(_Model(name), name, listen=_free_endpoint())
    server.start()
    client = RemoteInferenceModel(name, endpoint=server.endpoint)
    try:
        for _ in range(3):
            assert client.predict_action_chunk({"state": np.zeros(3, dtype=np.float32)}).shape == (4, 3)
    finally:
        client.close()
        server.stop()


def test_client_uses_rtc_chunk_size_from_handshake_without_manifest() -> None:
    name = "rtc-handshake"
    model = _Model(name)
    model.manifest.model_extra["rtc"] = {"chunk_size": 6}
    server = InferenceServer(model, name, listen=_free_endpoint())
    server.start()
    client = RemoteInferenceModel(name, endpoint=server.endpoint)
    try:
        assert client.chunk_size == 6
        assert not hasattr(client, "manifest")
    finally:
        client.close()
        server.stop()


def test_client_uses_reliable_best_matching_queriers_for_all_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    name = "querier-options"
    server_id = "test-server-id"
    declared: dict[str, dict[str, Any]] = {}
    requested_keys: list[str] = []

    class _Querier:
        matching_status = SimpleNamespace(matching=True)

        def __init__(self, key: str) -> None:
            self.key = key

        def get(self, *, payload: bytes) -> Any:
            requested_keys.append(self.key)
            request = decode_message(payload)
            if self.key.endswith("/metadata"):
                assert request == {"protocol_version": 1}
                response = encode_message(
                    {
                        "protocol_version": 1,
                        "server_id": server_id,
                        "name": name,
                        "policy": {
                            "name": "pi05",
                            "manifest_sha256": "a" * 64,
                            "backend": "openvino",
                            "device": "GPU",
                        },
                        "chunk_size": 4,
                        "rtc": {"supported": False, "chunk_size": None},
                        "cameras": [],
                        "image_codecs": ["jpeg", "raw"],
                        "max_request_bytes": 32 * 2**20,
                    },
                )
            elif self.key.endswith("/predict"):
                seq = request["seq"]
                response = encode_predict_reply(
                    {"action": np.ones((1, 4, 3), dtype=np.float32)},
                    seq=seq,
                    server_id=server_id,
                    server_queue_s=0.0001,
                    server_compute_s=0.001,
                )
            else:
                assert request == {"protocol_version": 1}
                response = encode_message({"protocol_version": 1, "ok": True})
            return SimpleNamespace(recv=lambda: SimpleNamespace(err=None, ok=SimpleNamespace(payload=response)))

        def undeclare(self) -> None:
            pass

    class _Session:
        def declare_querier(self, key: str, **kwargs: Any) -> _Querier:
            declared[key] = kwargs
            return _Querier(key)

        def get(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("requests must use declared queriers")

        def close(self) -> None:
            pass

    monkeypatch.setattr(client_module, "open_zenoh_session", lambda *_args, **_kwargs: _Session())
    client = RemoteInferenceModel(name, endpoint="tcp/127.0.0.1:14568", request_timeout_s=0.5)
    try:
        client.connect()
        actions = client.predict_action_chunk({"state": np.zeros(3, dtype=np.float32)})
        timing = client.last_timing
        assert timing is not None
        assert timing.server_queue_s == pytest.approx(0.0001)
        assert timing.server_compute_s == pytest.approx(0.001)
        assert timing.encode_s >= 0
        assert timing.decode_s >= 0
        assert timing.request_bytes > 0
        assert timing.reply_bytes > 0
        client.reset()
    finally:
        client.close()

    expected_keys = {f"physicalai/inference/{name}/{suffix}" for suffix in ("metadata", "predict", "reset")}
    assert set(declared) == expected_keys
    assert requested_keys == [
        f"physicalai/inference/{name}/metadata",
        f"physicalai/inference/{name}/predict",
        f"physicalai/inference/{name}/reset",
    ]
    for options in declared.values():
        assert options["target"] == zenoh.QueryTarget.BEST_MATCHING
        assert options["timeout"] == 0.5
        assert options["congestion_control"] == zenoh.CongestionControl.BLOCK
        assert options["priority"] == zenoh.Priority.INTERACTIVE_HIGH
    assert actions.shape == (4, 3)


def test_client_predict_guard_rejects_overlapping_in_flight_request() -> None:
    client = RemoteInferenceModel("in-flight-guard", endpoint="tcp/127.0.0.1:14569")
    client._predict_in_flight = True  # noqa: SLF001

    with pytest.raises(RemoteInferenceError, match="already in flight"):
        client._predict({"state": np.zeros(3, dtype=np.float32)})  # noqa: SLF001


def test_client_rejects_action_batch_larger_than_one() -> None:
    class _BatchedModel(_Model):
        def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
            return {ACTION: np.ones((2, 4, 3), dtype=np.float32)}

    name = "batched-actions"
    server = InferenceServer(_BatchedModel(name), name, listen=_free_endpoint())
    server.start()
    client = RemoteInferenceModel(name, endpoint=server.endpoint)
    try:
        with pytest.raises(RemoteInferenceProtocolError, match="batch size 1"):
            client.predict_action_chunk({"state": np.zeros(3, dtype=np.float32)})
    finally:
        client.close()
        server.stop()


@pytest.mark.parametrize("mode", ["sync", "async", "rtc"])
def test_remote_policy_source_runs_fifty_fake_robot_ticks(mode: str) -> None:
    from physicalai.runtime import RobotRuntime

    class _FakeRobot:
        def __init__(self) -> None:
            self.connected = False
            self.sent_actions = 0

        @property
        def joint_names(self) -> list[str]:
            return ["j0", "j1", "j2"]

        @property
        def device_ids(self) -> tuple[str, ...]:
            return (f"fake:{mode}",)

        def connect(self) -> None:
            self.connected = True

        def disconnect(self) -> None:
            self.connected = False

        def is_connected(self) -> bool:
            return self.connected

        def get_observation(self) -> Any:
            state = np.zeros(3, dtype=np.float32)
            return SimpleNamespace(
                joint_positions=state,
                state=state,
                timestamp=time.monotonic(),
                sensor_data=None,
                images=None,
            )

        def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
            self.sent_actions += 1

    name = f"fifty-ticks-{mode}"
    served_model = _Model(name)
    server = InferenceServer(served_model, name, listen=_free_endpoint())
    server.start()
    client = RemoteInferenceModel(name, endpoint=server.endpoint)
    if mode == "sync":
        execution = SyncExecution()
        action_queue = ChunkedActionQueue()
    elif mode == "async":
        execution = AsyncExecution(request_threshold=0.5)
        action_queue = ChunkedActionQueue()
    else:
        execution = RTCExecution(chunk_size=4, execution_horizon=2, fps=30.0, queue_threshold=2)
        action_queue = RTCActionQueue()
    source = PolicySource(model=client, execution=execution, action_queue=action_queue, task="pick red")
    robot = _FakeRobot()
    runtime = RobotRuntime(robot=robot, action_source=source, fps=30.0)
    try:
        with runtime:
            steps = runtime.run(duration_s=2.0)
        assert steps >= 50
        assert robot.sent_actions == steps
        assert served_model.call_count >= 1
        if mode == "rtc":
            assert served_model.last_call_inputs is not None
            assert PREV_CHUNK_LEFT_OVER in served_model.last_call_inputs
            assert RTC_INFERENCE_DELAY in served_model.last_call_inputs
    finally:
        source.disconnect()
        server.stop()


def test_sync_async_and_rtc_use_the_same_remote_model_interface() -> None:
    model_name = "execution-mode-check"
    served_model = _Model(model_name)
    server = InferenceServer(served_model, model_name, listen=_free_endpoint())
    server.start()

    model = RemoteInferenceModel(
        model_name,
        endpoint=server.endpoint,
        request_timeout_s=0.5,
    )
    observation = {"state": np.zeros(3, dtype=np.float32)}
    try:
        sync_queue = ChunkedActionQueue()
        sync = SyncExecution()
        sync.start(model, sync_queue)
        sync.warmup(observation)
        assert sync_queue.remaining == model.chunk_size
        sync_queue.clear()
        sync.stop()

        async_queue = ChunkedActionQueue()
        asynchronous = AsyncExecution(request_threshold=0.75)
        asynchronous.start(model, async_queue)
        asynchronous.warmup(observation)
        for _ in range(3):
            async_queue.pop()
        asynchronous.maybe_request(observation)
        assert _wait_until(lambda: asynchronous.inference_count > 0 and async_queue.remaining > 0, 5.0)
        asynchronous.stop()

        rtc_queue = RTCActionQueue()
        rtc = RTCExecution(chunk_size=None, execution_horizon=2, fps=30.0, queue_threshold=2)
        rtc.start(model, rtc_queue)
        rtc.warmup(observation)
        assert rtc_queue.remaining == 4
        assert served_model.call_count > 0
        assert model.last_timing is not None
        rtc.stop()
    finally:
        model.close()
        server.stop()


def test_observations_use_msgpack_jpeg_images_and_lossless_other_arrays() -> None:
    image = np.empty((1, 128, 160, 3), dtype=np.uint8)
    image[:] = [220, 35, 18]
    state = np.array([[1.25, -2.5]], dtype=np.float32)
    task = ["pick the red cube"]

    payload = encode_predict_request(
        {"images.front": image, "state": state, "task": task},
        seq=1,
        budget_ms=2000,
        image_codec="jpeg",
        jpeg_quality=90,
        max_image_side=None,
    )
    message = decode_message(payload)
    _, decoded = decode_predict_request(payload)
    image_entry = message["images"]["images.front"]

    assert message["protocol_version"] == 1
    assert image_entry["codec"] == "jpeg"
    assert image_entry["dtype"] == "uint8"
    assert image_entry["shape"] == list(image.shape)
    assert image_entry["data"].startswith(b"\xff\xd8")
    assert len(image_entry["data"]) < image.nbytes
    assert decoded["images.front"].shape == image.shape
    assert decoded["images.front"][0, 32, 32, 0] > 200
    assert decoded["images.front"][0, 32, 32, 2] < 40
    np.testing.assert_array_equal(decoded["state"], state)
    assert decoded["task"] == task


def test_batched_rgb_images_roundtrip_shape() -> None:
    image = np.full((1, 72, 96, 3), 127, dtype=np.uint8)
    payload = encode_predict_request(
        {"images": image},
        seq=1,
        budget_ms=1000,
        image_codec="jpeg",
        jpeg_quality=90,
        max_image_side=None,
    )
    _, decoded = decode_predict_request(payload)

    assert decoded["images"].shape == image.shape
    assert abs(int(decoded["images"][0, 20, 20, 0]) - 127) <= 3


def test_raw_images_preserve_exact_bytes_and_batch_shape() -> None:
    image = np.arange(1 * 12 * 14 * 3, dtype=np.uint8).reshape(1, 12, 14, 3)
    payload = encode_predict_request(
        {"images.front": image},
        seq=1,
        budget_ms=1000,
        image_codec="raw",
        jpeg_quality=90,
        max_image_side=None,
    )
    message = decode_message(payload)
    _, decoded = decode_predict_request(payload)

    assert message["images"]["images.front"]["codec"] == "raw"
    assert message["images"]["images.front"]["data"] == image.tobytes()
    np.testing.assert_array_equal(decoded["images.front"], image)


def test_jpeg_natural_image_mean_absolute_error_is_below_three() -> None:
    y, x = np.mgrid[:128, :160]
    image = np.stack(
        (
            (x * 3 // 2) % 256,
            (y * 2) % 256,
            ((x + y) * 3 // 2) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    image[32:96, 48:112] = [220, 35, 18]
    payload = encode_predict_request(
        {"images.front": image},
        seq=1,
        budget_ms=1000,
        image_codec="jpeg",
        jpeg_quality=90,
        max_image_side=None,
    )
    _, decoded = decode_predict_request(payload)

    assert decoded["images.front"].shape == image.shape
    assert decoded["images.front"].dtype == image.dtype
    assert np.abs(decoded["images.front"].astype(np.int16) - image.astype(np.int16)).mean() < 3


def test_oversized_jpeg_shape_is_rejected_before_opencv_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_decoded(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("oversized image must be rejected before OpenCV allocation")

    monkeypatch.setattr(protocol_module.cv2, "imdecode", fail_if_decoded)
    images = {
        "images.front": {
            "codec": "jpeg",
            "shape": [1000, 1000, 3],
            "dtype": "uint8",
            "data": b"not-decoded",
        }
    }

    with pytest.raises(RemoteInferenceProtocolError, match="max_request_bytes"):
        protocol_module.decode_images(images, max_bytes=1024, already_decoded_bytes=0)


def test_max_image_side_resizes_and_preserves_rgb_shape() -> None:
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    payload = encode_predict_request(
        {"images.front": image},
        seq=1,
        budget_ms=1000,
        image_codec="raw",
        jpeg_quality=90,
        max_image_side=32,
    )
    _, inputs = decode_predict_request(payload)

    assert inputs["images.front"].shape == (24, 32, 3)


def _wait_until(predicate: Any, timeout_s: float) -> bool:
    end_time = time.monotonic() + timeout_s
    while time.monotonic() < end_time:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())
