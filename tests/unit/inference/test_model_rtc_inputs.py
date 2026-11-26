# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""One RTC export serves sync, async and RTC execution; RTC refuses an export without RTC inputs."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from typing_extensions import override
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from loguru import logger

from physicalai.inference import InferenceModel
from physicalai.inference.preprocessors import StatsNormalizer
from physicalai.inference.preprocessors.base import Preprocessor
from physicalai.runtime import (
    AsyncExecution,
    PolicySource,
    ChunkedActionQueue,
    RTCActionQueue,
    RTCExecution,
    SyncExecution,
)

RTC_INPUTS = ["prev_chunk_left_over", "inference_delay", "max_guidance_weight", "execution_horizon"]
CHUNK = 4
DIM = 2
# Only the tail: a normalized state comes out float64, which ONNX Runtime rejects on main too. The state is
# not an RTC input, so this change leaves it alone.
QUANTILE_STATS: dict[str, dict[str, list[float]]] = {
    "prev_chunk_left_over": {"q01": [-3.0, 0.5], "q99": [5.0, 1.5]},
}


def _feature(ftype: str, shape: list[int], name: str, dtype: str) -> dict[str, Any]:
    return {
        "class_path": "physicalai.inference.data.features.InferenceFeature",
        "init_args": {"ftype": ftype, "shape": shape, "name": name, "dtype": dtype},
    }


def _make_export_dir(
    tmp_path: Path,
    *,
    rtc: bool = True,
    declare_chunk_shape: bool = True,
    backend: str = "openvino",
    normalize: bool = False,
    denormalize: bool = False,
    rtc_block: bool = False,
) -> Path:
    export_dir = tmp_path / "export"
    export_dir.mkdir(exist_ok=True)
    artifact = "model.xml" if backend == "openvino" else "model.onnx"
    features = [_feature("STATE", [DIM], "state", "float32")]
    if rtc:
        if declare_chunk_shape:
            features.append(_feature("COMMON", [CHUNK, DIM], "prev_chunk_left_over", "float32"))
        features += [
            _feature("COMMON", [], "inference_delay", "int64"),
            _feature("COMMON", [], "max_guidance_weight", "float32"),
            _feature("COMMON", [], "execution_horizon", "int64"),
        ]
    model: dict[str, Any] = {
        "artifacts": {backend: artifact},
        "runner": {"class_path": "physicalai.inference.runners.SinglePass", "init_args": {}},
        "input_features": features,
    }
    if normalize:
        model["preprocessors"] = [{"type": "normalize", "stats": QUANTILE_STATS, "mode": "quantiles"}]
    if denormalize:
        # Manifest stats are float64, so the denormalized action, which RTC feeds back as the tail, is float64.
        action_stats = {"action": QUANTILE_STATS["prev_chunk_left_over"]}
        model["postprocessors"] = [{"type": "denormalize", "stats": action_stats, "mode": "quantiles"}]
    manifest: dict[str, Any] = {
        "format": "policy_package",
        "version": "1.0",
        "policy": {"name": "toy", "source": {"class_path": "toy.Toy"}},
        "model": model,
    }
    if rtc_block:
        manifest["rtc"] = {"chunk_size": CHUNK}
    (export_dir / "manifest.json").write_text(json.dumps(manifest))
    (export_dir / artifact).touch()
    if backend == "openvino":
        (export_dir / "model.bin").touch()
    return export_dir


# --- toy graphs: action = state + max_guidance_weight * prev_chunk_left_over; delay and horizon echoed ---------------


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte, n = n & 0x7F, n >> 7
        out.append(byte | 0x80 if n else byte)
        if not n:
            return bytes(out)


def _field(number: int, payload: bytes | str | int) -> bytes:
    if isinstance(payload, int):
        return _varint(number << 3) + _varint(payload)
    data = payload.encode() if isinstance(payload, str) else payload
    return _varint(number << 3 | 2) + _varint(len(data)) + data


def _value_info(name: str, elem_type: int, shape: list[int]) -> bytes:
    dims = b"".join(_field(1, _field(1, size)) for size in shape)
    return _field(1, name) + _field(2, _field(1, _field(1, elem_type) + _field(2, dims)))


def _toy_onnx_model(float_type: int = 1) -> bytes:
    """Encode the toy RTC graph as ONNX (protobuf wire format; the ``onnx`` package is not a dependency)."""
    float32, int64 = float_type, 7  # ONNX element types: 1 float, 10 float16, 11 double
    nodes = [
        ("Mul", ["prev_chunk_left_over", "max_guidance_weight"], ["guided"]),
        ("Add", ["guided", "state"], ["action"]),
        ("Identity", ["inference_delay"], ["delay_out"]),
        ("Identity", ["execution_horizon"], ["horizon_out"]),
    ]
    graph = b"".join(
        _field(1, b"".join(_field(1, i) for i in ins) + b"".join(_field(2, o) for o in outs) + _field(4, op))
        for op, ins, outs in nodes
    )
    graph += _field(2, "toy_rtc")
    for name, elem_type, shape in [
        ("state", float32, [1, DIM]),
        ("prev_chunk_left_over", float32, [1, CHUNK, DIM]),
        ("inference_delay", int64, []),
        ("max_guidance_weight", float32, []),
        ("execution_horizon", int64, []),
    ]:
        graph += _field(11, _value_info(name, elem_type, shape))
    for name, elem_type, shape in [
        ("action", float32, [1, CHUNK, DIM]),
        ("delay_out", int64, []),
        ("horizon_out", int64, []),
    ]:
        graph += _field(12, _value_info(name, elem_type, shape))
    return _field(1, 8) + _field(7, graph) + _field(8, _field(1, "") + _field(2, 13))


def _write_toy_openvino_model(path: Path, *, int32_input: bool = False) -> None:
    import openvino as ov  # noqa: PLC0415
    import openvino.opset13 as ops  # noqa: PLC0415

    state = ops.parameter([1, DIM], np.float32, name="state")
    prev = ops.parameter([1, CHUNK, DIM], np.float32, name="prev_chunk_left_over")
    delay = ops.parameter([], np.int64, name="inference_delay")
    weight = ops.parameter([], np.float32, name="max_guidance_weight")
    horizon = ops.parameter([], np.int64, name="execution_horizon")
    params = [state, prev, delay, weight, horizon]
    if int32_input:
        params.append(ops.parameter([1], np.int32, name="count"))
    for param in params:
        param.output(0).set_names({param.get_friendly_name()})
    zero = ops.constant(np.int64(0))
    results = [
        ops.result(ops.add(ops.multiply(prev, weight), state)),
        ops.result(ops.add(delay, zero)),  # its own tensor, so the input keeps its name
        ops.result(ops.add(horizon, zero)),
    ]
    names = ["action", "delay_out", "horizon_out"]
    if int32_input:
        results.append(ops.result(ops.add(params[-1], ops.constant(np.int32(0)))))
        names.append("count_out")
    for result, name in zip(results, names, strict=True):
        result.output(0).set_names({name})
    model = ov.Model(results, params, "toy_rtc")  # pyrefly: ignore[no-matching-overload]
    ov.save_model(model, path)


def _write_toy_model(export_dir: Path, backend: str) -> None:
    if backend == "onnx":
        (export_dir / "model.onnx").write_bytes(_toy_onnx_model())
    else:
        _write_toy_openvino_model(export_dir / "model.xml")


def _state(value: float = 0.5) -> dict[str, Any]:
    return {"state": np.full((1, DIM), value, dtype=np.float32)}


def _rtc_sent(weight: float = 5.0) -> dict[str, Any]:
    """What ``RTCExecution`` adds to an observation for a guided chunk: NumPy scalars included."""
    return {
        "prev_chunk_left_over": np.ones((1, CHUNK, DIM), dtype=np.float32),
        "inference_delay": np.int64(1),
        "max_guidance_weight": np.float32(weight),
        "execution_horizon": np.int64(2),
    }


def _wait(condition: Callable[[], bool], timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


class _Record(Preprocessor):
    """Records what it sees and doubles the previous chunk, like a preprocessor that rescales the tail."""

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    @override
    def __call__(self, inputs: dict) -> dict:
        self.seen.append(dict(inputs))
        if "prev_chunk_left_over" not in inputs:
            return inputs
        return {**inputs, "prev_chunk_left_over": inputs["prev_chunk_left_over"] * 2}


class _DropRtc(Preprocessor):
    """Builds a fresh dict without the RTC inputs, as XR0's preprocessor does."""

    @override
    def __call__(self, inputs: dict) -> dict:
        return {k: v for k, v in inputs.items() if k not in RTC_INPUTS}


class _CastState(Preprocessor):
    """Casts the state to a given dtype."""

    def __init__(self, dtype: np.dtype) -> None:
        self.dtype = dtype

    @override
    def __call__(self, inputs: dict) -> dict:
        return {**inputs, "state": inputs["state"].astype(self.dtype)}


class _SupplyRtcIfMissing(Preprocessor):
    """Supplies the RTC inputs only when they are missing."""

    @override
    def __call__(self, inputs: dict) -> dict:
        return inputs if "prev_chunk_left_over" in inputs else {**inputs, **_rtc_sent(weight=2.0)}


class _SupplyRtc(Preprocessor):
    """Supplies all four RTC inputs itself."""

    @override
    def __call__(self, inputs: dict) -> dict:
        return {**inputs, **_rtc_sent(weight=2.0)}


@pytest.fixture
def adapter() -> Iterator[MagicMock]:
    adapter = MagicMock()
    adapter.input_names = ["state", *RTC_INPUTS]
    adapter.output_names = ["action"]
    adapter.default_device.return_value = "cpu"
    adapter.predict.side_effect = lambda inputs: {
        "action": (
            np.repeat(inputs["state"][:, np.newaxis], CHUNK, axis=1)
            + inputs.get("max_guidance_weight", 0) * inputs.get("prev_chunk_left_over", 0)
        ).astype(np.float32)
    }
    with patch("physicalai.inference.model.get_adapter", return_value=adapter):
        yield adapter


def _feeds(adapter: MagicMock, call: int = -1) -> dict[str, Any]:
    return adapter.predict.call_args_list[call].args[0]


def _refill(execution: SyncExecution | AsyncExecution | RTCExecution, queue: Any, threshold: int) -> None:  # noqa: ANN401
    """Play the queue down below ``threshold`` and wait for the mode's refill."""
    before = execution.inference_count
    while not queue.below_threshold(threshold):
        queue.pop()
    execution.maybe_request(_state(0.25))
    _wait(lambda: execution.inference_count > before)


# Test 1
@pytest.mark.parametrize("backend", ["onnx", "openvino"])
@pytest.mark.parametrize("mode", [SyncExecution, AsyncExecution])
def test_sync_and_async_run_an_rtc_export_with_a_refill(tmp_path: Path, backend: str, mode: type) -> None:
    export_dir = _make_export_dir(tmp_path, backend=backend)
    _write_toy_model(export_dir, backend)
    model = InferenceModel(export_dir)
    queue = ChunkedActionQueue()
    execution = mode()
    execution.start(model, queue)
    try:
        execution.warmup(_state(0.5))
        _refill(execution, queue, CHUNK // 2)
    finally:
        execution.stop()

    # Guidance weight 1e-8 on a zero previous chunk leaves the state.
    action = queue.pop()
    assert action is not None
    np.testing.assert_allclose(action, [0.25, 0.25])


# Test 2
def test_feeds_match_rtc_first_chunk_through_every_preprocessor(tmp_path: Path, adapter: MagicMock) -> None:
    sync_record, rtc_record = _Record(), _Record()
    sync_model = InferenceModel(_make_export_dir(tmp_path), preprocessors=[sync_record])
    rtc_model = InferenceModel(_make_export_dir(tmp_path), preprocessors=[rtc_record])

    sync = SyncExecution()
    sync.start(sync_model, ChunkedActionQueue())
    sync.warmup(_state())
    sync_feeds = _feeds(adapter)
    rtc = RTCExecution(chunk_size=CHUNK, execution_horizon=2)
    rtc.start(rtc_model, RTCActionQueue())
    try:
        rtc.warmup(_state())
    finally:
        rtc.stop()
    rtc_feeds = _feeds(adapter)

    for name in RTC_INPUTS:
        seen_sync, seen_rtc = sync_record.seen[0][name], rtc_record.seen[0][name]
        assert type(seen_sync) is type(seen_rtc), name
        assert np.asarray(seen_sync).dtype == np.asarray(seen_rtc).dtype, name
        np.testing.assert_array_equal(seen_sync, seen_rtc, err_msg=name)
        assert np.asarray(sync_feeds[name]).dtype == np.asarray(rtc_feeds[name]).dtype, name
        np.testing.assert_array_equal(sync_feeds[name], rtc_feeds[name], err_msg=name)


# Test 3
def test_normalized_onnx_export_gets_float32(tmp_path: Path) -> None:
    export_dir = _make_export_dir(tmp_path, backend="onnx", normalize=True)
    _write_toy_model(export_dir, "onnx")
    model = InferenceModel(export_dir)

    outputs = model(_state(0.0))

    # The zero tail normalizes to (-0.25, -2.0) and is scaled by weight 1e-8; state 0 normalizes to 0.
    assert outputs["action"].dtype == np.float32
    np.testing.assert_allclose(outputs["action"], 0.0, atol=1e-7)


# Test 4
@pytest.mark.parametrize("drop", [False, True], ids=["plain", "xr0-style-drop"])
def test_rtc_refuses_an_export_without_rtc_inputs(tmp_path: Path, adapter: MagicMock, drop: bool) -> None:
    adapter.input_names = ["state"]
    preprocessors: list[Preprocessor] = [_DropRtc()] if drop else []
    model = InferenceModel(_make_export_dir(tmp_path, rtc=False), preprocessors=preprocessors)
    assert model.rtc_inputs == ()

    with pytest.raises(ValueError, match="takes no RTC inputs"):
        RTCExecution(chunk_size=CHUNK, execution_horizon=2).start(model, RTCActionQueue())

    adapter.predict.assert_not_called()


# Test 5
def test_graph_with_only_inference_delay_behaves_as_before(tmp_path: Path, adapter: MagicMock) -> None:
    adapter.input_names = ["state", "inference_delay"]
    model = InferenceModel(_make_export_dir(tmp_path, rtc=False))
    assert model.rtc_inputs == ("inference_delay",)

    inputs: dict[str, Any] = {**_state(), "inference_delay": np.int64(3)}
    model(inputs)

    assert set(_feeds(adapter)) == {"state", "inference_delay"}
    assert _feeds(adapter)["inference_delay"] == 3  # noqa: PLR2004
    with pytest.raises(KeyError, match="inference_delay"):
        model(_state())


def test_rtc_warns_and_starts_for_a_partial_rtc_signature(tmp_path: Path, adapter: MagicMock) -> None:
    # A graph with fixed delay and horizon can still use the tail and the weight.
    adapter.input_names = ["state", "prev_chunk_left_over", "max_guidance_weight"]
    model = InferenceModel(_make_export_dir(tmp_path))
    assert model.rtc_inputs == ("prev_chunk_left_over", "max_guidance_weight")
    rtc = RTCExecution(chunk_size=CHUNK, execution_horizon=2, max_guidance_weight=5.0)
    queue = RTCActionQueue()
    with patch("physicalai.runtime.execution.rtc.logger") as rtc_logger:
        rtc.start(model, queue)
    try:
        rtc.warmup(_state())
        _refill(rtc, queue, rtc.queue_threshold)
    finally:
        rtc.stop()

    assert any("only some RTC inputs" in str(call) for call in rtc_logger.warning.call_args_list)
    assert _feeds(adapter)["max_guidance_weight"] == np.float32(5.0)
    assert set(_feeds(adapter)) == {"state", "prev_chunk_left_over", "max_guidance_weight"}


# Test 6
def test_manifest_rtc_block_does_not_bypass_the_refusal(tmp_path: Path, adapter: MagicMock) -> None:
    adapter.input_names = ["state"]
    model = InferenceModel(_make_export_dir(tmp_path, rtc=False, rtc_block=True))

    with pytest.raises(ValueError, match="takes no RTC inputs"):
        RTCExecution(chunk_size=CHUNK, execution_horizon=2).start(model, RTCActionQueue())


# Test 7
@pytest.mark.parametrize("declare_chunk_shape", [False, True], ids=["no-shape", "shape"])
def test_preprocessor_that_always_supplies_rtc_inputs_wins(
    tmp_path: Path, adapter: MagicMock, declare_chunk_shape: bool
) -> None:
    model = InferenceModel(
        _make_export_dir(tmp_path, declare_chunk_shape=declare_chunk_shape), preprocessors=[_SupplyRtc()]
    )
    sync = SyncExecution()
    sync.start(model, ChunkedActionQueue())

    sync.warmup(_state())

    assert _feeds(adapter)["max_guidance_weight"] == np.float32(2.0)


def test_preprocessor_that_supplies_only_missing_rtc_inputs_sees_the_fill(tmp_path: Path, adapter: MagicMock) -> None:
    # The documented contract: the fill runs before the preprocessors, so they find the four keys present.
    model = InferenceModel(_make_export_dir(tmp_path), preprocessors=[_SupplyRtcIfMissing()])

    model(_state())

    assert _feeds(adapter)["max_guidance_weight"] == np.float32(1e-8)
    assert not _feeds(adapter)["prev_chunk_left_over"].any()


def test_fill_opt_out_keeps_the_hook_contract(tmp_path: Path, adapter: MagicMock) -> None:
    model = InferenceModel(_make_export_dir(tmp_path), preprocessors=[_SupplyRtcIfMissing()], fill_rtc_inputs=False)

    outputs = model(_state(0.0))

    assert _feeds(adapter)["max_guidance_weight"] == np.float32(2.0)
    # The hook's guidance reaches the output: action = state + 2 * ones.
    np.testing.assert_allclose(outputs["action"], 2.0)


def test_fill_needs_a_positive_declared_shape(tmp_path: Path, adapter: MagicMock) -> None:
    export_dir = _make_export_dir(tmp_path)
    manifest = json.loads((export_dir / "manifest.json").read_text())
    for feature in manifest["model"]["input_features"]:
        if feature["init_args"]["name"] == "prev_chunk_left_over":
            feature["init_args"]["shape"] = [-1, DIM]
    (export_dir / "manifest.json").write_text(json.dumps(manifest))
    model = InferenceModel(export_dir, preprocessors=[_SupplyRtcIfMissing()])

    model(_state())

    # Nothing was filled in, so the hook supplied its own values, as on main.
    assert _feeds(adapter)["max_guidance_weight"] == np.float32(2.0)


# Test 8
def test_guided_rtc_refill_keeps_its_inputs(tmp_path: Path, adapter: MagicMock) -> None:
    model = InferenceModel(_make_export_dir(tmp_path, normalize=True))
    queue = RTCActionQueue()
    rtc = RTCExecution(chunk_size=CHUNK, execution_horizon=2, max_guidance_weight=5.0)
    rtc.start(model, queue)
    try:
        rtc.warmup(_state())
        _refill(rtc, queue, rtc.queue_threshold)
    finally:
        rtc.stop()

    first, refill = _feeds(adapter, 0), _feeds(adapter, -1)
    assert first["max_guidance_weight"] == np.float32(1e-8)
    assert refill["max_guidance_weight"] == np.float32(5.0)
    assert refill["execution_horizon"] == 2  # noqa: PLR2004
    assert not np.array_equal(refill["prev_chunk_left_over"], first["prev_chunk_left_over"])


# Test 9
def test_rtc_warns_and_starts_when_capability_is_unknown(tmp_path: Path, adapter: MagicMock) -> None:
    adapter.input_names = []
    model = InferenceModel(_make_export_dir(tmp_path))
    assert model.rtc_inputs is None
    rtc = RTCExecution(chunk_size=CHUNK, execution_horizon=2)
    with patch("physicalai.runtime.execution.rtc.logger") as rtc_logger:
        rtc.start(model, RTCActionQueue())
    try:
        rtc.warmup(_state())
    finally:
        rtc.stop()

    assert any("Cannot tell" in str(call) for call in rtc_logger.warning.call_args_list)
    assert adapter.predict.call_count == 1


def test_remote_model_capability_is_unknown_without_connecting() -> None:
    pytest.importorskip("zenoh")
    pytest.importorskip("msgpack")
    from physicalai.inference.remote import RemoteInferenceModel  # noqa: PLC0415

    client = RemoteInferenceModel("cell1-toy", port=12345)
    with patch.object(RemoteInferenceModel, "connect") as connect:
        assert client.rtc_inputs is None
    connect.assert_not_called()


# Dtypes at the graph boundary: float64 from manifest stats meets a float32 graph input.
@pytest.mark.parametrize("backend", ["onnx", "openvino"])
def test_normalized_and_denormalized_export_runs_sync_and_a_guided_rtc_refill(tmp_path: Path, backend: str) -> None:
    export_dir = _make_export_dir(tmp_path, backend=backend, normalize=True, denormalize=True)
    _write_toy_model(export_dir, backend)
    model = InferenceModel(export_dir)

    sync_queue = ChunkedActionQueue()
    sync = SyncExecution()
    sync.start(model, sync_queue)
    sync.warmup(_state())
    _refill(sync, sync_queue, CHUNK // 2)
    sync.stop()

    with patch.object(model.adapter, "predict", wraps=model.adapter.predict) as predict:
        rtc_queue = RTCActionQueue()
        rtc = RTCExecution(chunk_size=CHUNK, execution_horizon=2, max_guidance_weight=5.0)
        rtc.start(model, rtc_queue)
        try:
            rtc.warmup(_state())
            while not rtc_queue.below_threshold(rtc.queue_threshold):
                rtc_queue.pop()
            left_over = rtc_queue.get_left_over()
            rtc.maybe_request(_state(0.25))
            _wait(lambda: rtc_queue.remaining == CHUNK)  # the guided chunk is merged
            merged = np.stack([rtc_queue.pop() for _ in range(CHUNK)])  # type: ignore[arg-type]
        finally:
            rtc.stop()

    assert left_over is not None
    feeds = predict.call_args_list[-1].args[0]
    # The tail RTC feeds back is the denormalized (float64) action; the graph gets it normalized, as float32.
    padded = np.zeros((1, CHUNK, DIM))
    padded[0, : len(left_over)] = left_over
    expected_tail = model.preprocessors[0]({"prev_chunk_left_over": padded})["prev_chunk_left_over"]
    assert feeds["prev_chunk_left_over"].dtype == np.float32
    np.testing.assert_array_equal(feeds["prev_chunk_left_over"], expected_tail.astype(np.float32))
    assert feeds["max_guidance_weight"] == np.float32(5.0)
    # The merged actions are the guided graph output, denormalized.
    raw = feeds["state"][:, np.newaxis] + feeds["max_guidance_weight"] * feeds["prev_chunk_left_over"]
    expected = model.postprocessors[0]({"action": raw})["action"][0]
    np.testing.assert_allclose(merged, expected, rtol=1e-6)


def test_bf16_input_is_left_to_openvino(tmp_path: Path) -> None:
    import openvino as ov  # noqa: PLC0415
    import openvino.opset13 as ops  # noqa: PLC0415

    x = ops.parameter([2], ov.Type.bf16, name="x")
    x.output(0).set_names({"x"})
    out = ops.result(ops.convert(x, np.float32))
    out.output(0).set_names({"action"})
    export_dir = _make_export_dir(tmp_path, rtc=False)
    ov.save_model(ov.Model([out], [x], "bf16"), export_dir / "model.xml")  # pyrefly: ignore[no-matching-overload]
    model = InferenceModel(export_dir)
    assert "x" not in model.adapter.input_dtypes

    outputs = model({"x": np.array([0x3F80, 0x4000], dtype=np.uint16)})  # bf16 bit patterns of 1 and 2

    np.testing.assert_array_equal(outputs["action"], [1.0, 2.0])


def test_partial_signature_warning_names_the_inputs(
    tmp_path: Path, adapter: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    adapter.input_names = ["state", "prev_chunk_left_over", "max_guidance_weight"]
    model = InferenceModel(_make_export_dir(tmp_path))
    rtc = RTCExecution(chunk_size=CHUNK, execution_horizon=2)
    with caplog.at_level(logging.WARNING, logger="physicalai.runtime.execution.rtc"):
        rtc.start(model, RTCActionQueue())
    rtc.stop()

    assert "only some RTC inputs (prev_chunk_left_over, max_guidance_weight)" in caplog.text


def test_policy_source_closes_a_model_rtc_refuses(tmp_path: Path, adapter: MagicMock) -> None:
    adapter.input_names = ["state"]
    model = InferenceModel(_make_export_dir(tmp_path, rtc=False))
    source = PolicySource(model=model, execution=RTCExecution(chunk_size=CHUNK, execution_horizon=2))

    with pytest.raises(ValueError, match="takes no RTC inputs"):
        source.connect(bus=MagicMock(), session_id="probe")
    source.disconnect()

    adapter.close.assert_called_once()


def test_policy_source_keeps_the_model_open_after_a_late_start_failure(tmp_path: Path, adapter: MagicMock) -> None:
    class _StartsThenFails(SyncExecution):
        """Stands in for an execution that has a worker inside the model when start() fails."""

        @override
        def start(self, model: InferenceModel, action_queue: Any) -> None:  # noqa: ANN401
            super().start(model, action_queue)
            msg = "late failure"
            raise ValueError(msg)

    model = InferenceModel(_make_export_dir(tmp_path))
    source = PolicySource(model=model, execution=_StartsThenFails())

    with pytest.raises(ValueError, match="late failure"):
        source.connect(bus=MagicMock(), session_id="probe")

    adapter.close.assert_not_called()


def test_policy_source_closes_a_model_once_after_a_normal_run(tmp_path: Path, adapter: MagicMock) -> None:
    model = InferenceModel(_make_export_dir(tmp_path))
    source = PolicySource(model=model, execution=SyncExecution())

    source.connect(bus=MagicMock(), session_id="probe")
    source.disconnect()

    adapter.close.assert_called_once()


@pytest.mark.parametrize(
    ("graph_shapes", "manifest_shape", "reason"),
    [
        ({"prev_chunk_left_over": (1, CHUNK, DIM + 1)}, [CHUNK, DIM], "the graph declares prev_chunk_left_over"),
        ({"max_guidance_weight": (1,)}, [CHUNK, DIM], "the graph declares max_guidance_weight"),
        ({}, [4096, 4096], "larger than"),
        ({}, [2**32, 2**32], "larger than"),  # wraps to 0 in fixed-width int64 arithmetic
        ({}, [-1, DIM], "no fixed (chunk_size, action_dim) shape"),
    ],
    ids=["tail-mismatch", "scalar-mismatch", "too-large", "int64-overflow", "not-positive"],
)
def test_fill_is_skipped_when_it_would_not_fit_the_graph(
    tmp_path: Path, adapter: MagicMock, graph_shapes: dict, manifest_shape: list[int], reason: str
) -> None:
    adapter.input_shapes = graph_shapes
    export_dir = _make_export_dir(tmp_path)
    manifest = json.loads((export_dir / "manifest.json").read_text())
    for feature in manifest["model"]["input_features"]:
        if feature["init_args"]["name"] == "prev_chunk_left_over":
            feature["init_args"]["shape"] = manifest_shape
    (export_dir / "manifest.json").write_text(json.dumps(manifest))
    messages: list[str] = []
    handler_id = logger.add(lambda message: messages.append(message.record["message"]), level="WARNING")
    try:
        model = InferenceModel(export_dir)
    finally:
        logger.remove(handler_id)

    assert any(reason in message for message in messages)
    with pytest.raises(KeyError, match="prev_chunk_left_over"):  # nothing filled in: main's error
        model(_state())


def test_fill_accepts_dynamic_graph_dimensions(tmp_path: Path, adapter: MagicMock) -> None:
    adapter.input_shapes = {"prev_chunk_left_over": (None, None, DIM), "max_guidance_weight": ()}
    model = InferenceModel(_make_export_dir(tmp_path))

    model(_state())

    assert _feeds(adapter)["prev_chunk_left_over"].shape == (1, CHUNK, DIM)


@pytest.mark.parametrize(("onnx_type", "dtype"), [(10, np.float16), (11, np.float64)], ids=["f16", "f64"])
def test_fill_and_rtc_follow_the_graph_float_type(tmp_path: Path, onnx_type: int, dtype: type) -> None:
    export_dir = _make_export_dir(tmp_path, backend="onnx")
    (export_dir / "model.onnx").write_bytes(_toy_onnx_model(onnx_type))
    # The pipeline brings the state (not an RTC input) to the graph's type; the RTC inputs are this change's job.
    normalizer = StatsNormalizer(stats=QUANTILE_STATS, mode="quantiles")
    model = InferenceModel(export_dir, preprocessors=[normalizer, _CastState(np.dtype(dtype))])

    with patch.object(model.adapter, "predict", wraps=model.adapter.predict) as predict:
        sync_queue = ChunkedActionQueue()
        sync = SyncExecution()
        sync.start(model, sync_queue)
        sync.warmup(_state())
        _refill(sync, sync_queue, CHUNK // 2)
        rtc_queue = RTCActionQueue()
        rtc = RTCExecution(chunk_size=CHUNK, execution_horizon=2, max_guidance_weight=5.0)
        rtc.start(model, rtc_queue)
        try:
            rtc.warmup(_state())
            _refill(rtc, rtc_queue, rtc.queue_threshold)
        finally:
            rtc.stop()

    for call in predict.call_args_list:
        feeds = call.args[0]
        for name in ["state", "prev_chunk_left_over", "max_guidance_weight"]:
            assert feeds[name].dtype == dtype, name
    assert predict.call_args_list[-1].args[0]["max_guidance_weight"] == dtype(5.0)


class TestDirectCalls:
    """Direct ``InferenceModel`` calls around the first-chunk fill."""

    def test_rtc_inputs_is_a_graph_fact(self, tmp_path: Path, adapter: MagicMock) -> None:
        assert InferenceModel(_make_export_dir(tmp_path)).rtc_inputs == tuple(RTC_INPUTS)
        adapter.input_names = ["state", "prev_chunk_left_over", "inference_delay"]
        assert InferenceModel(_make_export_dir(tmp_path)).rtc_inputs == ("prev_chunk_left_over", "inference_delay")
        adapter.input_names = []
        assert InferenceModel(_make_export_dir(tmp_path)).rtc_inputs is None

    def test_fill_is_rtc_first_chunk_and_logged_once(self, tmp_path: Path, adapter: MagicMock) -> None:
        adapter.input_dtypes = {"inference_delay": np.dtype(np.int64)}  # a typed adapter, like OpenVINO and ORT
        model = InferenceModel(_make_export_dir(tmp_path))
        messages: list[str] = []
        handler_id = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
        try:
            for _ in range(3):
                model(_state())
        finally:
            logger.remove(handler_id)

        feeds = _feeds(adapter)
        assert feeds["prev_chunk_left_over"].shape == (1, CHUNK, DIM)
        assert not feeds["prev_chunk_left_over"].any()
        for name, value, dtype in [
            ("inference_delay", 0, np.int64),
            ("max_guidance_weight", 1e-8, np.float32),
            ("execution_horizon", 1, np.int64),
        ]:
            assert isinstance(feeds[name], np.ndarray), name  # rank 0, which ONNX Runtime accepts
            assert feeds[name].dtype == dtype, name
            assert feeds[name] == dtype(value), name
        assert sum("takes RTC inputs and the observation has none" in m for m in messages) == 1

    def test_caller_rtc_inputs_are_kept(self, tmp_path: Path, adapter: MagicMock) -> None:
        model = InferenceModel(_make_export_dir(tmp_path))
        inputs = {**_state(), **_rtc_sent()}

        model(inputs)

        assert _feeds(adapter)["prev_chunk_left_over"] is inputs["prev_chunk_left_over"]
        assert _feeds(adapter)["max_guidance_weight"] == np.float32(5.0)

    def test_adapter_without_declared_types_gets_inputs_as_before(self, tmp_path: Path, adapter: MagicMock) -> None:
        inputs = {**_state(), **_rtc_sent()}

        InferenceModel(_make_export_dir(tmp_path))(inputs)

        assert isinstance(_feeds(adapter)["inference_delay"], np.int64)  # a NumPy scalar, untouched
        assert _feeds(adapter)["max_guidance_weight"] is inputs["max_guidance_weight"]

    def test_only_rtc_inputs_are_converted(self, tmp_path: Path, adapter: MagicMock) -> None:
        adapter.input_names = ["state", "count", *RTC_INPUTS]
        adapter.input_dtypes = {
            "state": np.dtype(np.float32),
            "count": np.dtype(np.int32),
            "prev_chunk_left_over": np.dtype(np.float32),
            "inference_delay": np.dtype(np.int32),
        }
        model = InferenceModel(_make_export_dir(tmp_path))
        state, count = np.full((1, DIM), 0.5), np.array([3], dtype=np.int64)

        model({"state": state, "count": count, **_rtc_sent(), "prev_chunk_left_over": np.ones((1, CHUNK, DIM))})

        feeds = _feeds(adapter)
        assert feeds["prev_chunk_left_over"].dtype == np.float32
        assert feeds["inference_delay"].dtype == np.int32
        assert feeds["state"] is state  # not an RTC input: exactly as on main
        assert feeds["count"] is count

    @pytest.mark.parametrize(
        ("name", "value", "dtype"),
        [
            ("inference_delay", np.array(2**40, dtype=np.int64), np.int32),
            ("max_guidance_weight", np.array(1e300), np.float32),
        ],
        ids=["int64-overflow", "float64-overflow"],
    )
    def test_rtc_values_that_do_not_fit_raise(
        self, tmp_path: Path, adapter: MagicMock, name: str, value: np.ndarray, dtype: type
    ) -> None:
        adapter.input_dtypes = {name: np.dtype(dtype)}
        model = InferenceModel(_make_export_dir(tmp_path))

        with pytest.raises(ValueError, match="outside the range"):
            model({**_state(), **_rtc_sent(), name: value})
        adapter.predict.assert_not_called()

    @pytest.mark.parametrize("fill", [True, False], ids=["fill", "no-fill"])
    def test_int32_non_rtc_input_behaves_as_on_main(self, tmp_path: Path, fill: bool) -> None:
        export_dir = _make_export_dir(tmp_path)
        _write_toy_openvino_model(export_dir / "model.xml", int32_input=True)
        model = InferenceModel(export_dir, fill_rtc_inputs=fill)
        count = np.array([7], dtype=np.int64)
        inputs = {**_state(), "count": count} if fill else {**_state(), "count": count, **_rtc_sent()}

        with patch.object(model.adapter, "predict", wraps=model.adapter.predict) as predict:
            outputs = model(inputs)

        assert predict.call_args.args[0]["count"] is count  # handed to OpenVINO unchanged, as on main
        np.testing.assert_array_equal(outputs["count_out"], [7])

    def test_onnx_still_rejects_an_out_of_range_uint8_feed(self, tmp_path: Path) -> None:
        export_dir = _make_export_dir(tmp_path, rtc=False, backend="onnx")
        uint8 = 2
        graph = _field(1, _field(1, "x") + _field(2, "action") + _field(4, "Identity")) + _field(2, "u8")
        graph += _field(11, _value_info("x", uint8, [2])) + _field(12, _value_info("action", uint8, [2]))
        onnx_model = _field(1, 8) + _field(7, graph) + _field(8, _field(1, "") + _field(2, 13))
        (export_dir / "model.onnx").write_bytes(onnx_model)
        model = InferenceModel(export_dir)

        with pytest.raises(Exception, match="(?i)type"):  # noqa: PT011  # ONNX Runtime's own error, as on main
            model({"x": np.array([300.5, 1.0])})

    @pytest.mark.parametrize("backend", ["onnx", "openvino"])
    def test_rtc_execution_scalars_reach_a_real_session(self, tmp_path: Path, backend: str) -> None:
        export_dir = _make_export_dir(tmp_path, backend=backend)
        _write_toy_model(export_dir, backend)
        model = InferenceModel(export_dir)

        outputs = model({**_state(0.0), **_rtc_sent(weight=5.0)})

        np.testing.assert_allclose(outputs["action"], 5.0)
        assert model.adapter.input_dtypes["state"] == np.float32
        assert model.adapter.input_dtypes["inference_delay"] == np.int64
        assert outputs["delay_out"] == 1
        assert outputs["horizon_out"] == 2  # noqa: PLR2004
