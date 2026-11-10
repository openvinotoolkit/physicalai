# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Observation history at PolicySource's received-update boundary."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from physicalai.capture import Frame
from physicalai.inference.adapters.base import RuntimeAdapter
from physicalai.inference.constants import ACTION, IMAGES, STATE, TASK
from physicalai.inference.manifest import Manifest, ModelSpec, ObservationHistorySpec
from physicalai.inference.model import InferenceModel
from physicalai.inference.preprocessors.base import Preprocessor
from physicalai.inference.runners import SinglePass
from physicalai.runtime import AsyncExecution, ChunkedActionQueue, PolicySource, SyncExecution
from physicalai.runtime._callback_bus import _CallbackBus
from tests.unit.runtime.conftest import FakeRobotObservation


@dataclass
class RecordingModel:
    manifest: Manifest
    chunk_size: int = 1
    calls: list[dict[str, Any]] = field(default_factory=list)
    resets: int = 0

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        # Retain exact inputs, not copies, so later aliasing remains observable.
        self.calls.append(observation)
        return np.full((self.chunk_size, 2), len(self.calls), dtype=np.float32)

    def reset(self) -> None:
        self.resets += 1


@pytest.fixture
def make_source() -> Iterator[Callable[..., tuple[PolicySource, RecordingModel]]]:
    sources: list[PolicySource] = []

    def create(
        n_obs_steps: int | None = 3, *, stride: int = 1, chunk_size: int = 1
    ) -> tuple[PolicySource, RecordingModel]:
        history = None if n_obs_steps is None else ObservationHistorySpec(n_obs_steps=n_obs_steps, stride=stride)
        model = RecordingModel(Manifest(model=ModelSpec(n_obs_steps=7, observation_history=history)), chunk_size)
        source = PolicySource(
            cast("InferenceModel", model),
            execution=SyncExecution(request_threshold=0.0),
            action_queue=ChunkedActionQueue(),
        )
        source.connect(bus=_CallbackBus([]), session_id="history-test")
        sources.append(source)
        return source, model

    yield create

    for source in sources:
        source.disconnect()


def _frame(value: int) -> Frame:
    return Frame(np.full((2, 4, 3), value, dtype=np.uint8), timestamp=0.0, sequence=0)


def _observation(value: int = 0, *, images: dict[str, Frame] | None = None) -> FakeRobotObservation:
    return FakeRobotObservation(np.array([value, value + 1], dtype=np.float32), images=images)


def _assert_window(inputs: dict[str, Any], expected: list[int], key: str = IMAGES) -> None:
    image = inputs[key]
    assert image.shape == (1, len(expected), 2, 4, 3)
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image, np.broadcast_to(np.array(expected)[None, :, None, None, None], image.shape))


def _assert_state_window(inputs: dict[str, Any], expected: list[int]) -> None:
    state = inputs[STATE]
    assert state.shape == (1, len(expected), 2)
    assert state.dtype == np.float32
    np.testing.assert_array_equal(state, [[[value, value + 1] for value in expected]])


class TemporalChannelsFirst(Preprocessor):
    """Test-only preprocessor preserving batch and time axes."""

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        images = inputs[IMAGES]
        assert images.shape == (1, 3, 2, 4, 3)
        return {**inputs, IMAGES: images.transpose(0, 1, 4, 2, 3)}


def test_manifest_loaded_history_reaches_adapter_through_single_pass(tmp_path: Path) -> None:
    manifest = {
        "policy": {"name": "history"},
        "model": {
            "artifacts": {"onnx": "history.onnx"},
            "n_obs_steps": 7,
            "observation_history": {"n_obs_steps": 3, "stride": 2},
            "runner": {"type": "single_pass"},
            "preprocessors": [{"class_path": f"{__name__}.TemporalChannelsFirst"}],
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    artifact = tmp_path / "history.onnx"
    artifact.touch()
    adapter = MagicMock(spec=RuntimeAdapter)
    adapter.input_names = [IMAGES, STATE]
    pixels = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    windows = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 3], [0, 2, 4], [1, 3, 5]]

    def predict(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        value = adapter.predict.call_count - 1
        expected = np.stack([pixels + index * 24 for index in windows[value]])
        assert set(inputs) == {IMAGES, STATE}
        assert inputs[IMAGES].shape == (1, 3, 3, 2, 4)
        assert inputs[IMAGES].dtype == np.uint8
        np.testing.assert_array_equal(inputs[IMAGES], expected.transpose(0, 3, 1, 2)[None])
        _assert_state_window(inputs, windows[value])
        return {ACTION: np.full((1, 1, 2), value, dtype=np.float32)}

    adapter.predict.side_effect = predict
    with (
        patch("physicalai.inference.model.get_adapter", return_value=adapter) as get_adapter,
        InferenceModel(tmp_path, device="cpu") as model,
    ):
        get_adapter.assert_called_once_with("onnx", device="cpu")
        adapter.load.assert_called_once_with(artifact)
        assert model.manifest.model.n_obs_steps == 7
        assert model.manifest.model.observation_history == ObservationHistorySpec(n_obs_steps=3, stride=2)
        assert isinstance(model.runner, SinglePass)
        assert len(model.preprocessors) == 1
        assert isinstance(model.preprocessors[0], TemporalChannelsFirst)
        source = PolicySource(model, execution=SyncExecution(request_threshold=0.0), action_queue=ChunkedActionQueue())
        try:
            source.connect(bus=_CallbackBus([]), session_id="manifest-history")
            for value in range(len(windows)):
                frame = Frame(pixels + value * 24, timestamp=0.0, sequence=value)
                action = source.update(_observation(value), {"top": frame}, step=value)
                np.testing.assert_array_equal(action, [value, value])
            assert adapter.predict.call_count == len(windows)
        finally:
            source.disconnect()
    adapter.close.assert_called_once()


@pytest.mark.parametrize("stride", [1, 2])
def test_async_worker_snapshot_is_detached_while_delivered_history_advances(stride: int) -> None:
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    fresh_completed = threading.Event()

    class BlockingModel(RecordingModel):
        def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
            actions = super().predict_action_chunk(observation)
            if len(self.calls) == 2:
                entered.set()
                assert release.wait(timeout=5.0)
            return actions

    model = BlockingModel(
        Manifest(model=ModelSpec(observation_history=ObservationHistorySpec(n_obs_steps=3, stride=stride)))
    )
    execution = AsyncExecution(request_threshold=1.0)
    source = PolicySource(cast("InferenceModel", model), execution=execution, action_queue=ChunkedActionQueue())
    frame = _frame(0)
    try:
        source.connect(bus=_CallbackBus([]), session_id="async-history")
        wait = execution._obs_ready.wait  # noqa: SLF001

        def wait_until_requested(timeout: float | None = None) -> bool:
            # The next idle wait happens after the real worker has queued its result.
            if len(model.calls) == 2:
                completed.set()
            elif len(model.calls) == 3:
                fresh_completed.set()
            return wait(timeout)

        with patch.object(execution._obs_ready, "wait", side_effect=wait_until_requested):  # noqa: SLF001
            np.testing.assert_array_equal(source.update(_observation(), {"top": frame}, step=0), [1, 1])
            frame.data.fill(1)
            source.update(_observation(1), {"top": frame}, step=1)
            assert entered.wait(timeout=5.0)
            blocked = model.calls[1]
            _assert_window(blocked, [0, 0, 1])
            _assert_state_window(blocked, [0, 0, 1])
            assert not np.shares_memory(blocked[IMAGES], frame.data)

            for value in range(2, 5):
                frame.data.fill(value)
                action = source.update(_observation(value), {"top": frame}, step=value)
                np.testing.assert_array_equal(action, [1, 1])
                assert len(model.calls) == 2
                assert source.action_queue.remaining == 0
                _assert_window(blocked, [0, 0, 1])
                _assert_state_window(blocked, [0, 0, 1])
            release.set()
            assert completed.wait(timeout=5.0)
            assert execution.inference_count == 1
            assert source.action_queue.remaining == 1
            frame.data.fill(5)
            action = source.update(_observation(5), {"top": frame}, step=5)
            np.testing.assert_array_equal(action, [2, 2])
            assert len(model.calls) == 2
            frame.data.fill(6)
            source.update(_observation(6), {"top": frame}, step=6)
            assert fresh_completed.wait(timeout=5.0)
            _assert_window(model.calls[2], [4, 5, 6] if stride == 1 else [2, 4, 6])
            _assert_state_window(model.calls[2], [4, 5, 6] if stride == 1 else [2, 4, 6])
            _assert_window(blocked, [0, 0, 1])
            _assert_state_window(blocked, [0, 0, 1])
            assert not np.shares_memory(blocked[STATE], model.calls[2][STATE])

            source.reset(reset_model=False)
            assert source.action_queue.remaining == 0
            frame.data.fill(9)
            action = source.update(_observation(9), {"top": frame}, step=0)
            np.testing.assert_array_equal(action, [4, 4])
            assert len(model.calls) == 4
            _assert_window(model.calls[-1], [9, 9, 9])
            _assert_state_window(model.calls[-1], [9, 9, 9])
    finally:
        release.set()
        source.disconnect()
    assert not execution.alive


@pytest.mark.parametrize("with_camera", [False, True], ids=["state-only", "with-camera"])
@pytest.mark.parametrize("n_obs_steps", [None, 1])
def test_image_and_state_time_axes_require_explicit_opt_in(
    make_source: Callable, n_obs_steps: int | None, with_camera: bool
) -> None:
    source, model = make_source(n_obs_steps)
    frame = _frame(11)
    source.set_task("current task")
    frames = {"top": frame} if with_camera else {}
    preview = source.to_model_input(_observation(), frames)
    source.update(_observation(), frames, step=0)

    assert model.manifest.model.n_obs_steps == 7
    for inputs in [preview, model.calls[0]]:
        if n_obs_steps is None:
            if with_camera:
                assert inputs[IMAGES].shape == (1, 2, 4, 3)
                np.testing.assert_array_equal(inputs[IMAGES][0], frame.data)
            assert inputs[STATE].shape == (1, 2)
            assert inputs[STATE].dtype == np.float32
            np.testing.assert_array_equal(inputs[STATE], [[0, 1]])
        else:
            if with_camera:
                _assert_window(inputs, [11])
            _assert_state_window(inputs, [0])
        if not with_camera:
            assert set(inputs) == {STATE, TASK}
        assert inputs[TASK] == ["current task"]


@pytest.mark.parametrize("with_camera", [False, True], ids=["state-only", "with-camera"])
@pytest.mark.parametrize(
    ("stride", "expected"),
    [
        (1, [[0, 0, 0], [0, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]),
        (2, [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 3], [0, 2, 4], [1, 3, 5]]),
    ],
)
def test_state_and_images_align_through_startup_stride_and_eviction(
    make_source: Callable, stride: int, expected: list[list[int]], with_camera: bool
) -> None:
    source, model = make_source(stride=stride)
    for value, window in enumerate(expected):
        source.update(_observation(value), {"top": _frame(value)} if with_camera else {}, step=100 + value * 10)
        assert len(model.calls) == value + 1
        _assert_state_window(model.calls[-1], window)
        if with_camera:
            _assert_window(model.calls[-1], window)
        else:
            assert set(model.calls[-1]) == {STATE}


@pytest.mark.parametrize(("stride", "expected"), [(1, [1, 2, 3]), (2, [0, 1, 3])])
def test_updates_record_even_when_queue_skips_inference(
    make_source: Callable, stride: int, expected: list[int]
) -> None:
    source, model = make_source(stride=stride, chunk_size=3)
    frame = _frame(0)
    for value in range(3):
        frame.data.fill(value)
        action = source.update(_observation(value), {"top": frame}, step=value * 100)
        np.testing.assert_array_equal(action, [1, 1])
        assert len(model.calls) == 1
    assert source.action_queue.remaining == 0

    frame.data.fill(3)
    source.update(_observation(3), {"top": frame}, step=999)

    assert len(model.calls) == 2
    _assert_window(model.calls[-1], expected)
    _assert_state_window(model.calls[-1], expected)


def test_camera_buffer_and_returned_snapshots_do_not_alias_history(make_source: Callable) -> None:
    source, model = make_source()
    frame = _frame(1)
    source.update(_observation(), {"top": frame}, step=0)
    first = model.calls[0][IMAGES]
    assert not np.shares_memory(first, frame.data)
    frame.data.fill(2)
    source.update(_observation(), {"top": frame}, step=1)
    second = model.calls[1][IMAGES]
    _assert_window(model.calls[0], [1, 1, 1])
    _assert_window(model.calls[1], [1, 1, 2])
    assert not np.shares_memory(first, second)

    first.fill(99)
    second.fill(98)
    frame.data.fill(3)
    source.update(_observation(), {"top": frame}, step=2)
    _assert_window(model.calls[-1], [1, 2, 3])

    for value in range(4, 9):
        frame.data.fill(value)
        source.update(_observation(), {"top": frame}, step=value)
    _assert_window(model.calls[2], [1, 2, 3])
    _assert_window(model.calls[-1], [6, 7, 8])


def test_state_driver_buffer_and_returned_snapshots_do_not_alias_history(make_source: Callable) -> None:
    source, model = make_source()
    observation = _observation(1)
    source.update(observation, {}, step=0)
    first = model.calls[0][STATE]
    assert not np.shares_memory(first, observation.state)

    observation.joint_positions[:] = [2, 3]
    source.update(observation, {}, step=1)
    second = model.calls[1][STATE]
    _assert_state_window(model.calls[0], [1, 1, 1])
    _assert_state_window(model.calls[1], [1, 1, 2])
    assert not np.shares_memory(first, second)

    first.fill(99)
    second.fill(98)
    preview = source.to_model_input(_observation(90), {})
    _assert_state_window(preview, [1, 2, 90])
    preview[STATE].fill(97)
    for value in range(3, 9):
        observation.joint_positions[:] = [value, value + 1]
        source.update(observation, {}, step=value)
    observation.joint_positions.fill(96)
    _assert_state_window(model.calls[2], [1, 2, 3])
    _assert_state_window(model.calls[-1], [6, 7, 8])


@pytest.mark.parametrize("with_camera", [False, True], ids=["state-only", "with-camera"])
@pytest.mark.parametrize("reset_model", [True, False])
def test_reset_reseeds_state_image_history_and_queue(
    make_source: Callable, reset_model: bool, with_camera: bool
) -> None:
    source, model = make_source(chunk_size=4)
    source.update(_observation(1), {"top": _frame(1)} if with_camera else {}, step=0)
    source.update(_observation(2), {"top": _frame(2)} if with_camera else {}, step=1)
    resets = model.resets

    source.reset(reset_model=reset_model)

    assert model.resets == resets + int(reset_model)
    assert source.action_queue.remaining == 0
    action = source.update(_observation(8), {"top": _frame(8)} if with_camera else {}, step=0)
    assert len(model.calls) == 2
    np.testing.assert_array_equal(action, [2, 2])
    _assert_state_window(model.calls[-1], [8, 8, 8])
    if with_camera:
        _assert_window(model.calls[-1], [8, 8, 8])
    else:
        assert set(model.calls[-1]) == {STATE}


def test_disconnect_reconnect_starts_a_fresh_history(make_source: Callable) -> None:
    source, model = make_source(chunk_size=4)
    source.update(_observation(1), {"top": _frame(1)}, step=0)
    source.disconnect()
    preview = source.to_model_input(_observation(7), {"top": _frame(7)})
    _assert_window(preview, [7, 7, 7])
    _assert_state_window(preview, [7, 7, 7])

    source.connect(bus=_CallbackBus([]), session_id="second-episode")
    source.update(_observation(9), {"top": _frame(9)}, step=0)

    assert len(model.calls) == 2
    assert model.resets == 2
    _assert_window(model.calls[-1], [9, 9, 9])
    _assert_state_window(model.calls[-1], [9, 9, 9])


def test_preview_and_explicit_warmup_do_not_record_or_double_count(make_source: Callable) -> None:
    source, model = make_source()
    for value in [90, 91]:
        preview = source.to_model_input(_observation(value), {"top": _frame(value)})
        _assert_window(preview, [value] * 3)
        _assert_state_window(preview, [value] * 3)
    preview = source.to_model_input(_observation(1), {"top": _frame(1)})
    source.warmup(preview)
    source.warmup(preview)
    assert len(model.calls) == 1
    source.update(_observation(1), {"top": _frame(1)}, step=0)
    assert len(model.calls) == 1

    preview = source.to_model_input(_observation(99), {"top": _frame(99)})
    _assert_window(preview, [1, 1, 99])
    _assert_state_window(preview, [1, 1, 99])
    preview[IMAGES].fill(42)
    preview[STATE].fill(42)
    source.update(_observation(2), {"top": _frame(2)}, step=1)
    source.update(_observation(3), {"top": _frame(3)}, step=2)

    _assert_window(model.calls[0], [1, 1, 1])
    _assert_window(model.calls[1], [1, 1, 2])
    _assert_window(model.calls[2], [1, 2, 3])
    _assert_state_window(model.calls[0], [1, 1, 1])
    _assert_state_window(model.calls[1], [1, 1, 2])
    _assert_state_window(model.calls[2], [1, 2, 3])


@pytest.mark.parametrize("change", ["keys", "missing", "extra", "shape", "dtype"])
def test_camera_changes_require_reset_and_rejection_does_not_record(make_source: Callable, change: str) -> None:
    source, model = make_source()
    source.update(_observation(), {"top": _frame(1), "wrist": _frame(11)}, step=0)
    frames = {"top": _frame(2), "wrist": _frame(12)}
    if change == "keys":
        frames["side"] = frames.pop("wrist")
    elif change == "missing":
        frames = {}
    elif change == "extra":
        frames["side"] = _frame(22)
    elif change == "shape":
        frames["top"] = Frame(np.zeros((4, 4, 3), dtype=np.uint8), 0.0, 0)
    else:
        frames["top"] = Frame(np.zeros((2, 4, 3), dtype=np.uint16), 0.0, 0)

    with pytest.raises(ValueError, match="must remain stable"):
        source.update(_observation(), frames, step=1)
    assert len(model.calls) == 1
    source.update(_observation(), {"top": _frame(3), "wrist": _frame(13)}, step=2)
    _assert_window(model.calls[-1], [1, 1, 3], f"{IMAGES}.top")

    source.reset(reset_model=False)
    source.update(_observation(), frames, step=0)
    assert len(model.calls) == 3


@pytest.mark.parametrize("shape", [(2, 4), (1, 2, 4, 3)])
def test_invalid_single_frame_rank_is_rejected(make_source: Callable, shape: tuple[int, ...]) -> None:
    source, model = make_source()
    frame = Frame(np.zeros(shape, dtype=np.uint8), 0.0, 0)
    with pytest.raises(ValueError, match="batched single-frame images"):
        source.update(_observation(), {"top": frame}, step=0)
    assert model.calls == []
    source.update(_observation(), {"top": _frame(5)}, step=1)
    _assert_window(model.calls[-1], [5, 5, 5])


def test_replacing_single_camera_requires_reset(make_source: Callable) -> None:
    source, _ = make_source()
    source.update(_observation(), {"top": _frame(1)}, step=0)
    with pytest.raises(ValueError, match="must remain stable"):
        source.update(_observation(), {"wrist": _frame(2)}, step=1)


@pytest.mark.parametrize("with_camera", [False, True], ids=["state-only", "with-camera"])
def test_changed_state_dimensions_require_reset_and_rejection_does_not_record(
    make_source: Callable, with_camera: bool
) -> None:
    source, model = make_source()
    source.update(_observation(1), {"top": _frame(1)} if with_camera else {}, step=0)
    changed = FakeRobotObservation(np.array([2, 3, 4], dtype=np.float32))
    with pytest.raises(ValueError, match="must remain stable"):
        source.update(changed, {"top": _frame(2)} if with_camera else {}, step=1)
    assert len(model.calls) == 1

    source.update(_observation(3), {"top": _frame(3)} if with_camera else {}, step=2)
    _assert_state_window(model.calls[-1], [1, 1, 3])
    if with_camera:
        _assert_window(model.calls[-1], [1, 1, 3])

    source.reset(reset_model=False)
    source.update(changed, {"top": _frame(2)} if with_camera else {}, step=0)
    assert len(model.calls) == 3
    assert model.calls[-1][STATE].shape == (1, 3, 3)
    np.testing.assert_array_equal(model.calls[-1][STATE], [[[2, 3, 4]] * 3])


@pytest.mark.parametrize("shape", [(), (1, 2), (1, 1, 2)])
def test_invalid_single_state_rank_is_rejected_without_recording(make_source: Callable, shape: tuple[int, ...]) -> None:
    source, model = make_source()
    observation = FakeRobotObservation(np.zeros(shape, dtype=np.float32))
    with pytest.raises(ValueError, match="batched single-frame state"):
        source.update(observation, {}, step=0)
    assert model.calls == []
    source.update(_observation(5), {}, step=1)
    _assert_state_window(model.calls[-1], [5, 5, 5])


def test_multiple_cameras_and_state_share_windows_but_task_is_current_only(make_source: Callable) -> None:
    source, model = make_source()
    for value in range(1, 4):
        source.set_task(f"task {value}")
        obs = _observation(value, images={"top": _frame(90), "wrist": _frame(value + 10)})
        source.update(obs, {"top": _frame(value)}, step=value)

    inputs = model.calls[-1]
    assert set(inputs) == {f"{IMAGES}.top", f"{IMAGES}.wrist", STATE, TASK}
    _assert_window(inputs, [1, 2, 3], f"{IMAGES}.top")
    _assert_window(inputs, [11, 12, 13], f"{IMAGES}.wrist")
    _assert_state_window(inputs, [1, 2, 3])
    assert inputs[TASK] == ["task 3"]
    assert model.calls[0][TASK] == ["task 1"]

    source.set_task(None)
    source.update(_observation(4, images={"wrist": _frame(14)}), {"top": _frame(4)}, step=4)
    assert TASK not in model.calls[-1]
    _assert_state_window(model.calls[-1], [2, 3, 4])
    _assert_window(model.calls[-1], [2, 3, 4], f"{IMAGES}.top")
