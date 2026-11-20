# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Request-time selection with inference-time stacking, including real workers."""

from __future__ import annotations

import threading
from typing import Any, cast
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from physicalai.inference.constants import ACTION, IMAGES, STATE, TASK
from physicalai.inference.manifest import ComponentSpec, Manifest, ModelSpec, ObservationHistorySpec
from physicalai.inference.model import InferenceModel
from physicalai.inference.preprocessors.observation_history import StackObservationHistory
from physicalai.runtime import (
    AsyncExecution,
    ChunkedActionQueue,
    PolicySource,
    RTCActionQueue,
    RTCExecution,
    SyncExecution,
)
from physicalai.runtime._callback_bus import _CallbackBus
from physicalai.runtime.action_sources.observation_history import ObservationHistory
from physicalai.runtime.execution import rtc
from tests.unit.runtime.test_observation_history import (
    RecordingModel,
    _assert_state_window,
    _assert_window,
    _frame,
    _observation,
)


@pytest.mark.parametrize("stride", [1, 2])
def test_deferred_window_is_fixed_readonly_owned_selection_without_stacking(stride: int) -> None:
    history = ObservationHistory(ObservationHistorySpec(n_obs_steps=3, stride=stride))
    images = np.zeros((1, 2, 4, 3), dtype=np.uint8)
    state = np.array([[0, 1]], dtype=np.float32)
    task = ["current task"]
    with patch.object(np, "stack", side_effect=AssertionError("selection must not stack")):
        first = history.window(
            {IMAGES: images, STATE: state, TASK: task}, record=True, camera_names=("top",), stack=False
        )
        images.fill(1)
        state[:] = [1, 2]
        selected = history.window({IMAGES: images, STATE: state}, record=True, camera_names=("top",), stack=False)
        for value in range(2, 8):
            images.fill(value)
            state[:] = [value, value + 1]
            history.window({IMAGES: images, STATE: state}, record=True, camera_names=("top",), stack=False)
        history.reset()

    assert first[TASK] is task
    for key, buffer in [(IMAGES, images), (STATE, state)]:
        assert isinstance(selected[key], tuple)
        assert len(selected[key]) == 3
        for frame in selected[key]:
            assert frame.flags.owndata
            assert not frame.flags.writeable
            assert not np.shares_memory(frame, buffer)
            with pytest.raises(ValueError, match="read-only"):
                frame.flat[0] = 99
        with pytest.raises(TypeError):
            selected[key][0] = buffer
    _assert_window(StackObservationHistory()(first), [0, 0, 0])
    _assert_state_window(StackObservationHistory()(first), [0, 0, 0])
    materialized = StackObservationHistory()(selected)
    _assert_window(materialized, [0, 0, 1])
    _assert_state_window(materialized, [0, 0, 1])


@pytest.mark.parametrize("processors", ["first", "later", "empty", "absent", "mock", "spec-only"])
def test_default_defers_history_regardless_of_preprocessors(processors: str) -> None:
    model = RecordingModel(
        Manifest(
            model=ModelSpec(
                observation_history=ObservationHistorySpec(n_obs_steps=3),
                preprocessors=[ComponentSpec(type="stack_observation_history")],
            )
        )
    )
    if processors != "absent":
        model.preprocessors = {  # type: ignore[attr-defined]
            "first": [StackObservationHistory()],
            "later": [object(), StackObservationHistory()],
            "empty": [],
            "mock": MagicMock(),
            "spec-only": [ComponentSpec(type="stack_observation_history")],
        }[processors]
    source = PolicySource(cast("InferenceModel", model))
    inputs = source.to_model_input(_observation(5), {"top": _frame(5)})
    assert isinstance(inputs[IMAGES], tuple)
    assert isinstance(inputs[STATE], tuple)
    inputs = StackObservationHistory()(inputs)
    _assert_window(inputs, [5, 5, 5])
    _assert_state_window(inputs, [5, 5, 5])


def test_null_history_formatter_supplies_full_tuples_without_inspecting_model() -> None:
    class RawModel(RecordingModel):
        def predict_action_chunk(self, inputs: dict[str, Any]) -> np.ndarray:
            self.calls.append(inputs)
            return np.ones((1, 2), dtype=np.float32)

    model = RawModel(
        Manifest(model=ModelSpec(observation_history=ObservationHistorySpec(n_obs_steps=3, preprocessor=None)))
    )
    with patch.object(RecordingModel, "preprocessors", new_callable=PropertyMock, create=True) as processors:
        processors.side_effect = AssertionError("PolicySource must not inspect model preprocessors")
        source = PolicySource(
            cast("InferenceModel", model),
            execution=SyncExecution(request_threshold=0.0),
            action_queue=ChunkedActionQueue(),
        )
        try:
            source.connect(bus=_CallbackBus([]), session_id="null-history-formatter")
            for value, expected in enumerate([[0, 0, 0], [0, 0, 1], [0, 1, 2], [1, 2, 3]]):
                source.update(_observation(value), {"top": _frame(value)}, step=value)
                inputs = model.calls[-1]
                for key in (IMAGES, STATE):
                    assert isinstance(inputs[key], tuple)
                    assert len(inputs[key]) == 3
                    assert [int(frame.flat[0]) for frame in inputs[key]] == expected
                _assert_window(inputs, expected)
                _assert_state_window(inputs, expected)
        finally:
            source.disconnect()
        processors.assert_not_called()


def test_no_history_supplies_single_frame_arrays() -> None:
    model = RecordingModel(Manifest(model=ModelSpec(n_obs_steps=7)))
    source = PolicySource(cast("InferenceModel", model), task="current task")
    frame = _frame(5)
    inputs = source.to_model_input(_observation(5), {"top": frame})
    assert set(inputs) == {IMAGES, STATE, TASK}
    assert inputs[IMAGES].shape == (1, 2, 4, 3)
    assert inputs[STATE].shape == (1, 2)
    np.testing.assert_array_equal(inputs[IMAGES], frame.data[None])
    np.testing.assert_array_equal(inputs[STATE], [[5, 6]])
    assert inputs[TASK] == ["current task"]


def test_default_window_still_stacks_into_writable_detached_arrays() -> None:
    history = ObservationHistory(ObservationHistorySpec(n_obs_steps=3))
    image = np.zeros((1, 2, 4, 3), dtype=np.uint8)
    state = np.array([[0, 1]], dtype=np.float32)
    with patch.object(np, "stack", wraps=np.stack) as stack:
        outputs = history.window({IMAGES: image, STATE: state}, record=True, camera_names=("top",))
        assert stack.call_count == 2
    for key, buffer in [(IMAGES, image), (STATE, state)]:
        assert isinstance(outputs[key], np.ndarray)
        assert outputs[key].flags.writeable
        assert not np.shares_memory(outputs[key], buffer)
        outputs[key].fill(99)
    preview = history.window({IMAGES: image, STATE: state}, record=False, camera_names=("top",))
    _assert_window(preview, [0, 0, 0])
    _assert_state_window(preview, [0, 0, 0])


@pytest.mark.parametrize("stride,expected", [(1, [1, 2, 3]), (2, [0, 1, 3])])
def test_deferred_queue_skips_and_invalid_updates_never_stack(stride: int, expected: list[int]) -> None:
    materialized: list[dict[str, Any]] = []

    class DeferredModel(RecordingModel):
        preprocessors = [StackObservationHistory()]

        def predict_action_chunk(self, inputs: dict[str, Any]) -> np.ndarray:
            materialized.append(self.preprocessors[0](inputs))
            return super().predict_action_chunk(materialized[-1])

    model = DeferredModel(
        Manifest(model=ModelSpec(observation_history=ObservationHistorySpec(n_obs_steps=3, stride=stride))),
        chunk_size=3,
    )
    source = PolicySource(
        cast("InferenceModel", model), execution=SyncExecution(request_threshold=0.0), action_queue=ChunkedActionQueue()
    )
    try:
        source.connect(bus=_CallbackBus([]), session_id="deferred-skips")
        source.update(_observation(), {"top": _frame(0)}, step=0)
        with patch.object(np, "stack", side_effect=AssertionError("skipped updates must not stack")):
            for value in [1, 2]:
                source.update(_observation(value), {"top": _frame(value)}, step=value)
            with pytest.raises(ValueError, match="must remain stable"):
                source.update(_observation(99), {"other": _frame(99)}, step=99)
        assert len(materialized) == 1
        source.update(_observation(3), {"top": _frame(3)}, step=3)
        assert len(materialized) == 2
        _assert_window(materialized[-1], expected)
        _assert_state_window(materialized[-1], expected)
    finally:
        source.disconnect()


@pytest.mark.parametrize("mode", ["async", "rtc"])
@pytest.mark.parametrize("stride", [1, 2])
def test_worker_stacks_request_selection_only_after_intervening_updates(
    mode: str, stride: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rtc, "_WARMUP_TIMEOUT_S", 2.0)
    entered = threading.Event()
    release = threading.Event()
    accepted = threading.Event()
    selections: list[dict[str, Any]] = []
    materialized: list[dict[str, Any]] = []
    stack_threads: list[threading.Thread] = []
    original_stack = np.stack

    def record_stack(*args: Any, **kwargs: Any) -> np.ndarray:
        stack_threads.append(threading.current_thread())
        return original_stack(*args, **kwargs)

    class BlockingModel(RecordingModel):
        preprocessors = [StackObservationHistory()]

        def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
            selections.append(inputs)
            assert isinstance(inputs[IMAGES], tuple)
            assert isinstance(inputs[STATE], tuple)
            if len(selections) == 2:
                entered.set()
                assert release.wait(timeout=5.0)
            materialized.append(self.preprocessors[0](inputs))
            return {ACTION: np.full((1, 4, 2), len(selections), dtype=np.float32)}

        def predict_action_chunk(self, inputs: dict[str, Any]) -> np.ndarray:
            return self(inputs)[ACTION][0]

    model = BlockingModel(
        Manifest(model=ModelSpec(observation_history=ObservationHistorySpec(n_obs_steps=3, stride=stride))),
        chunk_size=4,
    )
    execution = (
        AsyncExecution(request_threshold=1.0)
        if mode == "async"
        else RTCExecution(chunk_size=4, execution_horizon=2, queue_threshold=4)
    )
    queue = ChunkedActionQueue() if mode == "async" else RTCActionQueue()
    source = PolicySource(cast("InferenceModel", model), execution=execution, action_queue=queue)
    frame = _frame(0)
    try:
        source.connect(bus=_CallbackBus([]), session_id=f"deferred-{mode}")
        with patch.object(np, "stack", side_effect=record_stack):
            np.testing.assert_array_equal(source.update(_observation(), {"top": frame}, step=0), [1, 1])
            assert len(stack_threads) == 2
            # With a full queue, this update is recorded but not inferred.
            assert len(selections) == 1
            frame.data.fill(1)
            source.update(_observation(1), {"top": frame}, step=1)
            assert entered.wait(timeout=5.0)
            assert len(stack_threads) == 2
            selected = selections[1]

            for value in range(2, 7):
                frame.data.fill(value)
                source.update(_observation(value), {"top": frame}, step=value)
                assert len(selections) == 2
                assert len(stack_threads) == 2
            with pytest.raises(ValueError, match="must remain stable"):
                source.update(_observation(99), {"other": frame}, step=99)
            assert len(stack_threads) == 2
            for key, expected in [(IMAGES, [0, 0, 1]), (STATE, [0, 0, 1])]:
                assert isinstance(selected[key], tuple)
                assert [int(item.flat[0]) for item in selected[key]] == expected

            # Signal only after the actual worker has accepted its result.
            if mode == "async":
                accept = queue.push_chunk

                def signal_acceptance(*args: Any, **kwargs: Any) -> None:
                    accept(*args, **kwargs)
                    accepted.set()

                monkeypatch.setattr(queue, "push_chunk", signal_acceptance)
            else:
                assert isinstance(execution, RTCExecution)
                accept_result = execution._accept_result

                def signal_result(*args: Any, **kwargs: Any) -> Any:
                    result = accept_result(*args, **kwargs)
                    accepted.set()
                    return result

                monkeypatch.setattr(execution, "_accept_result", signal_result)
            release.set()
            assert accepted.wait(timeout=5.0)
            assert len(stack_threads) == 4
            assert all(thread is execution._thread for thread in stack_threads[2:])
            assert len(materialized) == 2
            _assert_window(materialized[1], [0, 0, 1])
            _assert_state_window(materialized[1], [0, 0, 1])
            frame.data.fill(99)
            _assert_window(materialized[0], [0, 0, 0])
            _assert_window(materialized[1], [0, 0, 1])
    finally:
        release.set()
        worker = execution._thread
        source.disconnect()
        assert worker is None or not worker.is_alive()
