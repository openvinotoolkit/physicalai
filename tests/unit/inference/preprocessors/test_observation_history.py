# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Deferred history materialization and its manifest alias."""

from typing import Any
from copy import deepcopy

import numpy as np
import pytest

from physicalai.inference.component_factory import component_registry, instantiate_component
from physicalai.inference.constants import IMAGES, STATE, TASK
from physicalai.inference.manifest import ComponentSpec
from physicalai.inference.preprocessors.base import Preprocessor
from physicalai.inference.preprocessors.observation_history import ObservationWindow, StackObservationHistory


def test_window_deepcopy_preserves_shape_and_readonly_snapshot() -> None:
    frame = np.ones((1, 2), dtype=np.float32)
    frame.setflags(write=False)
    selection = ObservationWindow((frame, frame))
    copied = deepcopy(selection)
    assert isinstance(copied, ObservationWindow)
    assert copied.shape == (1, 2, 2)
    assert copied[0] is copied[1]
    for value in copied:
        assert not value.flags.writeable
        assert not np.shares_memory(value, frame)
        np.testing.assert_array_equal(value, frame)


def test_window_deepcopy_shares_frame_copies_across_request_windows() -> None:
    frame = np.ones((1, 2), dtype=np.float32)
    other = np.zeros_like(frame)
    request = {
        "first": ObservationWindow((frame,) * 5),
        "second": ObservationWindow((frame, other)),
    }
    for _ in range(2):  # RTC copies during warmup submission and worker entry.
        copied = deepcopy(request)
        assert all(value is copied["first"][0] for value in copied["first"])
        assert copied["first"][0] is copied["second"][0]
        assert copied["second"][0] is not copied["second"][1]
        for key in request:
            for original, value in zip(request[key], copied[key], strict=True):
                assert not value.flags.writeable
                assert not np.shares_memory(value, original)
                np.testing.assert_array_equal(value, original)
        request = copied


def test_registry_constructs_history_preprocessor() -> None:
    assert component_registry.get_class("stack_observation_history") is StackObservationHistory
    processor = instantiate_component(Preprocessor, ComponentSpec(type="stack_observation_history"))
    assert isinstance(processor, StackObservationHistory)


@pytest.mark.parametrize("image_keys", [(IMAGES,), (f"{IMAGES}.top", f"{IMAGES}.wrist")])
def test_stacks_aligned_batched_images_and_state_without_changing_inputs(image_keys: tuple[str, ...]) -> None:
    inputs: dict[str, Any] = {
        key: tuple(np.full((2, 2, 4, 3), value, dtype=np.uint8) for value in range(3)) for key in image_keys
    }
    inputs[STATE] = tuple(np.full((2, 5), value, dtype=np.float32) for value in range(3))
    for selection in inputs.values():
        for frame in selection:
            frame.setflags(write=False)

    outputs = StackObservationHistory()(inputs)

    assert outputs is not inputs
    assert outputs.keys() == inputs.keys()
    for key, selection in inputs.items():
        assert isinstance(selection, tuple)
        expected_shape = (2, 3, 5) if key == STATE else (2, 3, 2, 4, 3)
        assert outputs[key].shape == expected_shape
        assert outputs[key].dtype == selection[0].dtype
        for index, frame in enumerate(selection):
            np.testing.assert_array_equal(outputs[key][:, index], frame)
            assert not frame.flags.writeable
            assert not np.shares_memory(outputs[key], frame)


def test_already_stacked_arrays_tasks_and_unrelated_tuples_pass_through_by_identity() -> None:
    inputs = {
        IMAGES: np.zeros((1, 3, 2, 4, 3), dtype=np.uint8),
        f"{IMAGES}.top": np.zeros((1, 3, 2, 4, 3), dtype=np.uint8),
        STATE: np.zeros((1, 3, 2), dtype=np.float32),
        TASK: ["pick up"],
        "task_tuple": ("pick up", "place"),
        "observation.images_extra": ("not a camera",),
    }
    outputs = StackObservationHistory()(inputs)
    assert outputs is not inputs
    assert outputs.keys() == inputs.keys()
    for key, value in inputs.items():
        assert outputs[key] is value
    task = ("current task",)
    assert StackObservationHistory()({TASK: task})[TASK] is task
    assert StackObservationHistory()({}) == {}


@pytest.mark.parametrize("key,rank", [(STATE, 2), (IMAGES, 4), (f"{IMAGES}.top", 4)])
@pytest.mark.parametrize("invalid", ["empty", "not-array", "low-rank", "high-rank"])
def test_rejects_empty_non_array_and_invalid_rank_selections(key: str, rank: int, invalid: str) -> None:
    frame = np.zeros((1,) * rank, dtype=np.float32)
    selection: tuple = (frame,)
    if invalid == "empty":
        selection = ()
    elif invalid == "not-array":
        selection = (frame, frame.tolist())
    else:
        selection = (frame, np.zeros((1,) * (rank + (1 if invalid == "high-rank" else -1))))
    with pytest.raises(ValueError, match="Invalid observation history selection"):
        StackObservationHistory()({key: selection})


@pytest.mark.parametrize("key,shape", [(STATE, (1, 2)), (IMAGES, (1, 2, 4, 3)), (f"{IMAGES}.top", (1, 2, 4, 3))])
@pytest.mark.parametrize("change", ["shape", "dtype", "batch"])
def test_rejects_inconsistent_frames_within_selection(key: str, shape: tuple[int, ...], change: str) -> None:
    first = np.zeros(shape, dtype=np.float32)
    changed_shape = shape
    if change == "shape":
        changed_shape = (*shape[:-1], shape[-1] + 1)
    elif change == "batch":
        changed_shape = (2, *shape[1:])
    second = np.zeros(changed_shape, dtype=np.float64 if change == "dtype" else np.float32)
    with pytest.raises(ValueError, match="consistent shapes/dtypes"):
        StackObservationHistory()({key: (first, second)})


@pytest.mark.parametrize("other_key", [STATE, f"{IMAGES}.wrist"])
@pytest.mark.parametrize("change", ["batch", "time"])
def test_rejects_misaligned_batches_and_times_across_selections(other_key: str, change: str) -> None:
    shape = (2 if change == "batch" else 1, 2)
    if other_key != STATE:
        shape = (*shape, 4, 3)
    inputs = {
        f"{IMAGES}.top": (np.zeros((1, 2, 4, 3)),) * 3,
        other_key: (np.zeros(shape),) * (2 if change == "time" else 3),
    }
    with pytest.raises(ValueError, match="aligned batches/times"):
        StackObservationHistory()(inputs)
