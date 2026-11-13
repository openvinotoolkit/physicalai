# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Opt-in PolicySource history through the real RTC worker and action queue."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pytest

from physicalai.capture import Frame
from physicalai.inference.constants import (
    ACTION,
    IMAGES,
    PREV_CHUNK_LEFT_OVER,
    RTC_EXECUTION_HORIZON,
    RTC_INFERENCE_DELAY,
    RTC_MAX_GUIDANCE_WEIGHT,
    STATE,
)
from physicalai.inference.manifest import Manifest, ModelSpec, ObservationHistorySpec
from physicalai.inference.model import InferenceModel
from physicalai.inference.preprocessors.observation_history import StackObservationHistory
from physicalai.runtime import PolicySource, RTCActionQueue, RTCExecution
from physicalai.runtime._callback_bus import _CallbackBus
from physicalai.runtime.execution import rtc
from tests.unit.runtime.conftest import FakeRobotObservation


@dataclass
class _RTCModel:
    manifest: Manifest
    chunk_size: int = 4
    calls: list[dict[str, Any]] = field(default_factory=list)
    entered: list[threading.Event] = field(default_factory=lambda: [threading.Event(), threading.Event()])
    release: list[threading.Event] = field(default_factory=lambda: [threading.Event(), threading.Event()])
    resets: int = 0

    def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        # Materialize at model entry, as the default inference formatter does.
        self.calls.append(StackObservationHistory()(inputs))
        call = len(self.calls)
        if call in (2, 3):
            self.entered[call - 2].set()
            assert self.release[call - 2].wait(timeout=5.0)
        return {ACTION: np.full((1, self.chunk_size, 2), call, dtype=np.float32)}

    def reset(self) -> None:
        self.resets += 1


@pytest.fixture(params=[1, 2], ids=["stride-1", "stride-2"])
def rtc_source(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[PolicySource, _RTCModel, RTCExecution]]:
    monkeypatch.setattr(rtc, "_WARMUP_TIMEOUT_S", 5.0)
    model = _RTCModel(
        Manifest(model=ModelSpec(observation_history=ObservationHistorySpec(n_obs_steps=3, stride=request.param)))
    )
    execution = RTCExecution(chunk_size=4, execution_horizon=2, queue_threshold=4, fps=30.0)
    source = PolicySource(cast("InferenceModel", model), execution=execution, action_queue=RTCActionQueue())
    try:
        source.connect(bus=_CallbackBus([]), session_id="rtc-history")
        yield source, model, execution
    finally:
        for release in model.release:
            release.set()
        worker = execution._thread  # noqa: SLF001
        source.disconnect()
        assert worker is None or not worker.is_alive()


def _update(source: PolicySource, frame: Frame, value: int) -> np.ndarray:
    frame.data.fill(value)
    observation = FakeRobotObservation(np.array([value, value + 1], dtype=np.float32))
    return source.update(observation, {"top": frame}, step=value)


def _assert_window(inputs: dict[str, Any], expected: list[int]) -> None:
    images = inputs[IMAGES]
    assert images.shape == (1, 3, 2, 4, 3)
    assert images.dtype == np.uint8
    np.testing.assert_array_equal(images, np.broadcast_to(np.array(expected)[None, :, None, None, None], images.shape))
    state = inputs[STATE]
    assert state.shape == (1, 3, 2)
    assert state.dtype == np.float32
    np.testing.assert_array_equal(state, [[[value, value + 1] for value in expected]])


def test_rtc_snapshot_is_detached_while_updates_continue_history(
    rtc_source: tuple[PolicySource, _RTCModel, RTCExecution],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, model, execution = rtc_source
    frame = Frame(np.zeros((2, 4, 3), dtype=np.uint8), timestamp=0.0, sequence=0)

    np.testing.assert_array_equal(_update(source, frame, 0), [1, 1])
    _assert_window(model.calls[0], [0, 0, 0])
    np.testing.assert_array_equal(model.calls[0][PREV_CHUNK_LEFT_OVER], np.zeros((1, 4, 2), dtype=np.float32))
    assert int(model.calls[0][RTC_INFERENCE_DELAY]) == 0
    assert int(model.calls[0][RTC_EXECUTION_HORIZON]) == 1
    assert float(model.calls[0][RTC_MAX_GUIDANCE_WEIGHT]) == pytest.approx(0.0, abs=1e-6)

    np.testing.assert_array_equal(_update(source, frame, 1), [1, 1])
    assert model.entered[0].wait(timeout=5.0)
    blocked = model.calls[1]
    _assert_window(blocked, [0, 0, 1])
    assert not np.shares_memory(blocked[IMAGES], frame.data)
    assert int(blocked[RTC_EXECUTION_HORIZON]) == 2
    assert float(blocked[RTC_MAX_GUIDANCE_WEIGHT]) == pytest.approx(5.0)

    for value in range(2, 7):
        np.testing.assert_array_equal(_update(source, frame, value), [1, 1])
        assert len(model.calls) == 2
        _assert_window(blocked, [0, 0, 1])
    assert source.action_queue.remaining == 0

    accepted = threading.Event()
    accept_result = execution._accept_result

    def signal_acceptance(*args: Any, **kwargs: Any) -> np.ndarray | None:
        result = accept_result(*args, **kwargs)
        accepted.set()
        return result

    monkeypatch.setattr(execution, "_accept_result", signal_acceptance)
    model.release[0].set()
    assert accepted.wait(timeout=5.0)
    # RTC discards observations published before result acceptance. Supply a
    # fresh observation; history must still include the intervening updates.
    _update(source, frame, 7)
    assert model.entered[1].wait(timeout=5.0)
    history = model.manifest.model.observation_history
    assert history is not None
    stride = history.stride
    _assert_window(model.calls[2], [5, 6, 7] if stride == 1 else [3, 5, 7])
    assert execution.inference_count == 2
    assert not np.shares_memory(blocked[IMAGES], model.calls[2][IMAGES])
    assert not np.shares_memory(blocked[STATE], model.calls[2][STATE])

    frame.data.fill(99)
    _assert_window(model.calls[0], [0, 0, 0])
    _assert_window(blocked, [0, 0, 1])
    _assert_window(model.calls[2], [5, 6, 7] if stride == 1 else [3, 5, 7])


def test_rtc_reset_without_model_reset_discards_old_result_and_reseeds_history(
    rtc_source: tuple[PolicySource, _RTCModel, RTCExecution],
) -> None:
    source, model, execution = rtc_source
    frame = Frame(np.zeros((2, 4, 3), dtype=np.uint8), timestamp=0.0, sequence=0)
    _update(source, frame, 0)
    _update(source, frame, 1)
    assert model.entered[0].wait(timeout=5.0)
    for value in range(2, 5):
        _update(source, frame, value)

    worker = execution._thread  # noqa: SLF001
    resets = model.resets
    # This returns while the old model call is still blocked.
    source.reset(reset_model=False)
    assert not model.release[0].is_set()
    assert model.resets == resets
    assert source.action_queue.remaining == 0
    assert execution._thread is worker  # noqa: SLF001
    assert worker is not None and worker.is_alive()

    model.release[0].set()
    model.release[1].set()
    np.testing.assert_array_equal(_update(source, frame, 9), [3, 3])
    assert model.entered[1].is_set()
    assert len(model.calls) == 3
    assert execution.inference_count == 2
    assert source.action_queue.remaining == 3
    _assert_window(model.calls[2], [9, 9, 9])
    # Neither the rejected result nor its predecessor may guide the new episode.
    np.testing.assert_array_equal(model.calls[2][PREV_CHUNK_LEFT_OVER], np.zeros((1, 4, 2), dtype=np.float32))
    assert int(model.calls[2][RTC_EXECUTION_HORIZON]) == 1
    assert float(model.calls[2][RTC_MAX_GUIDANCE_WEIGHT]) == pytest.approx(0.0, abs=1e-6)
    assert model.resets == resets
    _assert_window(model.calls[1], [0, 0, 1])
