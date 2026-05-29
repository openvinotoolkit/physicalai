# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for InferenceLatencyBenchmark."""

from __future__ import annotations

import time

import numpy as np
import pytest

from physicalai.benchmark.performance.inference_benchmark import InferenceLatencyBenchmark
from physicalai.inference.data.features import (
    InferenceFeature,
    InferenceFeatureDtype,
    InferenceFeatureType,
)
from physicalai.inference import InferenceModel


class FakeModel(InferenceModel):
    """Minimal stand-in for InferenceModel used by the benchmark.

    Records every call and optionally sleeps to simulate compute time.
    """

    def __init__(
        self,
        input_features: list[InferenceFeature] | None = None,
        sleep_s: float = 0.0,
    ) -> None:
        self.input_features = input_features or []
        self.sleep_s = sleep_s
        self.calls: list[dict[str, np.ndarray | str | list[str]]] = []

    def __call__(self, sample: dict[str, np.ndarray | str | list[str]]) -> dict[str, np.ndarray]:
        self.calls.append(sample)
        if self.sleep_s:
            time.sleep(self.sleep_s)
        return {"action": np.zeros((1, 1), dtype=np.float32)}


@pytest.fixture
def features() -> list[InferenceFeature]:
    return [
        InferenceFeature(
            ftype=InferenceFeatureType.STATE,
            shape=(6,),
            name="observation.state",
            dtype=InferenceFeatureDtype.FLOAT32,
        ),
        InferenceFeature(
            ftype=InferenceFeatureType.VISUAL,
            shape=(3, 16, 16),
            name="observation.image",
            dtype=InferenceFeatureDtype.FLOAT32,
        ),
        InferenceFeature(
            ftype=InferenceFeatureType.LANGUAGE,
            shape=(),
            name="task",
            dtype=InferenceFeatureDtype.STRING,
        ),
    ]


class TestInferenceLatencyBenchmark:
    def test_run_with_random_inputs_returns_expected_metrics(self, features: list[InferenceFeature]) -> None:
        model = FakeModel(input_features=features)
        benchmark = InferenceLatencyBenchmark(max_iters=5, warmup_iters=2, max_duration=None)

        metrics = benchmark.run(model)

        assert metrics["num_iters"] == 5
        assert len(model.calls) == 2 + 5
        for key in (
            "avg_warmup_iter_time",
            "min_iter_time",
            "max_iter_time",
            "median_iter_time",
            "std_iter_time",
        ):
            assert metrics[key] >= 0.0
        assert metrics["min_iter_time"] <= metrics["median_iter_time"] <= metrics["max_iter_time"]

    def test_random_inputs_match_feature_shapes_and_dtypes(self, features: list[InferenceFeature]) -> None:
        model = FakeModel(input_features=features)
        benchmark = InferenceLatencyBenchmark(max_iters=1, warmup_iters=1, max_duration=None)

        benchmark.run(model)

        sample = model.calls[0]
        assert set(sample) == {"observation.state", "observation.image", "task"}
        assert isinstance(sample["observation.state"], np.ndarray)
        assert sample["observation.state"].shape == (1, 6)
        assert sample["observation.state"].dtype == np.float32
        assert sample["observation.image"].shape == (1, 3, 16, 16)
        assert isinstance(sample["task"], list)
        assert len(sample["task"]) == 1
        assert isinstance(sample["task"][0], str)

    def test_run_with_explicit_inputs_consumes_iterable_once(self, features: list[InferenceFeature]) -> None:
        model = FakeModel(input_features=features)
        inputs = [{"observation.state": np.full((1, 6), i, dtype=np.float32)} for i in range(6)]
        benchmark = InferenceLatencyBenchmark(max_iters=None, warmup_iters=2, max_duration=None)

        metrics = benchmark.run(model, iter(inputs))

        assert metrics["num_iters"] == 4
        assert len(model.calls) == 6
        # Warmup uses items 0, 1; measured loop uses items 2..5.
        assert int(model.calls[0]["observation.state"][0, 0]) == 0
        assert int(model.calls[2]["observation.state"][0, 0]) == 2

    def test_max_iters_bounds_measured_loop(self, features: list[InferenceFeature]) -> None:
        model = FakeModel(input_features=features)
        benchmark = InferenceLatencyBenchmark(max_iters=3, warmup_iters=1, max_duration=None)

        metrics = benchmark.run(model)

        assert metrics["num_iters"] == 3
        assert len(model.calls) == 1 + 3

    def test_max_duration_bounds_measured_loop(self, features: list[InferenceFeature]) -> None:
        model = FakeModel(input_features=features, sleep_s=0.02)
        benchmark = InferenceLatencyBenchmark(max_iters=None, warmup_iters=1, max_duration=50)

        start = time.perf_counter()
        metrics = benchmark.run(model)
        elapsed = time.perf_counter() - start

        assert metrics["num_iters"] >= 1
        # Generous upper bound: budget + warmup + a couple iterations of slack.
        assert elapsed < 1.0

    def test_std_iter_time_is_zero_for_single_iteration(self, features: list[InferenceFeature]) -> None:
        model = FakeModel(input_features=features)
        benchmark = InferenceLatencyBenchmark(max_iters=1, warmup_iters=1, max_duration=None)

        metrics = benchmark.run(model)

        assert metrics["num_iters"] == 1
        assert metrics["std_iter_time"] == 0.0

    def test_run_without_inputs_raises_when_model_has_no_features(self) -> None:
        model = FakeModel(input_features=[])
        benchmark = InferenceLatencyBenchmark(max_iters=1, warmup_iters=1, max_duration=None)

        with pytest.raises(ValueError, match="inputs should be provided"):
            benchmark.run(model)

    def test_inputs_exhausted_during_warmup_raises(self, features: list[InferenceFeature]) -> None:
        model = FakeModel(input_features=features)
        benchmark = InferenceLatencyBenchmark(max_iters=5, warmup_iters=3, max_duration=None)

        with pytest.raises(ValueError, match="inputs exhausted during warmup"):
            benchmark.run(model, iter([{"observation.state": np.zeros((1, 6), dtype=np.float32)}]))

    def test_no_measured_iterations_raises(self, features: list[InferenceFeature]) -> None:
        model = FakeModel(input_features=features)
        benchmark = InferenceLatencyBenchmark(max_iters=5, warmup_iters=1, max_duration=None)

        warmup_only = [{"observation.state": np.zeros((1, 6), dtype=np.float32)}]
        with pytest.raises(ValueError, match="no inference iterations were executed"):
            benchmark.run(model, iter(warmup_only))
