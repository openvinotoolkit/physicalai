# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Latency benchmark for exported inference models."""

import time
from collections.abc import Iterable
from itertools import islice
from statistics import median, pstdev

import numpy as np

from physicalai.benchmark.performance.input_sources import RandomInputSource
from physicalai.inference import InferenceModel


class InferenceLatencyBenchmark:
    """Measure per-iteration inference latency of an :class:`InferenceModel`.

    The benchmark runs a configurable number of warmup iterations followed by
    a measured loop bounded by both an iteration cap and a wall-clock budget,
    whichever is reached first.  Per-iteration timings are collected and
    summarized into a metrics dict.

    Examples:
        >>> benchmark = InferenceLatencyBenchmark(max_iters=500, warmup_iters=5)
        >>> metrics = benchmark.run(model, inputs)
        >>> metrics["median_iter_time"]
    """

    def __init__(self, max_iters: int | None = 1000, warmup_iters: int = 1, max_duration: int | None = 60000) -> None:
        """Initialize the benchmark configuration.

        Args:
            max_iters: Maximum number of measured iterations to run.  ``None``
                disables the iteration cap (the run is then bounded only by
                ``max_duration`` or by ``inputs`` being exhausted).
            warmup_iters: Number of warmup iterations executed before
                measurement starts.  Their average duration is reported under
                ``avg_warmup_iter_time``.
            max_duration: Wall-clock budget for the measured loop, in
                milliseconds.  ``None`` disables the time cap.
        """
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.max_duration = max_duration

    def run(
        self,
        model: InferenceModel,
        inputs: Iterable[dict[str, np.ndarray | list[str]]] | None = None,
    ) -> dict[str, float | int]:
        """Run the benchmark against ``model`` using samples from ``inputs``.

        The iterable is consumed once: the first ``warmup_iters`` items are
        used for warmup, and the remaining items feed the measured loop until
        ``max_iters`` or ``max_duration`` is reached, or ``inputs`` is
        exhausted.

        Args:
            model: Loaded inference model invoked via ``model(sample)``.
            inputs: Iterable yielding input dicts compatible with ``model``.
                Values may be numpy arrays or lists of strings, depending on the
                model's declared input features. Must contain at least
                ``warmup_iters`` items plus at least one measured sample.
                When ``None``, a
                :class:`~physicalai.benchmark.performance.input_sources.RandomInputSource`
                is built from ``model.input_features``.

        Returns:
            Dictionary of latency metrics (seconds):

            - ``avg_warmup_iter_time``: Mean per-iteration time during warmup.
            - ``num_iters``: Number of measured iterations executed.
            - ``min_iter_time`` / ``max_iter_time``: Extremes of measured
              per-iteration times.
            - ``median_iter_time``: Median per-iteration time.
            - ``std_iter_time``: Population standard deviation of measured
              per-iteration times (``0.0`` when only a single iteration ran).

        Raises:
            ValueError: If ``inputs`` is ``None`` and the model declares no
                input features, is exhausted during warmup, or yields no
                measured iterations.
        """
        if inputs is None:
            if not model.input_features:
                msg = "inputs should be provided: the input model doesn't contain inputs information"
                raise ValueError(msg)
            inputs = RandomInputSource(model.input_features)

        results: dict[str, float] = {}

        input_iter = iter(inputs)
        warmup_batch = list(islice(input_iter, self.warmup_iters))
        if len(warmup_batch) < self.warmup_iters:
            msg = f"inputs exhausted during warmup: expected {self.warmup_iters} items, got {len(warmup_batch)}"
            raise ValueError(msg)

        warmup_start = time.perf_counter()
        for sample in warmup_batch:
            model(sample)
        warmup_elapsed = time.perf_counter() - warmup_start

        results["avg_warmup_iter_time"] = warmup_elapsed / self.warmup_iters

        iter_times: list[float] = []
        max_duration_s = self.max_duration / 1000.0 if self.max_duration is not None else None
        loop_start = time.perf_counter()
        while True:
            if self.max_iters is not None and len(iter_times) >= self.max_iters:
                break
            if max_duration_s is not None and (time.perf_counter() - loop_start) >= max_duration_s:
                break

            try:
                sample = next(input_iter)
            except StopIteration:
                break

            iter_start = time.perf_counter()
            model(sample)
            iter_times.append(time.perf_counter() - iter_start)

        if not iter_times:
            msg = "no inference iterations were executed; check inputs, max_iters, and max_duration"
            raise ValueError(msg)

        results["num_iters"] = len(iter_times)
        results["min_iter_time"] = min(iter_times)
        results["max_iter_time"] = max(iter_times)
        results["median_iter_time"] = median(iter_times)
        results["std_iter_time"] = pstdev(iter_times) if len(iter_times) > 1 else 0.0

        return results
