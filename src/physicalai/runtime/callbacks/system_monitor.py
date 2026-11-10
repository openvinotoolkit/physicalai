# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Nonblocking system/GPU sampling, independent of visualization backends."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from loguru import logger

from physicalai.config import export_config
from physicalai.runtime.callbacks.gpu_stats import collect_gpu_stats

if TYPE_CHECKING:
    from concurrent.futures import Future

    import psutil

    from physicalai.runtime.callbacks.gpu_stats import GpuStatsSample


@dataclass
class SystemStats:
    """Host/process counters; CPU percentages use total host logical-core capacity."""

    host_cpu_percent: float
    host_memory_percent: float
    host_memory_used_bytes: int
    host_memory_total_bytes: int
    process_cpu_percent: float
    process_rss_bytes: int


@dataclass
class SystemStatsSample:
    """Newly collected counters; None sections leave the viewer's previous data intact."""

    system: SystemStats | None = None
    gpu: GpuStatsSample | None = None


class SystemMonitor(Protocol):
    """Replaceable system/GPU sampler owned and closed by its consuming callback.

    Methods run on the control thread. Implementations must not block on external
    I/O: perform slow work in the background and return only completed samples.
    """

    def poll(self) -> SystemStatsSample | None:
        """Return new counters, or None when no sample is due."""
        ...

    def reset(self) -> None:
        """Discard previous-session work and re-prime counters without waiting."""
        ...

    def close(self) -> None:
        """Release resources without waiting for in-flight queries."""
        ...


@export_config
class LocalSystemMonitor:
    """Sample psutil counters and read-only vendor GPU queries at most once a second.

    Requires psutil from ``physicalai[observer-rerun]``. GPU queries use one lazy
    background worker, bounded subprocess timeouts, and no overlapping submissions
    while the current session's query is pending. No tools or drivers are required.
    """

    def __init__(self) -> None:
        """Initialize counters without querying hardware or starting a worker."""
        self._process: psutil.Process | None = None
        self._next_sample = 0.0
        self._gpu_executor: ThreadPoolExecutor | None = None
        self._gpu_future: Future[GpuStatsSample] | None = None

    def poll(self) -> SystemStatsSample | None:
        """Collect due system counters and consume completed GPU work without waiting.

        Returns:
            Newly sampled sections, or None before the next sampling deadline.
        """
        now = time.monotonic()
        if now < self._next_sample:
            return None
        self._next_sample = now + 1.0
        gpu = self._poll_gpu()
        return SystemStatsSample(system=self._sample_system(), gpu=gpu)

    def reset(self) -> None:
        """Discard previous-session GPU results and re-prime CPU counters."""
        self._process = None
        self._next_sample = 0.0
        if self._gpu_future is not None:
            self._gpu_future.cancel()
            self._gpu_future = None

    def close(self) -> None:
        """Cancel queued queries without waiting for bounded in-flight commands."""
        if self._gpu_executor is not None:
            self._gpu_executor.shutdown(wait=False, cancel_futures=True)
            self._gpu_executor = None
            self._gpu_future = None

    def _sample_system(self) -> SystemStats | None:
        import psutil  # noqa: PLC0415

        try:
            if self._process is None:
                self._process = psutil.Process()
                # The first nonblocking CPU sample is meaningless; prime both counters.
                self._process.cpu_percent(interval=None)
                psutil.cpu_percent(interval=None)
                return None
            cpu = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            process_cpu = self._process.cpu_percent(interval=None) / (psutil.cpu_count() or 1)
            rss = self._process.memory_info().rss
        except (OSError, psutil.Error):
            logger.opt(exception=True).debug("Failed to sample system stats")
            self._process = None
            return None
        return SystemStats(
            host_cpu_percent=cpu,
            host_memory_percent=memory.percent,
            host_memory_used_bytes=memory.total - memory.available,
            host_memory_total_bytes=memory.total,
            process_cpu_percent=process_cpu,
            process_rss_bytes=rss,
        )

    def _poll_gpu(self) -> GpuStatsSample | None:
        sample = None
        if self._gpu_future is not None:
            if not self._gpu_future.done():
                return None
            try:
                sample = self._gpu_future.result()
            except Exception:  # noqa: BLE001
                # A failed collector must not poison future polls or CPU/RAM sampling.
                logger.opt(exception=True).debug("Failed to sample GPU stats")
            self._gpu_future = None
        if self._gpu_executor is None:
            self._gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="physicalai-gpu-stats")
        self._gpu_future = self._gpu_executor.submit(collect_gpu_stats)
        return sample
