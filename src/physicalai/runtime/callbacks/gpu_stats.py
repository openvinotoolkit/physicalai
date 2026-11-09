# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Best-effort GPU monitoring through read-only vendor CLI queries."""

from __future__ import annotations

import csv
import math
import shutil
import subprocess  # noqa: S404  # nosec B404  # Only fixed read-only queries, without a shell.
from dataclasses import dataclass, field

from loguru import logger

_CSV_COLUMNS = 5


@dataclass
class GpuDeviceStats:
    """Numeric GPU counters; unsupported values are None, never synthetic zeros."""

    vendor: str
    index: str
    utilization: float | None = None
    memory_used_mib: float | None = None
    memory_total_mib: float | None = None
    temperature_c: float | None = None

    @property
    def device_id(self) -> str:
        """Stable vendor/device key for utilization history."""
        return f"{self.vendor.lower()}_{self.index}"


@dataclass
class GpuStatsSample:
    """GPU counters and availability messages, independent of any viewer."""

    devices: list[GpuDeviceStats] = field(default_factory=list)
    unavailable: dict[str, str] = field(default_factory=dict)


def collect_gpu_stats() -> GpuStatsSample:
    """Collect table rows and numeric samples, tolerating missing tools/counters.

    This performs bounded subprocess I/O and must run outside the control thread.

    Returns:
        Numeric device counters and availability messages.
    """
    sample = GpuStatsSample()
    queries = (
        (
            "NVIDIA",
            "nvidia-smi",
            [
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
        ),
        ("Intel", "xpu-smi", ["dump", "-d", "-1", "-m", "0,3,18", "-n", "1"]),
    )
    for vendor, tool, args in queries:
        executable = shutil.which(tool)
        if executable is None:
            continue
        try:
            result = subprocess.run(  # noqa: S603  # nosec B603  # Fixed query via trusted PATH, no shell.
                [executable, *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=3,
                encoding="utf-8",
                errors="replace",
            )
            devices = _parse_gpu_stats(vendor, result.stdout)
        except (OSError, subprocess.SubprocessError, csv.Error):
            logger.debug("{} GPU counters unavailable", vendor)
            devices = []
        if devices:
            sample.devices.extend(devices)
        else:
            sample.unavailable[vendor] = "Unavailable (tool, driver, or permissions)"
    if not sample.devices and not sample.unavailable:
        sample.unavailable["Status"] = "No Intel/NVIDIA monitoring tool found"
    return sample


def _parse_gpu_stats(vendor: str, output: str) -> list[GpuDeviceStats]:
    devices = []
    for values in csv.reader(output.splitlines(), skipinitialspace=True, strict=True):
        if len(values) != _CSV_COLUMNS:  # Also ignores banners and incomplete output.
            continue
        if vendor == "Intel":
            _, index, utilization, temperature, used = values
            total = ""
        else:
            index, utilization, used, total, temperature = values
        index = index.strip()
        if not index.isdecimal():  # Ignore Intel's CSV header.
            continue
        devices.append(
            GpuDeviceStats(
                vendor=vendor,
                index=index,
                utilization=_parse_metric(utilization),
                memory_used_mib=_parse_metric(used),
                memory_total_mib=_parse_metric(total),
                temperature_c=_parse_metric(temperature),
            )
        )
    return devices


def _parse_metric(raw: str) -> float | None:
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) and value >= 0 else None
