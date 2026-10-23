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
class GpuStatsSample:
    """GPU table rows and numeric utilization samples keyed by vendor/device ID."""

    rows: str
    utilization: dict[str, float] = field(default_factory=dict)


def collect_gpu_stats() -> GpuStatsSample:
    """Collect table rows and numeric samples, tolerating missing tools/counters.

    This performs bounded subprocess I/O and must run outside the control thread.

    Returns:
        Table rows (or availability messages) and supported utilization values.
    """
    rows = []
    utilization = {}
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
            parsed = _parse_gpu_stats(vendor, result.stdout)
        except (OSError, subprocess.SubprocessError, csv.Error):
            logger.debug("{} GPU counters unavailable", vendor)
            parsed = GpuStatsSample("")
        rows.append(parsed.rows or f"| {vendor} | Unavailable (tool, driver, or permissions) |")
        utilization.update(parsed.utilization)
    return GpuStatsSample("\n".join(rows) or "| Status | No Intel/NVIDIA monitoring tool found |", utilization)


def _parse_gpu_stats(vendor: str, output: str) -> GpuStatsSample:
    rows = []
    samples = {}
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
        value = _parse_metric(utilization)
        if value is not None:
            samples[f"{vendor.lower()}_{index}"] = value
        memory = _format_metric(used, " MiB")
        if vendor == "NVIDIA":
            memory += f" / {_format_metric(total, ' MiB')}"
        rows.extend([
            f"| **{vendor} GPU {index}** | |",
            f"| Utilization | {_format_metric(utilization, '%')} |",
            f"| Memory | {memory} |",
            f"| Temperature | {_format_metric(temperature, ' °C')} |",
        ])
    return GpuStatsSample("\n".join(rows), samples)


def _parse_metric(raw: str) -> float | None:
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _format_metric(raw: str, unit: str) -> str:
    value = _parse_metric(raw)
    return f"{value:g}{unit}" if value is not None else "N/A"
