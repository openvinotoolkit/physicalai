# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from typing import Any
from unittest.mock import patch

import pytest

from physicalai.runtime.callbacks.gpu_stats import collect_gpu_stats

NVIDIA_CSV = "0, 65, 2048, 8192, 58\n1, N/A, [Not Supported], 16384, N/A\n"
INTEL_CSV = (
    "Timestamp, DeviceId, GPU Utilization (%), GPU Core Temperature (Celsius Degree), GPU Memory Used (MiB)\n"
    "10:00:00.000, 0, 30.5, 45, 512\n"
    "10:00:00.000, 1, 0, N/A, 1024\n"
)


@pytest.fixture()
def vendor_tools() -> Any:
    outputs = {"nvidia-smi": NVIDIA_CSV, "xpu-smi": INTEL_CSV}

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        tool = command[0].rsplit("/", 1)[-1]
        return subprocess.CompletedProcess(command, 0, stdout=outputs[tool])

    with (
        patch(
            "physicalai.runtime.callbacks.gpu_stats.shutil.which", side_effect=lambda tool: f"/tools/{tool}"
        ) as which,
        patch("physicalai.runtime.callbacks.gpu_stats.subprocess.run", side_effect=run) as query,
    ):
        yield outputs, which, query


def test_collects_all_intel_and_nvidia_devices(vendor_tools: Any) -> None:
    _, _, query = vendor_tools
    sample = collect_gpu_stats()
    md = sample.rows
    assert sample.utilization == {"nvidia_0": 65.0, "intel_0": 30.5, "intel_1": 0.0}
    assert "**NVIDIA GPU 0**" in md
    assert "**NVIDIA GPU 1**" in md
    assert "**Intel GPU 0**" in md
    assert "**Intel GPU 1**" in md
    assert "| Utilization | 65% |" in md
    assert "| Memory | 2048 MiB / 8192 MiB |" in md
    assert "| Temperature | 58 °C |" in md
    assert "| Utilization | 30.5% |" in md
    assert "| Memory | 512 MiB |" in md
    assert "| Temperature | 45 °C |" in md
    assert "| Utilization | N/A |" in md
    assert "| Memory | N/A / 16384 MiB |" in md
    assert "| Temperature | N/A |" in md
    assert query.call_count == 2
    for call in query.call_args_list:
        assert call.kwargs["timeout"] == 3
        assert call.kwargs["check"] is True
        assert call.kwargs.get("shell", False) is False
    assert query.call_args_list[1].args[0] == ["/tools/xpu-smi", "dump", "-d", "-1", "-m", "0,3,18", "-n", "1"]


def test_missing_tools_do_not_launch_subprocesses(vendor_tools: Any) -> None:
    _, which, query = vendor_tools
    which.side_effect = None
    which.return_value = None
    sample = collect_gpu_stats()
    assert "No Intel/NVIDIA monitoring tool found" in sample.rows
    assert sample.utilization == {}
    query.assert_not_called()


@pytest.mark.parametrize("tool", ["nvidia-smi", "xpu-smi"])
def test_each_vendor_works_without_the_other_tool(vendor_tools: Any, tool: str) -> None:
    _, which, query = vendor_tools
    which.side_effect = lambda candidate: f"/tools/{candidate}" if candidate == tool else None
    md = collect_gpu_stats().rows
    assert ("NVIDIA GPU" in md) is (tool == "nvidia-smi")
    assert ("Intel GPU" in md) is (tool == "xpu-smi")
    query.assert_called_once()


@pytest.mark.parametrize(
    "error",
    [PermissionError("denied"), subprocess.TimeoutExpired("query", 3), subprocess.CalledProcessError(1, "query")],
)
def test_query_failure_does_not_hide_the_other_vendor(vendor_tools: Any, error: Exception) -> None:
    _, _, query = vendor_tools
    original = query.side_effect

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[0].endswith("nvidia-smi"):
            raise error
        return original(command, **kwargs)

    query.side_effect = run
    sample = collect_gpu_stats()
    md = sample.rows
    assert all(not key.startswith("nvidia") for key in sample.utilization)
    assert "| NVIDIA | Unavailable" in md
    assert "**Intel GPU 0**" in md


@pytest.mark.parametrize("output", ["", "unexpected output\n", '0, "unterminated\n', "bad|index, 10, 20, 30, 40\n"])
def test_malformed_output_is_unavailable(vendor_tools: Any, output: str) -> None:
    outputs, _, _ = vendor_tools
    outputs["nvidia-smi"] = output
    md = collect_gpu_stats().rows
    assert "| NVIDIA | Unavailable" in md
    assert "**Intel GPU 0**" in md


@pytest.mark.parametrize("invalid", ["N/A", "nan", "inf", "-1", "[Not Supported]"])
def test_unsupported_or_invalid_metrics_are_not_reported_as_zero(vendor_tools: Any, invalid: str) -> None:
    outputs, _, _ = vendor_tools
    outputs["nvidia-smi"] = f"0, {invalid}, {invalid}, 8192, {invalid}\n"
    sample = collect_gpu_stats()
    md = sample.rows
    assert "nvidia_0" not in sample.utilization
    assert "| Utilization | N/A |" in md
    assert "| Memory | N/A / 8192 MiB |" in md
    assert "| Temperature | N/A |" in md
