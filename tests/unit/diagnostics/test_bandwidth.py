# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from physicalai.diagnostics.bandwidth import format_ceiling, frame_payload_bytes, payload_demand
from physicalai.diagnostics.models import VideoMode


def test_uncompressed_demand_uses_wire_format_and_reported_padding() -> None:
    mode = VideoMode(width=640, height=480, fps=30, pixel_format="YUYV")
    assert frame_payload_bytes(mode) == 614400
    assert payload_demand(mode) == 18_432_000
    mode.frame_bytes = 620000
    assert payload_demand(mode) == 18_600_000
    assert format_ceiling(480) == "60 MB/s"  # Signaling bound, not an available payload budget.


@pytest.mark.parametrize("format", ["MJPG", "H264", "unrecognized"])
def test_compressed_buffer_capacity_is_never_used_as_a_payload_measurement(format: str) -> None:
    mode = VideoMode(width=1920, height=1080, fps=30, pixel_format=format, frame_bytes=8_000_000)
    assert frame_payload_bytes(mode) is None
    assert payload_demand(mode) is None


def test_format_derived_estimate_does_not_use_decoded_rgb_size() -> None:
    mode = VideoMode(width=640, height=480, fps=30, pixel_format="Z16")
    assert payload_demand(mode) == 640 * 480 * 2 * 30
