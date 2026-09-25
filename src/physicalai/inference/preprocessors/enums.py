# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Enums for inference preprocessors."""

from enum import StrEnum


class ImageLayout(StrEnum):
    """Axis order of a batched image array.

    Attributes:
        BCHW: Batch, channels, height, width (channels-first).
        BHWC: Batch, height, width, channels (channels-last).
    """

    BCHW = "BCHW"
    BHWC = "BHWC"
