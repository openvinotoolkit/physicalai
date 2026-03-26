# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""V4L2 camera backend."""

from __future__ import annotations

from ._camera import V4L2Camera
from ._discover import discover_v4l2

__all__ = ["V4L2Camera", "discover_v4l2"]
