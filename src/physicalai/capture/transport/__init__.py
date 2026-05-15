# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared-memory camera transport via iceoryx2.

Provides :class:`SharedCamera` as the public entry point for
multi-process camera sharing. Use the constructor for auto-spawn mode
(``SharedCamera(camera_type, ...)``) or
:meth:`SharedCamera.from_publisher` for subscribe-only mode.

Requires the ``transport`` extra::

    pip install physicalai[transport]
"""

from __future__ import annotations

from ._shared_camera import SharedCamera

__all__ = ["SharedCamera"]
