# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared-memory camera transport via iceoryx2.

Provides :class:`SharedCamera` as the public entry point for
multi-process camera sharing. Prefer :meth:`SharedCamera.from_config`
(or YAML), or use ``SharedCamera(camera=...)``. Use
:meth:`SharedCamera.from_publisher` for subscribe-only mode. The publisher
owns the device exclusively — do not keep a direct camera open while sharing.

Requires the ``transport`` extra::

    pip install physicalai[transport]
"""

from __future__ import annotations

from ._shared_camera import SharedCamera

__all__ = ["SharedCamera"]
