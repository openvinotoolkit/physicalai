# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Frame dataclass — the universal return type for all read operations."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Frame:
    """A captured image with metadata.

    Every :meth:`~physicalai.capture.camera.Camera.read` call returns a
    ``Frame``. The frozen dataclass prevents accidental mutation of
    metadata; the underlying ``data`` buffer is still mutable.

    Attributes:
        data: Image array. ``(H, W, 3)`` uint8 for colour,
            ``(H, W)`` uint8 for grayscale, ``(H, W)`` uint16 for depth.
        timestamp: Backend-local ``time.monotonic()`` acquisition time.
            For IP streams this is decode/conversion completion, not source
            capture time; network buffering is not represented. SharedCamera
            preserves the publisher's timestamp, including on cached reads.
        sequence: Monotonically increasing counter per connection (0, 1, 2, …).
            Gaps indicate skipped local frames, not necessarily source drops.
    """

    data: NDArray[np.uint8] | NDArray[np.uint16]
    timestamp: float
    sequence: int
