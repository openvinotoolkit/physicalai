# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor that resizes images to a target resolution."""

from __future__ import annotations

import numpy as np

from .base import Preprocessor


class ResizePreprocessor(Preprocessor):
    """Resize observation images to a target resolution.

    Args:
        image_resolution: Target (height, width) for images.
        padding: Whether to pad images to reach the target resolution.
        keep_aspect_ratio: Whether to preserve the aspect ratio when resizing.
    """

    def __init__(
        self,
        image_resolution: tuple[int, int],
        padding: bool,
        keep_aspect_ratio: bool,
    ) -> None:
        """Initialize the resize preprocessor."""
        super().__init__()
        self._image_resolution = image_resolution
        self._padding = padding
        self._keep_aspect_ratio = keep_aspect_ratio

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Resize observation images to the target resolution."""
        raise NotImplementedError
