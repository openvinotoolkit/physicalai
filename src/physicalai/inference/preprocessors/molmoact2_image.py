# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NumPy image patchification for MolmoAct2 inference."""

from __future__ import annotations

import numpy as np

_IMAGE_NDIM = 4
_NUM_CHANNELS = 3


class MolmoAct2ImageProcessor:
    """Normalize and patchify pre-resized BCHW images."""

    def __init__(
        self,
        *,
        crop_mode: str = "resize",
        size: dict[str, int] | None = None,
        patch_size: int = 14,
        pooling_size: list[int] | tuple[int, int] = (2, 2),
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
    ) -> None:
        """Store image settings and precompute pooling indices."""
        size = size or {"height": 378, "width": 378}
        self.crop_mode = crop_mode
        self.height = int(size["height"])
        self.width = int(size["width"])
        self.patch_size = int(patch_size)
        self.pool_h, self.pool_w = (int(pooling_size[0]), int(pooling_size[1]))
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]
        self._pooling, self.pooled_h, self.pooled_w = self._pooling_indices()

    def __call__(self, images: np.ndarray) -> dict[str, np.ndarray]:
        """Return patches, pooling indices, grids, and crop counts.

        Returns:
            Model-ready image arrays and layout metadata.

        Raises:
            ValueError: If image shape, size, or dtype is unsupported.
            NotImplementedError: If crop mode is not ``resize``.
        """
        images = np.asarray(images)
        if images.ndim != _IMAGE_NDIM or images.shape[1] != _NUM_CHANNELS:
            msg = f"Expected images of shape (M, 3, H, W), got {images.shape}."
            raise ValueError(msg)
        if images.shape[2:] != (self.height, self.width):
            msg = f"Expected images of size {(self.height, self.width)}, got {images.shape[2:]}."
            raise ValueError(msg)
        if self.crop_mode != "resize":
            msg = f"MolmoAct2ImageProcessor only supports crop_mode='resize', got {self.crop_mode!r}."
            raise NotImplementedError(msg)
        if images.dtype not in {np.dtype(np.float16), np.dtype(np.float32)}:
            msg = f"Expected images of dtype float16 or float32, got {images.dtype}."
            raise ValueError(msg)

        count = images.shape[0]
        mean = np.asarray(self.image_mean, dtype=images.dtype).reshape(1, 3, 1, 1)
        std = np.asarray(self.image_std, dtype=images.dtype).reshape(1, 3, 1, 1)
        pixel_values = self._patchify((images - mean) / std)
        pooling = np.tile(self._pooling, (count, 1))
        grid = np.asarray([self.pooled_h, self.pooled_w, 0, 0], dtype=np.int64)
        return {
            "pixel_values": pixel_values,
            "image_token_pooling": pooling,
            "image_grids": np.tile(grid, (count, 1)),
            "image_num_crops": np.ones(count, dtype=np.int64),
        }

    def _patchify(self, pixels: np.ndarray) -> np.ndarray:
        count, channels, height, width = pixels.shape
        patch = self.patch_size
        if height % patch or width % patch:
            msg = f"Image size {(height, width)} must be divisible by patch_size={patch}."
            raise ValueError(msg)
        pixels = pixels.transpose(0, 2, 3, 1)
        pixels = pixels.reshape(count, height // patch, patch, width // patch, patch, channels)
        return pixels.transpose(0, 1, 3, 2, 4, 5).reshape(count, -1, patch * patch * channels)

    def _pooling_indices(self) -> tuple[np.ndarray, int, int]:
        patch_h = self.height // self.patch_size
        patch_w = self.width // self.patch_size
        pooled_h = (patch_h + self.pool_h - 1) // self.pool_h
        pooled_w = (patch_w + self.pool_w - 1) // self.pool_w
        pad_h = pooled_h * self.pool_h - patch_h
        pad_w = pooled_w * self.pool_w - patch_w
        indices = np.arange(patch_h * patch_w, dtype=np.int64).reshape(patch_h, patch_w)
        indices = np.pad(
            indices,
            ((pad_h // 2, (pad_h + 1) // 2), (pad_w // 2, (pad_w + 1) // 2)),
            constant_values=-1,
        )
        indices = indices.reshape(pooled_h, self.pool_h, pooled_w, self.pool_w)
        return indices.transpose(0, 2, 1, 3).reshape(-1, self.pool_h * self.pool_w), pooled_h, pooled_w


__all__ = ["MolmoAct2ImageProcessor"]
