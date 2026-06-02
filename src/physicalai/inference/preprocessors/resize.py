# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor that resizes images to a target resolution."""

from __future__ import annotations

from enum import StrEnum

import cv2
import numpy as np

from physicalai.inference.constants import IMAGES

from .base import Preprocessor


class ResizeMode(StrEnum):
    """Resize strategy for :class:`ResizePreprocessor`."""

    STRETCH = "stretch"
    LETTERBOX = "letterbox"


class ResizePreprocessor(Preprocessor):
    """Resize observation images to a target resolution.

    Args:
        image_resolution: Target (height, width) for images.
        mode: Resize strategy.
            - ``stretch`` distorts to exact target size without padding.
            - ``letterbox`` preserves aspect ratio and pads to exact target.
        pad_value: Fill value used for letterbox padding.
    """

    def __init__(
        self,
        image_resolution: tuple[int, int],
        *,
        mode: ResizeMode | str = ResizeMode.LETTERBOX,
        pad_value: int = 0,
    ) -> None:
        """Initialize the resize preprocessor."""
        super().__init__()
        self._image_resolution = image_resolution
        self._mode = ResizeMode(mode)
        self._pad_value = pad_value

    def __call__(
        self,
        inputs: dict[str, np.ndarray | dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
        """Resize observation images to the target resolution.

        Images may be provided as a single array under the ``images`` key, a
        nested ``{camera: array}`` dict under ``images``, or flat ``images.*``
        keys. ``is_pad`` keys are left untouched.

        Args:
            inputs: Observation dict. Image arrays are expected to have shape
                ``(batch, channels, height, width)``.

        Returns:
            A new dict with the image arrays resized.
        """
        outputs: dict[str, np.ndarray | dict[str, np.ndarray]] = dict(inputs)
        images_value = outputs.get(IMAGES)

        if isinstance(images_value, dict):
            outputs[IMAGES] = {key: self._resize_with_ar_pad(value) for key, value in images_value.items()}
        elif isinstance(images_value, np.ndarray):
            outputs[IMAGES] = self._resize_with_ar_pad(images_value)
        else:
            image_keys = [key for key in outputs if key.startswith(IMAGES) and "is_pad" not in key]
            for key in image_keys:
                value = outputs[key]
                if isinstance(value, np.ndarray):
                    outputs[key] = self._resize_with_ar_pad(value)

        return outputs

    def _resize_with_ar_pad(self, img: np.ndarray) -> np.ndarray:  # noqa: PLR0914
        """Resize an image array to the target resolution.

                Behavior depends on the configured ``mode``:

                - ``stretch``: image is resized directly to the target dimensions.
                - ``letterbox``: image is scaled to fit while preserving aspect ratio,
                    then padded symmetrically to exactly match the target dimensions.

        Args:
            img: Input image array with shape ``(batch, channels, height, width)``.

        Returns:
            Resized image array.

        Raises:
            ValueError: If the input array does not have 4 dimensions
                (batch, channels, height, width).
        """
        img_dim = 4
        if img.ndim != img_dim:
            msg = f"(b,c,h,w) expected, but {img.shape}"
            raise ValueError(msg)

        target_height, target_width = self._image_resolution
        cur_height, cur_width = img.shape[2:]

        if self._mode == ResizeMode.LETTERBOX:
            ratio = max(cur_width / target_width, cur_height / target_height)
            resized_height = max(1, min(int(cur_height / ratio), target_height))
            resized_width = max(1, min(int(cur_width / ratio), target_width))
        else:  # ResizeMode.STRETCH
            resized_height = target_height
            resized_width = target_width

        if (resized_height, resized_width) != (cur_height, cur_width):
            img = self._resize_bchw(img, resized_width, resized_height)

        if self._mode == ResizeMode.STRETCH:
            return img

        pad_height = target_height - resized_height
        pad_width = target_width - resized_width
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        if pad_height == 0 and pad_width == 0:
            return img

        return np.pad(
            img,
            ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            constant_values=self._pad_value,
        )

    @staticmethod
    def _resize_bchw(img: np.ndarray, width: int, height: int) -> np.ndarray:
        """Bilinear resize of a ``(batch, channels, height, width)`` array.

        Args:
            img: Input array in channels-first layout.
            width: Target width.
            height: Target height.

        Returns:
            Resized array in channels-first layout.
        """
        img_hwc = np.transpose(img, (0, 2, 3, 1))  # (B, H, W, C)
        resized = []
        for i in range(img_hwc.shape[0]):
            out = cv2.resize(img_hwc[i], (width, height), interpolation=cv2.INTER_LINEAR)
            if out.ndim == 2:  # noqa: PLR2004
                out = out[:, :, np.newaxis]
            resized.append(out)
        stacked = np.stack(resized, axis=0)  # (B, H, W, C)
        return np.transpose(stacked, (0, 3, 1, 2))  # (B, C, H, W)
