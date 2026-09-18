# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor that resizes images to a target resolution."""

from __future__ import annotations

from enum import StrEnum

import cv2
import numpy as np

from physicalai.inference.constants import IMAGES

from .base import Preprocessor
from .image_layout import ImageLayout


class ResizeMode(StrEnum):
    """Resize strategy for :class:`ResizePreprocessor`."""

    STRETCH = "stretch"
    LETTERBOX = "letterbox"


class ResizePreprocessor(Preprocessor):
    """Resize observation images to a target resolution.

    Args:
        image_resolution: Target (height, width) for images.
        image_layout: Input axis order, ``BCHW`` or ``BHWC``. All images passed
            to this preprocessor must use this layout.
        mode: Resize strategy.
            - ``stretch`` distorts to exact target size without padding.
            - ``letterbox`` preserves aspect ratio and pads to exact target.
        pad_value: Fill value used for letterbox padding.
    """

    def __init__(
        self,
        image_resolution: tuple[int, int],
        *,
        image_layout: ImageLayout | str,
        mode: ResizeMode | str = ResizeMode.LETTERBOX,
        pad_value: float = 0,
    ) -> None:
        """Initialize the resize preprocessor with validated layout and mode."""
        super().__init__()
        self._image_resolution = image_resolution
        self._image_layout = ImageLayout(image_layout)
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

        Image arrays may be in channels-first ``(batch, channels, height,
        width)`` or channels-last ``(batch, height, width, channels)`` layout
        as specified by ``image_layout``,
        with ``uint8`` (normalized to ``float32`` in [0, 1]) or floating-point
        values. The output is always in channels-first layout and ``float32``.

        Args:
            inputs: Observation dict.

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

        Accepts channels-first ``(B,C,H,W)`` or
        channels-last ``(B,H,W,C)`` arrays with ``uint8``
        or floating-point values. The output is always in channels-first layout
        and ``fp32``.

        Args:
            img: Input image array in the configured ``image_layout``.

        Returns:
            Resized image array in channels-first layout.

        Raises:
            ValueError: If the input array does not have 4 dimensions, or if it
                has an unsupported dtype (not ``uint8`` or floating point),
                or if the ``pad_value`` is out of range for ``uint8`` inputs,
                or if the input image has a zero spatial dimension.
        """
        img_dim = 4
        if img.ndim != img_dim:
            msg = f"4D image in {self._image_layout} layout expected, but got shape {img.shape}"
            raise ValueError(msg)

        if img.dtype == np.uint8 and self._pad_value > np.iinfo(np.uint8).max:
            msg = f"pad_value {self._pad_value} is out of range for uint8 inputs"
            raise ValueError(msg)

        if np.issubdtype(img.dtype, np.floating):
            img = img.astype(np.float32)
        elif img.dtype != np.uint8:
            msg = f"Unsupported image dtype: {img.dtype}"
            raise ValueError(msg)

        if self._image_layout == ImageLayout.BCHW:
            img = np.transpose(img, (0, 2, 3, 1))  # (B, C, H, W) -> (B, H, W, C)

        target_height, target_width = self._image_resolution
        cur_height, cur_width = img.shape[1:3]

        if cur_height == 0 or cur_width == 0:
            msg = f"Input image has a zero spatial dimension: shape {img.shape}"
            raise ValueError(msg)

        if self._mode == ResizeMode.LETTERBOX:
            ratio = max(cur_width / target_width, cur_height / target_height)
            resized_height = max(1, min(int(cur_height / ratio), target_height))
            resized_width = max(1, min(int(cur_width / ratio), target_width))
        else:  # ResizeMode.STRETCH
            resized_height = target_height
            resized_width = target_width

        if (resized_height, resized_width) != (cur_height, cur_width):
            img = self._resize_bhwc(img, resized_width, resized_height)

        if self._mode == ResizeMode.LETTERBOX:
            pad_height = target_height - resized_height
            pad_width = target_width - resized_width
            if pad_height > 0 or pad_width > 0:
                pad_top = pad_height // 2
                pad_bottom = pad_height - pad_top
                pad_left = pad_width // 2
                pad_right = pad_width - pad_left
                img = np.pad(
                    img,
                    ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    constant_values=self._pad_value,
                )

        output_transposed = np.transpose(img, (0, 3, 1, 2))  # (B, H, W, C) -> (B, C, H, W)

        if output_transposed.dtype == np.uint8:
            output_transposed = output_transposed.astype(np.float32) / 255.0

        return output_transposed

    @staticmethod
    def _resize_bhwc(img: np.ndarray, width: int, height: int) -> np.ndarray:
        """Bilinear resize of a ``(batch, height, width, channels)`` array.

        Args:
            img: Input array in channels-last layout.
            width: Target width.
            height: Target height.

        Returns:
            Resized array in channels-last layout.
        """
        resized = []
        for i in range(img.shape[0]):
            out = cv2.resize(img[i], (width, height), interpolation=cv2.INTER_LINEAR)
            if out.ndim == 2:  # noqa: PLR2004
                out = out[:, :, np.newaxis]
            resized.append(out)
        return np.stack(resized, axis=0)  # (B, H, W, C)
