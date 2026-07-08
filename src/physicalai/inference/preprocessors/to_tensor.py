# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor that converts uint8 images to float32 in the [0, 1] range."""

from __future__ import annotations

import numpy as np

from physicalai.inference.constants import IMAGES

from .base import Preprocessor


class ToFloatTensorPreprocessor(Preprocessor):
    """Convert observation images to ``float32`` tensor in the [0, 1] range.

    ``uint8`` image arrays are divided by 255.0 and cast to ``float32``.
    Non-``uint8`` image arrays keep their dtype unchanged.

    Image arrays may be in channels-first ``(batch, channels, height, width)``
    or channels-last ``(batch, height, width, channels)`` layout. Channels-last
    arrays are transposed to channels-first so the output is always in
    channels-first layout.
    """

    def __call__(
        self,
        inputs: dict[str, np.ndarray | dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
        """Convert observation images to ``float32`` tensor in the [0, 1] range.

        Images may be provided as a single array under the ``images`` key, a
        nested ``{camera: array}`` dict under ``images``, or flat ``images.*``
        keys. ``is_pad`` keys are left untouched.

        Image arrays in channels-last ``(batch, height, width, channels)``
        layout are transposed to channels-first ``(batch, channels, height,
        width)``; the output is always in channels-first layout.

        Args:
            inputs: Observation dict.

        Returns:
            A new dict with image arrays converted to ``float32`` (when
            ``uint8``) and in channels-first layout.
        """
        outputs: dict[str, np.ndarray | dict[str, np.ndarray]] = dict(inputs)
        images_value = outputs.get(IMAGES)

        if isinstance(images_value, dict):
            outputs[IMAGES] = {key: self._convert(value) for key, value in images_value.items()}
        elif isinstance(images_value, np.ndarray):
            outputs[IMAGES] = self._convert(images_value)
        else:
            image_keys = [key for key in outputs if key.startswith(IMAGES) and "is_pad" not in key]
            for key in image_keys:
                value = outputs[key]
                if isinstance(value, np.ndarray):
                    outputs[key] = self._convert(value)

        return outputs

    @classmethod
    def _convert(cls, img: np.ndarray) -> np.ndarray:
        """Convert an image array to ``float32`` in [0, 1] and channels-first.

        Args:
            img: Input image array.

        Returns:
            The image scaled to ``float32`` [0, 1] when the input is ``uint8``,
            and transposed to channels-first layout when it is channels-last.
        """
        return cls._to_chw(cls._to_float(img))

    @staticmethod
    def _to_float(img: np.ndarray) -> np.ndarray:
        """Convert a ``uint8`` image array to ``float32`` in [0, 1].

        Args:
            img: Input image array.

        Returns:
            The image as ``float32`` scaled to [0, 1] if the input is
            ``uint8``, otherwise the array unchanged.

        Raises:
            ValueError: If the input array has an unsupported dtype (not ``uint8`` or floating point).
        """
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        if np.issubdtype(img.dtype, np.floating):
            img = img.astype(np.float32)
        else:
            msg = f"Unsupported image dtype: {img.dtype}"
            raise ValueError(msg)

        return img

    @staticmethod
    def _to_chw(img: np.ndarray) -> np.ndarray:
        """Transpose a channels-last image array to channels-first layout.

        Only 4D arrays are considered. An array is treated as channels-last
        ``(batch, height, width, channels)`` when its last dimension is 3 and
        its second dimension is not 3; such arrays are transposed to
        channels-first ``(batch, channels, height, width)``. All other arrays
        are returned unchanged.

        Args:
            img: Input image array.

        Returns:
            The image in channels-first layout.
        """
        img_dim = 4
        rgb_channels = 3
        if img.ndim != img_dim:
            return img
        channels_last = img.shape[-1] == rgb_channels and img.shape[1] != rgb_channels
        if channels_last:
            return np.transpose(img, (0, 3, 1, 2))  # (B, H, W, C) -> (B, C, H, W)
        return img
