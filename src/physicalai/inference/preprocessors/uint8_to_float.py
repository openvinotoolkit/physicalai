# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor that converts uint8 images to float32 in the [0, 1] range."""

from __future__ import annotations

import numpy as np

from physicalai.inference.constants import IMAGES

from .base import Preprocessor


class Uint8ToFloatPreprocessor(Preprocessor):
    """Convert ``uint8`` observation images to ``float32`` in the [0, 1] range.

    ``uint8`` image arrays are divided by 255.0 and cast to ``float32``.
    Non-``uint8`` image arrays are returned unchanged.
    """

    def __call__(
        self,
        inputs: dict[str, np.ndarray | dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
        """Convert observation images from ``uint8`` to ``float32`` in [0, 1].

        Images may be provided as a single array under the ``images`` key, a
        nested ``{camera: array}`` dict under ``images``, or flat ``images.*``
        keys. ``is_pad`` keys are left untouched.

        Args:
            inputs: Observation dict.

        Returns:
            A new dict with ``uint8`` image arrays converted to ``float32``.
        """
        outputs: dict[str, np.ndarray | dict[str, np.ndarray]] = dict(inputs)
        images_value = outputs.get(IMAGES)

        if isinstance(images_value, dict):
            outputs[IMAGES] = {key: self._to_float(value) for key, value in images_value.items()}
        elif isinstance(images_value, np.ndarray):
            outputs[IMAGES] = self._to_float(images_value)
        else:
            image_keys = [key for key in outputs if key.startswith(IMAGES) and "is_pad" not in key]
            for key in image_keys:
                value = outputs[key]
                if isinstance(value, np.ndarray):
                    outputs[key] = self._to_float(value)

        return outputs

    @staticmethod
    def _to_float(img: np.ndarray) -> np.ndarray:
        """Convert a ``uint8`` image array to ``float32`` in [0, 1].

        Args:
            img: Input image array.

        Returns:
            The image as ``float32`` scaled to [0, 1] if the input is
            ``uint8``, otherwise the array unchanged.
        """
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        return img
