# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor that builds VLA-Adapter's six-channel-per-camera image tensor.

VLA-Adapter's vision backbone fuses two towers (DINOv2 and SigLIP) that were
pretrained with *different* pixel statistics. Rather than run the towers on
separately normalised inputs, the reference implementation feeds each camera as
**six** channels — the same resized pixels normalised twice, three channels per
tower — and splits them inside the backbone. With two cameras the model input is
therefore ``(batch, 12, 224, 224)``.

Neither channel duplication nor camera concatenation can be expressed with the
generic ``resize`` and ``normalize`` components, which is why this exists.
It mirrors ``VLAAdapterPreprocessor`` in ``physicalai-train``; the two are
pinned against each other by a parity test in that distribution.
"""

from __future__ import annotations

import cv2
import numpy as np

from physicalai.inference.constants import IMAGES

from .base import Preprocessor

# An image array carries a leading time axis when the producer applied
# observation delta timestamps: (B, T, C, H, W) rather than (B, C, H, W).
_IMAGE_DIMS_WITH_TIME = 5

# Channels in a single RGB camera frame, and per fused camera slot (one RGB
# copy per tower).
_CHANNELS_PER_TOWER = 3
_NUM_VISION_TOWERS = 2


class VLAAdapterPreprocessor(Preprocessor):
    """Resize camera frames and stack two per-tower normalisations per camera.

    Reads every ``images`` / ``images.*`` entry, resizes each to
    ``image_resolution``, normalises it once with the primary tower's statistics
    and once with the secondary's, concatenates the two copies to six channels,
    then concatenates the cameras. The per-camera keys are removed and the
    result is written back under ``images``.

    Camera order matters — it decides token order in the fused sequence — so it
    is resolved the same way the torch preprocessor resolves it: sorted key
    order, or ``image_key_reorder_map`` when one is supplied.

    Args:
        image_resolution: Target ``(height, width)``.
        primary_mean: Per-channel mean of the primary (DINOv2) tower.
        primary_std: Per-channel standard deviation of the primary tower.
        secondary_mean: Per-channel mean of the secondary (SigLIP) tower.
        secondary_std: Per-channel standard deviation of the secondary tower.
        image_key_reorder_map: Image key to camera slot; empty means sorted order.
        num_cameras: Total camera slots. Missing slots are zero-filled and extra
            cameras dropped; ``<= 0`` keeps exactly the cameras supplied.
    """

    def __init__(
        self,
        image_resolution: tuple[int, int] = (224, 224),
        primary_mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        primary_std: tuple[float, ...] = (0.229, 0.224, 0.225),
        secondary_mean: tuple[float, ...] = (0.5, 0.5, 0.5),
        secondary_std: tuple[float, ...] = (0.5, 0.5, 0.5),
        image_key_reorder_map: dict[str, int] | None = None,
        num_cameras: int = 0,
    ) -> None:
        """Initialize the VLA-Adapter preprocessor."""
        super().__init__()
        self._image_resolution = tuple(image_resolution)
        self._primary_mean = self._as_channel_vector(primary_mean)
        self._primary_std = self._as_channel_vector(primary_std)
        self._secondary_mean = self._as_channel_vector(secondary_mean)
        self._secondary_std = self._as_channel_vector(secondary_std)
        self._image_key_reorder_map = {
            key if key.startswith(f"{IMAGES}.") else f"{IMAGES}.{key}": order
            for key, order in (image_key_reorder_map or {}).items()
        }
        self._num_cameras = num_cameras

    @staticmethod
    def _as_channel_vector(values: tuple[float, ...]) -> np.ndarray:
        """Shape per-channel statistics for broadcasting over ``(B, C, H, W)``.

        Args:
            values: One value per channel.

        Returns:
            A ``(1, C, 1, 1)`` float32 array.
        """
        return np.asarray(values, dtype=np.float32).reshape(1, -1, 1, 1)

    def _ordered_image_keys(self, batch_img_keys: list[str]) -> list[str]:
        """Resolve deterministic camera ordering.

        Args:
            batch_img_keys: Image keys present in the batch.

        Returns:
            Image keys in camera-slot order.

        Raises:
            ValueError: If ``image_key_reorder_map`` does not match the keys.
        """
        if not self._image_key_reorder_map:
            return sorted(batch_img_keys)

        if set(self._image_key_reorder_map) != set(batch_img_keys):
            msg = (
                "image_key_reorder_map keys must match the batch image keys exactly. "
                f"Expected {sorted(self._image_key_reorder_map)}, got {sorted(batch_img_keys)}."
            )
            raise ValueError(msg)
        return sorted(batch_img_keys, key=lambda key: self._image_key_reorder_map[key])

    def _resize(self, view: np.ndarray) -> np.ndarray:
        """Resize a channels-first batch to the target resolution.

        Uses bicubic interpolation to match the torch reference's
        ``F.interpolate(mode="bicubic")``.

        Args:
            view: ``(B, C, H, W)`` float32 images.

        Returns:
            ``(B, C, *image_resolution)`` float32 images.
        """
        target_h, target_w = self._image_resolution
        if view.shape[2:] == (target_h, target_w):
            return view

        # cv2 works on (H, W, C), one sample at a time.
        resized = [
            cv2.resize(
                np.transpose(sample, (1, 2, 0)),
                (target_w, target_h),
                interpolation=cv2.INTER_CUBIC,
            )
            for sample in view
        ]
        return np.transpose(np.stack(resized, axis=0), (0, 3, 1, 2)).astype(np.float32)

    def _stack_view(self, view: np.ndarray) -> np.ndarray:
        """Resize one camera and stack its two normalised copies.

        Args:
            view: ``(B, 3, H, W)``, or ``(B, T, 3, H, W)`` when a time axis is
                present.

        Returns:
            ``(B, 6, *image_resolution)`` float32.
        """
        if view.ndim == _IMAGE_DIMS_WITH_TIME:
            view = view[:, -1]

        if view.dtype == np.uint8:
            view = view.astype(np.float32) / 255.0
        elif view.dtype != np.float32:
            view = view.astype(np.float32)

        resized = self._resize(view)
        primary = (resized - self._primary_mean) / self._primary_std
        secondary = (resized - self._secondary_mean) / self._secondary_std
        return np.concatenate([primary, secondary], axis=1)

    def __call__(
        self,
        inputs: dict[str, np.ndarray | dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
        """Build the fused multi-camera image tensor.

        Accepts cameras as flat ``images.<name>`` keys, as a nested dict under
        ``images``, or as a single stacked array under ``images``. ``is_pad``
        entries are ignored and left untouched.

        Args:
            inputs: Observation dict.

        Returns:
            A new dict whose ``images`` entry is
            ``(B, 6 * num_cameras, *image_resolution)``, with the per-camera
            keys removed. Returned unchanged when it carries no images.
        """
        outputs = dict(inputs)

        flat_keys = [key for key in outputs if key.startswith(f"{IMAGES}.") and "is_pad" not in key]
        if flat_keys:
            ordered = self._ordered_image_keys(flat_keys)
            views = [outputs[key] for key in ordered]
            for key in flat_keys:
                outputs.pop(key, None)
        else:
            images_value = outputs.get(IMAGES)
            if images_value is None:
                return outputs
            if isinstance(images_value, dict):
                lookup = {
                    key if key.startswith(f"{IMAGES}.") else f"{IMAGES}.{key}": value
                    for key, value in images_value.items()
                    if "is_pad" not in key
                }
                views = [lookup[key] for key in self._ordered_image_keys(list(lookup))]
            elif images_value.ndim == _IMAGE_DIMS_WITH_TIME:
                # A single array holding several cameras on the second axis.
                views = list(np.moveaxis(images_value, 1, 0))
            else:
                views = [images_value]

        stacked = [self._stack_view(np.asarray(view)) for view in views]

        if self._num_cameras > 0:
            while len(stacked) < self._num_cameras:
                stacked.append(np.zeros_like(stacked[0]))
            stacked = stacked[: self._num_cameras]

        outputs[IMAGES] = np.concatenate(stacked, axis=1)
        return outputs
