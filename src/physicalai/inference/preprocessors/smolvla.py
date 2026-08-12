# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor that resizes images for SmolVLA."""

from __future__ import annotations

import cv2
import numpy as np

from physicalai.inference.constants import IMAGE_MASKS, IMAGES

from .base import Preprocessor

# Dummy camera slots assume RGB, matching the SigLIP vision tower.
_RGB_CHANNELS = 3


class ResizeSmolVLA(Preprocessor):
    """Preprocessor for resizing images for SmolVLA model using numpy operations.

    This preprocessor resizes input images to a specified resolution while maintaining
    aspect ratio through padding. It normalizes the pixel values to the range [-1, 1]
    and generates corresponding image masks.

    Attributes:
        image_resolution (tuple[int, int]): The target resolution for input images
            as (height, width). Defaults to (512, 512).
        image_key_reorder_map (dict[str, int]): Mapping from image key to camera slot index.
        num_cameras (int): Total number of camera slots expected by the model.
    """

    def __init__(
        self,
        image_resolution: tuple[int, int] = (512, 512),
        image_key_reorder_map: dict[str, int] | None = None,
        num_cameras: int = 0,
    ) -> None:
        """Initialize the SmolVLA numpy-based preprocessor.

        Args:
            image_resolution: The target resolution for input images
                as (height, width). Defaults to (512, 512).
            image_key_reorder_map: Optional mapping from source image keys to target
                camera indices used for deterministic ordering. Keys may be given with
                or without the ``images.`` prefix. When the input holds a single unnamed
                image under ``images``, a single-entry map applies to it regardless of
                its key name.
            num_cameras: Total number of camera slots expected by the model. Slots left
                unfilled by the input image keys are filled with masked dummy images
                shaped after the real cameras, or after ``image_resolution`` with a
                single RGB frame when no real camera is present. Values <= 0 keep only
                the input cameras, without any dummy images.

        Raises:
            ValueError: If ``image_key_reorder_map`` contains negative or duplicate slot
                indices.
        """
        super().__init__()
        self.image_resolution = image_resolution
        self.image_key_reorder_map = {
            self._normalize_image_key(key): order for key, order in (image_key_reorder_map or {}).items()
        }
        self.num_cameras = num_cameras

        negative = sorted(key for key, slot in self.image_key_reorder_map.items() if slot < 0)
        if negative:
            msg = f"image_key_reorder_map slot indices must be non-negative, got negative values for {negative}."
            raise ValueError(msg)

        slots = list(self.image_key_reorder_map.values())
        if len(set(slots)) != len(slots):
            duplicates = sorted({slot for slot in slots if slots.count(slot) > 1})
            msg = f"image_key_reorder_map slot indices must be unique, got duplicates {duplicates}."
            raise ValueError(msg)

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process and prepare images for model inference.

        Resizes images with padding, normalizes pixel values to [-1, 1] range,
        and generates corresponding attention masks. Supported image formats:
        - (B, C, H, W) or (B, H, W, C) with float32 values in [0, 1]
        - (B, C, H, W) or (B, H, W, C) with uint8 values in [0, 255]

        Args:
            inputs: Dictionary containing image arrays under the ``images`` key (single
                array or camera-name dict) or under flat ``images.<camera>`` keys.

        Returns:
            Dictionary with processed:
            - IMAGES: Stacked resized images of shape (n_cameras, batch, channels, height, width)
                        with pixel values normalized to [-1, 1].
            - IMAGE_MASKS: Boolean masks of shape (n_cameras, batch) marking which camera
                             slots hold a real image.

        Raises:
            ValueError: If input images have unsupported data types, have an ambiguous
                channel layout, or the camera slots cannot be resolved.
        """
        inputs = dict(inputs)

        target_height, target_width = self.image_resolution
        images_by_key = self._collect_images(inputs)

        img_masks: list[np.ndarray | None] = []
        resized_images: list[np.ndarray | None] = []

        for key in self._camera_slot_layout(list(images_by_key)):
            if key is None:
                resized_images.append(None)
                img_masks.append(None)
                continue

            img = images_by_key[key]
            if img.dtype == np.uint8:
                img_fp32 = img.astype(np.float32) / 255.0
            elif np.issubdtype(img.dtype, np.floating):
                # Replace NaN/Inf before clipping; clip enforces the [0, 1] precondition.
                img_fp32 = np.clip(np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            else:
                msg = f"Unsupported image dtype: {img.dtype}"
                raise ValueError(msg)

            # Heuristic: standard channel counts are {1, 2, 3, 4}; spatial dims are typically larger.
            if img_fp32.ndim == 4 and img_fp32.shape[-1] in {1, 2, 3, 4} and img_fp32.shape[1] in {1, 2, 3, 4}:  # noqa: PLR2004
                msg = (
                    f"ambiguous layout: both dim 1 ({img_fp32.shape[1]}) and dim -1 ({img_fp32.shape[-1]}) "
                    "look like standard channel counts; provide input with spatial dims > 4"
                )
                raise ValueError(msg)
            if img_fp32.ndim == 4 and img_fp32.shape[-1] in {1, 2, 3, 4} and img_fp32.shape[1] not in {1, 2, 3, 4}:  # noqa: PLR2004
                img_fp32 = np.transpose(img_fp32, (0, 3, 1, 2))  # (B, H, W, C) to (B, C, H, W)

            resized_img = self._resize_with_pad(img_fp32, target_width, target_height, pad_value=0)
            resized_img = resized_img * 2.0 - 1.0
            bsize = resized_img.shape[0]
            mask = np.ones(bsize, dtype=np.bool_)
            resized_images.append(resized_img)
            img_masks.append(mask)

        reference_img = next((img for img in resized_images if img is not None), None)
        if reference_img is not None:
            reference_mask = next(mask for mask in img_masks if mask is not None)
        elif resized_images:
            # All slots are dummies, so there is no real frame to copy shape from.
            reference_img = np.empty((1, _RGB_CHANNELS, target_height, target_width), dtype=np.float32)
            reference_mask = np.empty(1, dtype=np.bool_)
        else:
            inputs[IMAGES] = np.empty(0, dtype=np.float32)
            inputs[IMAGE_MASKS] = np.empty(0, dtype=np.bool_)
            return inputs

        inputs[IMAGES] = np.stack(
            [np.full_like(reference_img, -1.0) if img is None else img for img in resized_images],
            axis=0,
        )
        inputs[IMAGE_MASKS] = np.stack(
            [np.zeros_like(reference_mask) if mask is None else mask for mask in img_masks],
            axis=0,
        )

        return inputs

    @staticmethod
    def _normalize_image_key(key: str) -> str:
        """Prefix a bare camera name with ``images.``.

        Args:
            key: Image key or camera name.

        Returns:
            Key in canonical ``images.<camera>`` form, or ``images`` when already exact.
        """
        if key == IMAGES or key.startswith(f"{IMAGES}."):
            return key
        return f"{IMAGES}.{key}"

    @staticmethod
    def _collect_images(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Gather camera images from the input dict, keyed by canonical image key.

        Args:
            inputs: Preprocessor input dict.

        Returns:
            Mapping of image key to image array, in input order.
        """
        images_value = inputs.get(IMAGES)
        if isinstance(images_value, np.ndarray):
            return {IMAGES: images_value}
        if isinstance(images_value, dict):
            return {ResizeSmolVLA._normalize_image_key(name): img for name, img in images_value.items()}
        return {key: value for key, value in inputs.items() if key.startswith(f"{IMAGES}.")}

    def _camera_slot_layout(self, image_keys: list[str]) -> list[str | None]:
        """Resolve the camera slot layout for a set of image keys.

        Args:
            image_keys: Image keys present in the input.

        Returns:
            List of camera slots, where each entry is either an image key or ``None``
            for a slot that must be filled with an empty camera.

        Raises:
            ValueError: If ``image_key_reorder_map`` is set and its keys do not match
                the input image keys exactly, or if the resolved slots do not fit into
                ``num_cameras``. A single-entry map is exempt from the name check when
                the input is one unnamed image.
        """
        if self.image_key_reorder_map:
            # An unnamed single image cannot be matched by name, so a single-entry map applies to it.
            if image_keys == [IMAGES] and len(self.image_key_reorder_map) == 1:
                slot_by_key = {IMAGES: next(iter(self.image_key_reorder_map.values()))}
            elif set(self.image_key_reorder_map) != set(image_keys):
                msg = (
                    "image_key_reorder_map keys must match the input image keys exactly. "
                    f"Expected {sorted(self.image_key_reorder_map)}, got {sorted(image_keys)}."
                )
                raise ValueError(msg)
            else:
                slot_by_key = {key: self.image_key_reorder_map[key] for key in image_keys}
        else:
            slot_by_key = {key: index for index, key in enumerate(image_keys)}

        if self.num_cameras <= 0:
            return sorted(image_keys, key=lambda key: slot_by_key[key])

        if any(slot >= self.num_cameras for slot in slot_by_key.values()):
            msg = (
                f"num_cameras={self.num_cameras} is too small for the resolved camera slots "
                f"{sorted(slot_by_key.values())} of image keys {sorted(image_keys)}."
            )
            raise ValueError(msg)

        layout: list[str | None] = [None] * self.num_cameras
        for key, slot in slot_by_key.items():
            layout[slot] = key
        return layout

    @staticmethod
    def _resize_with_pad(img: np.ndarray, width: int, height: int, pad_value: int = -1) -> np.ndarray:
        """Resize a batch to fit ``width`` x ``height``, padding the top and left edges.

        Args:
            img: Image batch of shape (B, C, H, W).
            width: Target width.
            height: Target height.
            pad_value: Fill value for the padded region.

        Returns:
            Resized batch of shape (B, C, height, width).

        Raises:
            ValueError: If ``img`` is not 4D or has a zero spatial dimension.
        """
        img_dim = 4
        if img.ndim != img_dim:
            msg = f"(b,c,h,w) expected, but {img.shape}"
            raise ValueError(msg)

        cur_height, cur_width = img.shape[2:]

        if cur_height == 0 or cur_width == 0:
            msg = f"Input image has a zero spatial dimension: shape {img.shape}"
            raise ValueError(msg)

        ratio = max(cur_width / width, cur_height / height)
        resized_height = max(1, min(int(cur_height / ratio), height))
        resized_width = max(1, min(int(cur_width / ratio), width))

        # Per-image cv2 bilinear resize (matches F.interpolate align_corners=False)
        batch = []
        for i in range(img.shape[0]):
            # cv2.resize expects (H, W, C) so transpose from (C, H, W)
            hwc = np.transpose(img[i], (1, 2, 0))
            resized_hwc = cv2.resize(hwc, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
            batch.append(np.transpose(resized_hwc, (2, 0, 1)))
        resized_img = np.stack(batch, axis=0)

        pad_height = max(0, int(height - resized_height))
        pad_width = max(0, int(width - resized_width))

        # pad on left and top of image
        if pad_height > 0 or pad_width > 0:
            padded = np.full(
                (resized_img.shape[0], resized_img.shape[1], resized_height + pad_height, resized_width + pad_width),
                fill_value=pad_value,
                dtype=resized_img.dtype,
            )
            padded[:, :, pad_height:, pad_width:] = resized_img
            return padded
        return resized_img
