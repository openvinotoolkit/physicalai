# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor that resizes images for SmolVLA."""

from __future__ import annotations

import cv2
import numpy as np

from physicalai.inference.constants import IMAGE_MASKS, IMAGES

from .base import Preprocessor


class ResizeSmolVLA(Preprocessor):
    """Preprocessor for resizing images for SmolVLA model using numpy operations.

    This preprocessor resizes input images to a specified resolution while maintaining
    aspect ratio through padding. It normalizes the pixel values to the range [-1, 1]
    and generates corresponding image masks.

    Attributes:
        image_resolution (tuple[int, int]): The target resolution for input images
            as (height, width). Defaults to (512, 512).
    """

    def __init__(
        self,
        image_resolution: tuple[int, int] = (512, 512),
        image_key_reorder_map: dict[str, int] | None = None,
        num_cameras: int = 0,
    ) -> None:
        """Initialize the SmolVLA numpy-based preprocessor.

        Args:
            image_resolution (tuple[int, int]): The target resolution for input images
                as (height, width). Defaults to (512, 512).
            image_key_reorder_map (dict[str, int] | None): Maps input image key to camera
                slot index for reordering. Exported by physical-ai-studio when the policy
                uses a specific camera layout. ``None`` means preserve natural input order.
            num_cameras (int): Total number of camera slots. When > 0, the resolved image
                list has exactly this length, with ``None`` for unoccupied slots.
        """
        super().__init__()
        self.image_resolution = image_resolution
        # Normalise to bare keys so both "wrist" and "images.wrist" map entries match flat runtime keys.
        self._image_key_reorder_map: dict[str, int] | None = (
            {self._bare_key(k): v for k, v in image_key_reorder_map.items()} if image_key_reorder_map else None
        )
        self._num_cameras = num_cameras

    @staticmethod
    def _bare_key(key: str) -> str:
        """Strip the ``images.`` prefix so bare and prefixed keys match the same slot.

        Returns:
            Key with ``images.`` prefix removed, or the original key unchanged.
        """
        return key.removeprefix(f"{IMAGES}.")

    def _resolve_image_order(self, img_keys: list[str]) -> list[str | None]:
        """Return keys sorted by camera slot; None slots filled when num_cameras > 0.

        Raises:
            ValueError: If a key is absent from the map, a slot index is out of range,
                ``num_cameras`` is smaller than the number of provided keys, or two keys
                map to the same slot.
        """
        bare_keys = [self._bare_key(k) for k in img_keys]
        slot_by_key = self._image_key_reorder_map or {k: i for i, k in enumerate(bare_keys)}
        missing = [img_keys[i] for i, k in enumerate(bare_keys) if k not in slot_by_key]
        if missing:
            msg = f"Missing slot mapping for image keys: {missing}"
            raise ValueError(msg)
        if self._num_cameras > 0:
            if self._num_cameras < len(img_keys):
                msg = f"num_cameras ({self._num_cameras}) is smaller than provided image count ({len(img_keys)})"
                raise ValueError(msg)
            layout: list[str | None] = [None] * self._num_cameras
            for orig_key, bare_key in zip(img_keys, bare_keys, strict=False):
                slot = int(slot_by_key[bare_key])
                if slot < 0 or slot >= self._num_cameras:
                    msg = (
                        f"Camera slot index {slot} for key {orig_key!r} "
                        f"is out of range for num_cameras={self._num_cameras}"
                    )
                    raise ValueError(msg)
                if layout[slot] is not None:
                    msg = f"Duplicate camera slot index {slot} for keys {layout[slot]!r} and {orig_key!r}"
                    raise ValueError(msg)
                layout[slot] = orig_key
            return layout
        return sorted(img_keys, key=lambda k: slot_by_key[self._bare_key(k)])

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process and prepare images for model inference.

        Resizes images with padding, normalizes pixel values to [-1, 1] range,
        and generates corresponding attention masks. Supported image formats:
        - (B, C, H, W) or (B, H, W, C) with float32 values in [0, 1]
        - (B, C, H, W) or (B, H, W, C) with uint8 values in [0, 255]

        Args:
            inputs: Dictionary containing IMAGES key with numpy array(s) of shape
                    (height, width, channels) or list of such arrays.

        Returns:
            Dictionary with processed:
            - IMAGES: Stacked resized images of shape (batch_size, height, width, channels)
                        with pixel values normalized to [-1, 1].
            - IMAGE_MASKS: Boolean masks of shape (batch_size, height, width) indicating
                             valid image regions (all ones for padded images).

        Raises:
            ValueError: If input images have unsupported data types.
        """
        inputs = dict(inputs)

        if IMAGES in inputs and isinstance(inputs[IMAGES], np.ndarray):
            images = [inputs[IMAGES]]
        elif IMAGES in inputs and isinstance(inputs[IMAGES], dict):
            img_keys = list(inputs[IMAGES].keys())
            ordered = self._resolve_image_order(img_keys)
            images = [inputs[IMAGES][k] if k is not None else None for k in ordered]
        else:
            img_keys = [key for key in inputs if key.startswith(IMAGES)]
            if len(img_keys) == 1:
                images = [inputs[img_keys[0]]]
            else:
                ordered = self._resolve_image_order(img_keys)
                images = [inputs[k] if k is not None else None for k in ordered]

        img_masks = []
        resized_images = []

        for img in images:
            if img is None:
                # empty camera slot — filled with black image and zero mask
                bsize = next((int(a.shape[0]) for a in images if a is not None), 1)
                h, w = self.image_resolution
                resized_images.append(np.full((bsize, 3, h, w), -1.0, dtype=np.float32))
                img_masks.append(np.zeros(bsize, dtype=np.bool_))
                continue
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

            resized_img = self._resize_with_pad(
                img_fp32, self.image_resolution[1], self.image_resolution[0], pad_value=0
            )
            resized_img = resized_img * 2.0 - 1.0
            bsize = resized_img.shape[0]
            mask = np.ones(bsize, dtype=np.bool_)
            resized_images.append(resized_img)
            img_masks.append(mask)

        inputs[IMAGES] = np.stack(resized_images, axis=0)
        inputs[IMAGE_MASKS] = np.stack(img_masks, axis=0)

        return inputs

    @staticmethod
    def _resize_with_pad(img: np.ndarray, width: int, height: int, pad_value: int = -1) -> np.ndarray:
        # assume no-op when width height fits already
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
