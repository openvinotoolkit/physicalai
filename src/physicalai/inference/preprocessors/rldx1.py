# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor that builds Qwen3-VL VTC inputs for RLDX-1.

Ports the frozen-geometry half of RLDX-1's Qwen3-VL image pipeline to numpy:
patchify (mirrors ``Qwen2VLImageProcessor._preprocess``) plus task
normalization (lowercase + punctuation strip). Both are closed-form given a
fixed export resolution/frame/view count, so neither needs the live HF
processor at inference time.

Tokenization (``task`` string -> ``input_ids``/``attention_mask``) is a
separate downstream stage (``OVTokenizer``/``HFTokenizer``), same contract as
:class:`~physicalai.inference.preprocessors.pi05.Pi05Preprocessor`.

Not produced here: ``mm_token_type_ids``. The traced Qwen3-VL model
recomputes it internally from ``input_ids`` (elementwise comparisons against
the image/video token ids), so the preprocessor does not need to supply it.
"""

from __future__ import annotations

import re

import numpy as np

from physicalai.inference.constants import IMAGES, STATE, TASK

from .base import Preprocessor

# RLDX-1-VLM (Qwen3-VL) vision tiler constants. These must match the values
# actually baked into the exported checkpoint's tokenizer/processor config
# (RLWRLD/RLDX-1-VLM's preprocessor_config.json) -- verify before relying on
# these defaults, they are not yet cross-checked against that file.
_PATCH_SIZE = 16
_TEMPORAL_PATCH_SIZE = 2
_SPATIAL_MERGE_SIZE = 2
_RGB_CHANNELS = 3
# Verified against RLWRLD/RLDX-1-VLM's actual preprocessor_config.json (loaded
# instance, not the generic Qwen2VLImageProcessor class default of CLIP
# stats): image_mean/image_std are both (0.5, 0.5, 0.5), i.e. rescale to
# [-1, 1] rather than CLIP normalization.
_IMAGE_MEAN = (0.5, 0.5, 0.5)
_IMAGE_STD = (0.5, 0.5, 0.5)

# RLDX-1 model input keys (see Rldx1Model.forward / Rldx1Preprocessor upstream).
PIXEL_VALUES = "pixel_values"
IMAGE_GRID_THW = "image_grid_thw"
MAX_STATE_DIM = 64


def _formalize_language(text: str) -> str:
    """Lowercase and strip punctuation, matching Studio's training-time step.

    Returns:
        The lowercased, punctuation-stripped instruction.
    """
    return re.sub(r"[^\w\s]", "", text.lower())


class Rldx1Preprocessor(Preprocessor):
    """Builds Qwen3-VL VTC inputs (frozen image geometry) for RLDX-1.

    Assumes a fixed, export-time-frozen image resolution, a fixed number of
    camera views, and a fixed VTC frame count -- geometry that never varies
    per call, so ``image_grid_thw`` is precomputed once in ``__init__``.

    Does not assemble the VTC frame-history window itself: callers must
    supply frames already ordered frame-major / view-inner (``[t0v0, t0v1,
    ..., t1v0, ...]``), matching Studio's training-time ordering. A rolling
    frame-history runner is separate, deferred work.

    Args:
        image_resolution: Frozen ``(height, width)`` of each camera frame,
            after Stage 3 (``AspectAreaResizeAndCrop``) and Qwen-tiler
            alignment. Must be divisible by ``patch_size * merge_size``.
        num_views: Number of camera views per VTC step.
        num_frames: Number of VTC temporal frames per view.
        max_state_dim: Padded state dimension. Shorter states are zero-padded
            on the trailing feature axis after coercion to ``(B, 1, D)``.
        embodiment_id: Per-embodiment projector slot in the MSAT action head.
        patch_size: Qwen3-VL vision tower patch size.
        temporal_patch_size: Qwen3-VL temporal patch size.
        merge_size: Qwen3-VL spatial merge size.
    """

    def __init__(
        self,
        image_resolution: tuple[int, int],
        num_views: int = 1,
        num_frames: int = 4,
        max_state_dim: int = MAX_STATE_DIM,
        embodiment_id: int = 0,
        patch_size: int = _PATCH_SIZE,
        temporal_patch_size: int = _TEMPORAL_PATCH_SIZE,
        merge_size: int = _SPATIAL_MERGE_SIZE,
    ) -> None:
        """Precompute the frozen patch grid, image_grid_thw, and prompt template.

        Raises:
            ValueError: If ``image_resolution`` isn't aligned to
                ``patch_size * merge_size``.
        """
        super().__init__()
        height, width = image_resolution
        align = patch_size * merge_size
        if height % align != 0 or width % align != 0:
            msg = (
                f"image_resolution {image_resolution} must be divisible by "
                f"patch_size * merge_size ({align}); got grid {height % align}, {width % align} remainder."
            )
            raise ValueError(msg)

        self._image_resolution = image_resolution
        self._num_views = num_views
        self._num_frames = num_frames
        self._max_state_dim = max_state_dim
        self._embodiment_id = embodiment_id
        self._patch_size = patch_size
        self._temporal_patch_size = temporal_patch_size
        self._merge_size = merge_size

        self._grid_h = height // patch_size
        self._grid_w = width // patch_size
        self._num_images = num_views * num_frames

        # grid_t is always 1: every VTC frame is encoded as an independent
        # still image (temporal duplication happens inside _patchify, not by
        # treating the window as a single multi-frame video clip).
        self._image_grid_thw_row = np.array([1, self._grid_h, self._grid_w], dtype=np.int64)

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Build pixel_values/image_grid_thw and normalized task text.

        Args:
            inputs: Dict with per-view image arrays under ``images.<view>``
                (or a single ``images`` array when ``num_views == 1``), each
                shaped ``(B, T, H, W, C)`` / ``(B, T, C, H, W)`` uint8 or
                float32, plus ``task`` (list[str] or str) and ``state``.

        Returns:
            Updated dict with ``pixel_values`` ``(num_images * grid_h *
            grid_w, C * temporal_patch_size * patch_size**2)`` for the common
            single-sample inference path, or ``(B, num_images * grid_h * grid_w,
            C * temporal_patch_size * patch_size**2)`` for batched inputs,
            ``image_grid_thw`` ``(B, num_images, 3)``, ``task`` (normalized
            natural-language strings), and optionally ``state``.
        """
        inputs = dict(inputs)
        batch_frames = self._collect_frame_major_frames(inputs)
        batch_size = len(batch_frames)

        pixel_values = [self._patchify(frames) for frames in batch_frames]
        inputs[PIXEL_VALUES] = pixel_values[0] if batch_size == 1 else np.stack(pixel_values, axis=0)
        inputs[IMAGE_GRID_THW] = np.tile(self._image_grid_thw_row, (batch_size, self._num_images, 1))

        tasks = inputs.get(TASK)
        if tasks is None:
            tasks = [""] * batch_size
        elif isinstance(tasks, str):
            tasks = [tasks]
        inputs[TASK] = [_formalize_language(t) for t in tasks]

        state = inputs.get(STATE)
        if state is not None:
            state_arr = np.asarray(state)
            if state_arr.ndim == 2:  # noqa: PLR2004  (B, D) -> (B, 1, D)
                state_arr = np.expand_dims(state_arr, axis=1)
            if state_arr.ndim == 3:  # (B, T, D) VTC window -> current-step state only.  # noqa: PLR2004
                state_arr = state_arr[:, -1:, :]
            state_arr = _pad_last_dim(state_arr, self._max_state_dim)
            inputs[STATE] = state_arr

        return inputs

    def _collect_frame_major_frames(self, inputs: dict[str, np.ndarray]) -> list[list[np.ndarray]]:
        """Assemble per-sample frame-major / view-inner frame lists.

        Returns:
            One list of ``num_images`` ``(H, W, 3)`` uint8 frames per batch
            sample, ordered ``[t0v0, t0v1, ..., t1v0, ...]``.

        Raises:
            ValueError: If no camera image is found, or a view's frame count
                doesn't match ``num_frames``.
        """
        view_arrays = self._collect_view_arrays(inputs)
        if not view_arrays:
            msg = "RLDX-1 preprocessor requires at least one camera image."
            raise ValueError(msg)

        per_view_frames = [self._per_sample_frames(arr) for arr in view_arrays]
        batch_size = len(per_view_frames[0])

        batch_frames: list[list[np.ndarray]] = []
        for sample_idx in range(batch_size):
            per_view = [per_view_frames[v][sample_idx] for v in range(len(view_arrays))]
            for view_frames in per_view:
                if len(view_frames) != self._num_frames:
                    msg = (
                        f"Expected {self._num_frames} VTC frames per view, got {len(view_frames)}. "
                        "The frame-history window must be assembled by the caller before this preprocessor runs."
                    )
                    raise ValueError(msg)
            batch_frames.append(
                [per_view[view_idx][t] for t in range(self._num_frames) for view_idx in range(len(view_arrays))],
            )
        return batch_frames

    @staticmethod
    def _collect_view_arrays(inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Return per-view image arrays in a deterministic key order.

        Returns:
            List of raw ``(B, T, H, W, C)`` / ``(B, T, C, H, W)`` arrays, one
            per camera view.
        """
        images_value = inputs.get(IMAGES)
        if isinstance(images_value, np.ndarray):
            return [images_value]
        if isinstance(images_value, dict):
            return [images_value[key] for key in sorted(images_value)]
        keys = sorted(key for key in inputs if key.startswith(f"{IMAGES}."))
        return [inputs[key] for key in keys]

    @staticmethod
    def _per_sample_frames(view_array: np.ndarray) -> list[list[np.ndarray]]:
        """Split a ``(B, T, H, W, C)``/``(B, T, C, H, W)`` or 4-D array into per-sample HWC uint8 frame lists.

        Returns:
            ``batch_size`` lists of ``(H, W, 3)`` uint8 frames.

        Raises:
            ValueError: If the input does not have a supported 4-D or 5-D
                image layout.
        """
        arr = np.asarray(view_array)
        if arr.ndim == 4:  # noqa: PLR2004 - (B, C, H, W) or (B, H, W, C)
            arr = np.expand_dims(arr, axis=1)
        elif arr.ndim != 5:  # noqa: PLR2004
            msg = (
                "Expected a (B, T, H, W, C)/(B, T, C, H, W) or (B, H, W, C)/(B, C, H, W) view array, "
                f"got shape {arr.shape}"
            )
            raise ValueError(msg)
        channels_first = arr.shape[2] == _RGB_CHANNELS
        if channels_first:
            arr = np.transpose(arr, (0, 1, 3, 4, 2))  # -> (B, T, H, W, C)
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.size > 0 and np.nanmax(arr) <= 1.0:
                arr *= 255.0
            arr = np.clip(np.nan_to_num(arr, nan=0.0), 0, 255).astype(np.uint8)
        return [[arr[b, t] for t in range(arr.shape[1])] for b in range(arr.shape[0])]

    def _patchify(self, frames: list[np.ndarray]) -> np.ndarray:
        """Port of ``Qwen2VLImageProcessor._preprocess`` for a fixed-shape frame list.

        Args:
            frames: ``num_images`` ``(H, W, 3)`` uint8 frames, already
                resized/cropped to ``self._image_resolution``.

        Returns:
            ``(num_images * grid_h * grid_w, C * temporal_patch_size *
            patch_size**2)`` float32 patch tensor.
        """
        mean = np.asarray(_IMAGE_MEAN, dtype=np.float32).reshape(1, 1, 1, -1)
        std = np.asarray(_IMAGE_STD, dtype=np.float32).reshape(1, 1, 1, -1)
        imgs = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (N, H, W, C)
        imgs = (imgs - mean) / std
        imgs = np.transpose(imgs, (0, 3, 1, 2))  # -> (N, C, H, W)

        num_images, channels = imgs.shape[0], imgs.shape[1]
        patch_size, temporal, merge = self._patch_size, self._temporal_patch_size, self._merge_size
        grid_h, grid_w = self._grid_h, self._grid_w

        # Duplicate each still image across the temporal axis (grid_t stays 1
        # after the reshape below divides by temporal_patch_size).
        patches = np.repeat(imgs[:, None, :, :, :], temporal, axis=1)  # (N, temporal, C, H, W)

        patches = patches.reshape(
            num_images,
            1,
            temporal,
            channels,
            grid_h // merge,
            merge,
            patch_size,
            grid_w // merge,
            merge,
            patch_size,
        )
        patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        return patches.reshape(
            num_images * grid_h * grid_w,
            channels * temporal * patch_size * patch_size,
        ).astype(np.float32)


def _pad_last_dim(array: np.ndarray, new_dim: int) -> np.ndarray:
    """Zero-pad the trailing feature dimension up to ``new_dim``.

    Returns:
        Original array when no padding is needed, otherwise a new zero-padded
        array with trailing dimension ``new_dim``.
    """
    if array.shape[-1] >= new_dim:
        return array
    pad_width = [(0, 0)] * array.ndim
    pad_width[-1] = (0, new_dim - array.shape[-1])
    return np.pad(array, pad_width, mode="constant")
