# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NumPy preprocessor for the XR0 model."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np
from typing_extensions import override

from physicalai.inference.constants import IMAGES, STATE, TASK
from physicalai.inference.preprocessors.base import Preprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence

# Qwen3-VL image-normalization + geometry constants. These are baked into the
# manifest ``init_args`` at export time from the source image processor; the
# defaults mirror ``Qwen/Qwen3-VL-4B-Instruct`` so the component is usable
# standalone.
_QWEN3VL_IMAGE_MEAN = (0.5, 0.5, 0.5)
_QWEN3VL_IMAGE_STD = (0.5, 0.5, 0.5)
_QWEN3VL_RESCALE_FACTOR = 1.0 / 255.0
_QWEN3VL_PATCH_SIZE = 16
_QWEN3VL_MERGE_SIZE = 2
_QWEN3VL_TEMPORAL_PATCH_SIZE = 2

# Numerical epsilon added to the state std (matches the training convention).
_STATE_EPS = 1e-6

_TEMPORAL_IMAGE_NDIM = 5
_BATCHED_IMAGE_NDIM = 4
_CHANNELS_FIRST_NDIM = 3
_TEMPORAL_STATE_NDIM = 3

# Reject images whose aspect ratio exceeds this (matches the source preprocessor).
_MAX_ASPECT_RATIO = 200

# --- prompt text (mirrors the Qwen3-VL processor chat template) ------------
_MULTI_VIEW_HEADER = "The following observations are captured from multiple views.\n"
_TASK_TEMPLATE = "Generate robot actions for the task:\n{instruction} /no_cot"
_ASSISTANT_PRIMER = "<cot></cot>"

# View titles the model was trained with, e.g.
# "wrist_left" -> "Left-Wrist" so the prompt reads "# Left-Wrist View".
_VIEW_TITLES = {
    "base": "Base",
    "wrist_left": "Left-Wrist",
    "wrist_right": "Right-Wrist",
}

# Qwen3-VL chat special tokens rendered as literal text (a bare tokenizer maps
# each to its dedicated special-token id).
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_VISION_START = "<|vision_start|>"
_VISION_END = "<|vision_end|>"
_IMAGE_PAD = "<|image_pad|>"


def _view_title(view: str) -> str:
    """Human-readable view title matching the reference prompt.

    Returns:
        The human-readable view title.
    """
    key = view.replace("-", "_")
    if key in _VIEW_TITLES:
        return _VIEW_TITLES[key]
    return " ".join(word.capitalize() for word in key.split("_"))


def _image_pad_count(grid_t: int, grid_h: int, grid_w: int, merge_size: int) -> int:
    """Number of ``<|image_pad|>`` tokens the processor expands one image into.

    Returns:
        The image-pad token count for a single image.
    """
    return (grid_t * grid_h * grid_w) // (merge_size * merge_size)


def _render_chat_prompt(views: Sequence[str], pad_counts: Sequence[int], instruction: str) -> str:
    """Render the XR0 Qwen3-VL chat prompt as a raw string.

    Reproduces ``processor.apply_chat_template(..., tokenize=False)`` for the XR0
    multi-view message, with each image's ``<|image_pad|>`` already expanded to
    ``pad_counts[i]`` copies.

    Returns:
        The fully-rendered chat prompt string.

    Raises:
        ValueError: If ``views`` and ``pad_counts`` have different lengths.
    """
    if len(views) != len(pad_counts):
        msg = f"views ({len(views)}) and pad_counts ({len(pad_counts)}) must have the same length"
        raise ValueError(msg)

    parts: list[str] = [_MULTI_VIEW_HEADER]
    for view, count in zip(views, pad_counts, strict=True):
        parts.extend((f"# {_view_title(view)} View\n", _VISION_START + _IMAGE_PAD * count + _VISION_END, "\n"))
    parts.append(_TASK_TEMPLATE.format(instruction=instruction))
    user = "".join(parts)
    return f"{_IM_START}user\n{user}{_IM_END}\n{_IM_START}assistant\n{_ASSISTANT_PRIMER}{_IM_END}\n"


def _resize_image(image: np.ndarray, factor: int, max_pixels: int) -> np.ndarray:
    """Resize an ``(H, W, C)`` uint8 image to patch-aligned dims within an area budget.

    Both sides are rounded to multiples of ``factor`` and the area is kept within
    ``[factor**2, max_pixels]``, preserving aspect ratio for the VLM vision encoder.

    Returns:
        The resized ``(H, W, C)`` uint8 image.

    Raises:
        ValueError: If the image aspect ratio exceeds ``_MAX_ASPECT_RATIO``.
    """
    min_pixels = factor * factor
    height, width = image.shape[:2]
    ratio = max(height, width) / min(height, width)
    if ratio > _MAX_ASPECT_RATIO:
        msg = f"absolute aspect ratio must be smaller than 200, got {ratio}"
        raise ValueError(msg)

    new_height = max(factor, round(height / factor) * factor)
    new_width = max(factor, round(width / factor) * factor)

    if new_height * new_width > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        new_height = max(factor, math.floor(height / scale / factor) * factor)
        new_width = max(factor, math.floor(width / scale / factor) * factor)
    elif new_height * new_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        new_height = max(factor, math.ceil(height * scale / factor) * factor)
        new_width = max(factor, math.ceil(width * scale / factor) * factor)

    # cv2.resize takes dsize as (width, height); INTER_CUBIC is the closest match to
    # the reference PIL bicubic resample (exact parity is not required here).
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def _build_pixel_grid(
    images: Sequence[np.ndarray],
    image_mean: Sequence[float],
    image_std: Sequence[float],
    rescale_factor: float,
) -> np.ndarray:
    """Rescale + normalize already-resized images into a Qwen3-VL pixel grid.

    Returns:
        The normalized image grid of shape ``(num_images, C, H, W)`` as float32.
    """
    mean = np.asarray(image_mean, dtype=np.float32)
    std = np.asarray(image_std, dtype=np.float32)
    grid = [
        np.transpose((np.asarray(image, dtype=np.float32) * np.float32(rescale_factor) - mean) / std, (2, 0, 1))
        for image in images
    ]
    return np.stack(grid).astype(np.float32)


def _patchify_pixel_grid(
    grid: np.ndarray,
    grid_thw: Sequence[Sequence[int]],
    *,
    temporal_patch_size: int,
    patch_size: int,
    merge_size: int,
) -> np.ndarray:
    """Patchify a normalized image grid exactly like the Qwen3-VL image processor.

    Reproduces the transformers ``Qwen2VLImageProcessor`` patchify (temporal
    duplication + 9-D reshape/transpose) so the exported graph receives the flat
    ``pixel_values`` layout directly (patchify happens off-graph here).

    Returns:
        The flat ``pixel_values`` array of shape
        ``(sum(grid_t * grid_h * grid_w), C * temporal_patch_size * patch_size ** 2)``
        as float32.
    """
    flattened: list[np.ndarray] = []
    for index, (grid_t, grid_h, grid_w) in enumerate(grid_thw):
        image = grid[index]  # (C, H, W)
        channel = image.shape[0]
        feature = channel * temporal_patch_size * patch_size * patch_size
        patches = np.tile(image[np.newaxis], (temporal_patch_size, 1, 1, 1))  # (tp, C, H, W)
        patches = patches.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flattened.append(patches.reshape(grid_t * grid_h * grid_w, feature))
    return np.concatenate(flattened, axis=0).astype(np.float32)


def _to_rgb_frame(array: object) -> np.ndarray:
    """Normalize a NumPy image (``(H,W,C)`` / ``(C,H,W)`` / batched / temporal) to uint8 RGB.

    Collapses arbitrary incoming layouts/dtypes to a single canonical ``(H, W, 3)``
    uint8 RGB frame so the subsequent ``cv2.resize`` sees a consistent input. The
    exported graph bakes image geometry from the resized dimensions, so only the
    output size has to match the reference; the interpolation kernel does not.

    Returns:
        The canonical ``(H, W, 3)`` uint8 RGB frame.
    """
    arr = np.asarray(array)
    if arr.ndim == _TEMPORAL_IMAGE_NDIM:  # (B, T, C, H, W) -> last frame of first sample
        arr = arr[0, -1]
    elif arr.ndim == _BATCHED_IMAGE_NDIM:  # (T|B, C, H, W) -> last frame
        arr = arr[-1]
    if arr.ndim == _CHANNELS_FIRST_NDIM and arr.shape[0] in {1, 3}:  # channels-first
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.ascontiguousarray(arr)


class XR0Preprocessor(Preprocessor):
    """Build the exported XR0 graph inputs from a raw observation dict.

    Lightweight, torch-free NumPy preprocessor: resizes the camera views into the
    Qwen3-VL ``pixel_values`` grid, pads/normalizes the ``state`` and renders the
    multi-view chat prompt as a plain ``task`` string.

    Args:
        max_state_dim: State dimension after padding.
        image_factor: Patch-alignment factor for image resizing.
        image_max_pixels: Maximum image area for image resizing.
        image_mean: Per-channel image mean (baked from the source image processor).
        image_std: Per-channel image std (baked from the source image processor).
        rescale_factor: Pixel rescale factor (``1/255`` for Qwen3-VL).
        patch_size: Vision patch size used to derive the ``<|image_pad|>`` count.
        merge_size: Spatial merge size used to derive the ``<|image_pad|>`` count.
        temporal_patch_size: Number of frames grouped per temporal patch (the
            off-graph patchify duplicates a still image to this many frames).
        normalize_state: Whether the exported model expects normalized state.
            Defaults to False (raw state), matching the training default.
        state_mean: Baked ``max_state_dim`` state mean (identity when disabled).
        state_std: Baked ``max_state_dim`` state std (identity when disabled).

    Examples:
        Constructed via manifest (type-based resolution)::

            {"type": "xr0", "max_state_dim": 32, "patch_size": 16, "merge_size": 2}
    """

    def __init__(
        self,
        max_state_dim: int = 32,
        image_factor: int = 32,
        image_max_pixels: int = 90000,
        image_mean: Sequence[float] = _QWEN3VL_IMAGE_MEAN,
        image_std: Sequence[float] = _QWEN3VL_IMAGE_STD,
        rescale_factor: float = _QWEN3VL_RESCALE_FACTOR,
        patch_size: int = _QWEN3VL_PATCH_SIZE,
        merge_size: int = _QWEN3VL_MERGE_SIZE,
        temporal_patch_size: int = _QWEN3VL_TEMPORAL_PATCH_SIZE,
        *,
        normalize_state: bool = False,
        state_mean: Sequence[float] | None = None,
        state_std: Sequence[float] | None = None,
    ) -> None:
        """Initialize the XR0 inference preprocessor.

        Raises:
            ValueError: If ``patch_size`` / ``merge_size`` is not positive.
        """
        super().__init__()
        if int(patch_size) <= 0 or int(merge_size) <= 0:
            msg = f"patch_size and merge_size must be positive, got {patch_size!r} / {merge_size!r}"
            raise ValueError(msg)

        self._max_state_dim = int(max_state_dim)
        self._image_factor = int(image_factor)
        self._image_max_pixels = int(image_max_pixels)
        self._image_mean = tuple(float(v) for v in image_mean)
        self._image_std = tuple(float(v) for v in image_std)
        self._rescale_factor = float(rescale_factor)
        self._patch_size = int(patch_size)
        self._merge_size = int(merge_size)
        self._temporal_patch_size = int(temporal_patch_size)
        self._normalize_state = bool(normalize_state)

        # State normalization is opt-in; padded dims use identity stats (mean 0,
        # std 1) so they stay zero, mirroring the training preprocessor.
        if normalize_state and state_mean is not None and state_std is not None:
            self._state_mean = self._pad_state_stat(state_mean, 0.0)
            self._state_std = self._pad_state_stat(state_std, 1.0)
        else:
            self._state_mean = np.zeros(self._max_state_dim, dtype=np.float32)
            self._state_std = np.ones(self._max_state_dim, dtype=np.float32)

    def _pad_state_stat(self, values: Sequence[float], fill: float) -> np.ndarray:
        """Pad/truncate a state stat to ``max_state_dim`` (padded dims use ``fill``).

        Returns:
            The ``(max_state_dim,)`` float32 stat array.
        """
        arr = np.asarray(values, dtype=np.float32).flatten()
        out = np.full(self._max_state_dim, fill, dtype=np.float32)
        dim = min(self._max_state_dim, arr.shape[0])
        out[:dim] = arr[:dim]
        return out

    def _extract_images(self, inputs: dict[str, object]) -> tuple[list[str], list[np.ndarray]]:
        """Return the ordered view names and resized ``(H, W, C)`` uint8 views.

        The view order is taken directly from the observation image keys
        (``images.<view>``) in their natural insertion order, so ``pixel_values``
        stays aligned with the per-view prompt sections (title + pad count).

        Returns:
            A ``(views, images)`` tuple: the ordered view names and the resized
            uint8 RGB images (one per available camera view).

        Raises:
            ValueError: If the observation contains no image entry.
        """
        images_value = inputs.get(IMAGES)
        if isinstance(images_value, dict):
            image_items = {f"{IMAGES}.{view}": array for view, array in images_value.items()}
        else:
            image_items = {
                key: value
                for key, value in inputs.items()
                if isinstance(key, str) and key.startswith(f"{IMAGES}.") and "is_pad" not in key
            }
        if not image_items:
            msg = "XR0 inference requires at least one image observation"
            raise ValueError(msg)
        views = [key.removeprefix(f"{IMAGES}.") for key in image_items]
        images = [
            _resize_image(_to_rgb_frame(value), factor=self._image_factor, max_pixels=self._image_max_pixels)
            for value in image_items.values()
        ]
        return views, images

    def _prepare_state(self, inputs: dict[str, object]) -> np.ndarray:
        """Pad the state into ``(B, 1, max_state_dim)`` (optionally normalized).

        Returns:
            The padded ``(B, 1, max_state_dim)`` float32 state array.

        Raises:
            ValueError: If the observation has no state entry.
        """
        state_value = inputs.get(STATE)
        if state_value is None:
            msg = "XR0 inference requires a 'state' observation"
            raise ValueError(msg)
        state = np.asarray(state_value, dtype=np.float32)
        if state.ndim == 1:  # (D,) -> (1, D)
            state = state[None, :]
        if state.ndim == _TEMPORAL_STATE_NDIM:  # (B, T, D) -> last frame
            state = state[:, -1, :]
        dim = state.shape[-1]
        if dim < self._max_state_dim:
            state = np.pad(state, ((0, 0), (0, self._max_state_dim - dim)))
        state = state[:, : self._max_state_dim]
        if self._normalize_state:
            state = (state - self._state_mean) / (self._state_std + _STATE_EPS)
        return state[:, None, :].astype(np.float32)  # (B, 1, max_state_dim)

    @staticmethod
    def _instruction(inputs: dict[str, object]) -> str:
        """Extract the task instruction string from the observation.

        Returns:
            The (first) task instruction as a string (empty when absent).
        """
        task = inputs.get(TASK)
        if task is None:
            return ""
        if isinstance(task, str):
            return task
        if isinstance(task, np.ndarray):
            flat = np.atleast_1d(task).tolist()
            return str(flat[0]) if flat else ""
        if isinstance(task, (list, tuple)):
            return str(task[0]) if task else ""
        return str(task)

    @override
    def __call__(self, inputs: dict[str, object]) -> dict[str, object]:
        """Transform a raw observation into the exported graph inputs.

        Args:
            inputs: Observation dict with a ``state`` array, ``images`` (nested
                dict or flattened ``images.*`` keys) and a ``task`` string.

        Returns:
            Dict with ``pixel_values`` / ``state`` (float32 NumPy) and ``task``
            (a single-element list holding the rendered chat prompt string).
            ``pixel_values`` is the flat patchified layout
            ``(sum(t*h*w), C * temporal_patch_size * patch_size ** 2)`` the exported
            graph consumes directly (patchify happens here, off-graph); the sibling
            OpenVINO tokenizer turns ``task`` into the graph's ``tokenized_prompt``
            / ``tokenized_prompt_mask`` inputs.
        """
        views, images = self._extract_images(inputs)
        pixel_grid = _build_pixel_grid(images, self._image_mean, self._image_std, self._rescale_factor)
        grid_thw = [
            (1, image.shape[0] // self._patch_size, image.shape[1] // self._patch_size)  # image.shape == (H, W, C)
            for image in images
        ]
        pixel_values = _patchify_pixel_grid(
            pixel_grid,
            grid_thw,
            temporal_patch_size=self._temporal_patch_size,
            patch_size=self._patch_size,
            merge_size=self._merge_size,
        )
        pad_counts = [_image_pad_count(grid_t, grid_h, grid_w, self._merge_size) for grid_t, grid_h, grid_w in grid_thw]
        prompt = _render_chat_prompt(views, pad_counts, self._instruction(inputs))
        state = self._prepare_state(inputs)
        return {
            "pixel_values": np.ascontiguousarray(pixel_values.astype(np.float32)),
            "state": np.ascontiguousarray(state),
            TASK: [prompt],
        }
