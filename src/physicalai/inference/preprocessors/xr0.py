# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch-free NumPy preprocessor for the exported XR0 OpenVINO model.

The exported XR0 OpenVINO graph is self-contained: the Qwen3-VL vision tower, the
language model, the 3D MRoPE ``position_ids`` and the image-token scatter all run
*inside* the graph. Its inputs are the tokenized prompt plus the raw pixels/state
(``tokenized_prompt`` / ``tokenized_prompt_mask`` / ``pixel_values`` / ``state``)
and its output is the still-normalized, ``max_state_dim``-wide action chunk.

:class:`XR0Preprocessor` reconstructs the graph inputs from a raw observation dict
without loading Torch or the full Qwen3-VL processor: it resizes the camera views
into the Qwen3-VL ``pixel_values`` grid, pads/normalizes the ``state`` and renders
the multi-view chat prompt as a plain ``task`` string. It does **not** tokenize --
a sibling OpenVINO tokenizer (``tokenizer.xml``, exported next to the graph) turns
``task`` into ``tokenized_prompt`` / ``tokenized_prompt_mask``. The image geometry
and normalization constants are baked at export time into the manifest
``init_args``; ``image_grid_thw`` is carried inside the graph as a baked constant.

The Studio-side torch training preprocessor and export-baking reference live in
``physicalai.policies.xr0`` (physicalai-train); this component is the deploy-only
NumPy mirror registered under the ``"xr0"`` component type.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
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

# Numerical epsilon added to the state std (matches the training convention).
_ACTION_EPS = 1e-6

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

# View titles the model was trained with (Xiaomi reference server prompt), e.g.
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
        parts.append(f"# {_view_title(view)} View\n")
        parts.append(_VISION_START + _IMAGE_PAD * count + _VISION_END)
        parts.append("\n")
    parts.append(_TASK_TEMPLATE.format(instruction=instruction))
    user = "".join(parts)
    return f"{_IM_START}user\n{user}{_IM_END}\n{_IM_START}assistant\n{_ASSISTANT_PRIMER}{_IM_END}\n"


def _resize_image(image: Image.Image, factor: int, max_pixels: int) -> Image.Image:
    """Resize a PIL image to patch-aligned dimensions within an area budget.

    Both sides are rounded to multiples of ``factor`` and the area is kept within
    ``[factor**2, max_pixels]``, preserving aspect ratio for the VLM vision encoder.

    Returns:
        The resized PIL image.

    Raises:
        ValueError: If the image aspect ratio exceeds ``_MAX_ASPECT_RATIO``.
    """
    min_pixels = factor * factor
    width, height = image.size
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

    return image.resize((new_width, new_height))


def _build_pixel_grid(
    images: Sequence[Image.Image],
    image_mean: Sequence[float],
    image_std: Sequence[float],
    rescale_factor: float,
) -> np.ndarray:
    """Rescale + normalize already-resized images into a Qwen3-VL pixel grid.

    Reproduces the Qwen3-VL image processor's rescale + normalize + channel-first
    steps in pure NumPy and stacks the views into the ``(num_images, C, H, W)``
    normalized grid the exported graph patchifies.

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


def _to_pil(array: object) -> Image.Image:
    """Convert a NumPy image (``(H,W,C)`` / ``(C,H,W)`` / batched / temporal) to PIL.

    Mirrors the training preprocessor's channels-first detection, ``[0, 1]`` float
    rescaling and grayscale expansion so the resized geometry matches the baked
    export exactly.

    Returns:
        The image as an RGB PIL ``Image``.
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
    return Image.fromarray(arr)


class XR0Preprocessor(Preprocessor):
    """Build the exported XR0 graph inputs from a raw observation dict.

    Lightweight, torch-free NumPy preprocessor: resizes the camera views into the
    Qwen3-VL ``pixel_values`` grid, pads/normalizes the ``state`` and renders the
    multi-view chat prompt as a plain ``task`` string. It does **not** tokenize --
    a sibling OpenVINO tokenizer (``tokenizer.xml``) turns ``task`` into
    ``tokenized_prompt`` / ``tokenized_prompt_mask``. The image geometry and
    normalization constants are baked at export time so no HuggingFace processor
    is loaded at inference. ``image_grid_thw`` is carried inside the graph as a
    baked constant.

    Args:
        camera_views: Ordered view names embedded into the prompt (must match the
            views used at export time so the baked image geometry stays valid).
        max_state_dim: State dimension after padding.
        image_factor: Patch-alignment factor for image resizing.
        image_max_pixels: Maximum image area for image resizing.
        image_mean: Per-channel image mean (baked from the source image processor).
        image_std: Per-channel image std (baked from the source image processor).
        rescale_factor: Pixel rescale factor (``1/255`` for Qwen3-VL).
        patch_size: Vision patch size used to derive the ``<|image_pad|>`` count.
        merge_size: Spatial merge size used to derive the ``<|image_pad|>`` count.
        normalize_state: Whether the exported model expects normalized state.
            Defaults to False (raw state), matching the training default.
        state_mean: Baked ``max_state_dim`` state mean (identity when disabled).
        state_std: Baked ``max_state_dim`` state std (identity when disabled).

    Examples:
        Constructed via manifest (type-based resolution)::

            {"type": "xr0", "camera_views": ["base", "wrist_left"],
             "max_state_dim": 32, "patch_size": 16, "merge_size": 2}
    """

    def __init__(
        self,
        camera_views: Sequence[str] = ("base", "wrist_left"),
        max_state_dim: int = 32,
        image_factor: int = 32,
        image_max_pixels: int = 90000,
        image_mean: Sequence[float] = _QWEN3VL_IMAGE_MEAN,
        image_std: Sequence[float] = _QWEN3VL_IMAGE_STD,
        rescale_factor: float = _QWEN3VL_RESCALE_FACTOR,
        patch_size: int = _QWEN3VL_PATCH_SIZE,
        merge_size: int = _QWEN3VL_MERGE_SIZE,
        *,
        normalize_state: bool = False,
        state_mean: Sequence[float] | None = None,
        state_std: Sequence[float] | None = None,
    ) -> None:
        """Initialize the XR0 inference preprocessor.

        Raises:
            ValueError: If ``camera_views`` is empty or ``patch_size`` / ``merge_size``
                is not positive.
        """
        super().__init__()
        camera_views = tuple(camera_views)
        if not camera_views:
            msg = "XR0Preprocessor requires at least one camera view"
            raise ValueError(msg)
        if int(patch_size) <= 0 or int(merge_size) <= 0:
            msg = f"patch_size and merge_size must be positive, got {patch_size!r} / {merge_size!r}"
            raise ValueError(msg)

        self._camera_views = camera_views
        self._max_state_dim = int(max_state_dim)
        self._image_factor = int(image_factor)
        self._image_max_pixels = int(image_max_pixels)
        self._image_mean = tuple(float(v) for v in image_mean)
        self._image_std = tuple(float(v) for v in image_std)
        self._rescale_factor = float(rescale_factor)
        self._patch_size = int(patch_size)
        self._merge_size = int(merge_size)
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

    def _extract_images(self, inputs: dict[str, object]) -> list[Image.Image]:
        """Return the resized PIL views in ``camera_views`` (sorted-key) order.

        Returns:
            The list of resized PIL images (one per available camera view).

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
        keys = sorted(image_items)[: len(self._camera_views)]
        if not keys:
            msg = "XR0 inference requires at least one image observation"
            raise ValueError(msg)
        return [
            _resize_image(_to_pil(image_items[key]), factor=self._image_factor, max_pixels=self._image_max_pixels)
            for key in keys
        ]

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
            state = (state - self._state_mean) / (self._state_std + _ACTION_EPS)
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
            ``pixel_values`` is the pre-patchify normalized image grid
            ``(num_images, C, H, W)`` -- the exported graph bakes the Qwen3-VL
            temporal-duplication + patchify reshape/transpose; the sibling
            OpenVINO tokenizer turns ``task`` into the graph's ``tokenized_prompt``
            / ``tokenized_prompt_mask`` inputs.
        """
        images = self._extract_images(inputs)
        pixel_values = _build_pixel_grid(images, self._image_mean, self._image_std, self._rescale_factor)
        pad_counts = [
            _image_pad_count(
                1,
                image.size[1] // self._patch_size,  # image.size == (width, height)
                image.size[0] // self._patch_size,
                self._merge_size,
            )
            for image in images
        ]
        views = self._camera_views[: len(images)]
        prompt = _render_chat_prompt(views, pad_counts, self._instruction(inputs))
        state = self._prepare_state(inputs)
        return {
            "pixel_values": np.ascontiguousarray(pixel_values.astype(np.float32)),
            "state": np.ascontiguousarray(state),
            TASK: [prompt],
        }
