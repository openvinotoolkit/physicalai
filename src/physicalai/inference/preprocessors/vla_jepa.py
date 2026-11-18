# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NumPy preprocessor for VLA-JEPA OpenVINO exports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import override

from physicalai.inference.constants import IMAGES, STATE, TASK
from physicalai.inference.preprocessors.base import Preprocessor
from physicalai.inference.preprocessors.xr0 import _patchify_pixel_grid  # noqa: PLC2701

if TYPE_CHECKING:
    from collections.abc import Sequence

_HWC_NDIM = 3
_RGB = 3


def _area_weights(size_in: int, size_out: int) -> np.ndarray:
    """Return adaptive-average-pooling weights, matching torch ``interpolate(mode="area")``."""
    weights = np.zeros((size_out, size_in), dtype=np.float64)
    for index in range(size_out):
        start, end = index * size_in // size_out, -(-(index + 1) * size_in // size_out)
        weights[index, start:end] = 1 / (end - start)
    return weights.astype(np.float32)


def _bicubic_weights(size_in: int, size_out: int, a: float = -0.5) -> np.ndarray:
    """Return antialiased bicubic weights, matching torchvision ``resize(antialias=True)``."""
    scale = size_in / size_out
    stretch = max(scale, 1.0)
    weights = np.zeros((size_out, size_in), dtype=np.float64)
    for index in range(size_out):
        center = (index + 0.5) * scale
        start = max(int(center - 2 * stretch + 0.5), 0)
        end = min(int(center + 2 * stretch + 0.5), size_in)
        x = np.abs((np.arange(start, end) - center + 0.5) / stretch)
        kernel = np.where(
            x < 1,
            ((a + 2) * x - (a + 3)) * x * x + 1,
            np.where(x < 2, (((x - 5) * x + 8) * x - 4) * a, 0.0),  # noqa: PLR2004
        )
        weights[index, start:end] = kernel / kernel.sum()
    return weights.astype(np.float32)


def _resize(image: np.ndarray, weights: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Apply separable ``(rows, cols)`` resize weights to a ``(C, H, W)`` image.

    Returns:
        The resized ``(C, H', W')`` image.
    """
    return weights[0] @ image @ weights[1].T


class VLAJEPAPreprocessor(Preprocessor):
    """Build VLA-JEPA graph inputs from a raw single observation.

    Resizes each camera like the training processor (area resize to
    ``image_resolution``, then antialiased bicubic to ``vision_resolution``),
    normalizes and patchifies it, tokenizes the baked chat prompt, and samples
    the flow-matching noise. State passes through unchanged; the exported graph
    normalizes it.

    Args:
        artifact: Path to the exported ``tokenizer.json``.
        prompt: Rendered chat prompt with expanded image pads and an
            ``{instruction}`` placeholder for the task.
        image_keys: Camera keys (``images.<name>``) in training order.
        embodied_token_id: Token id whose positions the action head reads.
        num_embodied_tokens: Expected number of embodied-action tokens.
        noise_shape: Flow noise shape ``(chunk_size, action_dim)``.
        image_resolution: ``(height, width)`` of the first, area resize.
        vision_resolution: ``(height, width)`` the vision encoder receives.
        image_mean: Per-channel normalization mean.
        image_std: Per-channel normalization std.
        patch_size: Vision patch size.
        merge_size: Spatial merge size.
        temporal_patch_size: Frames per temporal patch.

    Examples:
        Constructed via manifest::

            {"type": "vla_jepa", "artifact": "tokenizer.json", "prompt": "...", ...}
    """

    def __init__(
        self,
        artifact: str,
        prompt: str,
        image_keys: Sequence[str],
        embodied_token_id: int,
        num_embodied_tokens: int,
        noise_shape: Sequence[int],
        image_resolution: Sequence[int] = (224, 224),
        vision_resolution: Sequence[int] = (256, 256),
        image_mean: Sequence[float] = (0.5, 0.5, 0.5),
        image_std: Sequence[float] = (0.5, 0.5, 0.5),
        patch_size: int = 16,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
    ) -> None:
        """Load the tokenizer and precompute vision geometry.

        Raises:
            ValueError: If the prompt or geometry cannot produce valid graph inputs.
        """
        super().__init__()
        if "{instruction}" not in prompt:
            msg = "VLA-JEPA prompt must contain an {instruction} placeholder."
            raise ValueError(msg)
        height, width = (int(size) for size in vision_resolution)
        if height % (patch_size * merge_size) or width % (patch_size * merge_size):
            msg = "vision_resolution must be a multiple of patch_size * merge_size."
            raise ValueError(msg)
        from transformers import PreTrainedTokenizerFast  # noqa: PLC0415

        self._tokenizer = PreTrainedTokenizerFast(tokenizer_file=artifact)
        self._prompt = prompt
        self._image_keys = list(image_keys)
        self._embodied_token_id = int(embodied_token_id)
        self._num_embodied_tokens = int(num_embodied_tokens)
        self._noise_shape = (1, *(int(size) for size in noise_shape))
        self._image_resolution = tuple(int(size) for size in image_resolution)
        self._vision_resolution = (height, width)
        self._mean = np.asarray(image_mean, dtype=np.float32)[:, None, None]
        self._std = np.asarray(image_std, dtype=np.float32)[:, None, None]
        self._patch_size = int(patch_size)
        self._merge_size = int(merge_size)
        self._temporal_patch_size = int(temporal_patch_size)
        self._weights: dict[tuple[int, ...], tuple[tuple[np.ndarray, np.ndarray], ...]] = {}

    def _pixels(self, image: object) -> np.ndarray:
        """Return a ``(C, H, W)`` vision-resolution image from one camera frame.

        Raises:
            ValueError: If the frame is not a single RGB image.
        """
        array = np.asarray(image)
        if array.ndim == _HWC_NDIM + 1:
            if array.shape[0] != 1:
                msg = "VLA-JEPA OpenVINO inference requires batch size one."
                raise ValueError(msg)
            array = array[0]
        if array.ndim != _HWC_NDIM or _RGB not in {array.shape[0], array.shape[-1]}:
            msg = f"Expected one RGB camera frame, got shape {array.shape}."
            raise ValueError(msg)
        if array.shape[0] != _RGB:
            array = array.transpose(2, 0, 1)
        pixels = array.astype(np.float32) / 255 if array.dtype == np.uint8 else array.astype(np.float32)
        size = pixels.shape[1:]
        if size not in self._weights:
            self._weights[size] = (
                (_area_weights(size[0], self._image_resolution[0]), _area_weights(size[1], self._image_resolution[1])),
                (
                    _bicubic_weights(self._image_resolution[0], self._vision_resolution[0]),
                    _bicubic_weights(self._image_resolution[1], self._vision_resolution[1]),
                ),
            )
        area, bicubic = self._weights[size]
        resized = _resize(_resize(pixels, area).astype(np.float32), bicubic).astype(np.float32)
        return (resized - self._mean) / self._std

    def _images(self, inputs: dict[str, Any]) -> list[object]:
        """Return camera frames in training order from flat or nested inputs.

        Raises:
            KeyError: If a trained camera is missing.
        """
        nested = inputs.get(IMAGES)
        frames = []
        for key in self._image_keys:
            frame = inputs.get(key)
            if frame is None and isinstance(nested, dict):
                frame = nested.get(key.removeprefix(f"{IMAGES}."))
            if frame is None and len(self._image_keys) == 1 and not isinstance(nested, dict):
                frame = nested
            if frame is None:
                msg = f"Missing camera input {key!r}."
                raise KeyError(msg)
            frames.append(frame)
        return frames

    @override
    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Return the exported graph inputs for one observation.

        Raises:
            ValueError: If the task, prompt or noise do not match the graph contract.
        """
        task = inputs.get(TASK)
        tasks = [task] if isinstance(task, str) else [] if task is None else list(np.atleast_1d(task))
        if len(tasks) != 1 or not isinstance(tasks[0], str):
            msg = "VLA-JEPA OpenVINO inference requires one task string."
            raise ValueError(msg)
        ids = np.asarray([self._tokenizer.encode(self._prompt.replace("{instruction}", tasks[0]))], dtype=np.int64)
        positions = np.flatnonzero(ids[0] == self._embodied_token_id)
        if positions.size != self._num_embodied_tokens:
            msg = "The task produced an invalid number of embodied-action tokens."
            raise ValueError(msg)
        grid = self._vision_resolution[0] // self._patch_size, self._vision_resolution[1] // self._patch_size
        images = np.stack([self._pixels(frame) for frame in self._images(inputs)])
        pixel_values = _patchify_pixel_grid(
            images,
            [(1, *grid)] * len(images),
            temporal_patch_size=self._temporal_patch_size,
            patch_size=self._patch_size,
            merge_size=self._merge_size,
        )
        noise = inputs.get("noise")
        noise = np.random.default_rng().standard_normal(self._noise_shape) if noise is None else np.asarray(noise)
        if noise.shape != self._noise_shape or not np.isfinite(noise).all():
            msg = f"Flow noise must be finite and have shape {self._noise_shape}."
            raise ValueError(msg)
        outputs = {
            "input_ids": ids,
            "attention_mask": np.ones_like(ids),
            "pixel_values": np.ascontiguousarray(pixel_values),
            "embodied_positions": positions[None].astype(np.int64),
            "noise": noise.astype(np.float32),
        }
        if STATE in inputs:
            outputs[STATE] = np.asarray(inputs[STATE], dtype=np.float32)
        return outputs
