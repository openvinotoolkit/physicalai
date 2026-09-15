# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NumPy model-input assembly for MolmoAct2 inference.

Mirrors the PyTorch ``build_model_inputs`` used during training/export so the
exported OpenVINO graph receives identical, fully-prepared tensors. Turns a
tokenized prompt (with ``<|image|>`` placeholders) and patchified images into
``input_ids`` (placeholders expanded), ``attention_mask``, ``token_type_ids``,
per-example batched ``images``, ``token_pooling`` and ``action_dim_is_pad``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from typing_extensions import override

from physicalai.inference.constants import IMAGES, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.inference.preprocessors.base import Preprocessor

from .molmoact2_image import MolmoAct2ImageProcessor

_PACKED_IMAGE_NDIM = 5


@dataclass
class MolmoAct2InputConfig:
    """Token ids and layout flags needed to assemble MolmoAct2 model inputs."""

    pad_token_id: int
    image_placeholder_token_id: int
    image_patch_id: int
    image_start_token_id: int
    image_end_token_id: int
    image_col_id: int | None = None
    low_res_image_start_token_id: int | None = None
    frame_start_token_id: int | None = None
    frame_end_token_id: int | None = None
    image_low_res_id: int | None = None
    image_use_col_tokens: bool = True
    use_single_crop_col_tokens: bool | None = False
    use_single_crop_start_token: bool = True
    max_action_dim: int = 32
    env_action_dim: int = 0
    image_token_ids: list[int] | None = None

    def __post_init__(self) -> None:
        """Collect the configured image token identifiers."""
        ids = [
            self.image_patch_id,
            self.image_col_id,
            self.image_start_token_id,
            self.low_res_image_start_token_id,
            self.frame_start_token_id,
            self.image_end_token_id,
            self.frame_end_token_id,
            self.image_low_res_id,
        ]
        if self.image_token_ids is None:
            self.image_token_ids = [int(token_id) for token_id in ids if token_id is not None]


def _image_token_ids_for_grid(config: MolmoAct2InputConfig, grid: np.ndarray) -> list[int]:
    """Expand a single image grid into its sequence of image token ids.

    Returns:
        Ordered image token identifiers for the grid.
    """
    resized_h, resized_w, height, width = (int(x) for x in np.asarray(grid).reshape(-1)[:4].tolist())

    image_patch_id = int(config.image_patch_id)
    image_start_token_id = int(config.image_start_token_id)
    image_end_token_id = int(config.image_end_token_id)
    image_col_id = None if config.image_col_id is None else int(config.image_col_id)
    low_res_start_id = (
        int(config.low_res_image_start_token_id)
        if config.low_res_image_start_token_id is not None
        else image_start_token_id
    )

    image_use_col_tokens = bool(config.image_use_col_tokens)
    use_single_crop_col_tokens = (
        image_use_col_tokens if config.use_single_crop_col_tokens is None else bool(config.use_single_crop_col_tokens)
    )
    use_single_crop_start_token = bool(config.use_single_crop_start_token)

    def make_rows(num_rows: int, num_cols: int, *, use_col: bool) -> list[int]:
        row = [image_patch_id] * num_cols
        if use_col and image_col_id is not None:
            row += [image_col_id]
        return row * num_rows

    if height == 0 or width == 0:
        return [
            image_start_token_id,
            *make_rows(resized_h, resized_w, use_col=use_single_crop_col_tokens),
            image_end_token_id,
        ]

    high_res = [image_start_token_id, *make_rows(height, width, use_col=image_use_col_tokens), image_end_token_id]
    low_start = low_res_start_id if use_single_crop_start_token else image_start_token_id
    low_res = [low_start, *make_rows(resized_h, resized_w, use_col=use_single_crop_col_tokens), image_end_token_id]
    return low_res + high_res


def _build_token_type_ids(
    config: MolmoAct2InputConfig, input_ids: np.ndarray, attention_mask: np.ndarray
) -> np.ndarray | None:
    """Mark image tokens (1) vs. text tokens (0), respecting the attention mask.

    Returns:
        Image-token indicators, or ``None`` when no image tokens are configured.
    """
    image_token_ids = config.image_token_ids
    if not image_token_ids:
        return None
    token_set = np.asarray(image_token_ids, dtype=input_ids.dtype)
    is_image = np.isin(input_ids, token_set).astype(np.int64)
    return is_image * attention_mask.astype(np.int64)


def expand_image_placeholders(
    *,
    config: MolmoAct2InputConfig,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    image_grids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Replace each ``<|image|>`` placeholder with its expanded image token ids.

    Args:
        config: Token identifiers and image layout configuration.
        input_ids: Padded prompt token identifiers.
        attention_mask: Mask identifying valid prompt tokens.
        image_grids: Image grid dimensions in placeholder order.

    Returns:
        Expanded token identifiers, attention mask, and optional token type identifiers.

    Raises:
        ValueError: If there are fewer image grids than image placeholders.
    """
    if int(image_grids.shape[0]) == 0:
        return input_ids, attention_mask, _build_token_type_ids(config, input_ids, attention_mask)

    placeholder_id = int(config.image_placeholder_token_id)

    expanded_rows: list[list[int]] = []
    expanded_widths: list[int] = []
    grid_idx = 0
    for batch_idx in range(int(input_ids.shape[0])):
        valid = attention_mask[batch_idx].astype(bool)
        expanded: list[int] = []
        for token in input_ids[batch_idx][valid].tolist():
            token_int = int(token)
            if token_int == placeholder_id:
                if grid_idx >= int(image_grids.shape[0]):
                    msg = "Not enough image grids to expand all <|image|> placeholders."
                    raise ValueError(msg)
                expanded.extend(_image_token_ids_for_grid(config, image_grids[grid_idx]))
                grid_idx += 1
            else:
                expanded.append(token_int)
        expanded_rows.append(expanded)
        expanded_widths.append(len(expanded) + int((~valid).sum()))

    max_len = max(expanded_widths, default=1)
    out_ids = np.full((len(expanded_rows), max_len), config.pad_token_id, dtype=input_ids.dtype)
    out_mask = np.zeros((len(expanded_rows), max_len), dtype=attention_mask.dtype)
    for batch_idx, row in enumerate(expanded_rows):
        if not row:
            continue
        row_arr = np.asarray(row, dtype=input_ids.dtype)
        out_ids[batch_idx, : row_arr.size] = row_arr
        out_mask[batch_idx, : row_arr.size] = 1

    return out_ids, out_mask, _build_token_type_ids(config, out_ids, out_mask)


@dataclass(frozen=True)
class _BatchLayout:
    counts: np.ndarray
    num_examples: int
    n_patches: int
    pixels_per_patch: int
    pooled_per_image: np.ndarray
    crops_per_example: np.ndarray
    pooled_per_example: np.ndarray
    patches_per_image: np.ndarray


def _batch_layout(
    counts: np.ndarray,
    pixel_values: np.ndarray,
    image_grids: np.ndarray,
    image_num_crops: np.ndarray,
) -> _BatchLayout:
    num_examples = counts.shape[0]
    _, n_patches, pixels_per_patch = pixel_values.shape
    grids = np.asarray(image_grids)
    pooled_per_image = (grids[:, 0] * grids[:, 1] + grids[:, 2] * grids[:, 3]).astype(np.int64)
    example_for_image = np.repeat(np.arange(num_examples, dtype=np.int64), counts).astype(np.int64)
    crops_per_example = np.zeros(num_examples, dtype=np.int64)
    np.add.at(crops_per_example, example_for_image, image_num_crops.astype(np.int64))
    pooled_per_example = np.zeros(num_examples, dtype=np.int64)
    np.add.at(pooled_per_example, example_for_image, pooled_per_image)
    return _BatchLayout(
        counts=counts,
        num_examples=num_examples,
        n_patches=n_patches,
        pixels_per_patch=pixels_per_patch,
        pooled_per_image=pooled_per_image,
        crops_per_example=crops_per_example,
        pooled_per_example=pooled_per_example,
        patches_per_image=image_num_crops.astype(np.int64) * n_patches,
    )


def _allocate_batched_outputs(
    layout: _BatchLayout, pixel_values: np.ndarray, image_token_pooling: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    max_crops = int(layout.crops_per_example.max()) if layout.num_examples > 0 else 0
    images = np.full(
        (layout.num_examples, max_crops, layout.n_patches, layout.pixels_per_patch),
        -1.0,
        dtype=pixel_values.dtype,
    )
    max_pooled = int(layout.pooled_per_example.max()) if layout.num_examples > 0 else 0
    token_pooling = np.full(
        (layout.num_examples, max_pooled, image_token_pooling.shape[-1]),
        -1,
        dtype=image_token_pooling.dtype,
    )
    return images, token_pooling


def _offset_example_pooling(
    layout: _BatchLayout,
    image_token_pooling: np.ndarray,
    *,
    example_idx: int,
    image_offset: int,
    pooled_offset: int,
) -> np.ndarray:
    example_pooling = image_token_pooling[
        pooled_offset : pooled_offset + int(layout.pooled_per_example[example_idx])
    ].copy()
    patch_offset = 0
    row = 0
    for local_image in range(int(layout.counts[example_idx])):
        num_pooled = int(layout.pooled_per_image[image_offset + local_image])
        block = example_pooling[row : row + num_pooled]
        example_pooling[row : row + num_pooled] = np.where(block >= 0, block + patch_offset, block)
        patch_offset += int(layout.patches_per_image[image_offset + local_image])
        row += num_pooled
    return example_pooling


def build_batched_images(
    config: MolmoAct2InputConfig,
    input_ids: np.ndarray,
    pixel_values: np.ndarray,
    image_token_pooling: np.ndarray,
    image_grids: np.ndarray,
    image_num_crops: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Regroup per-image crops/pooling into per-example padded tensors.

    Mirrors the PyTorch host-side reassembly: infers the image-to-example
    mapping from ``image_end`` tokens and offsets pooling indices into each
    example's stacked crop patches.

    Returns:
        ``(images, token_pooling)`` of shapes ``(N, max_crops, n_patches, pixels)``
        and ``(N, max_pooled, pool_area)``.

    Raises:
        ValueError: If image counts cannot be inferred from image-end tokens.
    """
    raw_counts = (input_ids == int(config.image_end_token_id)).sum(1)
    num_images = int(image_grids.shape[0])
    total_end_tokens = int(raw_counts.sum())
    if num_images == 0:
        counts = np.zeros_like(raw_counts)
    elif total_end_tokens == num_images:
        counts = raw_counts
    elif total_end_tokens == 2 * num_images:
        counts = raw_counts // 2
    else:
        msg = (
            "Could not infer image counts from image-end tokens: "
            f"end_tokens={total_end_tokens}, image_grids={num_images}."
        )
        raise ValueError(msg)

    layout = _batch_layout(counts, pixel_values, image_grids, image_num_crops)
    images, token_pooling = _allocate_batched_outputs(layout, pixel_values, image_token_pooling)

    crop_offset = 0
    pooled_offset = 0
    image_offset = 0
    for example_idx in range(layout.num_examples):
        num_example_images = int(layout.counts[example_idx])
        num_example_crops = int(layout.crops_per_example[example_idx])
        images[example_idx, :num_example_crops] = pixel_values[crop_offset : crop_offset + num_example_crops]

        example_pooling = _offset_example_pooling(
            layout,
            image_token_pooling,
            example_idx=example_idx,
            image_offset=image_offset,
            pooled_offset=pooled_offset,
        )
        token_pooling[example_idx, : example_pooling.shape[0]] = example_pooling

        crop_offset += num_example_crops
        pooled_offset += int(layout.pooled_per_example[example_idx])
        image_offset += num_example_images

    return images, token_pooling


def default_action_dim_is_pad(config: MolmoAct2InputConfig, *, batch_size: int) -> np.ndarray:
    """Mark action dimensions beyond the environment action dim as padding.

    Returns:
        A boolean mask with padding dimensions marked true.
    """
    action_dim_is_pad = np.ones((batch_size, int(config.max_action_dim)), dtype=bool)
    if int(config.env_action_dim) > 0:
        action_dim_is_pad[:, : int(config.env_action_dim)] = False
    return action_dim_is_pad


@dataclass(eq=False, repr=False, kw_only=True)
class MolmoAct2ModelInputs(Preprocessor):
    """Assemble tokenized prompts and packed images into model inputs."""

    max_action_dim: int
    action_dim: int
    bos_token_id: int
    pad_token_id: int
    image_placeholder_token_id: int
    image_start_token_id: int
    image_end_token_id: int
    image_patch_id: int
    image_col_id: int | None
    low_res_image_start_token_id: int | None
    frame_start_token_id: int | None = None
    frame_end_token_id: int | None = None
    image_low_res_id: int | None = None
    image_size: tuple[int, int] = (378, 378)
    patch_size: int = 14
    pooling_size: tuple[int, int] = (2, 2)
    image_mean: list[float] | None = None
    image_std: list[float] | None = None
    image_crop_mode: str = "resize"
    image_use_col_tokens: bool = True
    use_single_crop_col_tokens: bool | None = False
    use_single_crop_start_token: bool = True
    image_token_ids: list[int] | None = None

    def __post_init__(self) -> None:
        """Build the input layout and image processor."""
        self._layout = MolmoAct2InputConfig(
            pad_token_id=self.pad_token_id,
            image_placeholder_token_id=self.image_placeholder_token_id,
            image_patch_id=self.image_patch_id,
            image_start_token_id=self.image_start_token_id,
            image_end_token_id=self.image_end_token_id,
            image_col_id=self.image_col_id,
            low_res_image_start_token_id=self.low_res_image_start_token_id,
            frame_start_token_id=self.frame_start_token_id,
            frame_end_token_id=self.frame_end_token_id,
            image_low_res_id=self.image_low_res_id,
            image_use_col_tokens=self.image_use_col_tokens,
            use_single_crop_col_tokens=self.use_single_crop_col_tokens,
            use_single_crop_start_token=self.use_single_crop_start_token,
            max_action_dim=self.max_action_dim,
            env_action_dim=self.action_dim,
            image_token_ids=self.image_token_ids,
        )
        self._image_processor = MolmoAct2ImageProcessor(
            crop_mode=self.image_crop_mode,
            size={"height": self.image_size[0], "width": self.image_size[1]},
            patch_size=self.patch_size,
            pooling_size=self.pooling_size,
            image_mean=self.image_mean,
            image_std=self.image_std,
        )

    @override
    def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Convert tokenized prompts and packed images to graph inputs.

        Returns:
            Backbone-ready NumPy arrays.

        Raises:
            ValueError: If packed images do not have the expected layout.
        """
        input_ids = np.asarray(inputs[TOKENIZED_PROMPT], dtype=np.int64)
        attention_mask = np.asarray(inputs[TOKENIZED_PROMPT_MASK], dtype=np.int64)
        input_ids, attention_mask = self._insert_bos(input_ids, attention_mask)

        images = np.asarray(inputs[IMAGES], dtype=np.float32)
        if images.ndim != _PACKED_IMAGE_NDIM:
            msg = f"Expected packed images [N, B, C, H, W], got {images.shape}."
            raise ValueError(msg)
        num_images, batch_size, channels, height, width = images.shape
        flat_images = images.transpose(1, 0, 2, 3, 4).reshape(
            batch_size * num_images,
            channels,
            height,
            width,
        )
        image_output = self._image_processor(flat_images)
        input_ids, attention_mask, token_type_ids = expand_image_placeholders(
            config=self._layout,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grids=image_output["image_grids"],
        )
        batched_images, pooling = build_batched_images(
            self._layout,
            input_ids,
            image_output["pixel_values"],
            image_output["image_token_pooling"],
            image_output["image_grids"],
            image_output["image_num_crops"],
        )
        outputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            IMAGES: batched_images.astype(np.float32),
            "token_pooling": pooling.astype(np.int64),
            "action_dim_is_pad": default_action_dim_is_pad(self._layout, batch_size=batch_size),
        }
        if token_type_ids is not None:
            outputs["token_type_ids"] = token_type_ids
        return outputs

    def _insert_bos(self, ids: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = [row_ids[row_mask.astype(bool)] for row_ids, row_mask in zip(ids, mask, strict=True)]
        if all(row.size > 0 and int(row[0]) == self.bos_token_id for row in rows):
            return ids, mask
        rows = [
            row
            if row.size > 0 and int(row[0]) == self.bos_token_id
            else np.concatenate((np.asarray([self.bos_token_id], dtype=ids.dtype), row))
            for row in rows
        ]
        width = ids.shape[1] + 1
        output_ids = np.full((len(rows), width), self.pad_token_id, dtype=ids.dtype)
        output_mask = np.zeros((len(rows), width), dtype=mask.dtype)
        for index, row in enumerate(rows):
            output_ids[index, : row.size] = row
            output_mask[index, : row.size] = 1
        return output_ids, output_mask


__all__ = [
    "MolmoAct2InputConfig",
    "MolmoAct2ModelInputs",
    "build_batched_images",
    "default_action_dim_is_pad",
    "expand_image_placeholders",
]
