# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)
# ruff: noqa: PLR0912, PLR0914, PLR0915

"""RLDX-1 dynamic-prompt rope preprocessing and M-RoPE position-id math."""

from __future__ import annotations

import numpy as np

from physicalai.inference.constants import TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.inference.preprocessors.base import Preprocessor

INPUT_IDS = "input_ids"
POSITION_IDS = "position_ids"
ATTENTION_MASK = "attention_mask"
IMAGE_GRID_THW = "image_grid_thw"

# Cog-token M-RoPE placeholder id (matches the Studio graph-safe backbone).
_PLACEHOLDER_TOKEN_ID = 248068
_UNBATCHED_GRID_RANK = 2
_BATCHED_GRID_RANK = 3


def compute_mrope_position_ids(
    input_ids: np.ndarray,
    image_grid_thw: np.ndarray | None,
    attention_mask: np.ndarray | None = None,
    *,
    image_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
) -> np.ndarray:
    """Compute 3-axis M-RoPE position ids, mirroring Qwen3-VL ``get_rope_index``.

    Args:
        input_ids: ``(B, L)`` (or ``(L,)``) token ids, including any appended
            cognition-token placeholders and left padding.
        image_grid_thw: ``(num_images, 3)`` temporal/height/width grid per image,
            or ``None`` for the text-only path.
        attention_mask: ``(B, L)`` 1/0 mask (1 = real token). ``None`` means all
            ones. Padded positions keep the default position id of 1.
        image_token_id: Token id of the image placeholder (``<|image_pad|>``).
        vision_start_token_id: Token id of ``<|vision_start|>``.
        spatial_merge_size: Vision spatial merge factor.

    Returns:
        ``position_ids`` of shape ``(3, B, L)`` (int64).

    Raises:
        ValueError: If ``image_grid_thw`` has an invalid shape or does not
            contain enough rows for image blocks found in the prompt.
    """
    input_ids = np.asarray(input_ids)
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]
    batch, seq_len = input_ids.shape

    if image_grid_thw is None:
        return _text_only_position_ids(attention_mask, batch, seq_len)

    image_grid_thw = np.asarray(image_grid_thw)
    if image_grid_thw.ndim == _UNBATCHED_GRID_RANK:  # (num_images, 3)
        if image_grid_thw.shape[1] != 3:  # noqa: PLR2004
            msg = f"Expected image_grid_thw shape (num_images, 3) for unbatched input, got {image_grid_thw.shape!r}."
            raise ValueError(msg)
    elif image_grid_thw.ndim == _BATCHED_GRID_RANK:  # (B, num_images, 3)
        if image_grid_thw.shape[0] != batch or image_grid_thw.shape[2] != 3:  # noqa: PLR2004
            msg = (
                "Expected image_grid_thw shape (B, num_images, 3) for batched input with "
                f"B={batch}, got {image_grid_thw.shape!r}."
            )
            raise ValueError(msg)
    else:
        msg = (
            "Expected image_grid_thw to have shape (num_images, 3) or (B, num_images, 3), "
            f"got {image_grid_thw.shape!r}."
        )
        raise ValueError(msg)

    if attention_mask is None:
        attention_mask = np.ones_like(input_ids)
    attention_mask = np.asarray(attention_mask)

    position_ids = np.ones((3, batch, seq_len), dtype=np.int64)
    image_index = 0
    for i in range(batch):
        sample_image_index = 0
        sample_grid = image_grid_thw[i] if image_grid_thw.ndim == _BATCHED_GRID_RANK else image_grid_thw

        keep = attention_mask[i] == 1
        ids = input_ids[i][keep]
        vision_start_indices = np.nonzero(ids == vision_start_token_id)[0]
        vision_tokens = ids[vision_start_indices + 1] if vision_start_indices.size else ids[:0]
        image_nums = int(np.sum(vision_tokens == image_token_id))

        input_tokens = ids.tolist()
        llm_pos_ids_list: list[np.ndarray] = []
        st = 0
        remain_images = image_nums
        for _ in range(image_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed = input_tokens.index(image_token_id, st)
            else:
                ed = len(input_tokens) + 1

            if image_grid_thw.ndim == _BATCHED_GRID_RANK:
                if sample_image_index >= sample_grid.shape[0]:
                    msg = (
                        "image_grid_thw has fewer rows than image blocks in the prompt for "
                        f"sample {i}: needed at least {sample_image_index + 1}, got {sample_grid.shape[0]}."
                    )
                    raise ValueError(msg)
                grid_t, grid_h, grid_w = sample_grid[sample_image_index]
                sample_image_index += 1
            else:
                if image_index >= sample_grid.shape[0]:
                    msg = (
                        "image_grid_thw has fewer rows than image blocks in the prompt: "
                        f"needed at least {image_index + 1}, got {sample_grid.shape[0]}."
                    )
                    raise ValueError(msg)
                grid_t, grid_h, grid_w = sample_grid[image_index]
                image_index += 1

            remain_images -= 1

            llm_grid_t = int(grid_t)
            llm_grid_h = int(grid_h) // spatial_merge_size
            llm_grid_w = int(grid_w) // spatial_merge_size
            text_len = ed - st

            st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)

            t_index = np.repeat(np.arange(llm_grid_t), llm_grid_h * llm_grid_w)
            h_index = np.tile(np.repeat(np.arange(llm_grid_h), llm_grid_w), llm_grid_t)
            w_index = np.tile(np.arange(llm_grid_w), llm_grid_t * llm_grid_h)
            llm_pos_ids_list.append(np.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)

        llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        position_ids[:, i, keep] = llm_positions

    return position_ids


def _text_only_position_ids(
    attention_mask: np.ndarray | None,
    batch: int,
    seq_len: int,
) -> np.ndarray:
    """Position ids for the no-vision path (mirrors the torch ``else`` branch).

    Returns:
        ``position_ids`` of shape ``(3, B, L)`` (int64).
    """
    if attention_mask is not None:
        mask = np.asarray(attention_mask).astype(np.int64)
        pos = np.cumsum(mask, axis=-1) - 1
        pos = np.where(mask == 0, 1, pos)
        return np.broadcast_to(pos[None], (3, batch, seq_len)).copy()
    pos = np.broadcast_to(np.arange(seq_len), (3, batch, seq_len))
    return pos.copy()


class Rldx1RopePreprocessor(Preprocessor):
    """Build left-padded ``input_ids`` + ``position_ids`` + ``attention_mask``."""

    def __init__(
        self,
        image_token_id: int,
        vision_start_token_id: int,
        spatial_merge_size: int = 2,
        n_cog_tokens: int = 64,
        pad_token_id: int = 0,
    ) -> None:
        """Initialize the layout preprocessor.

        Args:
            image_token_id: Token id of ``<|image_pad|>``.
            vision_start_token_id: Token id of ``<|vision_start|>``.
            spatial_merge_size: Vision spatial merge factor.
            n_cog_tokens: Number of appended cognition-token placeholders.
            pad_token_id: Fill id for left padding (numerically irrelevant: pads
                are masked as attention keys and their outputs are discarded).
        """
        self._image_token_id = image_token_id
        self._vision_start_token_id = vision_start_token_id
        self._spatial_merge_size = spatial_merge_size
        self._n_cog_tokens = n_cog_tokens
        self._pad_token_id = pad_token_id

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Emit ``input_ids`` / ``position_ids`` / ``attention_mask``.

        Args:
            inputs: Must contain ``TOKENIZED_PROMPT`` (``(B, L)`` ids),
                ``TOKENIZED_PROMPT_MASK`` (``(B, L)`` 1/0), and
                ``image_grid_thw``.

        Returns:
            The inputs dict with the three model tensors added.
        """
        ids = np.asarray(inputs[TOKENIZED_PROMPT])
        mask = np.asarray(inputs[TOKENIZED_PROMPT_MASK]).astype(bool)
        grid_thw = np.asarray(inputs[IMAGE_GRID_THW])

        left_ids, left_mask = self._left_repad(ids, mask)

        batch = left_ids.shape[0]
        cog_ids = np.full((batch, self._n_cog_tokens), _PLACEHOLDER_TOKEN_ID, dtype=left_ids.dtype)
        cog_mask = np.ones((batch, self._n_cog_tokens), dtype=np.int64)
        extended_ids = np.concatenate([left_ids, cog_ids], axis=1)
        extended_mask = np.concatenate([left_mask, cog_mask], axis=1)

        position_ids = compute_mrope_position_ids(
            extended_ids,
            grid_thw,
            extended_mask,
            image_token_id=self._image_token_id,
            vision_start_token_id=self._vision_start_token_id,
            spatial_merge_size=self._spatial_merge_size,
        )

        outputs = dict(inputs)
        outputs[INPUT_IDS] = left_ids
        outputs[POSITION_IDS] = position_ids
        outputs[ATTENTION_MASK] = extended_mask
        return outputs

    def _left_repad(self, ids: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Right-align each row's real tokens (left padding), fixed width.

        Returns:
            ``(left_ids, left_mask)`` of the same ``(B, L)`` shape.
        """
        batch, length = ids.shape
        left_ids = np.full((batch, length), self._pad_token_id, dtype=ids.dtype)
        left_mask = np.zeros((batch, length), dtype=np.int64)
        for i in range(batch):
            real = ids[i][mask[i]]
            n = real.shape[0]
            left_ids[i, length - n :] = real
            left_mask[i, length - n :] = 1
        return left_ids, left_mask

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return f"{self.__class__.__name__}(image_token_id={self._image_token_id}, n_cog_tokens={self._n_cog_tokens})"
