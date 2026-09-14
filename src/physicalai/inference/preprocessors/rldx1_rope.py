# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RLDX-1 dynamic-prompt layout preprocessor.

For the dynamic-prompt export the OpenVINO/ONNX model consumes ``input_ids`` /
``position_ids`` / ``attention_mask`` as fixed-shape inputs (the untraceable
``get_rope_index`` and prompt are no longer baked). This component runs *after*
the tokenizer and produces those tensors:

- re-lays-out the tokenizer output to **left** padding (RLDX-1 trains with
  left padding; left padding also keeps the image block right-aligned so the
  exported model's compression / image-mask positions stay constant),
- appends the cognition-token placeholders and computes the 3-axis M-RoPE
  ``position_ids`` with :func:`compute_mrope_position_ids`,
- emits the extended ``attention_mask``.
"""

from __future__ import annotations

import numpy as np

from physicalai.inference.constants import TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK

from ._rope import compute_mrope_position_ids
from .base import Preprocessor

INPUT_IDS = "input_ids"
POSITION_IDS = "position_ids"
ATTENTION_MASK = "attention_mask"
IMAGE_GRID_THW = "image_grid_thw"

# Cog-token M-RoPE placeholder id (matches the Studio graph-safe backbone).
_PLACEHOLDER_TOKEN_ID = 248068


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
