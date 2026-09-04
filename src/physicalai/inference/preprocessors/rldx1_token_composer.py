# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compose final RLDX-1 prompt token IDs from task-only tokenization.

This stage patches around OpenVINO tokenizer special-token parity gaps for
Qwen3-VL marker tokens by injecting the multimodal wrapper IDs from an
exported contract artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from physicalai.inference.constants import TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK

from .base import Preprocessor

_TOKEN_IDS_RANK = 2


class Rldx1TokenComposer(Preprocessor):
    r"""Compose RLDX-1 multimodal token IDs from a tokenizer contract.

    Args:
        artifact: Optional path to ``tokenizer_contract.json``.
        prefix_ids: Chat-prefix token IDs (e.g., ``<|im_start|>user\n``).
        suffix_ids: Chat-suffix token IDs (e.g., ``<|im_end|>\n``).
        special_ids: Mapping for ``vision_start``, ``vision_end``, ``image_pad`` IDs.
        tokens_per_image: Count of ``image_pad`` IDs per image block.
        num_images: Number of image blocks per prompt.
        max_token_len: Final fixed token length after left padding/truncation.
        padding_side: Must be ``"left"``.
    """

    def __init__(
        self,
        artifact: str | Path | None = None,
        *,
        prefix_ids: list[int] | None = None,
        suffix_ids: list[int] | None = None,
        special_ids: dict[str, int] | None = None,
        tokens_per_image: int | None = None,
        num_images: int | None = None,
        max_token_len: int | None = None,
        padding_side: str = "left",
        **extra: object,
    ) -> None:
        """Initialize from a contract artifact or explicit manifest parameters.

        Args:
            artifact: Optional path to ``tokenizer_contract.json``.
            prefix_ids: Chat-prefix token IDs.
            suffix_ids: Chat-suffix token IDs.
            special_ids: Mapping containing vision marker IDs.
            tokens_per_image: Number of image-pad tokens per image.
            num_images: Number of image blocks per prompt.
            max_token_len: Final fixed token length after padding/truncation.
            padding_side: Padding direction; only ``"left"`` is supported.
            extra: Ignored compatibility kwargs from manifest loading.
        """
        del extra
        super().__init__()
        self._artifact = Path(artifact) if artifact is not None else None

        if self._artifact is not None:
            contract = self._load_contract(self._artifact)
        else:
            contract = {
                "prefix_ids": prefix_ids,
                "suffix_ids": suffix_ids,
                "special_ids": special_ids,
                "tokens_per_image": tokens_per_image,
                "num_images": num_images,
                "max_token_len": max_token_len,
                "padding_side": padding_side,
            }
            contract = self._validate_contract(contract)

        self._prefix_ids = np.asarray(contract["prefix_ids"], dtype=np.int64)
        self._suffix_ids = np.asarray(contract["suffix_ids"], dtype=np.int64)
        self._max_token_len = int(contract["max_token_len"])
        self._num_images = int(contract["num_images"])

        special = contract["special_ids"]
        vision_start = int(special["vision_start"])
        vision_end = int(special["vision_end"])
        image_pad = int(special["image_pad"])
        tokens_per_image = int(contract["tokens_per_image"])

        self._vision_block = np.concatenate(
            (
                np.asarray([vision_start], dtype=np.int64),
                np.full((tokens_per_image,), image_pad, dtype=np.int64),
                np.asarray([vision_end], dtype=np.int64),
            ),
        )

    @staticmethod
    def _load_contract(path: Path) -> dict[str, Any]:
        """Load and validate a tokenizer contract JSON artifact.

        Returns:
            Validated tokenizer contract payload.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        return Rldx1TokenComposer._validate_contract(payload)

    @staticmethod
    def _validate_contract(payload: dict[str, Any]) -> dict[str, Any]:
        """Validate a contract payload loaded from JSON or manifest params.

        Returns:
            Validated tokenizer contract payload.

        Raises:
            TypeError: If a contract field has the wrong runtime type.
            ValueError: If required keys are missing or values are invalid.
        """
        if not isinstance(payload, dict):
            msg = "Tokenizer contract must be a JSON object."
            raise TypeError(msg)

        required_keys = {
            "prefix_ids",
            "suffix_ids",
            "special_ids",
            "tokens_per_image",
            "num_images",
            "max_token_len",
            "padding_side",
        }
        missing = required_keys.difference(payload)
        if missing:
            msg = f"Tokenizer contract missing required keys: {sorted(missing)!r}"
            raise ValueError(msg)

        if payload["padding_side"] != "left":
            msg = "Tokenizer contract padding_side must be 'left'."
            raise ValueError(msg)

        for key in ("prefix_ids", "suffix_ids"):
            values = payload[key]
            if not isinstance(values, list) or not all(isinstance(v, int) for v in values):
                msg = f"Tokenizer contract key {key!r} must be a list of integers."
                raise TypeError(msg)

        special = payload["special_ids"]
        if not isinstance(special, dict):
            msg = "Tokenizer contract key 'special_ids' must be an object."
            raise TypeError(msg)
        for name in ("vision_start", "vision_end", "image_pad"):
            if not isinstance(special.get(name), int):
                msg = f"Tokenizer contract special_ids.{name} must be an integer."
                raise TypeError(msg)

        for key in ("tokens_per_image", "num_images", "max_token_len"):
            value = payload[key]
            if not isinstance(value, int) or value <= 0:
                msg = f"Tokenizer contract key {key!r} must be a positive integer."
                raise TypeError(msg)

        return payload

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Compose final token IDs and attention mask from task-only token IDs.

        Returns:
            Updated preprocessor inputs with composed ``tokenized_prompt`` and
            ``tokenized_prompt_mask`` arrays.

        Raises:
            ValueError: If token IDs or masks have unexpected shapes.
        """
        ids = np.asarray(inputs[TOKENIZED_PROMPT], dtype=np.int64)
        mask = np.asarray(inputs[TOKENIZED_PROMPT_MASK]).astype(np.bool_)
        if ids.ndim != _TOKEN_IDS_RANK:
            msg = f"Expected {TOKENIZED_PROMPT!r} to have shape (B, L), got {ids.shape!r}"
            raise ValueError(msg)
        if mask.shape != ids.shape:
            msg = f"Expected {TOKENIZED_PROMPT_MASK!r} to have shape {ids.shape!r}, got {mask.shape!r}"
            raise ValueError(msg)

        outputs = dict(inputs)
        batch_size = ids.shape[0]
        composed_ids = np.zeros((batch_size, self._max_token_len), dtype=np.int64)
        composed_mask = np.zeros((batch_size, self._max_token_len), dtype=np.bool_)
        vision_ids = np.tile(self._vision_block, self._num_images)

        for index in range(batch_size):
            task_ids = ids[index][mask[index]]
            full = np.concatenate((self._prefix_ids, task_ids, vision_ids, self._suffix_ids))
            if full.shape[0] > self._max_token_len:
                full = full[-self._max_token_len :]
            composed_ids[index, -full.shape[0] :] = full
            composed_mask[index, -full.shape[0] :] = True

        outputs[TOKENIZED_PROMPT] = composed_ids
        outputs[TOKENIZED_PROMPT_MASK] = composed_mask
        return outputs

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return f"{self.__class__.__name__}(artifact={self._artifact!r})"
