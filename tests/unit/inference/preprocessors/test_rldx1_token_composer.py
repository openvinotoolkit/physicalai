# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contract-driven RLDX-1 token composer preprocessor."""

from __future__ import annotations

import json

import numpy as np
import pytest

from physicalai.inference.constants import TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.inference.preprocessors.rldx1_token_composer import Rldx1TokenComposer


def _write_contract(tmp_path, **overrides):
    contract = {
        "model_id": "RLWRLD/RLDX-1-VLM",
        "revision": None,
        "formalize_language": True,
        "prefix_ids": [151644, 872, 198],
        "suffix_ids": [151645, 198],
        "special_ids": {
            "vision_start": 151652,
            "vision_end": 151653,
            "image_pad": 151655,
        },
        "tokens_per_image": 4,
        "num_images": 2,
        "max_token_len": 20,
        "padding_side": "left",
    }
    contract.update(overrides)
    path = tmp_path / "tokenizer_contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    return path


def test_composes_special_segments_after_task_only_tokenization(tmp_path) -> None:
    """Special multimodal IDs are injected after text-only tokenization."""
    contract_path = _write_contract(tmp_path)
    composer = Rldx1TokenComposer(contract_path)

    inputs = {
        TOKENIZED_PROMPT: np.asarray([[11, 12, 13, 14, 15]], dtype=np.int64),
        TOKENIZED_PROMPT_MASK: np.asarray([[False, False, True, True, True]], dtype=np.bool_),
    }
    outputs = composer(inputs)

    expected_unpadded = [
        151644,
        872,
        198,
        13,
        14,
        15,
        151652,
        151655,
        151655,
        151655,
        151655,
        151653,
        151652,
        151655,
        151655,
        151655,
        151655,
        151653,
        151645,
        198,
    ]

    np.testing.assert_array_equal(outputs[TOKENIZED_PROMPT][0], np.asarray(expected_unpadded, dtype=np.int64))
    np.testing.assert_array_equal(outputs[TOKENIZED_PROMPT_MASK][0], np.ones((20,), dtype=np.bool_))


def test_left_truncates_when_sequence_exceeds_max_length(tmp_path) -> None:
    """Overlong composed prompts keep the tail to preserve left-padding semantics."""
    contract_path = _write_contract(tmp_path, max_token_len=8)
    composer = Rldx1TokenComposer(contract_path)

    inputs = {
        TOKENIZED_PROMPT: np.asarray([[101, 102, 103, 104]], dtype=np.int64),
        TOKENIZED_PROMPT_MASK: np.asarray([[True, True, True, True]], dtype=np.bool_),
    }
    outputs = composer(inputs)

    expected_tail = outputs[TOKENIZED_PROMPT][0]
    assert expected_tail.shape == (8,)
    np.testing.assert_array_equal(outputs[TOKENIZED_PROMPT_MASK][0], np.ones((8,), dtype=np.bool_))
    assert expected_tail[-2:].tolist() == [151645, 198]


def test_raises_on_invalid_contract_padding_side(tmp_path) -> None:
    """Only left-padding contracts are supported by the composer."""
    contract_path = _write_contract(tmp_path, padding_side="right")

    with pytest.raises(ValueError, match="padding_side"):
        Rldx1TokenComposer(contract_path)


def test_supports_manifest_inline_configuration() -> None:
    """Composer can be configured directly from manifest params (no artifact)."""
    composer = Rldx1TokenComposer(
        prefix_ids=[151644, 872, 198],
        suffix_ids=[151645, 198],
        special_ids={"vision_start": 151652, "vision_end": 151653, "image_pad": 151655},
        tokens_per_image=2,
        num_images=1,
        max_token_len=10,
        padding_side="left",
    )
    inputs = {
        TOKENIZED_PROMPT: np.asarray([[111, 222]], dtype=np.int64),
        TOKENIZED_PROMPT_MASK: np.asarray([[True, False]], dtype=np.bool_),
    }

    outputs = composer(inputs)
    assert outputs[TOKENIZED_PROMPT].shape == (1, 10)
    assert outputs[TOKENIZED_PROMPT_MASK].shape == (1, 10)
    assert outputs[TOKENIZED_PROMPT][0, -2:].tolist() == [151645, 198]


def test_golden_rldx1_prompt_layout_matches_expected_ids() -> None:
    """Compose the known RLDX-1 prompt layout for the provided mug/plate task.

    This test checks the exact token-ID layout expected by the PyTorch path:
    prefix + task IDs + 8 vision blocks (64 image-pad IDs each) + suffix.
    """
    composer = Rldx1TokenComposer(
        prefix_ids=[151644, 872, 198],
        suffix_ids=[151645, 198],
        special_ids={"vision_start": 151652, "vision_end": 151653, "image_pad": 151655},
        tokens_per_image=64,
        num_images=8,
        max_token_len=551,
        padding_side="left",
    )

    # Token IDs for:
    # "put the white mug on the plate and put the chocolate pudding to the right of the plate"
    task_ids = np.asarray(
        [
            628,
            279,
            4158,
            51489,
            389,
            279,
            11968,
            323,
            2182,
            279,
            17931,
            81427,
            311,
            279,
            1290,
            315,
            279,
            11968,
        ],
        dtype=np.int64,
    )
    padded_task = np.zeros((1, 64), dtype=np.int64)
    padded_task[0, -task_ids.shape[0] :] = task_ids
    task_mask = np.zeros((1, 64), dtype=np.bool_)
    task_mask[0, -task_ids.shape[0] :] = True

    outputs = composer({TOKENIZED_PROMPT: padded_task, TOKENIZED_PROMPT_MASK: task_mask})

    expected = [151644, 872, 198, *task_ids.tolist()]
    for _ in range(8):
        expected.extend([151652, *([151655] * 64), 151653])
    expected.extend([151645, 198])

    assert len(expected) == 551
    np.testing.assert_array_equal(outputs[TOKENIZED_PROMPT], np.asarray([expected], dtype=np.int64))
    np.testing.assert_array_equal(outputs[TOKENIZED_PROMPT_MASK], np.ones((1, 551), dtype=np.bool_))