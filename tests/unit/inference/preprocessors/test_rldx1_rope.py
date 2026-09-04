# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Parity test: numpy M-RoPE port vs the torch Qwen3-VL ``get_rope_index``.

Validates ``compute_mrope_position_ids`` against a faithful torch reimplementation
of Qwen3-VL's ``get_rope_index`` (image path) on synthetic prompts, including
left-padding. No checkpoint required.
"""

from __future__ import annotations

import numpy as np
import pytest

from physicalai.inference.preprocessors._rope import compute_mrope_position_ids

torch = pytest.importorskip("torch")

IMAGE_TOKEN_ID = 5
VISION_START_TOKEN_ID = 4
MERGE = 2


def _torch_get_rope_index(input_ids, image_grid_thw, attention_mask, spatial_merge_size):  # noqa: ANN001
    """Reference: Qwen3-VL get_rope_index (image path), transcribed to standalone torch."""
    total_input_ids = input_ids
    if attention_mask is None:
        attention_mask = torch.ones_like(total_input_ids)
    position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype)
    image_index = 0
    for i, ids_row in enumerate(total_input_ids):
        ids = ids_row[attention_mask[i] == 1]
        vision_start_indices = torch.argwhere(ids == VISION_START_TOKEN_ID).squeeze(1)
        vision_tokens = ids[vision_start_indices + 1]
        image_nums = (vision_tokens == IMAGE_TOKEN_ID).sum()
        input_tokens = ids.tolist()
        llm_pos_ids_list: list = []
        st = 0
        remain_images = image_nums
        for _ in range(image_nums):
            ed = input_tokens.index(IMAGE_TOKEN_ID, st) if (IMAGE_TOKEN_ID in input_tokens and remain_images > 0) else len(input_tokens) + 1
            t, h, w = image_grid_thw[image_index]
            image_index += 1
            remain_images -= 1
            llm_grid_t, llm_grid_h, llm_grid_w = int(t), int(h) // spatial_merge_size, int(w) // spatial_merge_size
            text_len = ed - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w
        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.dtype)
    return position_ids


def _build_prompt(task_len: int, grid_hw: tuple[int, int], *, left_pad: int = 0):
    """[pad*] <vision_start> <image_pad>*T <text>* with an image block."""
    h, w = grid_hw
    tokens_per_image = (h // MERGE) * (w // MERGE)
    ids = [0] * left_pad  # pad id 0
    ids += [10, 11]  # some prompt text
    ids += [VISION_START_TOKEN_ID] + [IMAGE_TOKEN_ID] * tokens_per_image
    ids += [20 + i for i in range(task_len)]  # trailing text (e.g. cog placeholders)
    mask = [0] * left_pad + [1] * (len(ids) - left_pad)
    grid = [[1, h, w]]
    return ids, mask, grid


@pytest.mark.parametrize("left_pad", [0, 3, 7])
@pytest.mark.parametrize(("task_len", "grid_hw"), [(2, (4, 4)), (5, (6, 4)), (0, (4, 6))])
def test_numpy_rope_matches_torch(left_pad: int, task_len: int, grid_hw: tuple[int, int]) -> None:
    """Numpy position ids match the torch reference across padding/sizes."""
    ids, mask, grid = _build_prompt(task_len, grid_hw, left_pad=left_pad)
    ids_np = np.array([ids], dtype=np.int64)
    mask_np = np.array([mask], dtype=np.int64)
    grid_np = np.array(grid, dtype=np.int64)

    got = compute_mrope_position_ids(
        ids_np,
        grid_np,
        mask_np,
        image_token_id=IMAGE_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        spatial_merge_size=MERGE,
    )
    expected = _torch_get_rope_index(
        torch.tensor(ids_np),
        grid_np,
        torch.tensor(mask_np),
        MERGE,
    ).numpy()

    np.testing.assert_array_equal(got, expected)


def test_numpy_rope_batch() -> None:
    """Batched prompts of equal length match the torch reference row-wise."""
    ids_a, mask_a, grid_a = _build_prompt(3, (4, 4), left_pad=2)
    ids_b, mask_b, _ = _build_prompt(3, (4, 4), left_pad=0)
    ids_np = np.array([ids_a, ids_b], dtype=np.int64)
    mask_np = np.array([mask_a, mask_b], dtype=np.int64)
    grid_np = np.array(grid_a + grid_a, dtype=np.int64)  # one image per row

    got = compute_mrope_position_ids(
        ids_np,
        grid_np,
        mask_np,
        image_token_id=IMAGE_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        spatial_merge_size=MERGE,
    )
    expected = _torch_get_rope_index(torch.tensor(ids_np), grid_np, torch.tensor(mask_np), MERGE).numpy()

    np.testing.assert_array_equal(got, expected)


def test_numpy_rope_batch_with_batched_grid_thw() -> None:
    """Batched image_grid_thw (B, N, 3) matches flattened-grid behavior."""
    ids_a, mask_a, grid_a = _build_prompt(3, (4, 4), left_pad=2)
    ids_b, mask_b, grid_b = _build_prompt(3, (4, 4), left_pad=0)
    ids_np = np.array([ids_a, ids_b], dtype=np.int64)
    mask_np = np.array([mask_a, mask_b], dtype=np.int64)

    # Runtime rldx1 preprocessor emits batched grid_thw with shape (B, num_images, 3).
    grid_batched = np.array([grid_a, grid_b], dtype=np.int64)
    grid_flat = np.array(grid_a + grid_b, dtype=np.int64)

    got_batched = compute_mrope_position_ids(
        ids_np,
        grid_batched,
        mask_np,
        image_token_id=IMAGE_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        spatial_merge_size=MERGE,
    )
    got_flat = compute_mrope_position_ids(
        ids_np,
        grid_flat,
        mask_np,
        image_token_id=IMAGE_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        spatial_merge_size=MERGE,
    )

    np.testing.assert_array_equal(got_batched, got_flat)
