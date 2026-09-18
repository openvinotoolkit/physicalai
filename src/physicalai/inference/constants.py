# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference output field name constants.

Canonical key names for inference pipeline inputs and outputs, enabling IDE
autocomplete and safe refactoring across the inference module.
"""

IMAGES = "images"
ACTION = "action"
TASK = "task"
STATE = "state"
STATE_PASSTHROUGH = "state_passthrough"

TOKENIZED_PROMPT = "tokenized_prompt"
TOKENIZED_PROMPT_MASK = "tokenized_prompt_mask"
IMAGE_MASKS = "image_masks"

PREV_CHUNK_LEFT_OVER = "prev_chunk_left_over"
RTC_INFERENCE_DELAY = "inference_delay"
RTC_MAX_GUIDANCE_WEIGHT = "max_guidance_weight"
RTC_EXECUTION_HORIZON = "execution_horizon"


__all__ = [
    "ACTION",
    "IMAGES",
    "IMAGE_MASKS",
    "PREV_CHUNK_LEFT_OVER",
    "RTC_EXECUTION_HORIZON",
    "RTC_INFERENCE_DELAY",
    "RTC_MAX_GUIDANCE_WEIGHT",
    "STATE",
    "STATE_PASSTHROUGH",
    "TASK",
    "TOKENIZED_PROMPT",
    "TOKENIZED_PROMPT_MASK",
]
