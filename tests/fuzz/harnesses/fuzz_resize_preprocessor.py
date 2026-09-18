# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fuzz ResizePreprocessor — all image key presentations, zero spatial dims,
channels-first/last layouts, and extreme resolutions. Output must be float32 channels-first.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import atheris
import numpy as np

with atheris.instrument_imports():
    from physicalai.inference.preprocessors.resize import ResizePreprocessor
    from physicalai.inference.constants import IMAGES

from _helpers import make_image_array


def test_one_input(data: bytes) -> None:
    if len(data) < 10:
        return

    fdp = atheris.FuzzedDataProvider(data)

    target_h = fdp.ConsumeIntInRange(1, 512)  # clamped to avoid OOM while still testing extremes
    target_w = fdp.ConsumeIntInRange(1, 512)
    mode = fdp.PickValueInList(["stretch", "letterbox"])
    pad_value = float(fdp.ConsumeFloat())
    channels_first = fdp.ConsumeBool()

    try:
        preprocessor = ResizePreprocessor(
            image_resolution=(target_h, target_w),
            image_layout="BCHW" if channels_first else "BHWC",
            mode=mode,
            pad_value=pad_value,
        )
    except (ValueError, OverflowError):
        return

    img = make_image_array(fdp, max_spatial=64, channels_first=channels_first)

    # Three input presentation styles (flat array, nested dict, dotted key)
    presentation = fdp.ConsumeIntInRange(0, 2)
    if presentation == 0:
        inputs: dict = {IMAGES: img}
    elif presentation == 1:
        inputs = {IMAGES: {"cam0": img}}
    else:
        inputs = {f"{IMAGES}.cam0": img}

    try:
        outputs = preprocessor(inputs)
    except (ValueError, MemoryError):
        return
    except Exception as exc:
        # cv2 raises its own error type; suppress those, propagate the rest
        if "cv2" in type(exc).__module__ or "cv2" in type(exc).__qualname__:
            return
        raise

    # Oracle: non-empty ndarray outputs must be float32 channels-first
    img_out = outputs.get(IMAGES)
    if isinstance(img_out, np.ndarray) and img_out.size > 0:
        assert img_out.dtype == np.float32, (
            f"Expected float32 output, got {img_out.dtype}"
        )
        if img_out.ndim == 4:  # noqa: PLR2004
            # (B, C, H, W) — C should be small (≤4 for RGB/RGBA)
            assert img_out.shape[1] <= 4, (  # noqa: PLR2004
                f"Unexpected channel count {img_out.shape[1]} in channels-first output"
            )


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
