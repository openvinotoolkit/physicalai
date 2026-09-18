# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fuzz ResizeSmolVLA — IMAGES output must be float32 in [-1, 1]; IMAGE_MASKS must be bool-compatible."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import atheris
import numpy as np

with atheris.instrument_imports():
    from physicalai.inference.preprocessors.smolvla import ResizeSmolVLA
    from physicalai.inference.constants import IMAGE_MASKS, IMAGES

from _helpers import make_image_array

# Tolerance for [-1, 1] bound check — small epsilon for float arithmetic.
_PIXEL_TOLERANCE = 1e-5


@atheris.instrument_func
def test_one_input(data: bytes) -> None:
    if len(data) < 10:
        return

    fdp = atheris.FuzzedDataProvider(data)

    # Target resolution — clamp to [1, 256] to avoid OOM while still testing
    # aspect-ratio edge cases (very tall, very wide, square).
    target_h = fdp.ConsumeIntInRange(1, 256)
    target_w = fdp.ConsumeIntInRange(1, 256)
    channels_first = fdp.ConsumeBool()

    try:
        preprocessor = ResizeSmolVLA(
            image_resolution=(target_h, target_w),
            image_layout="BCHW" if channels_first else "BHWC",
        )
    except (ValueError, OverflowError):
        return

    img = make_image_array(fdp, max_spatial=64, channels_first=channels_first)

    # Three input presentation styles (matching ResizeSmolVLA.__call__ dispatch)
    presentation = fdp.ConsumeIntInRange(0, 2)
    if presentation == 0:
        inputs: dict = {IMAGES: img}
    elif presentation == 1:
        inputs = {IMAGES: {"cam0": img}}
    else:
        inputs = {f"{IMAGES}.cam0": img}

    try:
        outputs = preprocessor(inputs)
    except ValueError:
        # Only acceptable exception — unsupported dtype or shape
        return
    except Exception as exc:  # noqa: BLE001
        # Propagate cv2 / numpy errors that aren't ValueError
        module = type(exc).__module__ or ""
        qualname = type(exc).__qualname__ or ""
        if "cv2" in module or "cv2" in qualname:
            return  # cv2 internal errors are suppressed (same policy as resize harness)
        raise

    # Oracle 1: IMAGES output must be float32 when non-empty
    img_out = outputs.get(IMAGES)
    if isinstance(img_out, np.ndarray) and img_out.size > 0:
        assert img_out.dtype == np.float32, (
            f"ResizeSmolVLA IMAGES output dtype is {img_out.dtype}, expected float32"
        )

        # Oracle 2 (physical AI safety): pixel values must be in [-1, 1].
        # Values outside this range can produce out-of-distribution model inputs
        # that drive the robot with actions calibrated for a different input scale.
        px_min = float(img_out.min())
        px_max = float(img_out.max())
        assert px_min >= -1.0 - _PIXEL_TOLERANCE, (
            f"ResizeSmolVLA output pixel min {px_min:.6f} is below -1.0"
        )
        assert px_max <= 1.0 + _PIXEL_TOLERANCE, (
            f"ResizeSmolVLA output pixel max {px_max:.6f} is above 1.0"
        )

    # Oracle 3: IMAGE_MASKS must be bool-compatible when present
    masks_out = outputs.get(IMAGE_MASKS)
    if isinstance(masks_out, np.ndarray) and masks_out.size > 0:
        assert np.issubdtype(masks_out.dtype, np.bool_) or np.issubdtype(
            masks_out.dtype, np.integer
        ), (
            f"ResizeSmolVLA IMAGE_MASKS dtype is {masks_out.dtype}, expected bool or integer"
        )


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
