# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Joint-frame action postprocessing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from physicalai.inference.postprocessors.base import Postprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence


class JointFrameTransform:
    """Apply an invertible affine transform to leading joint values."""

    def __init__(self, *, signs: Sequence[float], offsets: Sequence[float]) -> None:
        """Store the joint signs and offsets.

        Raises:
            ValueError: If signs and offsets differ in length or a sign is not +/-1.
        """
        if len(signs) != len(offsets):
            msg = f"signs ({len(signs)}) and offsets ({len(offsets)}) must match"
            raise ValueError(msg)
        if any(sign not in {-1.0, 1.0} for sign in signs):
            msg = "Joint frame transform signs must be either -1 or 1."
            raise ValueError(msg)
        self._signs = np.asarray(signs, dtype=np.float32)
        self._offsets = np.asarray(offsets, dtype=np.float32)

    def forward(self, values: np.ndarray) -> np.ndarray:
        """Apply ``sign * value + offset`` to leading joint values.

        Returns:
            A transformed copy of ``values``.
        """
        return self._apply(values, inverse=False)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Apply ``sign * (value - offset)`` to leading joint values.

        Returns:
            An inverse-transformed copy of ``values``.
        """
        return self._apply(values, inverse=True)

    def _apply(self, values: np.ndarray, *, inverse: bool) -> np.ndarray:
        count = min(self._signs.size, values.shape[-1])
        output = np.array(values, copy=True)
        joints = values[..., :count]
        output[..., :count] = (
            self._signs[:count] * (joints - self._offsets[:count])
            if inverse
            else self._signs[:count] * joints + self._offsets[:count]
        )
        return output


class JointFramePostprocessor(Postprocessor):
    """Map one output feature from checkpoint to robot joint coordinates."""

    def __init__(self, *, feature: str, signs: Sequence[float], offsets: Sequence[float]) -> None:
        """Configure the feature and calibration frame."""
        self._feature = feature
        self._transform = JointFrameTransform(signs=signs, offsets=offsets)

    @override
    def __call__(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Transform the configured feature while preserving all other outputs.

        Returns:
            A shallow copy with the transformed feature.

        Raises:
            ValueError: If the configured feature is absent.
        """
        if self._feature not in outputs:
            msg = f"Joint frame postprocessor expected feature {self._feature!r}"
            raise ValueError(msg)
        result = dict(outputs)
        result[self._feature] = self._transform.inverse(np.asarray(outputs[self._feature]))
        return result


__all__ = ["JointFramePostprocessor"]
