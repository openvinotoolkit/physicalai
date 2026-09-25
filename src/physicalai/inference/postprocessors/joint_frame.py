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
    """Apply an invertible per-joint affine transform to leading joint values.

    Forward maps robot joints to the checkpoint frame with ``sign * scale * value + offset``.
    Inverse maps them back with ``sign * (value - offset) / scale``.
    """

    def __init__(
        self,
        *,
        signs: Sequence[float],
        offsets: Sequence[float],
        scales: Sequence[float] | None = None,
    ) -> None:
        """Store the joint signs, offsets and scales.

        Args:
            signs: Per-joint direction, either -1 or 1.
            offsets: Per-joint offset added in the checkpoint frame.
            scales: Optional per-joint positive unit scale from robot to checkpoint. Defaults to 1.

        Raises:
            ValueError: If signs, offsets and scales differ in length, a sign is not +/-1,
                or a scale is not positive.
        """
        scales = [1.0] * len(signs) if scales is None else list(scales)
        if not len(signs) == len(offsets) == len(scales):
            msg = f"signs ({len(signs)}), offsets ({len(offsets)}) and scales ({len(scales)}) must match"
            raise ValueError(msg)
        if any(sign not in {-1.0, 1.0} for sign in signs):
            msg = "Joint frame transform signs must be either -1 or 1."
            raise ValueError(msg)
        scales_array = np.asarray(scales, dtype=np.float32)
        if not np.all(np.isfinite(scales_array)) or np.any(scales_array <= 0):
            msg = "Joint frame transform scales must be finite and positive."
            raise ValueError(msg)
        self._signs = np.asarray(signs, dtype=np.float32)
        self._offsets = np.asarray(offsets, dtype=np.float32)
        self._scales = scales_array

    def forward(self, values: np.ndarray) -> np.ndarray:
        """Apply ``sign * scale * value + offset`` to leading joint values.

        Returns:
            A transformed copy of ``values``.
        """
        return self._apply(values, inverse=False)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Apply ``sign * (value - offset) / scale`` to leading joint values.

        Returns:
            An inverse-transformed copy of ``values``.
        """
        return self._apply(values, inverse=True)

    def _apply(self, values: np.ndarray, *, inverse: bool) -> np.ndarray:
        count = min(self._signs.size, values.shape[-1])
        signs = self._signs[:count]
        offsets = self._offsets[:count]
        scales = self._scales[:count]

        output = np.array(values, copy=True)
        joints = values[..., :count]
        output[..., :count] = signs * (joints - offsets) / scales if inverse else signs * scales * joints + offsets
        return output


class JointFramePostprocessor(Postprocessor):
    """Map one output feature from checkpoint to robot joint coordinates."""

    def __init__(
        self,
        *,
        feature: str,
        signs: Sequence[float],
        offsets: Sequence[float],
        scales: Sequence[float] | None = None,
    ) -> None:
        """Configure the feature and calibration frame."""
        self._feature = feature
        self._transform = JointFrameTransform(signs=signs, offsets=offsets, scales=scales)

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
