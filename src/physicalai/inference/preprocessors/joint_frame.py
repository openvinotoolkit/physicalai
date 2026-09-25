# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Joint-frame observation preprocessing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import override

from physicalai.inference.postprocessors.joint_frame import JointFrameTransform
from physicalai.inference.preprocessors.base import Preprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence


class JointFramePreprocessor(Preprocessor):
    """Map one observation feature from robot to checkpoint joint coordinates."""

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
    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Transform the configured feature while preserving all other inputs.

        Returns:
            A shallow copy with the transformed feature.
        """
        key = self._resolve_key(inputs)
        outputs = dict(inputs)
        outputs[key] = self._transform.forward(np.asarray(inputs[key]))
        return outputs

    def _resolve_key(self, inputs: dict[str, Any]) -> str:
        if self._feature in inputs:
            return self._feature
        observation_key = f"observation.{self._feature}"
        if observation_key in inputs:
            return observation_key
        msg = f"Joint frame preprocessor expected feature {self._feature!r}"
        raise ValueError(msg)


__all__ = ["JointFramePreprocessor"]
