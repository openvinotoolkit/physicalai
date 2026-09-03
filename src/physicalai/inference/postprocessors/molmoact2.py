# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NumPy postprocessing for MolmoAct2 inference."""

from __future__ import annotations

from typing import Any

import numpy as np
from typing_extensions import override

from physicalai.inference.constants import ACTION
from physicalai.inference.postprocessors.base import Postprocessor
from physicalai.inference.postprocessors.stats_denormalizer import StatsDenormalizer
from physicalai.inference.preprocessors.molmoact2 import JointFrameTransform, normalization_stats


class MolmoAct2Postprocessor(Postprocessor):
    """Clamp, denormalize, and optionally convert actions to robot frame."""

    def __init__(
        self,
        *,
        action_stats: dict[str, Any] | None = None,
        normalization_mode: str = "QUANTILES",
        adapt_to_so101: bool = False,
        joint_signs: list[float] | None = None,
        joint_offsets: list[float] | None = None,
    ) -> None:
        """Store action postprocessing settings."""
        self.denormalizer = (
            StatsDenormalizer(
                stats={ACTION: normalization_stats(action_stats)},
                mode=normalization_mode.lower(),
                features=[ACTION],
            )
            if action_stats
            else None
        )
        self.joint_transform = JointFrameTransform(joint_signs, joint_offsets) if adapt_to_so101 else None

    @override
    def __call__(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Postprocess the model action output.

        Returns:
            Outputs with the canonical denormalized action.

        Raises:
            ValueError: If no action output is present.
        """
        result = dict(outputs)
        action = result.get(ACTION, result.get("actions"))
        if action is None:
            msg = "MolmoAct2 postprocessor expected an action tensor"
            raise ValueError(msg)
        action = np.clip(np.asarray(action), -1.0, 1.0)
        if self.denormalizer is not None:
            action = self.denormalizer({ACTION: action})[ACTION]
        if self.joint_transform is not None:
            action = self.joint_transform.apply(action, inverse=True)
        result.pop("actions", None)
        result[ACTION] = action
        return result


__all__ = ["MolmoAct2Postprocessor"]
