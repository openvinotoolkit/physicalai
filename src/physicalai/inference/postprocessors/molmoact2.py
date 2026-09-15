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
from physicalai.inference.preprocessors.molmoact2 import normalization_stats


class MolmoAct2Postprocessor(Postprocessor):
    """Clamp and denormalize MolmoAct2 actions."""

    def __init__(
        self,
        *,
        action_key: str,
        action_stats: dict[str, Any] | None = None,
        normalization_mode: str = "QUANTILES",
    ) -> None:
        """Store action postprocessing settings.

        Args:
            action_key: Adapter output key containing the action tensor.
            action_stats: Statistics used to denormalize actions.
            normalization_mode: Normalization strategy used during training.
        """
        self._action_key = action_key
        self.denormalizer = (
            StatsDenormalizer(
                stats={ACTION: normalization_stats(action_stats)},
                mode=normalization_mode.lower(),
                features=[ACTION],
            )
            if action_stats
            else None
        )

    @override
    def __call__(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Postprocess the model action output.

        Returns:
            Outputs with the canonical denormalized action.

        Raises:
            ValueError: If the configured action output is absent.
        """
        result = dict(outputs)
        if self._action_key not in result:
            msg = f"MolmoAct2 postprocessor expected action key {self._action_key!r}"
            raise ValueError(msg)
        action = result.pop(self._action_key)
        action = np.clip(np.asarray(action), -1.0, 1.0)
        if self.denormalizer is not None:
            action = self.denormalizer({ACTION: action})[ACTION]
        result[ACTION] = action
        return result


__all__ = ["MolmoAct2Postprocessor"]
