# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action chunk trimmer postprocessor.

Some policies are trained with a longer action chunk than is used during
inference. Training with a longer chunk makes the action sequence smoother,
but at inference time, the tail of a long chunk is mostly useless.
This postprocessor trims the action chunk to a specified length,
following the common `n_action_steps` notation used in policy configs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from physicalai.inference.constants import ACTION
from physicalai.inference.postprocessors.base import Postprocessor

if TYPE_CHECKING:
    import numpy as np

_NDIM_WITH_TEMPORAL = 3


class ActionChunkTrimmer(Postprocessor):
    """Trim action chunk to a specified length.

    Args:
        n_action_steps: Number of action steps to trim the action chunk to.

    Examples:
        >>> trimmer = ActionChunkTrimmer(n_action_steps=10)
        >>> trimmer({"action": np.zeros((1, 50, 6))})["action"].shape
        (1, 10, 6)
    """

    def __init__(self, n_action_steps: int) -> None:
        """Initialize with the number of action steps.

        Args:
            n_action_steps: Number of action steps to trim the action chunk to.
        """
        self._n_action_steps = n_action_steps

    @override
    def __call__(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        actions = outputs[ACTION]
        if actions.ndim == _NDIM_WITH_TEMPORAL and actions.shape[1] > self._n_action_steps:
            outputs[ACTION] = actions[:, : self._n_action_steps, :]
        return outputs

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(n_action_steps={self._n_action_steps})"
