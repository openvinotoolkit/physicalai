# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Stateful low-pass filter callback for smoothing outgoing actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.config import export_config

if TYPE_CHECKING:
    import numpy as np


@export_config(class_path="physicalai.runtime.LowPassFilterCallback")
class LowPassFilterCallback:
    """Stateful low-pass filter (Exponential Moving Average) callback for smooth actions.

    Filters outgoing multidimensional joint positions/actions using a simple
    discrete one-pole IIR filter (exponential moving average):
        y_t = alpha * x_t + (1 - alpha) * y_{t-1}

    Args:
        alpha: Smoothing factor in range (0, 1]. A lower value introduces
            more smoothing (heavy low-pass filter), whereas 1.0 is a no-op.
    """

    def __init__(self, alpha: float = 0.5) -> None:  # noqa: D107
        if not (0.0 < alpha <= 1.0):
            msg = f"alpha must be in (0, 1], got {alpha}"
            raise ValueError(msg)
        self.alpha = alpha
        self._last_action: np.ndarray | None = None

    def on_action_ready(self, *, action: np.ndarray, step: int) -> np.ndarray:  # noqa: ARG002
        """Filter target action vector using previous action state.

        Args:
            action: The target raw/unfiltered joint configuration.
            step: The iteration step index in the control loop.

        Returns:
            The smoothed/filtered action target configuration.
        """
        if self._last_action is None or self._last_action.shape != action.shape:
            # First tick or shape mismatch: initialize filter state to current action
            self._last_action = action.copy()
            return action

        # Apply low-pass recursive formula
        filtered_action = self.alpha * action + (1.0 - self.alpha) * self._last_action
        self._last_action = filtered_action.copy()
        return filtered_action

    def on_action_sent(self, *, action: np.ndarray, step: int) -> None:
        """No-op."""
