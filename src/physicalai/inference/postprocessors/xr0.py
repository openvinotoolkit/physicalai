# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch-free NumPy postprocessor for the exported XR0 OpenVINO model.

The exported XR0 graph outputs a still-normalized, ``max_action_dim``-wide action
chunk. :class:`XR0Postprocessor` inverts the source action normalization
(``action * (std + eps) + mean``) and slices the padded action back to its real
dimension, mirroring the Studio-side training ``XR0Postprocessor``.  It is the
deploy-only NumPy mirror registered under the ``"xr0_denormalize"`` component type.

In ``action_mode="delta"`` the graph's denormalized output is a *delta* action
(``action[t] - state``) and the graph additionally emits the current-frame
``state`` as a second output; this postprocessor then re-adds the state
(``delta + state`` on the overlapping leading channels) to reconstruct the
absolute action, matching the pretrained flow head's delta prior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from physicalai.inference.constants import ACTION, STATE
from physicalai.inference.postprocessors.base import Postprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence

# Numerical epsilon added to the action std (matches the training convention).
_ACTION_EPS = 1e-6


class XR0Postprocessor(Postprocessor):
    """Denormalize the exported XR0 graph's action output.

    Inverts the source action normalization (``action * (std + eps) + mean``) and
    slices the padded action back to its real dimension, mirroring the training
    ``XR0Postprocessor`` in physicalai-train.

    In ``action_mode="delta"`` the denormalized output is a delta and the
    current-frame ``state`` (a second graph output) is re-added on the
    overlapping leading channels before slicing to the real action dimension.

    Args:
        action_mean: Per-dimension action mean (padded to ``max_action_dim``).
            Per-timestep ``(chunk_size, max_action_dim)`` in delta mode.
        action_std: Per-dimension action std (padded to ``max_action_dim``).
            Per-timestep ``(chunk_size, max_action_dim)`` in delta mode.
        action_dim: Real (unpadded) action dimension; when set, the output is
            sliced to it. ``None`` keeps the padded width.
        action_mode: ``"absolute"`` (default) returns the denormalized action;
            ``"delta"`` re-adds the current-frame ``state`` to the denormalized
            delta before slicing.
        action_eps: Numerical epsilon added to the std (matches training).

    Examples:
        Constructed via manifest (type-based resolution)::

            {"type": "xr0_denormalize", "action_mean": [...],
             "action_std": [...], "action_dim": 30, "action_mode": "delta"}
    """

    def __init__(
        self,
        action_mean: Sequence[float],
        action_std: Sequence[float],
        action_dim: int | None = None,
        action_mode: str = "absolute",
        action_eps: float = _ACTION_EPS,
    ) -> None:
        """Initialize the XR0 inference postprocessor.

        Raises:
            ValueError: If ``action_mean`` and ``action_std`` shapes differ.
        """
        super().__init__()
        self._mean = np.asarray(action_mean, dtype=np.float32)
        self._std = np.asarray(action_std, dtype=np.float32)
        if self._mean.shape != self._std.shape:
            msg = f"action_mean {self._mean.shape} and action_std {self._std.shape} must have the same shape"
            raise ValueError(msg)
        self._action_dim = int(action_dim) if action_dim is not None else None
        self._action_mode = str(action_mode)
        self._eps = float(action_eps)

    @override
    def __call__(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Denormalize and unpad the predicted action chunk.

        Args:
            outputs: Runner output dict containing an ``action`` array (and, in
                delta mode, the current-frame ``state`` echoed by the graph).

        Returns:
            The outputs dict with the ``action`` denormalized (and sliced to the
            real action dimension when known). In delta mode the current-frame
            state is re-added to reconstruct the absolute action.

        Raises:
            ValueError: If ``action_mode == "delta"`` but no ``state`` output is
                present to invert the delta prediction.
        """
        action = outputs.get(ACTION)
        if action is None:
            return outputs
        action = np.asarray(action, dtype=np.float32)
        # Mirror io.denormalize_action: action * (std + eps) + mean.
        action = action * (self._std + self._eps) + self._mean
        if self._action_mode == "delta":
            action = self._add_current_state(action, outputs.get(STATE))
        if self._action_dim is not None:
            action = action[..., : self._action_dim]
        result = dict(outputs)
        result[ACTION] = action
        return result

    def _add_current_state(self, action: np.ndarray, state: np.ndarray | None) -> np.ndarray:
        """Re-add the current-frame state to a denormalized delta action.

        Mirrors the Studio-side ``XR0Postprocessor.forward`` delta inverse: the
        current (last) state frame is added on the overlapping leading channels,
        broadcast over the action chunk.

        Returns:
            The reconstructed absolute action.

        Raises:
            ValueError: If ``state`` is missing.
        """
        if state is None:
            msg = "action_mode='delta' requires the graph's 'state' output to invert the delta prediction."
            raise ValueError(msg)
        current = np.asarray(state, dtype=np.float32)
        if current.ndim == 3:  # (B, T, D) -> current (last) frame  # noqa: PLR2004
            current = current[:, -1, :]
        overlap = min(action.shape[-1], current.shape[-1])
        # Insert a chunk axis so (..., overlap) broadcasts over the action chunk.
        current = np.expand_dims(current[..., :overlap], axis=-2)
        action = action.copy()
        action[..., :overlap] = action[..., :overlap] + current
        return action
