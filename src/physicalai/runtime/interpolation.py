# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action interpolators that run the control loop faster than the action source."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from typing_extensions import override

from physicalai.config import export_config

if TYPE_CHECKING:
    import numpy as np


class ActionInterpolator(ABC):
    """Splits each action-source action into ``multiplier`` robot commands.

    ``RobotRuntime`` calls :meth:`add` once per action-source update and
    :meth:`get` once per control tick with the substep that tick is due for.
    The last substep returns the added action unchanged, and the first action
    after :meth:`reset` is returned as-is because there is nothing to
    interpolate from.
    """

    def __init__(self, multiplier: int = 1) -> None:
        """Create an interpolator emitting ``multiplier`` commands per action.

        Raises:
            ValueError: If ``multiplier`` is less than 1.
        """
        if multiplier < 1:
            msg = f"multiplier must be >= 1, got {multiplier}"
            raise ValueError(msg)
        self.multiplier = multiplier
        self._prev: np.ndarray | None = None
        self._target: np.ndarray | None = None
        self._done = True

    def reset(self) -> None:
        """Forget the previous action so the next one is sent without interpolation."""
        self._prev = None
        self._target = None
        self._done = True

    def needs_new_action(self) -> bool:
        """Whether the current action has been fully emitted.

        Returns:
            ``True`` before the first :meth:`add` and after :meth:`get` returned
            the added action itself.
        """
        return self._done

    def add(self, action: np.ndarray) -> None:
        """Start a new segment from the previous action towards ``action``."""
        self._prev = self._target
        self._target = action.copy()
        self._done = False

    def get(self, substep: int) -> np.ndarray:
        """Return the command for ``substep`` (0-based) of the current segment.

        Substeps at or beyond ``multiplier - 1`` return the added action and
        complete the segment, so a late tick skips straight to where it is due.

        Returns:
            The command to send this tick.

        Raises:
            RuntimeError: If called before :meth:`add`.
        """
        if self._target is None:
            msg = "get() called before add()"
            raise RuntimeError(msg)
        if self._prev is None or substep >= self.multiplier - 1:
            self._done = True
            return self._target
        return self._interpolate(self._prev, self._target, (substep + 1) / self.multiplier)

    @abstractmethod
    def _interpolate(self, prev: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
        """Return the command a fraction ``alpha`` in (0, 1) of the way from ``prev`` to ``target``."""
        raise NotImplementedError


@export_config(class_path="physicalai.runtime.LinearInterpolator")
class LinearInterpolator(ActionInterpolator):
    """Linearly interpolates between consecutive actions."""

    def __init__(self, multiplier: int = 1) -> None:
        """Create a linear interpolator emitting ``multiplier`` commands per action."""
        super().__init__(multiplier=multiplier)

    @override
    def _interpolate(self, prev: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
        return prev + alpha * (target - prev)
