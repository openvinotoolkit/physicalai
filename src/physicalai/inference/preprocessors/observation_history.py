# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Materialize observation windows only when inference runs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from physicalai.inference.constants import IMAGES, STATE
from physicalai.inference.preprocessors.base import Preprocessor


class ObservationWindow(tuple[np.ndarray, ...]):
    """Fixed frame selection with temporal shape metadata, without stacking."""

    __slots__ = ()

    def __deepcopy__(self, memo: dict[int, Any]) -> ObservationWindow:
        """Preserve read-only ownership when RTC snapshots a request.

        Returns:
            A detached selection whose frames remain read-only.
        """
        # Startup padding repeats frames; copy each distinct frame only once.
        copied = ObservationWindow(deepcopy(frame, memo) for frame in self)
        for frame in copied:
            frame.setflags(write=False)
        memo[id(self)] = copied
        return copied

    @property
    def shape(self) -> tuple[int, ...]:
        """Temporal array dimensions available before materialization."""
        return (self[0].shape[0], len(self), *self[0].shape[1:])


class StackObservationHistory(Preprocessor):
    """Stack request-time frame selections before other preprocessors.

    Put this first in the manifest to enable deferred stacking in PolicySource.
    Already-stacked arrays and non-temporal task inputs pass through unchanged.
    """

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Materialize selected image/state tuples along the time axis.

        Returns:
            A new input dictionary with temporal arrays instead of tuples.

        Raises:
            ValueError: If selected frames have invalid ranks or inconsistent
                shapes, dtypes, batch sizes, or window lengths.
        """
        outputs = dict(inputs)
        reference = None
        for key, value in inputs.items():
            if not (key in {IMAGES, STATE} or key.startswith(f"{IMAGES}.")) or not isinstance(value, tuple):
                continue
            rank = 2 if key == STATE else 4
            if not value or any(not isinstance(frame, np.ndarray) or frame.ndim != rank for frame in value):
                msg = f"Invalid observation history selection for {key!r}"
                raise ValueError(msg)
            first = value[0]
            dimensions = (first.shape[0], len(value))
            if (reference is not None and dimensions != reference) or any(
                frame.shape != first.shape or frame.dtype != first.dtype for frame in value
            ):
                msg = "Observation history selections must have aligned batches/times and consistent shapes/dtypes"
                raise ValueError(msg)
            reference = dimensions
            outputs[key] = np.stack(value, axis=1)
        return outputs
