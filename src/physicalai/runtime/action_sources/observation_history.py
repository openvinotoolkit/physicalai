# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Image and state windows sampled at action-source update cadence."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

from physicalai.inference.constants import IMAGES, STATE

if TYPE_CHECKING:
    from physicalai.inference.manifest import ObservationHistorySpec


class ObservationHistory:
    """Retain aligned image and state observations for temporal model inputs."""

    def __init__(self, spec: ObservationHistorySpec) -> None:
        """Initialize bounded history from the manifest sampling settings."""
        self._spec = spec
        self._capacity = (spec.n_obs_steps - 1) * spec.stride + 1
        self._frames: deque[dict[str, np.ndarray]] = deque(maxlen=self._capacity)
        self._camera_names: tuple[str, ...] | None = None

    def reset(self) -> None:
        """Discard observations and camera identity for a new episode."""
        self._frames.clear()
        self._camera_names = None

    def window(self, inputs: dict[str, Any], *, record: bool, camera_names: tuple[str, ...]) -> dict[str, Any]:
        """Build a detached window; previews do not advance stored history.

        Returns:
            Current task inputs and aligned image/state windows.

        Raises:
            ValueError: If observation keys, shapes, or dtypes change within an
                episode, or image/state inputs have invalid ranks.
        """
        observations = {
            key: value for key, value in inputs.items() if key in {IMAGES, STATE} or key.startswith(f"{IMAGES}.")
        }
        for key, value in observations.items():
            single_frame_rank = 2 if key == STATE else 4
            if not isinstance(value, np.ndarray) or value.ndim != single_frame_rank:
                msg = (
                    "Observation history requires batched single-frame state (B,D)"
                    if key == STATE
                    else f"Observation history requires batched single-frame images for {key!r}"
                )
                raise ValueError(msg)
        if self._frames:
            previous = self._frames[-1]
            if (
                camera_names != self._camera_names
                or observations.keys() != previous.keys()
                or any(
                    value.shape != previous[key].shape or value.dtype != previous[key].dtype
                    for key, value in observations.items()
                )
            ):
                msg = "Observation history keys, shapes, and dtypes must remain stable; reset between episodes"
                raise ValueError(msg)

        # Capture ownership before camera drivers can reuse their array buffers.
        current = {key: value.copy() for key, value in observations.items()}
        frames = [*self._frames, current][-self._capacity :]
        if record:
            self._frames.append(current)
            self._camera_names = camera_names
        indices = [
            max(0, len(frames) - 1 - offset * self._spec.stride) for offset in reversed(range(self._spec.n_obs_steps))
        ]
        return {**inputs, **{key: np.stack([frames[i][key] for i in indices], axis=1) for key in observations}}
