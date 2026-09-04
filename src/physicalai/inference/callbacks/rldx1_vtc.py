# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""VTC frame-window callback for RLDX-1 runtime inference.

Builds temporal camera windows from single-frame observations before
preprocessing. This mirrors Studio's rollout-time VTC assembly without
introducing torch dependencies into runtime.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from typing_extensions import override

from physicalai.config import export_config
from physicalai.inference.callbacks.base import Callback
from physicalai.inference.constants import IMAGES

_OBS_IMAGES_PREFIX = "observation.images."
_OBS_IMAGE = "observation.image"


@export_config(class_path="physicalai.inference.callbacks.Rldx1VtcWindowCallback")
class Rldx1VtcWindowCallback(Callback):
    """Assemble a per-view VTC window from single-frame observations.

    Keeps per-camera ring buffers across calls and rewrites image inputs to
    ``(B, T, C, H, W)`` or ``(B, T, H, W, C)`` before preprocessors execute.

    Args:
        video_length: Number of temporal frames in the output window.
        video_stride: Env-step stride between sampled frames.
    """

    def __init__(self, video_length: int = 4, video_stride: int = 2) -> None:
        """Initialize callback state."""
        self._video_length = int(video_length)
        self._video_stride = int(video_stride)
        if self._video_length < 1 or self._video_stride < 1:
            msg = "video_length and video_stride must be positive integers."
            raise ValueError(msg)
        self._history: dict[str, deque[np.ndarray]] | None = None

    @override
    def on_reset(self) -> None:
        """Clear camera frame history for a new episode."""
        self._history = None

    @override
    def on_predict_start(self, inputs: dict[str, Any]) -> dict[str, Any] | None:
        """Apply temporal windowing to image inputs when needed.

        Returns:
            A rewritten input dict when windowing is applied, otherwise
            ``None`` to keep inputs unchanged.
        """
        if self._video_length <= 1:
            return None

        entries = self._collect_view_entries(inputs)
        if not entries:
            return None

        if _is_multiframe(entries[0][1]):
            return None

        self._record(entries)
        return self._apply_window(inputs, entries)

    def _record(self, entries: list[tuple[str, np.ndarray]]) -> None:
        if self._history is None:
            span = (self._video_length - 1) * self._video_stride
            self._history = {key: deque(maxlen=span + 1) for key, _ in entries}

        if self._history is None:
            msg = "History should be initialized before recording."
            raise RuntimeError(msg)

        window_size = (self._video_length - 1) * self._video_stride + 1
        for key, value in entries:
            if key not in self._history:
                msg = (
                    "Camera view keys changed during an active VTC window: "
                    f"received {key!r}, known keys are {sorted(self._history)}. "
                    "Call reset() at episode boundaries and keep camera keys stable within an episode."
                )
                raise ValueError(msg)
            frame = _as_batched_single_frame(value)
            history = self._history[key]
            if not history:
                history.extend([frame.copy() for _ in range(window_size)])
            else:
                history.append(frame.copy())

    def _apply_window(
        self,
        inputs: dict[str, Any],
        entries: list[tuple[str, np.ndarray]],
    ) -> dict[str, Any]:
        if self._history is None:
            msg = "History should be initialized before applying window."
            raise RuntimeError(msg)

        out = dict(inputs)
        images_value = out.get(IMAGES)
        if isinstance(images_value, dict):
            out[IMAGES] = dict(images_value)

        offsets = [(i - (self._video_length - 1)) * self._video_stride for i in range(self._video_length)]

        for key, _ in entries:
            history = self._history[key]
            count = len(history)
            frames = [history[max(0, count - 1 + offset)] for offset in offsets]
            _assign_view(out, key, np.stack(frames, axis=1))

        return out

    @staticmethod
    def _collect_view_entries(inputs: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
        entries: list[tuple[str, np.ndarray]] = []

        images_value = inputs.get(IMAGES)
        if isinstance(images_value, np.ndarray):
            return [(IMAGES, images_value)]

        if isinstance(images_value, dict):
            entries.extend(
                (f"{IMAGES}.{view_name}", np.asarray(images_value[view_name])) for view_name in sorted(images_value)
            )
            return entries

        image_keys = sorted(k for k in inputs if k.startswith(f"{IMAGES}.") and "is_pad" not in k)
        if image_keys:
            return [(key, np.asarray(inputs[key])) for key in image_keys]

        obs_image_keys = sorted(k for k in inputs if k.startswith(_OBS_IMAGES_PREFIX) and "is_pad" not in k)
        if obs_image_keys:
            return [(key, np.asarray(inputs[key])) for key in obs_image_keys]

        if _OBS_IMAGE in inputs:
            return [(_OBS_IMAGE, np.asarray(inputs[_OBS_IMAGE]))]

        return entries

    @override
    def __repr__(self) -> str:
        """Return concise callback state for debugging."""
        return f"Rldx1VtcWindowCallback(video_length={self._video_length}, video_stride={self._video_stride})"


def _as_batched_single_frame(value: np.ndarray) -> np.ndarray:
    """Normalize a view array to a batched single-frame tensor-like array.

    Accepts ``(C, H, W)``/``(H, W, C)`` or batched
    ``(B, C, H, W)``/``(B, H, W, C)``. Raises when a temporal axis is already
    present.

    Returns:
        A 4-D array shaped ``(B, C, H, W)`` or ``(B, H, W, C)``.

    Raises:
        ValueError: If the input shape is not a supported single-frame layout.
    """
    arr = np.asarray(value)
    if arr.ndim == 3:  # noqa: PLR2004
        arr = np.expand_dims(arr, axis=0)
    if arr.ndim != 4:  # noqa: PLR2004
        msg = f"Expected single-frame image input with shape (C,H,W)/(H,W,C) or (B,C,H,W)/(B,H,W,C), got {arr.shape}."
        raise ValueError(msg)
    return arr


def _is_multiframe(value: np.ndarray) -> bool:
    """Return ``True`` when the input already includes a temporal axis."""
    return np.asarray(value).ndim == 5  # noqa: PLR2004


def _assign_view(inputs: dict[str, Any], key: str, value: np.ndarray) -> None:
    """Assign a rewritten view back to nested or flat input structures."""
    images_value = inputs.get(IMAGES)
    if isinstance(images_value, dict) and key.startswith(f"{IMAGES}."):
        _, view_name = key.split(".", 1)
        images_value[view_name] = value
        return
    inputs[key] = value
