# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NumPy image layout processing for exported Patch Policy models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from typing_extensions import override

from physicalai.inference.constants import IMAGES
from physicalai.inference.preprocessors.base import Preprocessor

_SINGLE_IMAGE_NDIM = 3
_BATCHED_IMAGE_NDIM = 4
_TEMPORAL_IMAGE_NDIM = 5


class PatchPolicyPreprocessor(Preprocessor):
    """Order camera images and add batch and temporal axes for Patch Policy.

    Each output ``images.<camera>`` array has shape ``(B, T, ...)``.
    Pixel values, spatial layout, and resolution are unchanged. Compose with
    ``ToFloatTensorPreprocessor``, ``ResizePreprocessor``, and ``StatsNormalizer``
    for model-ready images.

    Args:
        camera_names: Camera names in the same order used during training.
        n_obs_steps: Full observation window length; one current frame is also accepted.
    """

    def __init__(
        self,
        camera_names: list[str],
        *,
        n_obs_steps: int = 1,
    ) -> None:
        """Initialize camera order and the accepted observation window.

        Raises:
            ValueError: If no cameras are configured or camera names are duplicated.
        """
        if not camera_names:
            msg = "Patch Policy requires at least one camera name."
            raise ValueError(msg)
        if len(set(camera_names)) != len(camera_names):
            msg = f"Patch Policy camera names must be unique, got {camera_names!r}."
            raise ValueError(msg)
        if n_obs_steps <= 0:
            msg = f"n_obs_steps must be positive, got {n_obs_steps}."
            raise ValueError(msg)
        self._camera_names = list(camera_names)
        self._n_obs_steps = n_obs_steps

    @override
    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        views = [self._prepare_view(self._resolve_view(inputs, name), name) for name in self._camera_names]
        reference_shape = views[0].shape[:2]
        for name, view in zip(self._camera_names[1:], views[1:], strict=True):
            if view.shape[:2] != reference_shape:
                msg = (
                    "Patch Policy camera batch and temporal dimensions must match; "
                    f"camera {name!r} has {view.shape[:2]}, expected {reference_shape}."
                )
                raise ValueError(msg)
        outputs = {f"{IMAGES}.{name}": view for name, view in zip(self._camera_names, views, strict=True)}
        camera_keys = {
            key for name in self._camera_names for key in (name, f"{IMAGES}.{name}", f"observation.{IMAGES}.{name}")
        }
        for key, value in inputs.items():
            if key == IMAGES or key in camera_keys:
                continue
            outputs[key] = value
        return outputs

    @staticmethod
    def _resolve_view(inputs: Mapping[str, Any], camera_name: str) -> np.ndarray:
        images = inputs.get(IMAGES)
        if isinstance(images, Mapping):
            for key in (camera_name, f"{IMAGES}.{camera_name}", f"observation.{IMAGES}.{camera_name}"):
                if key in images:
                    return np.asarray(images[key])
        elif isinstance(images, np.ndarray):
            return images

        for key in (f"{IMAGES}.{camera_name}", f"observation.{IMAGES}.{camera_name}", camera_name):
            if key in inputs:
                return np.asarray(inputs[key])
        msg = f"Missing Patch Policy camera {camera_name!r}."
        raise KeyError(msg)

    def _prepare_view(self, value: np.ndarray, camera_name: str) -> np.ndarray:
        image = np.asarray(value)
        if image.ndim == _SINGLE_IMAGE_NDIM:
            image = image[None, None]
        elif image.ndim == _BATCHED_IMAGE_NDIM:
            image = image[:, None]
        elif image.ndim != _TEMPORAL_IMAGE_NDIM:
            msg = (
                f"Patch Policy camera {camera_name!r} must have shape (C,H,W), (H,W,C), "
                f"(B,C,H,W), (B,H,W,C), or the corresponding temporal layout; got {image.shape}."
            )
            raise ValueError(msg)

        timesteps = image.shape[1]
        if timesteps not in {1, self._n_obs_steps}:
            msg = (
                f"Patch Policy camera {camera_name!r} has {timesteps} observation timestep(s); "
                f"expected 1 or {self._n_obs_steps}."
            )
            raise ValueError(msg)
        return image

    def __repr__(self) -> str:
        """Return the configured camera order and observation window."""
        return f"{self.__class__.__name__}(camera_names={self._camera_names!r}, n_obs_steps={self._n_obs_steps!r})"


__all__ = ["PatchPolicyPreprocessor"]
