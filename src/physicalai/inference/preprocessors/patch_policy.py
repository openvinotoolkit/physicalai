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
from physicalai.inference.preprocessors.resize import ResizeMode, ResizePreprocessor

_SINGLE_IMAGE_NDIM = 3
_BATCHED_IMAGE_NDIM = 4
_TEMPORAL_IMAGE_NDIM = 5


class PatchPolicyPreprocessor(Preprocessor):
    """Order, resize, and stack camera images for Patch Policy.

    The output ``images`` array has shape ``(B, T, V, C, H, W)``. Pixel
    normalization is deliberately left to the generic
    :class:`~physicalai.inference.preprocessors.StatsNormalizer` stage.

    Args:
        camera_names: Camera names in the same order used during training.
        image_resolution: Target image height and width.
        image_layout: Input image layout. When omitted, the shared resize
            processor infers the layout.
        n_obs_steps: Required number of temporal observation frames.
    """

    def __init__(
        self,
        camera_names: list[str],
        image_resolution: tuple[int, int],
        *,
        image_layout: str | None = None,
        n_obs_steps: int = 1,
    ) -> None:
        """Initialize the camera layout and shared stretch resizer.

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
        self._image_resolution = tuple(image_resolution)
        self._n_obs_steps = n_obs_steps
        self._resizer = ResizePreprocessor(
            self._image_resolution,
            image_layout=image_layout,
            mode=ResizeMode.STRETCH,
        )

    @override
    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        outputs = dict(inputs)
        views = [self._prepare_view(self._resolve_view(inputs, name), name) for name in self._camera_names]
        reference_shape = views[0].shape[:2]
        for name, view in zip(self._camera_names[1:], views[1:], strict=True):
            if view.shape[:2] != reference_shape:
                msg = (
                    "Patch Policy camera batch and temporal dimensions must match; "
                    f"camera {name!r} has {view.shape[:2]}, expected {reference_shape}."
                )
                raise ValueError(msg)
        outputs[IMAGES] = np.stack(views, axis=2)
        return outputs

    @staticmethod
    def _resolve_view(inputs: Mapping[str, Any], camera_name: str) -> np.ndarray:
        images = inputs.get(IMAGES)
        if isinstance(images, Mapping):
            for key in (camera_name, f"observation.images.{camera_name}"):
                if key in images:
                    return np.asarray(images[key])
        elif isinstance(images, np.ndarray):
            return images

        for key in (f"{IMAGES}.{camera_name}", f"observation.images.{camera_name}", camera_name):
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

        batch, timesteps = image.shape[:2]
        if timesteps != self._n_obs_steps:
            msg = (
                f"Patch Policy camera {camera_name!r} has {timesteps} observation timestep(s); "
                f"expected {self._n_obs_steps}."
            )
            raise ValueError(msg)
        flattened = image.reshape(batch * timesteps, *image.shape[2:])
        resized = self._resizer({IMAGES: flattened})[IMAGES]
        if not isinstance(resized, np.ndarray):
            msg = "Patch Policy image resize produced an invalid nested result."
            raise TypeError(msg)
        return resized.reshape(batch, timesteps, *resized.shape[1:])

    def __repr__(self) -> str:
        """Return the configured camera order and target resolution."""
        return (
            f"{self.__class__.__name__}(camera_names={self._camera_names!r}, "
            f"image_resolution={self._image_resolution!r})"
        )


__all__ = ["PatchPolicyPreprocessor"]
