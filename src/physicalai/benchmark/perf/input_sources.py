# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Input data sources for inference benchmarks.

An input source is any iterable yielding input dicts compatible with
:class:`~physicalai.inference.InferenceModel`. Wrapping the source
behind a small abstraction lets the benchmark consume samples from a
recorded dataset, a live capture pipeline, or a synthetic generator
without changing its measurement logic.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

from physicalai.inference.data.features import InferenceFeatureType

if TYPE_CHECKING:
    from physicalai.inference.data.features import InferenceFeature


class RandomInputSource:
    """Generate random inputs from :class:`InferenceFeature` descriptors.

    Visual features are sampled as ``uint8`` arrays in ``[0, 255]``; all
    other feature types are sampled as ``float32`` standard-normal arrays.
    A leading batch dimension of size 1 is prepended to each feature's
    declared shape.

    Examples:
        >>> source = RandomInputSource(model.input_features, seed=0)
        >>> metrics = InferenceBenchmark().run(model, source)
    """

    def __init__(
        self,
        features: list[InferenceFeature],
        seed: int | None = None,
        num_samples: int | None = None,
    ) -> None:
        """Initialize the random input source.

        Args:
            features: Feature descriptors used to determine sample shapes
                and dtypes.
            seed: Seed for the underlying :class:`numpy.random.Generator`.
                ``None`` selects a fresh, non-deterministic seed.
            num_samples: Maximum number of samples to yield. ``None``
                yields an unbounded stream.
        """
        if not features:
            msg = "RandomInputSource requires at least one InferenceFeature"
            raise ValueError(msg)
        self._features = features
        self._seed = seed
        self._num_samples = num_samples

    def __iter__(self) -> Iterator[dict[str, np.ndarray | str]]:
        rng = np.random.default_rng(self._seed)
        count = 0
        while self._num_samples is None or count < self._num_samples:
            yield {feature.name: self._sample(feature, rng) for feature in self._features}
            count += 1

    _LANGUAGE_SAMPLE_LENGTH = 30
    _LANGUAGE_ALPHABET = np.array(
        list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "),
    )

    @staticmethod
    def _sample(feature: InferenceFeature, rng: np.random.Generator) -> np.ndarray | str:
        shape = (1, *feature.shape)
        if feature.ftype is InferenceFeatureType.VISUAL:
            return rng.standard_normal(size=shape, dtype=np.float32)
        if feature.ftype is InferenceFeatureType.LANGUAGE:
            chars = rng.choice(RandomInputSource._LANGUAGE_ALPHABET, size=RandomInputSource._LANGUAGE_SAMPLE_LENGTH)
            return "".join(chars.tolist())
        return rng.standard_normal(size=shape, dtype=np.float32)
