# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Synthetic input data sources for inference benchmarks.

These helpers generate input payloads compatible with
:class:`~physicalai.inference.InferenceModel` from exported feature
descriptors so benchmarks can run without a recorded dataset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from physicalai.inference.data.features import InferenceFeatureDtype

if TYPE_CHECKING:
    from collections.abc import Iterator

    from physicalai.inference.data.features import InferenceFeature


class RandomInputSource:
    """Generate random inputs from :class:`InferenceFeature` descriptors.

    Tensor-valued features are sampled with a leading batch dimension of
    size 1 prepended to each feature's declared shape, using the dtype
    declared by the feature (``float32`` standard-normal or ``int64``
    uniform integers). ``STRING`` features are sampled as random strings.

    Examples:
        >>> source = RandomInputSource(model.input_features, seed=0)
        >>> metrics = InferenceLatencyBenchmark().run(model, source)
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

        Raises:
            ValueError: If ``features`` is empty.
        """
        if not features:
            msg = "RandomInputSource requires at least one InferenceFeature"
            raise ValueError(msg)
        self._features = features
        self._seed = seed
        self._num_samples = num_samples

    def __iter__(self) -> Iterator[dict[str, np.ndarray | list[str]]]:
        """Yield randomly generated benchmark samples."""
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
    def _sample(feature: InferenceFeature, rng: np.random.Generator) -> np.ndarray | list[str]:
        """Generate one random value compatible with ``feature`` metadata.

        Returns:
            A random sample matching the feature's declared dtype: a numpy
            array for tensor-valued features, or a list of strings for ``STRING``
            dtype features.

        Raises:
            ValueError: If the feature declares an unsupported dtype.
        """
        if feature.dtype == InferenceFeatureDtype.STRING:
            chars = rng.choice(RandomInputSource._LANGUAGE_ALPHABET, size=RandomInputSource._LANGUAGE_SAMPLE_LENGTH)
            return ["".join(chars.tolist())]
        shape = (1, *feature.shape)
        if feature.dtype == InferenceFeatureDtype.FLOAT32:
            return rng.standard_normal(size=shape, dtype=np.float32)
        if feature.dtype == InferenceFeatureDtype.INT64:
            return rng.integers(low=0, high=2, size=shape, dtype=np.int64)
        msg = f"Unsupported feature dtype: {feature.dtype!r}"
        raise ValueError(msg)
