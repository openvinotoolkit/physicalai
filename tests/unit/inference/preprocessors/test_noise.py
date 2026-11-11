# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from physicalai.inference.component_factory import component_registry
from physicalai.inference.constants import NOISE, STATE
from physicalai.inference.preprocessors import NoisePreprocessor


def _inputs(batch: int = 1) -> dict[str, np.ndarray]:
    return {STATE: np.zeros((batch, 6), dtype=np.float32)}


def test_registered_type() -> None:
    assert component_registry.resolve("noise") == "physicalai.inference.preprocessors.NoisePreprocessor"


def test_shape_and_dtype() -> None:
    result = NoisePreprocessor(shape=(50, 32))(_inputs(batch=2))
    assert result[NOISE].shape == (2, 50, 32)
    assert result[NOISE].dtype == np.float32


def test_batch_follows_state_with_history() -> None:
    inputs = {STATE: np.zeros((3, 2, 6), dtype=np.float32)}
    assert NoisePreprocessor(shape=(4, 2))(inputs)[NOISE].shape == (3, 4, 2)


def test_fresh_on_every_call() -> None:
    prep = NoisePreprocessor(shape=(50, 32))
    assert not np.array_equal(prep(_inputs())[NOISE], prep(_inputs())[NOISE])


def test_standard_normal() -> None:
    prep = NoisePreprocessor(shape=(50, 32), seed=0)
    noise = np.concatenate([prep(_inputs())[NOISE].ravel() for _ in range(20)])
    assert abs(noise.mean()) < 0.03
    assert abs(noise.std() - 1.0) < 0.03
    assert noise.min() < -3.0


def test_seed_makes_noise_reproducible() -> None:
    a, b = NoisePreprocessor(shape=(50, 32), seed=7), NoisePreprocessor(shape=(50, 32), seed=7)
    np.testing.assert_array_equal(a(_inputs())[NOISE], b(_inputs())[NOISE])


def test_caller_noise_is_kept() -> None:
    inputs = _inputs()
    given = np.ones((1, 50, 32), dtype=np.float32)
    inputs[NOISE] = given
    assert NoisePreprocessor(shape=(50, 32))(inputs)[NOISE] is given
