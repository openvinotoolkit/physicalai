# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fuzz StatsNormalizer — all four modes, extreme stat values, compatible array
shapes, passthrough-key preservation, and non-finite-stat rejection.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import atheris
import numpy as np

with atheris.instrument_imports():
    from physicalai.inference.preprocessors.stats_normalizer import StatsNormalizer

from _helpers import make_float_array, make_stats_dict

_MODES = ["mean_std", "min_max", "quantiles", "identity"]
_EPS = 1e-8


def _make_compatible_array(
    fdp: atheris.FuzzedDataProvider,
    stat_dim: int,
) -> np.ndarray:
    """Build a float32 array whose trailing dimension matches the statistics."""
    prefix_ndim = fdp.ConsumeIntInRange(0, 2)
    shape = tuple(fdp.ConsumeIntInRange(0, 16) for _ in range(prefix_ndim)) + (stat_dim,)
    n_bytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
    raw = fdp.ConsumeBytes(n_bytes)
    if len(raw) < n_bytes:
        raw += b"\x00" * (n_bytes - len(raw))
    return np.frombuffer(raw, dtype=np.float32).copy().reshape(shape)


def _result_is_representable(
    array: np.ndarray,
    stats: dict[str, np.ndarray],
    mode: str,
) -> bool:
    """Return whether the float64 reference result fits finite float32."""
    values = array.astype(np.float64)
    with np.errstate(all="ignore"):
        if mode == "mean_std":
            transformed = (values - stats["mean"].astype(np.float64)) / (stats["std"].astype(np.float64) + _EPS)
        elif mode == "min_max":
            minimum = stats["min"].astype(np.float64)
            denominator = stats["max"].astype(np.float64) - minimum + _EPS
            transformed = 2.0 * (values - minimum) / denominator - 1.0
        else:
            lower = stats["q01"].astype(np.float64)
            denominator = stats["q99"].astype(np.float64) - lower
            denominator = np.where(denominator == 0, _EPS, denominator)
            transformed = 2.0 * (values - lower) / denominator - 1.0

        mask = stats.get("mask")
        if mask is not None:
            transformed = np.where(mask.astype(np.bool_), transformed, values)

    return bool(np.all(np.isfinite(transformed)) and np.all(np.abs(transformed) <= np.finfo(np.float32).max))


def test_one_input(data: bytes) -> None:
    if len(data) < 8:
        return

    fdp = atheris.FuzzedDataProvider(data)
    mode = fdp.PickValueInList(_MODES)
    feature_name = fdp.ConsumeUnicodeNoSurrogates(32) or "observation.state"

    stat_dim = fdp.ConsumeIntInRange(1, 16)
    stats = make_stats_dict(fdp, feature_name, stat_dim=stat_dim, mode=mode)

    arr = _make_compatible_array(fdp, stat_dim)
    other_key = "passthrough_feature"
    if feature_name == other_key:
        other_key = "passthrough_feature.other"
    other_arr = make_float_array(fdp, max_ndim=2, max_dim=16)

    inputs = {feature_name: arr, other_key: other_arr}

    stats_are_non_finite = mode != "identity" and any(
        not np.all(np.isfinite(v)) for v in stats.get(feature_name, {}).values() if isinstance(v, np.ndarray)
    )

    try:
        normalizer = StatsNormalizer(mode=mode, features=[feature_name], stats=stats)
        outputs = normalizer(inputs)
    except ValueError:
        if stats_are_non_finite:
            return
        raise

    assert not stats_are_non_finite, f"StatsNormalizer accepted non-finite statistics for mode={mode!r}"

    assert other_key in outputs, f"StatsNormalizer dropped key {other_key!r} which was not in features"

    np.testing.assert_array_equal(
        outputs[other_key],
        other_arr,
        err_msg=f"StatsNormalizer modified non-listed key {other_key!r}",
    )

    if mode == "identity" and feature_name in outputs:
        np.testing.assert_array_equal(
            outputs[feature_name],
            arr,
            err_msg="identity mode should not modify the input array",
        )

    # Finite, float32-representable reference results must remain finite.
    if (
        mode != "identity"
        and arr.size > 0
        and np.all(np.isfinite(arr))
        and feature_name in outputs
        and outputs[feature_name].size > 0
    ):
        feature_stats = stats[feature_name]
        if _result_is_representable(arr, feature_stats, mode):
            assert np.all(np.isfinite(outputs[feature_name])), (
                f"StatsNormalizer produced non-finite output for a representable float32 result (mode={mode!r})"
            )


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
