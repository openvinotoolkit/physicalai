# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for Physical AI Atheris fuzz harnesses."""
from __future__ import annotations

import numpy as np


def make_float_array(
    fdp,  # atheris.FuzzedDataProvider
    *,
    max_ndim: int = 4,
    max_dim: int = 64,
    dtype: type = np.float32,
) -> np.ndarray:
    """Return an ndarray of fuzz-derived random shape and float values."""
    ndim = fdp.ConsumeIntInRange(1, max_ndim)
    shape = tuple(fdp.ConsumeIntInRange(0, max_dim) for _ in range(ndim))
    total = int(np.prod(shape)) if shape else 0
    if total == 0:
        return np.zeros(shape, dtype=dtype)
    item_size = np.dtype(dtype).itemsize
    n_bytes = total * item_size
    raw = fdp.ConsumeBytes(n_bytes)
    if len(raw) < n_bytes:
        raw = raw + b"\x00" * (n_bytes - len(raw))
    return np.frombuffer(raw[:n_bytes], dtype=dtype).copy().reshape(shape)


def make_2d_float_array(
    fdp,
    *,
    max_rows: int = 64,
    max_cols: int = 32,
) -> np.ndarray:
    """Return a 2-D float32 array (rows, cols) from fuzz bytes."""
    rows = fdp.ConsumeIntInRange(0, max_rows)
    cols = fdp.ConsumeIntInRange(0, max_cols)
    total = rows * cols
    if total == 0:
        return np.zeros((rows, cols), dtype=np.float32)
    n_bytes = total * 4
    raw = fdp.ConsumeBytes(n_bytes)
    if len(raw) < n_bytes:
        raw = raw + b"\x00" * (n_bytes - len(raw))
    return np.frombuffer(raw[:n_bytes], dtype=np.float32).copy().reshape((rows, cols))


def make_2d_same_cols(
    fdp,
    cols: int,
    *,
    max_rows: int = 64,
) -> np.ndarray:
    """Return a 2-D float32 array with exactly *cols* columns."""
    rows = fdp.ConsumeIntInRange(0, max_rows)
    if rows == 0 or cols == 0:
        return np.zeros((rows, cols), dtype=np.float32)
    n_bytes = rows * cols * 4
    raw = fdp.ConsumeBytes(n_bytes)
    if len(raw) < n_bytes:
        raw = raw + b"\x00" * (n_bytes - len(raw))
    return np.frombuffer(raw[:n_bytes], dtype=np.float32).copy().reshape((rows, cols))


def make_image_array(fdp, *, max_spatial: int = 128, channels_first: bool | None = None) -> np.ndarray:
    """Return a plausible image array — channels-first (B,C,H,W) or channels-last (B,H,W,C).

    Dtype is uint8 or float32; chosen randomly from fuzz data.
    Pass ``channels_first`` when the caller needs to declare the input layout.
    """
    if channels_first is None:
        channels_first = fdp.ConsumeBool()
    B = fdp.ConsumeIntInRange(0, 4)
    C = fdp.ConsumeIntInRange(1, 4)
    H = fdp.ConsumeIntInRange(0, max_spatial)
    W = fdp.ConsumeIntInRange(0, max_spatial)
    shape = (B, C, H, W) if channels_first else (B, H, W, C)
    total = B * C * H * W
    dtype = np.uint8 if fdp.ConsumeBool() else np.float32
    if total == 0:
        return np.zeros(shape, dtype=dtype)
    item_size = np.dtype(dtype).itemsize
    n_bytes = total * item_size
    raw = fdp.ConsumeBytes(n_bytes)
    if len(raw) < n_bytes:
        raw = raw + b"\x00" * (n_bytes - len(raw))
    return np.frombuffer(raw[:n_bytes], dtype=dtype).copy().reshape(shape)


def make_stats_dict(
    fdp,
    feature_name: str,
    *,
    stat_dim: int | None = None,
    mode: str | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Return a pre-loaded stats dict suitable for StatsNormalizer/StatsDenormalizer.

    Bypasses safetensors file loading by passing stats directly via the ``stats``
    constructor kwarg.  The stat arrays use fuzz-derived float32 values including edge
    cases (inf, nan, negative std, inverted quantiles).

    Args:
        fdp: FuzzedDataProvider to consume bytes from.
        feature_name: Feature key to use in the returned stats dict.
        stat_dim: Fixed dimension for stat arrays; if None, derived from fuzz data.
        mode: Normalizer mode (``"mean_std"``, ``"min_max"``, ``"quantiles"``,
            ``"identity"``).  When provided the returned stat dict has exactly the
            keys that the corresponding normalizer expects.  When None a mode is
            picked from fuzz data — only use None when the caller does not pass
            the mode to the normalizer, otherwise stats and normalizer will be
            mismatched, causing spurious KeyError crashes.
    """
    dim = stat_dim if stat_dim is not None else fdp.ConsumeIntInRange(1, 16)
    # Use the caller-supplied mode when available; fall back to fuzz-derived choice.
    # Callers that pass the same mode to StatsNormalizer MUST also pass it here so
    # the stat dict has the correct keys and no spurious KeyError is raised.
    resolved_mode = mode if mode is not None else fdp.PickValueInList(
        ["mean_std", "min_max", "quantiles", "identity"]
    )

    def _stat() -> np.ndarray:
        n_bytes = dim * 4
        raw = fdp.ConsumeBytes(n_bytes)
        if len(raw) < n_bytes:
            raw = raw + b"\x00" * (n_bytes - len(raw))
        return np.frombuffer(raw[:n_bytes], dtype=np.float32).copy()

    if resolved_mode == "mean_std":
        return {feature_name: {"mean": _stat(), "std": _stat()}}
    elif resolved_mode == "min_max":
        return {feature_name: {"min": _stat(), "max": _stat()}}
    elif resolved_mode == "quantiles":
        return {feature_name: {"q01": _stat(), "q99": _stat()}}
    else:  # identity — no stat arrays needed; return an empty stats dict
        return {feature_name: {}}
