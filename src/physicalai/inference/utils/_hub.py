# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Helpers for loading exported policy packages from the Hugging Face Hub.

A policy package on the Hub is a repository whose snapshot mirrors a local
export directory: a ``manifest.json`` alongside the model artifacts and any
processor assets.  :func:`download_from_hub` materialises that snapshot to a
local cache directory so the rest of the inference stack can load it exactly
like a local export.
"""

from __future__ import annotations

from pathlib import Path


def download_from_hub(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | None = None,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Download an exported policy package from the Hugging Face Hub.

    Fetches a repository snapshot to a local directory and returns its path.
    The returned directory mirrors a local export directory and can be passed
    straight to :class:`~physicalai.inference.model.InferenceModel`.

    Args:
        repo_id: Hub repository identifier, e.g. ``"OpenVINO/act-fp16-ov"``.
        revision: Optional git revision (branch, tag, or commit SHA). Pin to a
            commit SHA for reproducible, tamper-evident loads.
        cache_dir: Optional cache directory for the download. Defaults to the
            standard Hugging Face cache location.
        token: Optional Hugging Face access token for private repositories.
        allow_patterns: Optional glob patterns limiting which files are
            downloaded. When ``None``, the entire snapshot is fetched so that
            arbitrary processor assets (stats, tokenizer files, ...) are not
            accidentally omitted.

    Returns:
        Path to the local snapshot directory.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
    """
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised via patched import in tests
        msg = (
            "Loading a policy from the Hugging Face Hub requires the "
            "'huggingface_hub' package. Install it with "
            "'pip install huggingface_hub'."
        )
        raise ImportError(msg) from exc

    kwargs: dict[str, object] = {}
    if token is not None:
        kwargs["token"] = token

    local_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        allow_patterns=allow_patterns,
        **kwargs,
    )

    return Path(local_path)
