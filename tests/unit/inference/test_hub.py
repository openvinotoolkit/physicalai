# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Hugging Face Hub loading helpers."""

from __future__ import annotations

import builtins
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from physicalai.inference.utils._hub import download_from_hub


class TestDownloadFromHub:
    def test_passes_arguments_to_snapshot_download(self) -> None:
        fake_snapshot = MagicMock(return_value="/cache/snapshots/abc")
        fake_module = MagicMock(snapshot_download=fake_snapshot)

        with patch.dict("sys.modules", {"huggingface_hub": fake_module}):
            result = download_from_hub(
                "physical-ai/act-cube",
                revision="v1.0",
                cache_dir="/cache",
                allow_patterns=["*.json"],
            )

        assert result == Path("/cache/snapshots/abc")
        fake_snapshot.assert_called_once_with(
            repo_id="physical-ai/act-cube",
            revision="v1.0",
            cache_dir="/cache",
            allow_patterns=["*.json"],
        )

    def test_defaults(self) -> None:
        fake_snapshot = MagicMock(return_value="/cache/snapshots/abc")
        fake_module = MagicMock(snapshot_download=fake_snapshot)

        with patch.dict("sys.modules", {"huggingface_hub": fake_module}):
            download_from_hub("physical-ai/act-cube")

        fake_snapshot.assert_called_once_with(
            repo_id="physical-ai/act-cube",
            revision=None,
            cache_dir=None,
            allow_patterns=None,
        )

    def test_missing_huggingface_hub_raises_importerror(self) -> None:
        real_import = builtins.__import__

        def _fake_import(name: str, *args, **kwargs) -> object:
            if name == "huggingface_hub":
                msg = "No module named 'huggingface_hub'"
                raise ImportError(msg)
            return real_import(name, *args, **kwargs)

        with patch.dict("sys.modules", {"huggingface_hub": None}), patch(
            "builtins.__import__", side_effect=_fake_import
        ), pytest.raises(ImportError, match="huggingface_hub"):
            download_from_hub("physical-ai/act-cube")
