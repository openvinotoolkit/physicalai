# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Prepare real uv projects offline to check that stale lockfiles are rejected."""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from physicalai.isolate import BundledProject, Environment, SyncProfile, UvProvider, prepare

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("uv") is None, reason="requires the uv executable"),
]


def _locked_project(path: Path) -> Path:
    path.mkdir()
    (path / "pyproject.toml").write_text(
        '[project]\nname = "worker"\nversion = "0.1.0"\nrequires-python = ">=3.10"\ndependencies = []\n',
        encoding="utf-8",
    )
    subprocess.run(
        ["uv", "lock", "--offline", "--quiet", "--python", sys.executable, "--project", str(path)],
        check=True,
    )
    return path


def test_prepare_syncs_a_project_whose_lock_is_current(tmp_path: Path) -> None:
    project = _locked_project(tmp_path / "project")
    environment = Environment(BundledProject(project), SyncProfile(python=sys.executable))

    prepared = prepare(environment, provider=UvProvider(cache_dir=tmp_path / "cache"), offline=True)

    assert prepared.python.is_file()


def test_prepare_rejects_a_lock_that_is_out_of_date(tmp_path: Path) -> None:
    project = _locked_project(tmp_path / "project")
    pyproject = project / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            "dependencies = []",
            'dependencies = ["physicalai-isolate-missing-pkg"]',
        ),
        encoding="utf-8",
    )
    environment = Environment(BundledProject(project), SyncProfile(python=sys.executable))
    cache = tmp_path / "cache"

    with pytest.raises(subprocess.CalledProcessError):
        prepare(environment, provider=UvProvider(cache_dir=cache), offline=True)

    assert not (cache / environment.key()).exists()
