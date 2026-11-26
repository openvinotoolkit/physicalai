# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Prepare real uv projects offline to check which sources have their lockfile verified."""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from physicalai.isolate import (
    BundledProject,
    Environment,
    ProjectSource,
    SyncProfile,
    UvProvider,
    list_prepared,
    prepare,
    remove_prepared,
)

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



def test_labelled_environments_are_listed_and_removed(tmp_path: Path) -> None:
    environment = Environment(BundledProject(_locked_project(tmp_path / "project")), SyncProfile(python=sys.executable))
    cache = tmp_path / "cache"

    prepared = prepare(environment, provider=UvProvider(cache_dir=cache), offline=True, labels={"owner": "test"})
    (listed,) = list_prepared(cache_dir=cache)

    assert listed == prepared
    assert listed.labels == {"owner": "test"}
    assert listed.python.is_file()
    remove_prepared(listed, cache_dir=cache)
    assert list_prepared(cache_dir=cache) == []

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


_LFS_POINTER = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
    "size 1024\n"
)


@dataclass(frozen=True)
class _CopiedProject(ProjectSource):
    """A non-bundled source, standing in for an upstream checkout such as a GitProject."""

    path: Path

    def identity(self) -> dict[str, str]:
        return {"kind": "copied", "path": str(self.path)}

    def materialize(self, destination: Path, *, offline: bool = False) -> None:
        del offline
        shutil.copytree(self.path, destination)


def _wheel(directory: Path, name: str = "otherplatform", version: str = "0.1.0") -> Path:
    directory.mkdir(parents=True)
    wheel = directory / f"{name}-{version}-py3-none-any.whl"
    info = f"{name}-{version}.dist-info"
    files = {
        f"{name}/__init__.py": "",
        f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        f"{info}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, content in files.items():
            archive.writestr(path, content)
        archive.writestr(f"{info}/RECORD", "".join(f"{path},,\n" for path in [*files, f"{info}/RECORD"]))
    return wheel


def _project_with_other_platform_wheel_as_lfs_pointer(path: Path) -> Path:
    """Lock a project whose path wheel is only for another platform, then leave only its LFS pointer.

    This is what a checkout of Isaac-GR00T looks like without git-lfs: its lock is current,
    but the aarch64 wheels it references are pointer files.
    """
    path.mkdir()
    wheel = _wheel(path / "wheels")
    (path / "pyproject.toml").write_text(
        '[project]\nname = "worker"\nversion = "0.1.0"\nrequires-python = ">=3.10"\n'
        "dependencies = [\"otherplatform; platform_machine == 'physicalai-test-machine'\"]\n\n"
        f'[tool.uv.sources]\notherplatform = {{ path = "wheels/{wheel.name}" }}\n',
        encoding="utf-8",
    )
    subprocess.run(
        ["uv", "lock", "--offline", "--quiet", "--python", sys.executable, "--project", str(path)],
        check=True,
    )
    wheel.write_text(_LFS_POINTER, encoding="utf-8")
    return path


def test_upstream_lock_is_installed_as_is_when_an_unused_wheel_is_an_lfs_pointer(tmp_path: Path) -> None:
    project = _project_with_other_platform_wheel_as_lfs_pointer(tmp_path / "project")
    environment = Environment(_CopiedProject(project), SyncProfile(python=sys.executable))

    prepared = prepare(environment, provider=UvProvider(cache_dir=tmp_path / "cache"), offline=True)

    assert prepared.python.is_file()


def test_verifying_the_lock_reads_the_unused_wheel(tmp_path: Path) -> None:
    project = _project_with_other_platform_wheel_as_lfs_pointer(tmp_path / "project")
    environment = Environment(BundledProject(project), SyncProfile(python=sys.executable))

    with pytest.raises(subprocess.CalledProcessError):
        prepare(environment, provider=UvProvider(cache_dir=tmp_path / "cache"), offline=True)
