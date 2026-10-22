from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from physicalai.isolate import (
    BundledPatches,
    BundledProject,
    GitProject,
    PatchFile,
    PublishedPatches,
)

from physicalai.isolate import _sources

_REPOSITORY = "https://github.com/open-edge-platform/robotics-ai-suite.git"
_REVISION = "a" * 40
_TREE = "b" * 40


def test_published_patches_read_pinned_mirror_files_and_reject_tampering(tmp_path: Path) -> None:
    patch_bytes = b"diff --git a/model.py b/model.py\n"
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "0001-enable.patch").write_bytes(patch_bytes)
    patch = PatchFile("0001-enable.patch", hashlib.sha256(patch_bytes).hexdigest())
    series = PublishedPatches(
        repository=_REPOSITORY,
        revision=_REVISION,
        directory="patches/gr00t",
        files=(patch,),
        tree=_TREE,
        mirror=mirror,
    )

    assert series.patches() == [(patch.name, patch_bytes)]
    assert series.identity()["tree"] == _TREE

    (mirror / patch.name).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        series.patches()


def test_git_project_applies_pinned_patch_series_and_checks_the_resulting_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is required")
    assert git is not None
    repository = tmp_path / "origin"
    repository.mkdir()

    def git_command(*arguments: str) -> str:
        result = subprocess.run(
            [git, "-c", "user.name=test", "-c", "user.email=test@example.invalid", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
        return result.stdout.strip()

    (repository / "pyproject.toml").write_text("[project]\nname='worker'\n", encoding="utf-8")
    (repository / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (repository / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    git_command("init", "--quiet")
    git_command("add", ".")
    git_command("commit", "--quiet", "-m", "base")
    base_revision = git_command("rev-parse", "HEAD")

    (repository / "model.py").write_text("VALUE = 2\n", encoding="utf-8")
    git_command("commit", "--quiet", "-am", "enable feature")
    expected_tree = git_command("rev-parse", "HEAD^{tree}")
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    git_command("format-patch", "-1", "HEAD", "-o", str(patch_dir))
    git_command("reset", "--quiet", "--hard", base_revision)
    series = BundledPatches(patch_dir, tree=expected_tree)
    source = GitProject("https://github.com/example/worker.git", base_revision, patches=series)

    def fetch_local_repository(
        arguments: list[str],
        *,
        cwd: Path,
        capture: bool = False,
        extra_env: dict[str, str] | None = None,
    ) -> str:
        actual_arguments = arguments
        if arguments[:1] == ["fetch"]:
            actual_arguments = [
                "fetch",
                "--quiet",
                "--depth",
                "1",
                "--no-tags",
                str(repository),
                base_revision,
            ]
        result = subprocess.run(
            [git, *actual_arguments],
            cwd=cwd,
            env={
                **os.environ,
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_TERMINAL_PROMPT": "0",
                **(extra_env or {}),
            },
            check=True,
            capture_output=capture,
            text=True,
            shell=False,
        )
        return result.stdout.strip() if capture else ""

    monkeypatch.setattr(_sources, "_git", fetch_local_repository)

    checkout = tmp_path / "materialized"
    source.materialize(checkout)

    assert (checkout / "model.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (checkout / "uv.lock").is_file()


def test_git_project_materializes_the_pinned_revision_without_resolving_a_branch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "checkout"
    calls: list[list[str]] = []
    project = GitProject("https://github.com/example/worker.git", _REVISION)

    def fake_git(
        arguments: list[str],
        *,
        cwd: Path,
        capture: bool = False,
        extra_env: dict[str, str] | None = None,
    ) -> str:
        del extra_env
        calls.append(arguments)
        if "checkout" in arguments:
            (cwd / "pyproject.toml").write_text("[project]\nname='worker'\n", encoding="utf-8")
            (cwd / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        if capture:
            return _REVISION
        return ""

    monkeypatch.setattr(_sources, "_git", fake_git)

    project.materialize(destination)

    assert calls[1:4] == [
        ["remote", "add", "origin", project.repository],
        ["fetch", "--quiet", "--depth", "1", "--no-tags", "origin", _REVISION],
        ["-c", "advice.detachedHead=false", "checkout", "--quiet", "FETCH_HEAD"],
    ]
    assert calls[-1] == ["rev-parse", "HEAD"]
    assert (destination / "uv.lock").is_file()


def test_published_patch_fetch_falls_back_to_its_pinned_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contents = b"published patch"
    patch = PatchFile("0001-enable.patch", hashlib.sha256(contents).hexdigest())
    series = PublishedPatches(
        repository=_REPOSITORY,
        revision=_REVISION,
        directory="patches",
        files=(patch,),
        tree=_TREE,
        mirror=tmp_path / "empty-mirror",
    )
    downloaded: list[str] = []

    def fake_download(url: str) -> bytes:
        downloaded.append(url)
        return contents

    monkeypatch.setattr(_sources, "_download", fake_download)

    assert series.read(patch) == contents
    assert downloaded == [
        f"https://raw.githubusercontent.com/open-edge-platform/robotics-ai-suite/{_REVISION}/patches/{patch.name}"
    ]


def test_patch_paths_and_remote_revisions_must_be_immutable() -> None:
    with pytest.raises(ValueError, match="Invalid patch file name"):
        PatchFile("../escape.patch", "0" * 64)

    with pytest.raises(ValueError, match="full lowercase commit"):
        GitProject("https://github.com/NVIDIA/Isaac-GR00T.git", "main")

    with pytest.raises(ValueError, match="lowercase Git commit"):
        PublishedPatches(
            repository=_REPOSITORY,
            revision="main",
            directory="patches",
            files=(PatchFile("0001-enable.patch", "0" * 64),),
            tree=_TREE,
        )

    with pytest.raises(ValueError, match="HTTPS repository URL"):
        GitProject("http://github.com/example/worker.git", _REVISION)

    with pytest.raises(ValueError, match="credentials"):
        GitProject("https://user@github.com/example/worker.git", _REVISION)


def test_bundled_patch_identity_tracks_patch_content(tmp_path: Path) -> None:
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    patch = patch_dir / "0001-enable.patch"
    patch.write_bytes(b"first patch")
    series = BundledPatches(patch_dir)
    before = series.identity()

    patch.write_bytes(b"changed patch")

    assert before["files"][0]["sha256"] == hashlib.sha256(b"first patch").hexdigest()
    assert series.identity() != before


def test_bundled_project_rejects_symlinks_instead_of_following_them(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = tmp_path / "outside.py"
    target.write_text("EXTERNAL = True\n", encoding="utf-8")
    try:
        (project / "worker.py").symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ValueError, match="symbolic link"): 
        BundledProject(project).identity()


def test_bundled_project_identity_tracks_all_project_files(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='worker'\n", encoding="utf-8")
    (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (project / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
    source = BundledProject(project)
    identity = source.identity()
    copied = tmp_path / "copy"

    source.materialize(copied)
    (project / "worker.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert (copied / "worker.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert source.identity() != identity
