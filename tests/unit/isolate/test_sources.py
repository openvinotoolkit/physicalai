from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import urllib.request
from http.client import HTTPMessage
from pathlib import Path
from typing import Any

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


def test_git_commands_ignore_the_callers_repository(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # A git hook sets these for the caller's repository; git would act on it instead of cwd.
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY"):
        monkeypatch.setenv(name, str(tmp_path / "caller" / name.lower()))
    monkeypatch.setenv("GIT_SSL_CAINFO", str(tmp_path / "ca.pem"))
    environments: list[dict[str, str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        environments.append(kwargs["env"])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(_sources.subprocess, "run", fake_run)
    monkeypatch.setattr(_sources.shutil, "which", lambda name: f"/usr/bin/{name}")

    _sources._git(["status"], cwd=tmp_path)

    [env] = environments
    assert {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY"}.isdisjoint(env)
    assert env["GIT_SSL_CAINFO"] == str(tmp_path / "ca.pem")


def _run_git(arguments: list[str], cwd: Path) -> None:
    git = shutil.which("git")
    assert git is not None
    env = {
        key: value for key, value in os.environ.items() if key.upper() not in _sources._GIT_REPOSITORY_VARIABLES
    }
    env.update(_sources._GIT_IDENTITY, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    subprocess.run([git, *arguments], cwd=cwd, env=env, check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is unavailable")
def test_patches_apply_byte_for_byte_including_trailing_whitespace(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _run_git(["init", "--quiet"], checkout)
    (checkout / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    _run_git(["add", "model.py"], checkout)
    _run_git(["commit", "--quiet", "-m", "base"], checkout)
    (checkout / "model.py").write_text("VALUE = 1\nTRAILING = 2  \n", encoding="utf-8")
    _run_git(["commit", "--quiet", "-am", "add trailing whitespace"], checkout)
    _run_git(["format-patch", "--quiet", "-1", "-o", str(tmp_path / "patches")], checkout)
    _run_git(["reset", "--quiet", "--hard", "HEAD~1"], checkout)

    _sources._apply(BundledPatches(tmp_path / "patches"), checkout)

    assert (checkout / "model.py").read_text(encoding="utf-8") == "VALUE = 1\nTRAILING = 2  \n"


def test_bundled_patch_files_must_not_be_a_single_string(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="single string"):
        BundledPatches(tmp_path, files="0001-enable.patch")  # type: ignore[arg-type]


@pytest.mark.parametrize("suffix", ["\n", "\t", "\r", " ", "\x00"])
def test_repository_urls_reject_whitespace_and_control_characters(suffix: str) -> None:
    with pytest.raises(ValueError, match="whitespace or control characters"):
        GitProject(f"https://github.com/example/worker.git{suffix}", _REVISION)

    with pytest.raises(ValueError, match="whitespace or control characters"):
        PublishedPatches(
            repository=f"https://github.com/example/{suffix}worker",
            revision=_REVISION,
            directory="patches",
            files=(PatchFile("0001-enable.patch", "0" * 64),),
            tree=_TREE,
        )


def test_patch_downloads_refuse_redirects_away_from_the_pinned_host() -> None:
    handler = _sources._PinnedRedirectHandler()
    request = urllib.request.Request(f"https://raw.githubusercontent.com/example/worker/{_REVISION}/a.patch")
    body = io.BytesIO(b"")

    with pytest.raises(ValueError, match="unpinned location"):
        handler.redirect_request(request, body, 302, "Found", HTTPMessage(), "https://example.com/a.patch")
    assert body.closed

    pinned = f"https://raw.githubusercontent.com/example/worker/{_REVISION}/b.patch"
    followed = handler.redirect_request(request, io.BytesIO(b""), 302, "Found", HTTPMessage(), pinned)
    assert followed is not None
    assert followed.full_url == pinned


def test_bundled_project_skips_caches_and_local_environments(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='worker'\n", encoding="utf-8")
    (project / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
    source = BundledProject(project)
    identity = source.identity()

    for name in ("__pycache__", ".venv", ".git", ".pytest_cache"):
        (project / name).mkdir()
        (project / name / "state").write_bytes(b"local state")
    (project / "stale.pyc").write_bytes(b"bytecode")
    copied = tmp_path / "copy"
    source.materialize(copied)

    assert source.identity() == identity
    assert sorted(path.name for path in copied.iterdir()) == ["pyproject.toml", "worker.py"]


def test_published_to_bundled_tree_pinned_patches_keep_identity_and_differ_in_provenance(tmp_path: Path) -> None:
    patch_bytes = b"diff --git a/model.py b/model.py\n"
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    (patch_dir / "0001-enable.patch").write_bytes(patch_bytes)
    published = PublishedPatches(
        repository=_REPOSITORY,
        revision=_REVISION,
        directory="patches/gr00t",
        files=(PatchFile("0001-enable.patch", hashlib.sha256(patch_bytes).hexdigest()),),
        tree=_TREE,
        license="Apache-2.0",
    )
    bundled = BundledPatches(patch_dir, tree=_TREE, license="Apache-2.0")
    from_published = GitProject("https://github.com/NVIDIA/Isaac-GR00T.git", _REVISION, patches=published)
    from_bundled = GitProject("https://github.com/NVIDIA/Isaac-GR00T.git", _REVISION, patches=bundled)

    assert from_published.identity() == from_bundled.identity()
    assert from_published.provenance() != from_bundled.provenance()
    assert from_bundled.provenance()["patches"][0]["directory"] == str(patch_dir)
    assert from_bundled.provenance()["patches"][0]["license"] == "Apache-2.0"


def test_patch_license_metadata_does_not_change_unpinned_identity(tmp_path: Path) -> None:
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    (patch_dir / "0001-enable.patch").write_bytes(b"diff --git a/a b/a\n")
    without_license = BundledPatches(patch_dir)
    with_license = BundledPatches(patch_dir, license="Apache-2.0")

    assert without_license.identity() == with_license.identity()
    assert without_license.provenance()["license"] is None
    assert with_license.provenance()["license"] == "Apache-2.0"


@pytest.mark.parametrize(
    "bad_url",
    [
        "https://raw.githubusercontent.com:8443/org/repo/rev/a.patch",
        "https://user:pass@raw.githubusercontent.com/org/repo/rev/a.patch",
        "https://raw.githubusercontent.com/org/repo/\ra.patch",
    ],
)
def test_check_download_url_rejects_ports_credentials_and_control_characters(bad_url: str) -> None:
    with pytest.raises(ValueError, match="unpinned location|whitespace or control characters"):
        _sources._check_download_url(bad_url)


@pytest.mark.skipif(os.name != "posix", reason="POSIX executable mode bits")
def test_bundled_project_materialize_preserves_executable_bits(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    script = project / "run.sh"
    script.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    script.chmod(0o755)
    copied = tmp_path / "copy"

    BundledProject(project).materialize(copied)

    assert os.access(copied / "run.sh", os.X_OK)
