# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess  # noqa: S404  # ruff: ignore[noqa-comments] - git uses fixed argv lists with shell disabled.
import tempfile
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable
    from typing import Any

_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PATCH_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}")
_PATH_PART_RE = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_DOWNLOAD_TIMEOUT_SECONDS = 60
_MAX_PATCH_BYTES = 16 * 1024 * 1024
_GITHUB_REPOSITORY_PARTS = 2
_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "physicalai-isolate",
    "GIT_AUTHOR_EMAIL": "physicalai-isolate@localhost",
    "GIT_COMMITTER_NAME": "physicalai-isolate",
    "GIT_COMMITTER_EMAIL": "physicalai-isolate@localhost",
}


class ProjectSource(ABC):
    """Source materialized as a locked uv project."""

    @abstractmethod
    def identity(self) -> dict[str, Any]:
        """Return JSON-serializable values identifying the materialized project.

        Returns:
            Source values that determine the project's contents.
        """

    def provenance(self) -> dict[str, Any]:
        """Return source details suitable for the prepared-environment marker.

        Returns:
            The source identity, unless a source records additional details.
        """
        return self.identity()

    @abstractmethod
    def materialize(self, destination: Path, *, offline: bool = False) -> None:
        """Write this source into a new *destination* directory."""


@dataclass(frozen=True)
class BundledProject(ProjectSource):
    """A uv project shipped with the calling distribution."""

    path: Path | Traversable

    def identity(self) -> dict[str, str]:
        """Identify every file in the bundled project by its content.

        Returns:
            A deterministic tree SHA-256 digest.
        """
        return {"kind": "bundled", "tree_sha256": _tree_sha256(self.path)}

    def lock(self) -> bytes:
        """Read the bundled ``uv.lock`` file.

        Returns:
            The raw lockfile bytes.

        Raises:
            FileNotFoundError: If the project has no lockfile.
            ValueError: If the lockfile is a symbolic link.
        """
        lock = self.path.joinpath("uv.lock")
        if isinstance(lock, Path) and lock.is_symlink():
            msg = f"Bundled project lockfile cannot be a symbolic link: {lock}."
            raise ValueError(msg)
        if not lock.is_file():
            msg = f"Bundled project {self.path} does not include uv.lock."
            raise FileNotFoundError(msg)
        return lock.read_bytes()

    def materialize(self, destination: Path, *, offline: bool = False) -> None:
        """Copy the bundled project without following symbolic links.

        Raises:
            FileExistsError: If the destination already exists.
        """
        del offline
        if destination.exists() or destination.is_symlink():
            msg = f"Project destination already exists: {destination}."
            raise FileExistsError(msg)
        _copy_tree(self.path, destination)


@dataclass(frozen=True)
class PatchFile:
    """One patch file and its expected lowercase SHA-256 digest."""

    name: str
    sha256: str

    def __post_init__(self) -> None:
        """Reject unsafe file names and malformed digests.

        Raises:
            ValueError: If the file name or digest is malformed.
        """
        _validate_patch_name(self.name)
        if _SHA256_RE.fullmatch(self.sha256) is None:
            msg = f"Patch {self.name!r} needs a lowercase hexadecimal SHA-256 digest."
            raise ValueError(msg)


@dataclass(frozen=True)
class PatchSeries(ABC):
    """An ordered set of patches applied to a Git project."""

    tree: str | None = field(default=None, kw_only=True)
    license: str | None = field(default=None, kw_only=True)

    @abstractmethod
    def identity(self) -> dict[str, Any]:
        """Return JSON-serializable values identifying this patch series.

        Returns:
            Patch-source identity and any immutable pins.
        """

    @abstractmethod
    def patches(self) -> list[tuple[str, bytes]]:
        """Return verified patch names and contents in application order.

        Returns:
            Verified ``(filename, contents)`` pairs.
        """


@dataclass(frozen=True)
class BundledPatches(PatchSeries):
    """Patches owned by and shipped with the calling distribution."""

    directory: Path | Traversable
    files: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate tree pins and explicit patch order.

        Raises:
            ValueError: If the tree pin or an explicit patch name is invalid.
        """
        _validate_tree(self.tree)
        if self.files is not None:
            object.__setattr__(self, "files", tuple(self.files))
            for name in self.files:
                _validate_patch_name(name)
            if len(self.files) != len(set(self.files)):
                msg = "A bundled patch series cannot repeat a file name."
                raise ValueError(msg)

    def identity(self) -> dict[str, Any]:
        """Identify the ordered patch contents, expected tree, and license.

        Returns:
            Patch names and SHA-256 digests, plus tree and license pins.

        """
        return {
            "kind": "bundled",
            "files": [{"name": name, "sha256": hashlib.sha256(data).hexdigest()} for name, data in self.patches()],
            "tree": self.tree,
            "license": self.license,
        }

    def patches(self) -> list[tuple[str, bytes]]:
        """Read patch files in name order or the explicit order supplied.

        Returns:
            Patch names and contents in application order.

        Raises:
            FileNotFoundError: If the directory or a patch file is missing.
            ValueError: If a patch path is a symbolic link or malformed.
        """
        if isinstance(self.directory, Path) and self.directory.is_symlink():
            msg = f"Bundled patch directory cannot be a symbolic link: {self.directory}."
            raise ValueError(msg)
        if not self.directory.is_dir():
            msg = f"Bundled patch directory does not exist: {self.directory}."
            raise FileNotFoundError(msg)
        names = self.files
        if names is None:
            names = tuple(sorted(child.name for child in self.directory.iterdir() if child.name.endswith(".patch")))
        if not names:
            msg = f"No patch files found in {self.directory}."
            raise FileNotFoundError(msg)
        result: list[tuple[str, bytes]] = []
        for name in names:
            _validate_patch_name(name)
            path = self.directory.joinpath(name)
            if isinstance(path, Path) and path.is_symlink():
                msg = f"Patch file cannot be a symbolic link: {path}."
                raise ValueError(msg)
            if not path.is_file():
                msg = f"Patch file {name!r} is missing from {self.directory}."
                raise FileNotFoundError(msg)
            result.append((name, path.read_bytes()))
        return result


@dataclass(frozen=True)
class PublishedPatches(PatchSeries):
    """Patches published at an immutable GitHub revision and pinned by digest."""

    repository: str
    revision: str
    directory: str
    files: tuple[PatchFile, ...]
    mirror: Path | Traversable | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Validate immutable repository, file, and resulting-tree pins.

        Raises:
            TypeError: If a patch file is not a :class:`PatchFile`.
            ValueError: If any repository, revision, path, or tree pin is invalid.
        """
        _github_slug(self.repository)
        if _COMMIT_RE.fullmatch(self.revision) is None:
            msg = "Published patch revision must be a full lowercase Git commit ID."
            raise ValueError(msg)
        _validate_tree(self.tree, required=True)
        _validate_relative_directory(self.directory)
        object.__setattr__(self, "files", tuple(self.files))
        if not self.files:
            msg = "A patch series needs at least one file."
            raise ValueError(msg)
        if any(not isinstance(patch, PatchFile) for patch in self.files):
            msg = "Published patch files must be PatchFile instances."
            raise TypeError(msg)
        if len({patch.name for patch in self.files}) != len(self.files):
            msg = "A published patch series cannot repeat a file name."
            raise ValueError(msg)

    def identity(self) -> dict[str, Any]:
        """Return immutable pins that determine the resulting source tree.

        Returns:
            Repository, revision, directory, patch digests, tree, and license.
        """
        return {
            "kind": "published",
            "repository": self.repository,
            "revision": self.revision,
            "directory": self.directory,
            "files": [{"name": patch.name, "sha256": patch.sha256} for patch in self.files],
            "tree": self.tree,
            "license": self.license,
        }

    def patches(self) -> list[tuple[str, bytes]]:
        """Read and verify all files in application order.

        Returns:
            Verified patch names and contents.
        """
        return [(patch.name, self.read(patch)) for patch in self.files]

    def url(self, patch: PatchFile) -> str:
        """Return the raw-content URL pinned to this repository revision.

        Returns:
            An HTTPS URL on ``raw.githubusercontent.com``.
        """
        _validate_patch_name(patch.name)
        return (
            f"https://raw.githubusercontent.com/{_github_slug(self.repository)}/{self.revision}/"
            f"{self.directory}/{patch.name}"
        )

    def read(self, patch: PatchFile) -> bytes:
        """Read a mirror copy or download the pinned file, then verify its digest.

        Returns:
            The verified patch bytes.

        Raises:
            ValueError: If the file is oversized or its digest does not match its pin.
        """
        local = _mirror_path(self.mirror, patch.name) if self.mirror is not None else None
        data = local.read_bytes() if local is not None and local.is_file() else _download(self.url(patch))
        if len(data) > _MAX_PATCH_BYTES:
            msg = f"Patch {patch.name} exceeded the {_MAX_PATCH_BYTES}-byte limit."
            raise ValueError(msg)
        digest = hashlib.sha256(data).hexdigest()
        if digest != patch.sha256:
            msg = f"SHA-256 mismatch for patch {patch.name}: expected {patch.sha256}, got {digest}."
            raise ValueError(msg)
        return data


@dataclass(frozen=True)
class GitProject(ProjectSource):
    """A Git project checked out at an immutable commit, with optional patches."""

    repository: str
    revision: str
    patches: PatchSeries | tuple[PatchSeries, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the HTTPS repository, immutable commit, and patch layers.

        Raises:
            TypeError: If patch layers are not :class:`PatchSeries` instances.
            ValueError: If the repository URL or revision is invalid.
        """
        _validate_git_repository(self.repository)
        if _COMMIT_RE.fullmatch(self.revision) is None:
            msg = f"Git project revision must be a full lowercase commit ID, got {self.revision!r}."
            raise ValueError(msg)
        if isinstance(self.patches, tuple):
            if any(not isinstance(layer, PatchSeries) for layer in self.patches):
                msg = "Git project patch layers must be PatchSeries instances."
                raise TypeError(msg)
        elif self.patches is not None and not isinstance(self.patches, PatchSeries):
            msg = "Git project patches must be a PatchSeries or a tuple of PatchSeries instances."
            raise TypeError(msg)

    @property
    def layers(self) -> tuple[PatchSeries, ...]:
        """Patch series in the order they are applied."""
        if self.patches is None:
            return ()
        if isinstance(self.patches, PatchSeries):
            return (self.patches,)
        return self.patches

    def identity(self) -> dict[str, Any]:
        """Identify the upstream commit and its reviewed patch result.

        Returns:
            The immutable upstream revision and patch identity.
        """
        layers = self.layers
        if layers and layers[-1].tree is not None:
            return {"kind": "git", "revision": self.revision, "tree": layers[-1].tree}
        return {"kind": "git", "revision": self.revision, "patches": [layer.identity() for layer in layers]}

    def provenance(self) -> dict[str, Any]:
        """Record the repository and pins for each patch layer.

        Returns:
            Repository URL, commit, and patch identities.
        """
        return {
            "kind": "git",
            "repository": self.repository,
            "revision": self.revision,
            "patches": [layer.identity() for layer in self.layers],
        }

    def materialize(self, destination: Path, *, offline: bool = False) -> None:
        """Fetch the pinned commit and apply verified patch series.

        Raises:
            RuntimeError: If offline or the fetched commit/tree differs from its pin.
        """
        if offline:
            msg = f"Cannot fetch {self.repository} offline; prepare the environment with network access first."
            raise RuntimeError(msg)
        destination.mkdir(parents=True, exist_ok=False)
        with tempfile.TemporaryDirectory(prefix="physicalai-isolate-git-template-") as template:
            _git(["init", "--quiet", f"--template={template}"], cwd=destination)
        _git(["remote", "add", "origin", self.repository], cwd=destination)
        _git(["fetch", "--quiet", "--depth", "1", "--no-tags", "origin", self.revision], cwd=destination)
        _git(["-c", "advice.detachedHead=false", "checkout", "--quiet", "FETCH_HEAD"], cwd=destination)
        head = _git(["rev-parse", "HEAD"], cwd=destination, capture=True)
        if head != self.revision:
            msg = f"Fetched commit {head} does not match pinned commit {self.revision}."
            raise RuntimeError(msg)
        for layer in self.layers:
            _apply(layer, destination)


def _apply(series: PatchSeries, checkout: Path) -> None:
    patch_contents = series.patches()
    with tempfile.TemporaryDirectory(prefix="physicalai-isolate-patches-") as temporary:
        temporary_path = Path(temporary)
        hooks = temporary_path / "hooks"
        hooks.mkdir()
        patch_paths: list[str] = []
        for name, contents in patch_contents:
            _validate_patch_name(name)
            path = temporary_path / name
            path.write_bytes(contents)
            patch_paths.append(str(path))
        _git(
            [
                "-c",
                f"core.hooksPath={hooks}",
                "-c",
                "commit.gpgsign=false",
                "am",
                "--quiet",
                "--whitespace=fix",
                "--committer-date-is-author-date",
                *patch_paths,
            ],
            cwd=checkout,
            extra_env=_GIT_IDENTITY,
        )
    if series.tree is None:
        return
    tree = _git(["rev-parse", "HEAD^{tree}"], cwd=checkout, capture=True)
    if tree != series.tree:
        msg = f"Patched source tree {tree} does not match the reviewed tree {series.tree}."
        raise RuntimeError(msg)


def _validate_patch_name(name: str) -> None:
    if not isinstance(name, str) or _PATCH_NAME_RE.fullmatch(name) is None or name in {".", ".."}:
        msg = f"Invalid patch file name {name!r}."
        raise ValueError(msg)


def _validate_tree(tree: str | None, *, required: bool = False) -> None:
    if tree is None and not required:
        return
    if tree is None or _COMMIT_RE.fullmatch(tree) is None:
        msg = "A patch series tree must be a full lowercase Git object ID."
        raise ValueError(msg)


def _validate_relative_directory(directory: str) -> None:
    parts = directory.split("/")
    if not directory or any(_PATH_PART_RE.fullmatch(part) is None or part in {".", ".."} for part in parts):
        msg = f"Invalid relative patch directory {directory!r}."
        raise ValueError(msg)


def _github_slug(repository: str) -> str:
    try:
        parsed = urlsplit(repository)
        port = parsed.port
    except ValueError as exc:
        msg = f"Published patch repository must be an HTTPS GitHub repository: {repository!r}."
        raise ValueError(msg) from exc
    if parsed.scheme != "https" or parsed.hostname != "github.com":
        msg = f"Patch series must be published in an HTTPS GitHub repository, got {repository!r}."
        raise ValueError(msg)
    if parsed.username is not None or parsed.password is not None or port not in {None, 443}:
        msg = f"Patch repository URLs cannot contain credentials or a custom port: {repository!r}."
        raise ValueError(msg)
    if parsed.query or parsed.fragment:
        msg = f"Patch repository URLs cannot contain a query or fragment: {repository!r}."
        raise ValueError(msg)
    parts = parsed.path.strip("/").removesuffix(".git").split("/")
    invalid_part = any(_PATH_PART_RE.fullmatch(part) is None or part in {".", ".."} for part in parts)
    if len(parts) != _GITHUB_REPOSITORY_PARTS or invalid_part:
        msg = f"Patch series must name a GitHub owner and repository, got {repository!r}."
        raise ValueError(msg)
    return "/".join(parts)


def _validate_git_repository(repository: str) -> None:
    try:
        parsed = urlsplit(repository)
        port = parsed.port
    except ValueError as exc:
        msg = f"Git projects must use a valid HTTPS repository URL, got {repository!r}."
        raise ValueError(msg) from exc
    invalid_url = (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
    )
    if invalid_url:
        msg = f"Git projects must use an HTTPS repository URL without credentials: {repository!r}."
        raise ValueError(msg)
    invalid_path = (
        parsed.query
        or parsed.fragment
        or not parsed.path.strip("/")
        or any(part in {".", ".."} for part in parsed.path.split("/"))
    )
    if invalid_path:
        msg = f"Git project URL cannot contain a query, fragment, or path traversal: {repository!r}."
        raise ValueError(msg)


def _mirror_path(mirror: Path | Traversable, name: str) -> Path | Traversable:
    _validate_patch_name(name)
    if isinstance(mirror, Path) and mirror.is_symlink():
        msg = f"Published patch mirror cannot be a symbolic link: {mirror}."
        raise ValueError(msg)
    path = mirror.joinpath(name)
    if isinstance(path, Path):
        if path.is_symlink():
            msg = f"Mirrored patch cannot be a symbolic link: {path}."
            raise ValueError(msg)
        if isinstance(mirror, Path) and not path.resolve().is_relative_to(mirror.resolve()):
            msg = f"Mirrored patch escapes its directory: {path}."
            raise ValueError(msg)
    return path


def _download(url: str) -> bytes:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "raw.githubusercontent.com":
        msg = f"Refusing to download from an unpinned location: {url}."
        raise ValueError(msg)
    request = urllib.request.Request(  # noqa: S310  # ruff: ignore[noqa-comments] - URL is pinned to the validated HTTPS host.
        url,
        headers={"User-Agent": "physicalai-isolate"},
    )
    response_context = urllib.request.urlopen(  # noqa: S310  # ruff: ignore[noqa-comments] - HTTPS host, redirect, and digest are checked.
        request,
        timeout=_DOWNLOAD_TIMEOUT_SECONDS,
    )
    with response_context as response:
        final_url = urlsplit(response.geturl())
        if final_url.scheme != "https" or final_url.hostname != "raw.githubusercontent.com":
            msg = f"Patch download redirected outside raw.githubusercontent.com: {response.geturl()}."
            raise ValueError(msg)
        data = response.read(_MAX_PATCH_BYTES + 1)
    if len(data) > _MAX_PATCH_BYTES:
        msg = f"Patch download exceeded the {_MAX_PATCH_BYTES}-byte limit."
        raise ValueError(msg)
    return data


def _git(
    arguments: list[str],
    *,
    cwd: Path,
    capture: bool = False,
    extra_env: dict[str, str] | None = None,
) -> str:
    executable = shutil.which("git")
    if executable is None:
        msg = "Git project sources require the 'git' executable on PATH."
        raise RuntimeError(msg)
    child_env = {
        **os.environ,
        **_GIT_IDENTITY,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_COUNT": "0",
        "GIT_CONFIG_PARAMETERS": "",
        "GIT_TERMINAL_PROMPT": "0",
        **(extra_env or {}),
    }
    result = subprocess.run(  # noqa: S603  # ruff: ignore[noqa-comments] - fixed git argv and shell disabled.
        [executable, *arguments],
        cwd=cwd,
        env=child_env,
        check=True,
        shell=False,
        capture_output=capture,
        text=True,
        timeout=600,
    )
    return result.stdout.strip() if capture else ""


def _tree_sha256(root: Path | Traversable) -> str:
    if isinstance(root, Path) and root.is_symlink():
        msg = f"Bundled project cannot be a symbolic link: {root}."
        raise ValueError(msg)
    if not root.is_dir():
        msg = f"Bundled project is not a directory: {root}."
        raise FileNotFoundError(msg)

    digest = hashlib.sha256()

    def visit(path: Path | Traversable, relative: str) -> None:
        if isinstance(path, Path) and path.is_symlink():
            msg = f"Bundled project contains a symbolic link: {path}."
            raise ValueError(msg)
        if path.is_dir():
            digest.update(b"D\0")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            for child in sorted(path.iterdir(), key=lambda item: item.name):
                child_relative = f"{relative}/{child.name}" if relative else child.name
                visit(child, child_relative)
            return
        if not path.is_file():
            msg = f"Bundled project contains a special file: {path}."
            raise ValueError(msg)
        data = path.read_bytes()
        digest.update(b"F\0")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(data).to_bytes(8, byteorder="big"))
        digest.update(data)

    visit(root, "")
    return digest.hexdigest()


def _copy_tree(source: Path | Traversable, destination: Path) -> None:
    if isinstance(source, Path) and source.is_symlink():
        msg = f"Bundled project cannot be a symbolic link: {source}."
        raise ValueError(msg)
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            if not child.name or child.name in {".", ".."} or "/" in child.name or "\\" in child.name:
                msg = f"Invalid bundled project path component {child.name!r}."
                raise ValueError(msg)
            _copy_tree(child, destination / child.name)
        return
    if not source.is_file():
        msg = f"Bundled project resource is not a file or directory: {source}."
        raise FileNotFoundError(msg)
    if isinstance(source, Path) and source.is_symlink():
        msg = f"Bundled project contains a symbolic link: {source}."
        raise ValueError(msg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


__all__ = [
    "BundledPatches",
    "BundledProject",
    "GitProject",
    "PatchFile",
    "PatchSeries",
    "ProjectSource",
    "PublishedPatches",
]
