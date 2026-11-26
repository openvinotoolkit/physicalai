# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import errno
import json
import os
import re
import shutil
import stat
import subprocess  # noqa: S404  # nosec B404  # ruff: ignore[noqa-comments] - uv uses fixed argv with shell disabled.
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from ._sources import BundledProject, ProjectSource, _bundled_identity, _bundled_provenance, _tree_sha256
from ._types import Environment, SyncProfile

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows does not provide fcntl.
    _fcntl = None

if sys.platform == "win32":  # pragma: no cover - msvcrt exists only on Windows.
    import msvcrt as _msvcrt
else:
    _msvcrt = None

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping
    from typing import Any

_MARKER = ".physicalai-isolate.json"
_MARKER_VERSION = 1
_WINDOWS = os.name == "nt"
_KEY_RE = re.compile(r"[0-9a-f]{64}")
_LABEL_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_LABEL_VALUE_MAX_LENGTH = 1024
# Host uv settings that contradict the sync command or change the environment outside its cache key.
_UV_OVERRIDES = frozenset({
    "UV_CONFIG_FILE",
    "UV_DEV",
    "UV_FROZEN",
    "UV_LOCKED",
    "UV_NO_DEFAULT_GROUPS",
    "UV_NO_DEV",
    "UV_NO_EDITABLE",
    "UV_NO_GROUP",
    "UV_NO_INSTALL_LOCAL",
    "UV_NO_INSTALL_PROJECT",
    "UV_NO_INSTALL_WORKSPACE",
    "UV_NO_SOURCES",
    "UV_NO_SOURCES_PACKAGE",
    "UV_OFFLINE",
    "UV_PYTHON",
    "VIRTUAL_ENV",
})


@dataclass(frozen=True)
class PreparedEnvironment:
    """A cached, synchronized Python environment.

    Attributes:
        path: Cache directory containing ``project`` and ``venv``.
        key: SHA-256 identity of the project source and sync profile.
        profile: Profile used to synchronize the environment.
        labels: Caller metadata recorded with the environment, such as the application that
            prepared it. Labels are not part of the key and do not affect equality.
    """

    path: Path
    key: str
    profile: SyncProfile
    labels: Mapping[str, str] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        """Keep labels read-only."""
        object.__setattr__(self, "labels", MappingProxyType(dict(self.labels)))

    @property
    def project(self) -> Path:
        """Materialized uv project directory."""
        return self.path / "project"

    @property
    def venv(self) -> Path:
        """Prepared virtual environment directory."""
        return self.path / "venv"

    @property
    def python(self) -> Path:
        """Prepared Python interpreter path."""
        return _python_path(self.venv)


class Provider(Protocol):
    """Interface for preparing an environment from a project declaration."""

    def prepare(
        self,
        environment: Environment,
        *,
        offline: bool = False,
        labels: Mapping[str, str] | None = None,
    ) -> PreparedEnvironment:
        """Materialize and prepare *environment*, recording *labels* unless they are ``None``.

        Args:
            environment: Project source and sync profile to prepare.
            offline: Prevent network access during preparation.
            labels: Metadata to record with the environment. ``None`` keeps recorded labels.
                :func:`prepare` passes it only when the caller gives labels, so a provider
                without label support still works for callers that do not use them.

        Returns:
            The synchronized environment.
        """


@dataclass(frozen=True)
class UvProvider:
    """Prepare locked Python projects with the ``uv`` command-line tool.

    Attributes:
        cache_dir: Optional directory storing prepared environments. Defaults to
            ``PHYSICALAI_ISOLATE_CACHE`` or ``$XDG_CACHE_HOME/physicalai/isolate``.
        uv: Optional explicit path or command name for the ``uv`` executable.
    """

    cache_dir: Path | None = None
    uv: str | Path | None = None

    def prepare(
        self,
        environment: Environment,
        *,
        offline: bool = False,
        labels: Mapping[str, str] | None = None,
    ) -> PreparedEnvironment:
        """Create or reuse a cached environment for *environment*.

        A bundled project's ``uv.lock`` must be current: ``uv sync --locked`` fails with
        :class:`subprocess.CalledProcessError` when it is out of date with ``pyproject.toml``.
        Other sources, such as an upstream :class:`GitProject`, install their lock as is
        (``uv sync --frozen``); see :attr:`ProjectSource.verify_lock`.

        Args:
            environment: Project source and sync profile to prepare.
            offline: Prevent network access during project synchronization.
            labels: Metadata to record with the environment, replacing any recorded labels.
                ``None`` keeps the labels of a cached environment.

        Returns:
            The prepared environment.

        Raises:
            ValueError: If the cache path or a label is invalid.
        """
        new_labels = None if labels is None else _validate_labels(labels)
        # The configured cache root may be a symbolic link; entries inside it may not.
        cache_root = _cache_root(self.cache_dir)
        if cache_root.exists() and not cache_root.is_dir():
            msg = f"Environment cache must be a directory: {cache_root}."
            raise ValueError(msg)
        cache_root.mkdir(parents=True, exist_ok=True)

        key = environment.key()
        root = cache_root / key
        if root.is_symlink():
            msg = f"Environment cache path cannot be a symbolic link: {root}."
            raise ValueError(msg)
        lock_path = cache_root / f".{key}.lock"

        with _exclusive(lock_path):
            if root.is_symlink():
                msg = f"Environment cache path cannot be a symbolic link: {root}."
                raise ValueError(msg)
            cached = _cached_environment(root, environment, key)
            if cached is not None:
                if new_labels is None or new_labels == cached.labels:
                    return cached
                _write_labels(root, new_labels)
                return replace(cached, labels=new_labels)
            _remove_cache_path(root)
            return _create_environment(
                root, environment, key, self._uv_executable(), offline=offline, labels=new_labels or {}
            )

    def _uv_executable(self) -> str:
        executable = shutil.which(os.fspath(self.uv) if self.uv is not None else "uv")
        if executable is None:
            msg = "Preparing isolated environments requires the 'uv' executable on PATH."
            raise RuntimeError(msg)
        return executable


def prepare(
    environment: Environment,
    *,
    provider: Provider | None = None,
    offline: bool = False,
    labels: Mapping[str, str] | None = None,
) -> PreparedEnvironment:
    """Prepare or reuse the locked interpreter environment.

    Args:
        environment: Project source and sync profile to prepare.
        provider: Environment provider. Defaults to :class:`UvProvider`.
        offline: Prevent network access while materializing and synchronizing.
        labels: Metadata to record with the environment, such as the application that prepared it,
            so :func:`list_prepared` callers can recognize their own environments. Labels replace
            any recorded labels and are not part of the cache key. ``None`` keeps recorded labels.

    Returns:
        The prepared environment.
    """
    selected_provider = UvProvider() if provider is None else provider
    if labels is None:
        # Providers written before labels existed accept only environment and offline.
        return selected_provider.prepare(environment, offline=offline)
    return selected_provider.prepare(environment, offline=offline, labels=labels)


def list_prepared(*, cache_dir: Path | None = None) -> list[PreparedEnvironment]:
    """List the complete environments in the :class:`UvProvider` cache.

    Environments that are still being prepared, or whose preparation was interrupted, are not
    listed; preparing them again replaces them.

    Args:
        cache_dir: Cache to read, as passed to :class:`UvProvider`. Defaults to its cache.

    Returns:
        Prepared environments with their recorded labels, sorted by key.
    """
    cache_root = _cache_root(cache_dir)
    if not cache_root.is_dir():
        return []
    listed = []
    for root in sorted(cache_root.iterdir()):
        # Cache entries are named by their key; this also skips the per-key lock files.
        if _KEY_RE.fullmatch(root.name) is None:
            continue
        environment = _listed_environment(root)
        if environment is not None:
            listed.append(environment)
    return listed


def remove_prepared(prepared: PreparedEnvironment, *, cache_dir: Path | None = None) -> None:
    """Remove a prepared environment from the :class:`UvProvider` cache.

    Removal waits for any preparation of the same key to finish. Sessions already started from
    the environment are not stopped, so remove only environments that are no longer in use.
    Removing an environment that is already gone does nothing. The key's small ``.<key>.lock``
    file stays in the cache, since another process may be waiting on it.

    The environment's marker is removed first, so an environment whose removal fails partway,
    such as on a file in use on Windows, is no longer listed or reused; preparing it again
    rebuilds it.

    Args:
        prepared: Environment returned by :func:`prepare` or :func:`list_prepared`.
        cache_dir: Cache that holds the environment, as passed to :class:`UvProvider`.

    Raises:
        ValueError: If the environment is not an entry of that cache or is a symbolic link.
    """
    cache_root = _cache_root(cache_dir)
    path = Path(prepared.path)
    if _KEY_RE.fullmatch(prepared.key) is None or path.name != prepared.key or path.parent.resolve() != cache_root:
        msg = f"Environment {path} is not an entry of the cache {cache_root}."
        raise ValueError(msg)
    if not cache_root.is_dir():
        return
    root = cache_root / prepared.key
    # Checked before taking the lock too, so rejecting a link leaves no lock file behind.
    if root.is_symlink():
        msg = f"Refusing to remove a symbolic-link cache path: {root}."
        raise ValueError(msg)
    with _exclusive(cache_root / f".{prepared.key}.lock"):
        # Preparation writes the marker last, so removal deletes it first: an entry that is only
        # partly removed never looks complete to list_prepared() or prepare().
        if not root.is_symlink() and root.is_dir():
            marker = root / _MARKER
            if not marker.is_symlink():
                marker.unlink(missing_ok=True)
        _remove_cache_path(root)


def _create_environment(
    root: Path,
    environment: Environment,
    key: str,
    uv: str,
    *,
    offline: bool,
    labels: Mapping[str, str],
) -> PreparedEnvironment:
    root.mkdir()
    try:
        return _build_environment(root, environment, key, uv, offline=offline, labels=labels)
    except BaseException:
        _remove_cache_path(root)
        raise


def _build_environment(
    root: Path,
    environment: Environment,
    key: str,
    uv: str,
    *,
    offline: bool,
    labels: Mapping[str, str],
) -> PreparedEnvironment:
    project = root / "project"
    _materialize_once(environment.source, project, offline=offline)
    _validate_project(project)
    source_provenance = _verify_materialized_source(environment, project, key)

    # Duck-typed sources only need identity(), materialize(), and provenance().
    verify_lock = getattr(environment.source, "verify_lock", False) is True
    command = _sync_command(uv, project, environment.profile, offline=offline, verify_lock=verify_lock)
    child_env = {name: value for name, value in os.environ.items() if name.upper() not in _UV_OVERRIDES}
    child_env["UV_PROJECT_ENVIRONMENT"] = str(root / "venv")
    subprocess.run(  # noqa: S603  # nosec B603  # ruff: ignore[noqa-comments] - fixed uv subcommand, no shell.
        command,
        cwd=project,
        env=child_env,
        check=True,
        shell=False,
    )
    prepared = PreparedEnvironment(path=root, key=key, profile=environment.profile, labels=labels)
    if not prepared.venv.is_dir() or not prepared.python.is_file():
        msg = f"uv did not create a Python environment at {prepared.venv}."
        raise RuntimeError(msg)
    _write_marker(root, environment, key, source_provenance, labels)
    return prepared


def _verify_materialized_source(environment: Environment, project: Path, key: str) -> dict[str, Any]:
    if isinstance(environment.source, BundledProject):
        # Hash the materialized copy once instead of re-reading the live source tree twice.
        tree_sha256 = _tree_sha256(project)
        snapshot = Environment(source=_BundledSnapshot(tree_sha256), profile=environment.profile)
        if snapshot.key() != key:
            msg = "The environment source changed while it was being prepared."
            raise RuntimeError(msg)
        return _bundled_provenance(environment.source.path, tree_sha256)
    if environment.key() != key:
        msg = "The environment source changed while it was being prepared."
        raise RuntimeError(msg)
    return environment.source.provenance()


@dataclass(frozen=True)
class _BundledSnapshot(ProjectSource):
    """The identity of an already materialized bundled project, used only to rebuild its key."""

    tree_sha256: str

    def identity(self) -> dict[str, str]:
        return _bundled_identity(self.tree_sha256)

    def provenance(self) -> dict[str, str]:
        return self.identity()

    @staticmethod
    def materialize(destination: Path, *, offline: bool = False) -> None:
        del destination, offline
        raise NotImplementedError


def _cache_root(configured: Path | None) -> Path:
    if configured is not None:
        return configured.expanduser().resolve()
    override = os.environ.get("PHYSICALAI_ISOLATE_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    cache_home_setting = os.environ.get("XDG_CACHE_HOME")
    cache_home = Path(cache_home_setting).expanduser() if cache_home_setting else Path.home() / ".cache"
    return (cache_home / "physicalai" / "isolate").resolve()


def _sync_command(uv: str, project: Path, profile: SyncProfile, *, offline: bool, verify_lock: bool) -> list[str]:
    command = [uv, "sync", "--locked" if verify_lock else "--frozen", "--project", str(project)]
    if profile.python is not None:
        command.extend(["--python", profile.python])
    if profile.no_default_groups:
        command.append("--no-default-groups")
    for group in profile.groups:
        command.extend(["--group", group])
    for extra in profile.extras:
        command.extend(["--extra", extra])
    if offline:
        command.append("--offline")
    return command


def _materialize_once(source: ProjectSource, destination: Path, *, offline: bool) -> None:
    if destination.is_symlink():
        msg = f"Project cache path cannot be a symbolic link: {destination}."
        raise ValueError(msg)
    if destination.exists():
        msg = f"Project cache path already exists: {destination}."
        raise FileExistsError(msg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".project-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "project"
        source.materialize(staged, offline=offline)
        if not staged.is_dir() or staged.is_symlink():
            msg = f"Project source did not create a regular directory: {staged}."
            raise ValueError(msg)
        staged.rename(destination)


def _validate_project(project: Path) -> None:
    if project.is_symlink() or not project.is_dir():
        msg = f"Materialized project must be a regular directory: {project}."
        raise ValueError(msg)
    for name in ("pyproject.toml", "uv.lock"):
        path = project / name
        if path.is_symlink() or not path.is_file():
            msg = f"Materialized project must include a regular {name}: {project}."
            raise FileNotFoundError(msg)


def _cached_environment(root: Path, environment: Environment, key: str) -> PreparedEnvironment | None:
    marker = _complete_entry_marker(root, key)
    if marker is None or marker.get("profile") != environment.profile.identity():
        return None
    return PreparedEnvironment(path=root, key=key, profile=environment.profile, labels=_marker_labels(marker))


def _listed_environment(root: Path) -> PreparedEnvironment | None:
    try:
        marker = _complete_entry_marker(root, root.name)
    except UnicodeDecodeError:
        return None
    if marker is None:
        return None
    profile = _profile_from_identity(marker.get("profile"))
    if profile is None:
        return None
    return PreparedEnvironment(path=root, key=root.name, profile=profile, labels=_marker_labels(marker))


def _complete_entry_marker(root: Path, key: str) -> dict[str, object] | None:
    """Return the marker of a fully prepared cache entry for *key*, or ``None``."""
    if root.is_symlink() or not root.is_dir():
        return None
    marker = _read_marker(root / _MARKER)
    if (
        marker is None
        or type(marker.get("format_version")) is not int
        or marker.get("format_version") != _MARKER_VERSION
        or marker.get("key") != key
    ):
        return None
    project = root / "project"
    venv = root / "venv"
    python = _python_path(venv)
    if project.is_symlink() or not project.is_dir() or venv.is_symlink() or not venv.is_dir() or not python.is_file():
        return None
    return marker


def _profile_from_identity(identity: object) -> SyncProfile | None:
    if not isinstance(identity, dict) or set(identity) != {"python", "extras", "groups", "no_default_groups"}:
        return None
    extras, groups = identity["extras"], identity["groups"]
    if not isinstance(extras, list) or not isinstance(groups, list):
        return None
    try:
        profile = SyncProfile(
            python=identity["python"],
            extras=tuple(extras),
            groups=tuple(groups),
            no_default_groups=identity["no_default_groups"],
        )
    except (TypeError, ValueError):
        return None
    return profile if profile.identity() == identity else None


def _validate_labels(labels: Mapping[str, str]) -> dict[str, str]:
    if not hasattr(labels, "items"):
        msg = "Environment labels must be a mapping of names to strings."
        raise TypeError(msg)
    validated = dict(labels.items())
    for name, value in validated.items():
        if not isinstance(name, str) or _LABEL_NAME_RE.fullmatch(name) is None:
            msg = f"Invalid environment label name {name!r}."
            raise ValueError(msg)
        if not isinstance(value, str) or len(value) > _LABEL_VALUE_MAX_LENGTH:
            msg = f"Environment label {name!r} must be a string of at most {_LABEL_VALUE_MAX_LENGTH} characters."
            raise ValueError(msg)
    return validated


def _marker_labels(marker: dict[str, object]) -> dict[str, str]:
    # Labels are advisory: a marker without valid labels still describes a usable environment.
    labels = marker.get("labels")
    if not isinstance(labels, dict):
        return {}
    try:
        return _validate_labels(labels)
    except (TypeError, ValueError):
        return {}


def _read_marker(path: Path) -> dict[str, object] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return marker if isinstance(marker, dict) else None


def _write_marker(
    root: Path,
    environment: Environment,
    key: str,
    source_provenance: dict[str, Any],
    labels: Mapping[str, str],
) -> None:
    _write_marker_data(
        root,
        {
            "format_version": _MARKER_VERSION,
            "key": key,
            "labels": dict(labels),
            "profile": environment.profile.identity(),
            "source": source_provenance,
        },
    )


def _write_labels(root: Path, labels: Mapping[str, str]) -> None:
    # Keep the recorded provenance: it describes how the cached environment was built.
    marker = _read_marker(root / _MARKER)
    if marker is None:
        msg = f"Environment cache entry has no readable marker: {root}."
        raise RuntimeError(msg)
    _write_marker_data(root, {**marker, "labels": dict(labels)})


def _write_marker_data(root: Path, marker: dict[str, object]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=root, delete=False) as file:
        json.dump(marker, file, sort_keys=True)
        file.write("\n")
        temporary = Path(file.name)
    temporary.replace(root / _MARKER)


def _handle_remove_readonly(func: Callable[[str], object], path: str, exc_info: object) -> None:
    # Windows refuses to delete read-only files such as Git packfiles. Elsewhere the parent
    # directory's permissions decide, so clearing the file's mode would not help. Never follow a
    # symbolic link: its target may live outside the cache.
    exc = exc_info[1] if isinstance(exc_info, tuple) else exc_info
    if not _WINDOWS or not isinstance(exc, PermissionError) or Path(path).is_symlink():
        raise exc if isinstance(exc, BaseException) else PermissionError(path)
    Path(path).chmod(stat.S_IWRITE)
    func(path)


def _remove_cache_path(path: Path) -> None:
    if path.is_symlink():
        msg = f"Refusing to remove a symbolic-link cache path: {path}."
        raise ValueError(msg)
    if path.is_dir():
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=_handle_remove_readonly)
        else:
            shutil.rmtree(path, onerror=_handle_remove_readonly)
    elif path.exists():
        try:
            path.unlink()
        except PermissionError:
            if not _WINDOWS:
                raise
            path.chmod(stat.S_IWRITE)
            path.unlink()


@contextlib.contextmanager
def _exclusive(path: Path) -> Generator[None, None, None]:
    """Serialize cache preparation across processes.

    Raises:
        ValueError: If the lock path is a symbolic link.
        RuntimeError: If the platform provides no file locking.
        OSError: If Windows refuses the lock for a reason other than another holder.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        msg = f"Environment cache lock cannot be a symbolic link: {path}."
        raise ValueError(msg)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "a+b") as handle:
        fd = handle.fileno()
        if _fcntl is not None:
            _fcntl.flock(fd, _fcntl.LOCK_EX)
            try:
                yield
            finally:
                _fcntl.flock(fd, _fcntl.LOCK_UN)
        elif _msvcrt is not None:
            while True:
                os.lseek(fd, 0, os.SEEK_SET)
                try:
                    _msvcrt.locking(fd, _msvcrt.LK_LOCK, 1)
                    break
                except OSError as exc:
                    # LK_LOCK gives up after about 10 seconds; keep waiting like flock does.
                    if exc.errno != errno.EDEADLOCK:
                        raise
            try:
                yield
            finally:
                os.lseek(fd, 0, os.SEEK_SET)
                _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
        else:
            msg = "Preparing isolated environments requires file locking (fcntl or msvcrt)."
            raise RuntimeError(msg)


def _python_path(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


__all__ = ["PreparedEnvironment", "Provider", "UvProvider", "list_prepared", "prepare", "remove_prepared"]
