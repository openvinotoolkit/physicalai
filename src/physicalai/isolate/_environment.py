# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess  # noqa: S404  # ruff: ignore[noqa-comments] - uv uses fixed argv with shell disabled.
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows does not provide fcntl.
    _fcntl = None

if TYPE_CHECKING:
    from collections.abc import Generator

    from ._sources import ProjectSource
    from ._types import Environment, SyncProfile

_MARKER = ".physicalai-isolate.json"
_MARKER_VERSION = 1


@dataclass(frozen=True)
class PreparedEnvironment:
    """A cached, synchronized Python environment.

    Attributes:
        path: Cache directory containing ``project`` and ``venv``.
        key: SHA-256 identity of the project source and sync profile.
        profile: Profile used to synchronize the environment.
    """

    path: Path
    key: str
    profile: SyncProfile

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

    def prepare(self, environment: Environment, *, offline: bool = False) -> PreparedEnvironment:
        """Materialize and prepare *environment*.

        Returns:
            The synchronized environment.
        """


@dataclass(frozen=True)
class UvProvider:
    """Prepare locked Python projects with the ``uv`` command-line tool."""

    cache_dir: Path | None = None
    uv: str | Path | None = None

    def prepare(self, environment: Environment, *, offline: bool = False) -> PreparedEnvironment:
        """Create or reuse a cached environment for *environment*.

        Args:
            environment: Project source and sync profile to prepare.
            offline: Prevent network access during project synchronization.

        Returns:
            The prepared environment.

        Raises:
            ValueError: If the cache path is invalid.
        """
        cache_root = _cache_root(self.cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        if cache_root.is_symlink() or not cache_root.is_dir():
            msg = f"Environment cache must be a directory, not a symbolic link: {cache_root}."
            raise ValueError(msg)

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
                return cached
            _remove_cache_path(root)
            return _create_environment(root, environment, key, self._uv_executable(), offline=offline)

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
) -> PreparedEnvironment:
    """Prepare or reuse the locked interpreter environment.

    Args:
        environment: Project source and sync profile to prepare.
        provider: Environment provider. Defaults to :class:`UvProvider`.
        offline: Prevent network access while materializing and synchronizing.

    Returns:
        The prepared environment and its content key.
    """
    selected_provider = UvProvider() if provider is None else provider
    return selected_provider.prepare(environment, offline=offline)


def _create_environment(
    root: Path,
    environment: Environment,
    key: str,
    uv: str,
    *,
    offline: bool,
) -> PreparedEnvironment:
    root.mkdir()
    try:
        return _build_environment(root, environment, key, uv, offline=offline)
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
) -> PreparedEnvironment:
    project = root / "project"
    _materialize_once(environment.source, project, offline=offline)
    _validate_project(project)
    if environment.key() != key:
        msg = "The environment source changed while it was being prepared."
        raise RuntimeError(msg)

    command = _sync_command(uv, project, environment.profile, offline=offline)
    child_env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(root / "venv")}
    subprocess.run(  # noqa: S603  # ruff: ignore[noqa-comments] - fixed uv subcommand, no shell.
        command,
        cwd=project,
        env=child_env,
        check=True,
        shell=False,
    )
    prepared = PreparedEnvironment(path=root, key=key, profile=environment.profile)
    if not prepared.venv.is_dir() or not prepared.python.is_file():
        msg = f"uv did not create a Python environment at {prepared.venv}."
        raise RuntimeError(msg)
    _write_marker(root, environment, key)
    return prepared


def _cache_root(configured: Path | None) -> Path:
    if configured is not None:
        return configured.expanduser().resolve()
    override = os.environ.get("PHYSICALAI_ISOLATE_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    cache_home_setting = os.environ.get("XDG_CACHE_HOME")
    cache_home = Path(cache_home_setting).expanduser() if cache_home_setting else Path.home() / ".cache"
    return (cache_home / "physicalai" / "isolate").resolve()


def _sync_command(uv: str, project: Path, profile: SyncProfile, *, offline: bool) -> list[str]:
    command = [uv, "sync", "--frozen", "--project", str(project)]
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
    if root.is_symlink() or not root.is_dir():
        return None
    marker = _read_marker(root / _MARKER)
    if (
        marker is None
        or type(marker.get("format_version")) is not int
        or marker.get("format_version") != _MARKER_VERSION
        or marker.get("key") != key
        or marker.get("profile") != environment.profile.identity()
    ):
        return None
    project = root / "project"
    venv = root / "venv"
    python = _python_path(venv)
    if project.is_symlink() or not project.is_dir() or venv.is_symlink() or not venv.is_dir() or not python.is_file():
        return None
    return PreparedEnvironment(path=root, key=key, profile=environment.profile)


def _read_marker(path: Path) -> dict[str, object] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return marker if isinstance(marker, dict) else None


def _write_marker(root: Path, environment: Environment, key: str) -> None:
    marker = {
        "format_version": _MARKER_VERSION,
        "key": key,
        "profile": environment.profile.identity(),
        "source": environment.source.provenance(),
    }
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=root, delete=False) as file:
        json.dump(marker, file, sort_keys=True)
        file.write("\n")
        temporary = Path(file.name)
    temporary.replace(root / _MARKER)


def _remove_cache_path(path: Path) -> None:
    if path.is_symlink():
        msg = f"Refusing to remove a symbolic-link cache path: {path}."
        raise ValueError(msg)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


@contextlib.contextmanager
def _exclusive(path: Path) -> Generator[None, None, None]:
    """Serialize cache preparation across processes where file locking is available.

    Raises:
        ValueError: If the lock path is a symbolic link.
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
        if _fcntl is None:
            yield
            return
        _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX)
        try:
            yield
        finally:
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)


def _python_path(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


__all__ = ["PreparedEnvironment", "Provider", "UvProvider", "prepare"]
