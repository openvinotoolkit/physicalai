from __future__ import annotations

import errno
import os
import shutil
import stat
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from physicalai.isolate import (
    BundledProject,
    Environment,
    GitProject,
    ProjectSource,
    SyncProfile,
    UvProvider,
    _environment,
    prepare,
)


def _bundled_project(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "pyproject.toml").write_text("[project]\nname = 'worker'\nversion = '0.1.0'\n", encoding="utf-8")
    (path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (path / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
    return path


def test_prepare_rejects_projects_without_a_lock_and_cleans_partial_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "unlocked"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='worker'\n", encoding="utf-8")
    environment = Environment(source=BundledProject(project))
    cache = tmp_path / "cache"
    provider = UvProvider(cache_dir=cache, uv=sys.executable)

    def unexpected_uv(*args: object, **kwargs: object) -> None:
        pytest.fail("uv must not run for a project without uv.lock")

    monkeypatch.setattr("physicalai.isolate._environment.subprocess.run", unexpected_uv)

    with pytest.raises(FileNotFoundError, match="uv.lock"):
        prepare(environment, provider=provider)

    assert not (cache / environment.key()).exists()


def test_prepare_caches_by_locked_project_and_sync_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = _bundled_project(tmp_path / "project")
    environment = Environment(
        source=BundledProject(project),
        profile=SyncProfile(python="3.12", extras=("cpu",), groups=("worker",), no_default_groups=True),
    )
    provider = UvProvider(cache_dir=tmp_path / "cache", uv=sys.executable)
    calls: list[list[str]] = []

    def fake_uv(command: list[str], *, cwd: Path, env: dict[str, str], check: bool, shell: bool) -> None:
        assert cwd == Path(command[command.index("--project") + 1])
        assert check
        assert not shell
        calls.append(command)
        venv = Path(env["UV_PROJECT_ENVIRONMENT"])
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_bytes(b"prepared interpreter")

    monkeypatch.setattr("physicalai.isolate._environment.subprocess.run", fake_uv)

    first = prepare(environment, provider=provider, offline=True)
    cached = prepare(environment, provider=provider)

    assert first == cached
    assert first.project.joinpath("worker.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert first.python.is_file()
    assert calls == [
        [
            sys.executable,
            "sync",
            "--locked",
            "--project",
            str(first.project),
            "--python",
            "3.12",
            "--no-default-groups",
            "--group",
            "worker",
            "--extra",
            "cpu",
            "--offline",
        ],
    ]

    alternate_profile = Environment(
        source=BundledProject(project),
        profile=SyncProfile(python="3.12", extras=("cu128",)),
    )
    alternate = prepare(alternate_profile, provider=provider)
    assert alternate.key != first.key
    assert alternate.path != first.path

    (project / "worker.py").write_text("VALUE = 2\n", encoding="utf-8")
    updated = prepare(environment, provider=provider)

    assert updated.key != first.key
    assert updated.path != first.path
    assert len(calls) == 3



@dataclass(frozen=True)
class _CopiedProject(ProjectSource):
    """A non-bundled source, standing in for an upstream checkout such as a GitProject."""

    path: Path

    def identity(self) -> dict[str, str]:
        return {"kind": "copied", "path": str(self.path)}

    def materialize(self, destination: Path, *, offline: bool = False) -> None:
        del offline
        shutil.copytree(self.path, destination)


def test_only_bundled_projects_have_their_lock_verified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert BundledProject.verify_lock
    assert not GitProject.verify_lock
    project = _bundled_project(tmp_path / "project")
    provider = UvProvider(cache_dir=tmp_path / "cache", uv=sys.executable)
    modes: list[str] = []

    def fake_uv(command: list[str], *, cwd: Path, env: dict[str, str], check: bool, shell: bool) -> None:
        del cwd, check, shell
        modes.append(command[2])
        python = Path(env["UV_PROJECT_ENVIRONMENT"]) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_bytes(b"prepared interpreter")

    monkeypatch.setattr("physicalai.isolate._environment.subprocess.run", fake_uv)

    prepare(Environment(source=BundledProject(project)), provider=provider)
    prepare(Environment(source=_CopiedProject(project)), provider=provider)

    assert modes == ["--locked", "--frozen"]

def test_concurrent_prepares_for_the_same_key_build_the_environment_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = Environment(source=BundledProject(_bundled_project(tmp_path / "project")))
    provider = UvProvider(cache_dir=tmp_path / "cache", uv=sys.executable)
    syncs: list[list[str]] = []

    def slow_uv(command: list[str], *, cwd: Path, env: dict[str, str], check: bool, shell: bool) -> None:
        del cwd, check, shell
        syncs.append(command)
        python = Path(env["UV_PROJECT_ENVIRONMENT"]) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_bytes(b"prepared interpreter")
        time.sleep(0.5)
        if not python.is_file():
            msg = "Another prepare removed this environment while it was being built."
            raise RuntimeError(msg)

    monkeypatch.setattr("physicalai.isolate._environment.subprocess.run", slow_uv)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(lambda _: prepare(environment, provider=provider), range(2))

    assert first == second
    assert first.python.is_file()
    assert len(syncs) == 1


class _FakeMsvcrt:
    LK_UNLCK = 0
    LK_LOCK = 1

    def __init__(self, failures: list[OSError]) -> None:
        self.calls: list[tuple[int, int, int]] = []
        self._failures = failures

    def locking(self, fd: int, mode: int, nbytes: int) -> None:
        self.calls.append((mode, nbytes, os.lseek(fd, 0, os.SEEK_CUR)))
        if self._failures:
            raise self._failures.pop(0)


def test_windows_lock_keeps_waiting_for_a_busy_lock_and_releases_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake = _FakeMsvcrt([OSError(errno.EDEADLOCK, "locked by another process")])
    monkeypatch.setattr(_environment, "_fcntl", None)
    monkeypatch.setattr(_environment, "_msvcrt", fake)

    with _environment._exclusive(tmp_path / "key.lock"):
        assert fake.calls == [(fake.LK_LOCK, 1, 0), (fake.LK_LOCK, 1, 0)]

    assert fake.calls[-1] == (fake.LK_UNLCK, 1, 0)


def test_windows_lock_passes_on_unexpected_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _FakeMsvcrt([PermissionError(errno.EACCES, "denied")])
    monkeypatch.setattr(_environment, "_fcntl", None)
    monkeypatch.setattr(_environment, "_msvcrt", fake)

    with pytest.raises(PermissionError), _environment._exclusive(tmp_path / "key.lock"):
        pytest.fail("The lock must not be entered.")

    assert fake.calls == [(fake.LK_LOCK, 1, 0)]


def test_prepare_refuses_to_run_without_file_locking(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    environment = Environment(source=BundledProject(_bundled_project(tmp_path / "project")))
    monkeypatch.setattr(_environment, "_fcntl", None)
    monkeypatch.setattr(_environment, "_msvcrt", None)

    def unexpected_uv(*args: object, **kwargs: object) -> None:
        pytest.fail("uv must not run without a lock")

    monkeypatch.setattr("physicalai.isolate._environment.subprocess.run", unexpected_uv)

    with pytest.raises(RuntimeError, match="file locking"):
        prepare(environment, provider=UvProvider(cache_dir=tmp_path / "cache", uv=sys.executable))


def _recording_uv(records: list[tuple[list[str], dict[str, str]]]) -> Any:
    def fake_uv(command: list[str], *, cwd: Path, env: dict[str, str], check: bool, shell: bool) -> None:
        del cwd, check, shell
        records.append((command, env))
        python = Path(env["UV_PROJECT_ENVIRONMENT"]) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_bytes(b"prepared interpreter")

    return fake_uv


def test_prepare_ignores_host_uv_settings_that_change_the_sync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in ("UV_FROZEN", "UV_LOCKED", "UV_NO_DEFAULT_GROUPS", "UV_NO_SOURCES"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("UV_PYTHON", "3.10")
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "host-venv"))
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    records: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr("physicalai.isolate._environment.subprocess.run", _recording_uv(records))
    environment = Environment(source=BundledProject(_bundled_project(tmp_path / "project")))

    prepare(environment, provider=UvProvider(cache_dir=tmp_path / "cache", uv=sys.executable))

    [(_, env)] = records
    removed = {"UV_FROZEN", "UV_LOCKED", "UV_NO_DEFAULT_GROUPS", "UV_NO_SOURCES", "UV_PYTHON", "VIRTUAL_ENV"}
    assert removed.isdisjoint(env)
    assert env["UV_CACHE_DIR"] == str(tmp_path / "uv-cache")


def test_prepare_strips_extra_uv_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in ("UV_NO_GROUP", "UV_NO_INSTALL_WORKSPACE", "UV_NO_SOURCES_PACKAGE", "UV_OFFLINE", "UV_CONFIG_FILE"):
        monkeypatch.setenv(name, "1")
    records: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr("physicalai.isolate._environment.subprocess.run", _recording_uv(records))
    environment = Environment(source=BundledProject(_bundled_project(tmp_path / "project")))

    prepare(environment, provider=UvProvider(cache_dir=tmp_path / "cache", uv=sys.executable))

    [(_, env)] = records
    assert {"UV_NO_GROUP", "UV_NO_INSTALL_WORKSPACE", "UV_NO_SOURCES_PACKAGE", "UV_OFFLINE", "UV_CONFIG_FILE"}.isdisjoint(
        env,
    )


def test_prepare_records_materialized_provenance_even_if_source_changes_during_sync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = _bundled_project(tmp_path / "project")
    environment = Environment(source=BundledProject(project))
    original_tree = environment.source.identity()["tree_sha256"]

    def mutating_uv(command: list[str], *, cwd: Path, env: dict[str, str], check: bool, shell: bool) -> None:
        del command, cwd, check, shell
        (project / "worker.py").write_text("VALUE = 999\n", encoding="utf-8")
        python = Path(env["UV_PROJECT_ENVIRONMENT"]) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_bytes(b"prepared interpreter")

    monkeypatch.setattr("physicalai.isolate._environment.subprocess.run", mutating_uv)

    prepared = prepare(environment, provider=UvProvider(cache_dir=tmp_path / "cache", uv=sys.executable))
    marker = _environment._read_marker(prepared.path / _environment._MARKER)

    assert marker is not None
    assert marker["source"] == {"kind": "bundled", "path": str(project), "tree_sha256": original_tree}
    assert (prepared.project / "worker.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_remove_cache_path_clears_readonly_files(tmp_path: Path) -> None:
    root = tmp_path / "cache-entry"
    pack_dir = root / "project" / ".git" / "objects" / "pack"
    pack_dir.mkdir(parents=True)
    readonly = pack_dir / "pack-123.idx"
    readonly.write_bytes(b"pack")
    readonly.chmod(0o444)

    _environment._remove_cache_path(root)

    assert not root.exists()


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs POSIX permission checks")
def test_remove_cache_path_leaves_symlink_targets_and_directory_modes_alone(tmp_path: Path) -> None:
    """On POSIX a read-only directory blocks removal; cleanup must not chmod its way around it."""
    outside = tmp_path / "base-interpreter"
    outside.write_text("#!/bin/sh\n", encoding="utf-8")
    outside.chmod(0o755)
    bin_dir = tmp_path / "cache-entry" / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").symlink_to(outside)
    bin_dir.chmod(0o555)

    try:
        with pytest.raises(PermissionError):
            _environment._remove_cache_path(tmp_path / "cache-entry")

        assert stat.S_IMODE(outside.stat().st_mode) == 0o755
        assert stat.S_IMODE(bin_dir.stat().st_mode) == 0o555
    finally:
        bin_dir.chmod(0o755)


def test_readonly_handler_clears_readonly_files_on_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(_environment, "_WINDOWS", True)
    readonly = tmp_path / "pack-123.idx"
    readonly.write_bytes(b"pack")
    readonly.chmod(0o444)
    removed: list[str] = []

    _environment._handle_remove_readonly(removed.append, str(readonly), PermissionError(str(readonly)))

    assert removed == [str(readonly)]
    assert os.access(readonly, os.W_OK)


@pytest.mark.skipif(os.name != "posix", reason="creating symbolic links may need privileges on Windows")
def test_readonly_handler_never_follows_symlinks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(_environment, "_WINDOWS", True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"x")
    outside.chmod(0o444)
    link = tmp_path / "link"
    link.symlink_to(outside)
    error = PermissionError(str(link))

    with pytest.raises(PermissionError) as raised:
        _environment._handle_remove_readonly(os.unlink, str(link), error)

    assert raised.value is error
    assert stat.S_IMODE(outside.stat().st_mode) == 0o444


def test_prepare_rejects_regular_file_as_cache_dir(tmp_path: Path) -> None:
    cache_file = tmp_path / "not-a-dir"
    cache_file.write_text("file", encoding="utf-8")
    environment = Environment(source=BundledProject(_bundled_project(tmp_path / "project")))

    with pytest.raises(ValueError, match="must be a directory"):
        prepare(environment, provider=UvProvider(cache_dir=cache_file, uv=sys.executable))


def test_sync_profile_rejects_control_characters_in_python_selector() -> None:
    with pytest.raises(ValueError, match="control characters"):
        SyncProfile(python="3.12\n--evil")


@dataclass(frozen=True)
class _DuckTypedProject:
    """A source that provides only the methods Environment requires."""

    path: Path

    def identity(self) -> dict[str, str]:
        return {"kind": "duck", "path": str(self.path)}

    def provenance(self) -> dict[str, str]:
        return self.identity()

    def materialize(self, destination: Path, *, offline: bool = False) -> None:
        del offline
        shutil.copytree(self.path, destination)


def test_prepare_installs_the_lock_as_is_for_sources_without_a_lock_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    records: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr("physicalai.isolate._environment.subprocess.run", _recording_uv(records))
    environment = Environment(source=_DuckTypedProject(_bundled_project(tmp_path / "project")))  # type: ignore[arg-type]

    prepare(environment, provider=UvProvider(cache_dir=tmp_path / "cache", uv=sys.executable))

    [(command, _)] = records
    assert command[2] == "--frozen"


def test_prepare_accepts_a_cache_directory_behind_a_symbolic_link(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "disk" / "cache"
    target.mkdir(parents=True)
    link = tmp_path / "cache"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")
    monkeypatch.setattr("physicalai.isolate._environment.subprocess.run", _recording_uv([]))
    environment = Environment(source=BundledProject(_bundled_project(tmp_path / "project")))

    prepared = prepare(environment, provider=UvProvider(cache_dir=link, uv=sys.executable))

    assert prepared.path.parent == target.resolve()
