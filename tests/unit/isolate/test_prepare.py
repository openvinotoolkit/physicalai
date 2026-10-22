from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from physicalai.isolate import BundledProject, Environment, SyncProfile, UvProvider, prepare


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
            "--frozen",
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
