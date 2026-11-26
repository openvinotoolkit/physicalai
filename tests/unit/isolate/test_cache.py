# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Environment labels, cache listing, and removal."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from physicalai.isolate import (
    BundledProject,
    Environment,
    PreparedEnvironment,
    SyncProfile,
    UvProvider,
    _environment,
    list_prepared,
    prepare,
    remove_prepared,
)

if TYPE_CHECKING:
    from collections.abc import Generator


def _bundled_project(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "pyproject.toml").write_text("[project]\nname = 'worker'\nversion = '0.1.0'\n", encoding="utf-8")
    (path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    return path


@pytest.fixture
def uv_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Replace ``uv sync`` with a stub that creates the interpreter and records each call."""
    calls: list[list[str]] = []

    def fake_uv(command: list[str], *, cwd: Path, env: dict[str, str], check: bool, shell: bool) -> None:
        del cwd, check, shell
        calls.append(command)
        python = Path(env["UV_PROJECT_ENVIRONMENT"]) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_bytes(b"prepared interpreter")

    monkeypatch.setattr("physicalai.isolate._environment.subprocess.run", fake_uv)
    return calls


def _marker(prepared: PreparedEnvironment) -> dict[str, object]:
    return json.loads((prepared.path / ".physicalai-isolate.json").read_text(encoding="utf-8"))


def test_prepare_records_labels_without_changing_the_key(tmp_path: Path, uv_calls: list[list[str]]) -> None:
    environment = Environment(BundledProject(_bundled_project(tmp_path / "project")), SyncProfile(extras=("cpu",)))
    cache = tmp_path / "cache"
    provider = UvProvider(cache_dir=cache, uv=sys.executable)

    labelled = prepare(environment, provider=provider, labels={"owner": "studio", "framework": "lerobot"})
    reused = prepare(environment, provider=provider)
    source = _marker(labelled)["source"]
    relabelled = prepare(environment, provider=provider, labels={"owner": "runtime"})

    assert len(uv_calls) == 1
    assert labelled.key == reused.key == relabelled.key == environment.key()
    assert labelled.labels == {"owner": "studio", "framework": "lerobot"}
    assert reused.labels == {"owner": "studio", "framework": "lerobot"}
    assert relabelled.labels == {"owner": "runtime"}
    assert _marker(relabelled)["labels"] == {"owner": "runtime"}
    assert _marker(relabelled)["source"] == source
    # Labels are metadata: they neither change equality nor make the environment unhashable.
    assert labelled == relabelled
    assert len({labelled, relabelled}) == 1
    with pytest.raises(TypeError):
        labelled.labels["owner"] = "changed"  # type: ignore[index]  # ruff: ignore[unsupported-assignment]


@pytest.mark.parametrize(
    ("labels", "error"),
    [
        ({"": "x"}, ValueError),
        ({"-owner": "x"}, ValueError),
        ({"owner": 1}, ValueError),
        ({"owner": "x" * 1025}, ValueError),
        ([("owner", "studio")], TypeError),
    ],
)
def test_prepare_rejects_invalid_labels_before_preparing(
    labels: object,
    error: type[Exception],
    tmp_path: Path,
    uv_calls: list[list[str]],
) -> None:
    environment = Environment(BundledProject(_bundled_project(tmp_path / "project")))
    cache = tmp_path / "cache"

    with pytest.raises(error, match="label"):
        prepare(environment, provider=UvProvider(cache_dir=cache, uv=sys.executable), labels=labels)  # type: ignore[arg-type]

    assert uv_calls == []
    assert not cache.exists()


def test_prepare_passes_labels_only_to_callers_that_give_them(tmp_path: Path) -> None:
    """Providers written before labels existed keep working when no labels are given."""
    environment = Environment(BundledProject(_bundled_project(tmp_path / "project")))
    prepared = PreparedEnvironment(path=tmp_path / "env", key="a" * 64, profile=environment.profile)
    calls: list[dict[str, object]] = []

    class ProviderWithoutLabels:
        def prepare(self, environment: Environment, *, offline: bool = False) -> PreparedEnvironment:
            del environment
            calls.append({"offline": offline})
            return prepared

    class ProviderWithLabels:
        def prepare(
            self,
            environment: Environment,
            *,
            offline: bool = False,
            labels: object = None,
        ) -> PreparedEnvironment:
            del environment
            calls.append({"offline": offline, "labels": labels})
            return prepared

    assert prepare(environment, provider=ProviderWithoutLabels(), offline=True) is prepared  # type: ignore[arg-type]
    assert prepare(environment, provider=ProviderWithLabels(), labels={"owner": "studio"}) is prepared  # type: ignore[arg-type]
    assert calls == [{"offline": True}, {"offline": False, "labels": {"owner": "studio"}}]


def test_environments_prepared_before_labels_are_reused_and_can_be_labelled(
    tmp_path: Path,
    uv_calls: list[list[str]],
) -> None:
    environment = Environment(BundledProject(_bundled_project(tmp_path / "project")))
    provider = UvProvider(cache_dir=tmp_path / "cache", uv=sys.executable)
    prepared = prepare(environment, provider=provider)
    marker_path = prepared.path / ".physicalai-isolate.json"
    marker = _marker(prepared)
    del marker["labels"]
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    assert prepare(environment, provider=provider).labels == {}
    assert prepare(environment, provider=provider, labels={"owner": "studio"}).labels == {"owner": "studio"}
    assert len(uv_calls) == 1


def test_list_prepared_reports_only_complete_cache_entries(tmp_path: Path, uv_calls: list[list[str]]) -> None:
    del uv_calls
    cache = tmp_path / "cache"
    assert list_prepared(cache_dir=cache) == []

    provider = UvProvider(cache_dir=cache, uv=sys.executable)
    project = BundledProject(_bundled_project(tmp_path / "project"))
    cpu = prepare(Environment(project, SyncProfile(python="3.12", extras=("cpu",))), provider=provider)
    xpu = prepare(
        Environment(project, SyncProfile(extras=("xpu",), groups=("worker",), no_default_groups=True)),
        provider=provider,
        labels={"owner": "studio"},
    )
    unlabelled = prepare(Environment(project, SyncProfile(extras=("cuda",))), provider=provider)

    def copy_of(prepared: PreparedEnvironment, key: str) -> Path:
        copy = cache / key
        (copy / "project").mkdir(parents=True)
        python = copy / prepared.python.relative_to(prepared.path)
        python.parent.mkdir(parents=True)
        python.write_bytes(b"prepared interpreter")
        marker = _marker(prepared)
        marker["key"] = key
        (copy / ".physicalai-isolate.json").write_text(json.dumps(marker), encoding="utf-8")
        return copy

    # Still being prepared: the marker is written last.
    (cache / ("1" * 64) / "project").mkdir(parents=True)
    # A marker recording another key.
    mismatched = copy_of(cpu, "2" * 64)
    marker = json.loads((mismatched / ".physicalai-isolate.json").read_text(encoding="utf-8"))
    marker["key"] = "3" * 64
    (mismatched / ".physicalai-isolate.json").write_text(json.dumps(marker), encoding="utf-8")
    # Its interpreter is gone.
    broken = copy_of(cpu, "4" * 64)
    (broken / cpu.python.relative_to(cpu.path)).unlink()
    # A profile that SyncProfile rejects.
    bad_profile = copy_of(cpu, "5" * 64)
    marker = json.loads((bad_profile / ".physicalai-isolate.json").read_text(encoding="utf-8"))
    marker["profile"] = {"python": "-bad", "extras": [], "groups": [], "no_default_groups": False}
    (bad_profile / ".physicalai-isolate.json").write_text(json.dumps(marker), encoding="utf-8")
    # Labels that cannot be read are dropped rather than hiding the environment.
    bad_labels = copy_of(xpu, "6" * 64)
    marker = json.loads((bad_labels / ".physicalai-isolate.json").read_text(encoding="utf-8"))
    marker["labels"] = "studio"
    (bad_labels / ".physicalai-isolate.json").write_text(json.dumps(marker), encoding="utf-8")
    # Entries behind symbolic links, and anything not named like a key.
    (cache / ("7" * 64)).symlink_to(cpu.path, target_is_directory=True)
    (cache / "notes").mkdir()

    listed = {environment.key: environment for environment in list_prepared(cache_dir=cache)}

    assert set(listed) == {cpu.key, xpu.key, unlabelled.key, "6" * 64}
    assert [environment.key for environment in list_prepared(cache_dir=cache)] == sorted(listed)
    for prepared in (cpu, xpu, unlabelled):
        assert listed[prepared.key] == prepared
        assert listed[prepared.key].path == prepared.path
        assert listed[prepared.key].profile == prepared.profile
        assert listed[prepared.key].labels == prepared.labels
    assert listed["6" * 64].labels == {}


def test_list_prepared_matches_a_profile_given_in_any_order(tmp_path: Path, uv_calls: list[list[str]]) -> None:
    del uv_calls
    cache = tmp_path / "cache"
    profile = SyncProfile(extras=("xpu", "cpu"), groups=("worker", "dev"))
    prepared = prepare(
        Environment(BundledProject(_bundled_project(tmp_path / "project")), profile),
        provider=UvProvider(cache_dir=cache, uv=sys.executable),
    )

    assert profile == SyncProfile(extras=("cpu", "xpu"), groups=("dev", "worker"))
    assert (profile.extras, profile.groups) == (("cpu", "xpu"), ("dev", "worker"))
    [listed] = list_prepared(cache_dir=cache)
    assert listed == prepared
    assert listed.profile == prepared.profile


def test_list_prepared_reads_the_default_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    uv_calls: list[list[str]],
) -> None:
    del uv_calls
    monkeypatch.setenv("PHYSICALAI_ISOLATE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(UvProvider, "_uv_executable", lambda self: sys.executable)
    prepared = prepare(Environment(BundledProject(_bundled_project(tmp_path / "project"))), labels={"owner": "x"})

    assert list_prepared() == [prepared]
    remove_prepared(prepared)
    assert list_prepared() == []


def test_remove_prepared_deletes_only_entries_of_the_cache(tmp_path: Path, uv_calls: list[list[str]]) -> None:
    del uv_calls
    cache = tmp_path / "cache"
    provider = UvProvider(cache_dir=cache, uv=sys.executable)
    prepared = prepare(Environment(BundledProject(_bundled_project(tmp_path / "project"))), provider=provider)
    outside = tmp_path / "elsewhere" / prepared.key
    outside.mkdir(parents=True)
    profile = prepared.profile

    with pytest.raises(ValueError, match="not an entry of the cache"):
        remove_prepared(PreparedEnvironment(path=outside, key=prepared.key, profile=profile), cache_dir=cache)
    with pytest.raises(ValueError, match="not an entry of the cache"):
        remove_prepared(PreparedEnvironment(path=prepared.path, key="0" * 64, profile=profile), cache_dir=cache)
    with pytest.raises(ValueError, match="not an entry of the cache"):
        remove_prepared(prepared, cache_dir=tmp_path / "other-cache")
    assert outside.is_dir()
    assert prepared.python.is_file()

    remove_prepared(prepared, cache_dir=cache)

    assert not prepared.path.exists()
    assert list_prepared(cache_dir=cache) == []
    remove_prepared(prepared, cache_dir=cache)


def test_remove_prepared_refuses_symbolic_links(tmp_path: Path, uv_calls: list[list[str]]) -> None:
    del uv_calls
    cache = tmp_path / "cache"
    prepared = prepare(
        Environment(BundledProject(_bundled_project(tmp_path / "project"))),
        provider=UvProvider(cache_dir=cache, uv=sys.executable),
    )
    link = cache / ("8" * 64)
    link.symlink_to(prepared.path, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic-link"):
        remove_prepared(PreparedEnvironment(path=link, key=link.name, profile=prepared.profile), cache_dir=cache)

    assert link.is_symlink()
    assert prepared.python.is_file()
    assert not (cache / f".{link.name}.lock").exists()


def test_remove_prepared_hides_an_environment_it_could_not_finish_removing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    uv_calls: list[list[str]],
) -> None:
    """A removal that fails partway, such as on a file in use on Windows, never leaves an entry that looks complete."""
    cache = tmp_path / "cache"
    environment = Environment(BundledProject(_bundled_project(tmp_path / "project")))
    provider = UvProvider(cache_dir=cache, uv=sys.executable)
    prepared = prepare(environment, provider=provider)

    def rmtree_in_use(path: Path, **kwargs: object) -> None:
        del kwargs
        msg = f"In use: {path}"
        raise PermissionError(msg)

    with monkeypatch.context() as patch, pytest.raises(PermissionError):
        patch.setattr(_environment.shutil, "rmtree", rmtree_in_use)
        remove_prepared(prepared, cache_dir=cache)

    assert prepared.python.is_file()
    assert list_prepared(cache_dir=cache) == []
    assert prepare(environment, provider=provider) == prepared
    assert len(uv_calls) == 2
    assert list_prepared(cache_dir=cache) == [prepared]


def test_remove_prepared_holds_the_preparation_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    uv_calls: list[list[str]],
) -> None:
    """Removal waits for, and blocks, preparation of the same key."""
    del uv_calls
    cache = tmp_path / "cache"
    prepared = prepare(
        Environment(BundledProject(_bundled_project(tmp_path / "project"))),
        provider=UvProvider(cache_dir=cache, uv=sys.executable),
    )
    held: list[tuple[Path, bool]] = []
    exclusive = _environment._exclusive

    @contextlib.contextmanager
    def recording_exclusive(path: Path) -> Generator[None, None, None]:
        with exclusive(path):
            yield
            held.append((path, prepared.path.exists()))

    monkeypatch.setattr(_environment, "_exclusive", recording_exclusive)

    remove_prepared(prepared, cache_dir=cache)

    assert held == [(cache.resolve() / f".{prepared.key}.lock", False)]
