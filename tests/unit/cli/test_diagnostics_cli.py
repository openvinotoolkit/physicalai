# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import shlex
from pathlib import Path

import pytest

from physicalai.cli.main import main
from physicalai.diagnostics.models import CollectionError, Device, Endpoint, Inventory


@pytest.fixture
def snapshot(monkeypatch: pytest.MonkeyPatch) -> Inventory:
    inventory = Inventory(platform="linux", devices=[Device(id="arm", name="Serial", endpoints=[
        Endpoint(id="serial", path="/dev/ttyUSB0", kind="serial"),
    ])])

    class Collector:
        def collect(self):
            return inventory

    monkeypatch.setattr("physicalai.diagnostics.collect_inventory", lambda: inventory)
    monkeypatch.setattr("physicalai.diagnostics.platforms.create_collector", Collector)
    return inventory


def test_snapshot_json_is_clean_and_warns_without_failing(snapshot: Inventory, capsys: pytest.CaptureFixture) -> None:
    assert main(["doctor", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema_version"] == 1
    assert data["inventory"]["devices"][0]["id"] == "arm"
    assert main(["doctor", "--strict", "--json"]) == 1
    json.loads(capsys.readouterr().out)


def test_devices_json_and_inspect(snapshot: Inventory, capsys: pytest.CaptureFixture) -> None:
    assert main(["devices", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["platform"] == "linux"
    assert main(["devices", "inspect", "/dev/ttyUSB0", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["selection"]["endpoint"]["kind"] == "serial"
    snapshot.errors.append(CollectionError(source="sysfs", message="Denied", essential=True))
    assert main(["devices", "--json"]) == 3


@pytest.mark.parametrize("args", [
    ["--stream"], ["--stream", "--camera", "/dev/video0"], ["--camera", "/dev/video0"],
    ["--duration", "1"], ["--watch", "--stream"], ["--interval", "nan"],
    ["--watch", "--duration", "-1"],
])
def test_invalid_active_options_fail_before_collecting(args: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("physicalai.diagnostics.platforms.create_collector", lambda: pytest.fail("must validate first"))
    with pytest.raises(SystemExit) as exc:
        main(["doctor", *args])
    assert exc.value.code == 2


def test_discovery_noninteractive_never_collects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("physicalai.diagnostics.collect_inventory", lambda: pytest.fail("must not scan"))
    with pytest.raises(SystemExit) as exc:
        main(["discover", "--output", "runtime.yaml"])
    assert exc.value.code == 2


def test_interactive_camera_export_writes_only_approved_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit.diagnostics.test_discovery import inventory, prompts
    from physicalai.diagnostics.plugins import Catalog
    import yaml

    destination = tmp_path / "chosen-folder/hardware.json"
    ui = prompts(["no", "yes", "wrist", "1", "1", "1", "no", "yes", "yes"])
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("physicalai.diagnostics.collect_inventory", inventory)
    monkeypatch.setattr("physicalai.diagnostics.plugins.load_catalog", Catalog)
    monkeypatch.setattr("physicalai.diagnostics.discovery.Prompts", lambda: ui)
    assert main(["discover", "--components-only", "--output", str(destination)]) == 0
    document = yaml.safe_load(destination.read_text())
    assert document["kind"] == "physicalai.components"
    assert document["cameras"]["wrist"]["init_args"]["device"] == "/dev/v4l/by-path/wrist"
    assert list(destination.parent.iterdir()) == [destination]


def test_interactive_export_preserves_declined_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit.diagnostics.test_discovery import inventory, prompts
    from physicalai.diagnostics.plugins import Catalog

    destination = tmp_path / "hardware.yaml"
    destination.write_text("user work")
    ui = prompts(["no", "yes", "wrist", "1", "1", "1", "no", "yes", "no"])
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("physicalai.diagnostics.collect_inventory", inventory)
    monkeypatch.setattr("physicalai.diagnostics.plugins.load_catalog", Catalog)
    monkeypatch.setattr("physicalai.diagnostics.discovery.Prompts", lambda: ui)
    assert main(["discover", "--components-only", "--output", str(destination)]) == 0
    assert destination.read_text() == "user work"


def test_run_hint_quotes_posix_shell_metacharacters(monkeypatch: pytest.MonkeyPatch) -> None:
    from physicalai.cli.discover import _run_hint
    monkeypatch.setattr("sys.platform", "linux")
    path = Path("/some directory/robot's $(not-a-command).yaml")
    hint = _run_hint(path)
    command = hint.split("run: ", 1)[1]
    assert shlex.split(command) == ["pai", "run", "--config", str(path)]


def test_color_can_be_forced_for_human_output_but_never_leaks_into_json(snapshot: Inventory, capsys: pytest.CaptureFixture) -> None:
    assert main(["doctor", "--color", "always"]) == 0
    assert "\x1b[" in capsys.readouterr().out
    assert main(["doctor", "--color", "always", "--json"]) == 0
    raw = capsys.readouterr().out
    assert "\x1b[" not in raw and json.loads(raw)["schema_version"] == 1
    assert main(["devices", "--color", "always", "--json"]) == 0
    assert "\x1b[" not in capsys.readouterr().out
    assert main(["devices", "--color", "never"]) == 0
    assert "\x1b[" not in capsys.readouterr().out


def test_tree_is_default_and_flat_is_an_explicit_alternative(snapshot: Inventory, capsys: pytest.CaptureFixture) -> None:
    assert main(["devices"]) == 0
    assert "└──" in capsys.readouterr().out
    assert main(["devices", "--flat"]) == 0
    assert "└──" not in capsys.readouterr().out
